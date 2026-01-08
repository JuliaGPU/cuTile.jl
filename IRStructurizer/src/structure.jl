# Control Tree to Structured IR

using AbstractTrees: PreOrderDFS

#=============================================================================
 Helpers
=============================================================================#

"""
    get_loop_blocks(tree::ControlTree, ir::IRCode) -> Set{Int}

Get all block indices contained in a loop control tree.
"""
function get_loop_blocks(tree::ControlTree, ir::IRCode)
    loop_blocks = Set{Int}()
    nblocks = length(ir.cfg.blocks)
    for subtree in PreOrderDFS(tree)
        idx = node_index(subtree)
        if 1 <= idx <= nblocks
            push!(loop_blocks, idx)
        end
    end
    return loop_blocks
end

"""
    convert_phi_value(val) -> IRValue

Convert a phi node value to an IRValue. Most values pass through unchanged;
only QuoteNode needs unwrapping to extract the quoted value.
"""
function convert_phi_value(val)
    val isa QuoteNode ? val.value : val
end

"""
    get_value_type(val, ir::IRCode) -> Type

Get the Julia type of a value that could be SSAValue, SlotNumber, Argument, or a constant.
"""
function get_value_type(val, ir::IRCode)
    if val isa SSAValue
        return ir.stmts.type[val.id]
    elseif val isa SlotNumber
        return ir.argtypes[val.id]
    elseif val isa Argument
        # Argument(n) maps directly to slottypes[n]
        return ir.argtypes[val.n]
    else
        # Constant value
        return typeof(val)
    end
end

#=============================================================================
 Control Tree to Structured IR
=============================================================================#

"""
    control_tree_to_structured_ir(ctree::ControlTree, ir::IRCode, ctx::StructurizationContext) -> Block

Convert a control tree to structured IR entry block.
All loops become LoopOp (no pattern matching yet, no substitutions).
"""
function control_tree_to_structured_ir(ctree::ControlTree, ir::IRCode,
                                       ctx::StructurizationContext)
    return tree_to_block(ctree, ir, ctx)
end

"""
    tree_to_block(tree::ControlTree, ir::IRCode, ctx::StructurizationContext) -> Block

Convert a control tree node to a Block with raw expressions (no substitutions).
"""
function tree_to_block(tree::ControlTree, ir::IRCode, ctx::StructurizationContext)
    idx = node_index(tree)
    rtype = region_type(tree)
    block = Block()

    if rtype == REGION_BLOCK
        handle_block_region!(block, tree, ir, ctx)
    elseif rtype == REGION_IF_THEN_ELSE
        handle_if_then_else!(block, tree, ir, ctx)
    elseif rtype == REGION_IF_THEN
        handle_if_then!(block, tree, ir, ctx)
    elseif rtype == REGION_TERMINATION
        handle_termination!(block, tree, ir, ctx)
    elseif rtype == REGION_FOR_LOOP || rtype == REGION_WHILE_LOOP || rtype == REGION_NATURAL_LOOP
        handle_loop!(block, tree, ir, ctx)
    elseif rtype == REGION_SWITCH
        handle_switch!(block, tree, ir, ctx)
    else
        error("Unknown region type: $rtype")
    end

    # Set terminator if not already set
    set_block_terminator!(block, ir)

    return block
end

#=============================================================================
 Region Handlers
=============================================================================#

"""
    handle_block_region!(block::Block, tree::ControlTree, ir::IRCode, ctx::StructurizationContext)

Handle REGION_BLOCK - a linear sequence of blocks.
"""
function handle_block_region!(block::Block, tree::ControlTree, ir::IRCode,
                              ctx::StructurizationContext)
    nblocks = length(ir.cfg.blocks)
    if isempty(children(tree))
        # Leaf node - collect statements from the block
        idx = node_index(tree)
        if 1 <= idx <= nblocks
            collect_block_statements!(block, idx, ir)
        end
    else
        # Non-leaf - process children in order
        for child in children(tree)
            child_rtype = region_type(child)
            if child_rtype == REGION_BLOCK
                handle_block_region!(block, child, ir, ctx)
            else
                # Nested control flow - create appropriate op
                handle_nested_region!(block, child, ir, ctx)
            end
        end
    end
end

"""
    handle_nested_region!(block::Block, tree::ControlTree, ir::IRCode, ctx::StructurizationContext)

Handle a nested control flow region.
"""
function handle_nested_region!(block::Block, tree::ControlTree, ir::IRCode,
                               ctx::StructurizationContext)
    rtype = region_type(tree)

    if rtype == REGION_IF_THEN_ELSE
        handle_if_then_else!(block, tree, ir, ctx)
    elseif rtype == REGION_IF_THEN
        handle_if_then!(block, tree, ir, ctx)
    elseif rtype == REGION_TERMINATION
        handle_termination!(block, tree, ir, ctx)
    elseif rtype == REGION_FOR_LOOP || rtype == REGION_WHILE_LOOP || rtype == REGION_NATURAL_LOOP
        handle_loop!(block, tree, ir, ctx)
    elseif rtype == REGION_SWITCH
        handle_switch!(block, tree, ir, ctx)
    else
        error("Unknown region type in nested region: $rtype")
    end
end

"""
    handle_if_then_else!(block::Block, tree::ControlTree, ir::IRCode, ctx::StructurizationContext)

Handle REGION_IF_THEN_ELSE.
"""
function handle_if_then_else!(block::Block, tree::ControlTree, ir::IRCode,
                              ctx::StructurizationContext)
    tree_children = children(tree)
    length(tree_children) >= 3 || return handle_block_region!(block, tree, ir, ctx)

    nblocks = length(ir.cfg.blocks)

    # First child is the condition block
    cond_tree = tree_children[1]
    cond_idx = node_index(cond_tree)

    # Find the GotoIfNot's SSA index for keying (fallback if no merge phi)
    gotoifnot_idx = nothing
    if 1 <= cond_idx <= nblocks
        bb = ir.cfg.blocks[cond_idx]
        for si in first(bb.stmts):last(bb.stmts)
            stmt = ir.stmts.stmt[si]
            if stmt isa GotoIfNot
                gotoifnot_idx = si
            end
            if !(stmt isa GotoNode || stmt isa GotoIfNot || stmt isa ReturnNode || stmt isa PhiNode)
                push!(block, si, stmt, ir.stmts.type[si])
            end
        end
    end

    cond_value = find_condition_value(cond_idx, ir)

    # Then and else blocks
    then_tree = tree_children[2]
    else_tree = tree_children[3]

    then_blk = tree_to_block(then_tree, ir, ctx)
    else_blk = tree_to_block(else_tree, ir, ctx)

    # Find merge block and detect merge phis
    then_block_idx = node_index(then_tree)
    else_block_idx = node_index(else_tree)
    merge_phis = find_merge_phis(ir, then_block_idx, else_block_idx)

    # Add YieldOp terminators with phi values
    if !isempty(merge_phis)
        then_values = [phi.then_val for phi in merge_phis]
        else_values = [phi.else_val for phi in merge_phis]
        then_blk.terminator = YieldOp(then_values)
        else_blk.terminator = YieldOp(else_values)
    end

    if_op = IfOp(cond_value, then_blk, else_blk)

    # Allocate synthesized SSA index for IfOp
    if_result_idx = ctx.next_ssa_idx
    ctx.next_ssa_idx += 1

    # Always use Tuple type for IfOp results (uniform handling in codegen)
    if !isempty(merge_phis)
        phi_types = [ctx.ssavaluetypes[phi.ssa_idx] for phi in merge_phis]
        result_type = Tuple{phi_types...}
        push!(block, if_result_idx, if_op, result_type)

        # Generate getfield statements at original phi indices
        # This preserves SSA reference semantics: `return %7` still works because getfield is at %7
        for (i, phi) in enumerate(merge_phis)
            getfield_expr = Expr(:call, Core.getfield, SSAValue(if_result_idx), i)
            push!(block, phi.ssa_idx, getfield_expr, phi_types[i])
        end
    else
        # No results - still use Tuple{} for uniformity
        result_type = Tuple{}
        push!(block, if_result_idx, if_op, result_type)
    end
end

"""
    find_merge_phis(ir, then_block_idx, else_block_idx)

Find phis in the merge block (common successor of then and else blocks)
that receive values from both branches.

Returns a vector of NamedTuples: (ssa_idx, then_val, else_val)
"""
function find_merge_phis(ir::IRCode, then_block_idx::Int, else_block_idx::Int)
    merge_phis = NamedTuple{(:ssa_idx, :then_val, :else_val), Tuple{Int, Any, Any}}[]
    nblocks = length(ir.cfg.blocks)

    # Find common successor (merge block)
    then_succs = 1 <= then_block_idx <= nblocks ? ir.cfg.blocks[then_block_idx].succs : Int[]
    else_succs = 1 <= else_block_idx <= nblocks ? ir.cfg.blocks[else_block_idx].succs : Int[]
    merge_blocks = intersect(then_succs, else_succs)
    isempty(merge_blocks) && return merge_phis

    merge_block_idx = first(merge_blocks)
    1 <= merge_block_idx <= nblocks || return merge_phis
    merge_bb = ir.cfg.blocks[merge_block_idx]

    # Look for phis that have edges from both then and else blocks
    # In IRCode, phi edges are BLOCK indices directly
    for si in first(merge_bb.stmts):last(merge_bb.stmts)
        stmt = ir.stmts.stmt[si]
        stmt isa PhiNode || continue

        then_val = nothing
        else_val = nothing
        for (edge_idx, edge) in enumerate(stmt.edges)
            # edge is a block index in IRCode
            if edge == then_block_idx
                then_val = stmt.values[edge_idx]
            elseif edge == else_block_idx
                else_val = stmt.values[edge_idx]
            end
        end

        # Only include if we have values from both branches
        if then_val !== nothing && else_val !== nothing
            push!(merge_phis, (ssa_idx=si, then_val=then_val, else_val=else_val))
        end
    end

    return merge_phis
end

"""
    handle_if_then!(block::Block, tree::ControlTree, ir::IRCode, ctx::StructurizationContext)

Handle REGION_IF_THEN.
"""
function handle_if_then!(block::Block, tree::ControlTree, ir::IRCode,
                         ctx::StructurizationContext)
    tree_children = children(tree)
    length(tree_children) >= 2 || return handle_block_region!(block, tree, ir, ctx)

    nblocks = length(ir.cfg.blocks)

    # First child is the condition block
    cond_tree = tree_children[1]
    cond_idx = node_index(cond_tree)

    # Find the GotoIfNot's SSA index for keying
    gotoifnot_idx = nothing
    if 1 <= cond_idx <= nblocks
        bb = ir.cfg.blocks[cond_idx]
        for si in first(bb.stmts):last(bb.stmts)
            stmt = ir.stmts.stmt[si]
            if stmt isa GotoIfNot
                gotoifnot_idx = si
            end
            if !(stmt isa GotoNode || stmt isa GotoIfNot || stmt isa ReturnNode || stmt isa PhiNode)
                push!(block, si, stmt, ir.stmts.type[si])
            end
        end
    end

    cond_value = find_condition_value(cond_idx, ir)

    # Then block
    then_tree = tree_children[2]
    then_blk = tree_to_block(then_tree, ir, ctx)

    # Empty else block
    else_blk = Block()

    if_op = IfOp(cond_value, then_blk, else_blk)
    bb = ir.cfg.blocks[cond_idx]
    result_idx = gotoifnot_idx !== nothing ? gotoifnot_idx : last(bb.stmts)
    push!(block, result_idx, if_op, Nothing)
end

"""
    handle_termination!(block::Block, tree::ControlTree, ir::IRCode, ctx::StructurizationContext)

Handle REGION_TERMINATION - branches where some paths terminate.
"""
function handle_termination!(block::Block, tree::ControlTree, ir::IRCode,
                             ctx::StructurizationContext)
    tree_children = children(tree)
    isempty(tree_children) && return handle_block_region!(block, tree, ir, ctx)

    nblocks = length(ir.cfg.blocks)

    # First child is the condition block
    cond_tree = tree_children[1]
    cond_idx = node_index(cond_tree)

    # Find the GotoIfNot's SSA index for keying
    gotoifnot_idx = nothing
    if 1 <= cond_idx <= nblocks
        bb = ir.cfg.blocks[cond_idx]
        for si in first(bb.stmts):last(bb.stmts)
            stmt = ir.stmts.stmt[si]
            if stmt isa GotoIfNot
                gotoifnot_idx = si
            end
            if !(stmt isa GotoNode || stmt isa GotoIfNot || stmt isa ReturnNode || stmt isa PhiNode)
                push!(block, si, stmt, ir.stmts.type[si])
            end
        end
    end

    cond_value = find_condition_value(cond_idx, ir)
    bb = ir.cfg.blocks[cond_idx]
    result_idx = gotoifnot_idx !== nothing ? gotoifnot_idx : last(bb.stmts)

    # Build then and else blocks from remaining children
    if length(tree_children) >= 3
        then_tree = tree_children[2]
        else_tree = tree_children[3]
        then_blk = tree_to_block(then_tree, ir, ctx)
        else_blk = tree_to_block(else_tree, ir, ctx)
        if_op = IfOp(cond_value, then_blk, else_blk)
        push!(block, result_idx, if_op, Nothing)
    elseif length(tree_children) == 2
        then_tree = tree_children[2]
        then_blk = tree_to_block(then_tree, ir, ctx)
        else_blk = Block()
        if_op = IfOp(cond_value, then_blk, else_blk)
        push!(block, result_idx, if_op, Nothing)
    end
end

"""
    handle_loop!(block::Block, tree::ControlTree, ir::IRCode, ctx::StructurizationContext)

Handle REGION_FOR_LOOP, REGION_WHILE_LOOP, and REGION_NATURAL_LOOP.

For REGION_FOR_LOOP: Creates ForOp directly using metadata from CFG analysis.
For REGION_WHILE_LOOP: Creates WhileOp directly with before/after regions.
For REGION_NATURAL_LOOP: Creates LoopOp with internal IfOp (fallback for complex loops).

The loop is keyed at a synthesized SSA index, and getfield statements are generated
at the original phi node indices. This ensures that references like `return %2`
continue to work because getfield is placed at %2.
"""
function handle_loop!(block::Block, tree::ControlTree, ir::IRCode,
                      ctx::StructurizationContext)
    rtype = region_type(tree)

    # Dispatch based on region type
    if rtype == REGION_FOR_LOOP
        for_op, phi_indices, phi_types, is_inclusive, original_upper = build_for_op(tree, ir, ctx)

        # Handle inclusive bounds (Julia's for i in 1:n) by inserting add_int(upper, 1)
        if is_inclusive
            # Allocate SSA for adjusted upper bound
            adj_ssa_idx = ctx.next_ssa_idx
            ctx.next_ssa_idx += 1

            # Insert add_int expression before the loop
            adj_upper = convert_phi_value(original_upper)
            upper_type = get_value_type(original_upper, ir)
            add_int_expr = Expr(:call, GlobalRef(Base, :add_int), adj_upper, one(upper_type))
            push!(block, adj_ssa_idx, add_int_expr, upper_type)

            # Update ForOp's upper bound to the adjusted value
            for_op.upper = SSAValue(adj_ssa_idx)
        end

        loop_op = for_op
    elseif rtype == REGION_WHILE_LOOP
        loop_op, phi_indices, phi_types = build_while_op(tree, ir, ctx)
    else  # REGION_NATURAL_LOOP or other cyclic regions
        loop_op, phi_indices, phi_types = build_loop_op(tree, ir, ctx)
    end

    # Allocate new SSA index for loop's tuple result
    loop_result_idx = ctx.next_ssa_idx
    ctx.next_ssa_idx += 1

    # Always use Tuple type for loop results (uniform handling in codegen)
    # Empty phi_indices produces Tuple{} which is fine
    result_type = Tuple{phi_types...}

    # Push loop op at synthesized index
    push!(block, loop_result_idx, loop_op, result_type)

    # Generate getfield statements at original phi indices
    # This preserves SSA reference semantics: `return %2` still works because getfield is at %2
    for (i, (phi_idx, phi_type)) in enumerate(zip(phi_indices, phi_types))
        getfield_expr = Expr(:call, Core.getfield, SSAValue(loop_result_idx), i)
        push!(block, phi_idx, getfield_expr, phi_type)
    end
end

"""
    handle_switch!(block::Block, tree::ControlTree, ir::IRCode, ctx::StructurizationContext)

Handle REGION_SWITCH by treating it as a nested if-else chain.
"""
function handle_switch!(block::Block, tree::ControlTree, ir::IRCode,
                        ctx::StructurizationContext)
    # For now, handle as a nested if-else chain
    # TODO: Implement proper switch handling if needed
    handle_block_region!(block, tree, ir, ctx)
end

#=============================================================================
 Statement Collection Helpers
=============================================================================#

"""
    collect_block_statements!(block::Block, block_idx::Int, ir::IRCode)

Collect statements from a basic block into a Block, excluding control flow.
Stores raw expressions (no substitutions) with their SSA indices.
"""
function collect_block_statements!(block::Block, block_idx::Int, ir::IRCode)
    stmts = ir.stmts.stmt
    types = ir.stmts.type
    bb = ir.cfg.blocks[block_idx]
    for si in first(bb.stmts):last(bb.stmts)
        stmt = stmts[si]
        if stmt isa ReturnNode
            block.terminator = stmt
        elseif !(stmt isa GotoNode || stmt isa GotoIfNot || stmt isa PhiNode)
            push!(block, si, stmt, types[si])
        end
    end
end

"""
    find_condition_value(block_idx::Int, ir::IRCode) -> IRValue

Find the condition value for a GotoIfNot in the given block.
"""
function find_condition_value(block_idx::Int, ir::IRCode)
    nblocks = length(ir.cfg.blocks)
    @assert 1 <= block_idx <= nblocks "Invalid block index: $block_idx"

    bb = ir.cfg.blocks[block_idx]
    for si in first(bb.stmts):last(bb.stmts)
        stmt = ir.stmts.stmt[si]
        if stmt isa GotoIfNot
            cond = stmt.cond
            @assert cond isa SSAValue || cond isa SlotNumber || cond isa Argument "Unexpected condition type: $(typeof(cond))"
            return cond
        end
    end

    error("No GotoIfNot found in block $block_idx")
end

"""
    find_condition_chain(stmts, header_range, cond_ssa::SSAValue) -> Set{Int}

Walk backwards from the condition SSA to find all SSA indices in the header
that contribute to computing the condition. These should be excluded from
the ForOp body since they're part of the loop control, not the loop body.
"""
function find_condition_chain(stmts, header_range, cond_ssa::SSAValue)
    chain = Set{Int}()
    worklist = [cond_ssa.id]
    while !isempty(worklist)
        idx = popfirst!(worklist)
        idx in header_range || continue
        idx in chain && continue
        push!(chain, idx)
        # Add operands that are SSAValues in header
        stmt = stmts[idx]
        if stmt isa Expr
            for arg in stmt.args
                if arg isa SSAValue && arg.id in header_range
                    push!(worklist, arg.id)
                end
            end
        end
    end
    return chain
end

"""
    set_block_terminator!(block::Block, ir::IRCode)

Set the block terminator based on statements.
"""
function set_block_terminator!(block::Block, ir::IRCode)
    block.terminator !== nothing && return

    last_idx = nothing
    for (idx, entry) in block.body
        if !(entry.stmt isa ControlFlowOp)
            if last_idx === nothing || idx > last_idx
                last_idx = idx
            end
        end
    end
    if last_idx !== nothing && last_idx < length(ir.stmts.stmt)
        next_stmt = ir.stmts.stmt[last_idx + 1]
        if next_stmt isa ReturnNode
            block.terminator = next_stmt
        end
    end
end

#=============================================================================
 Loop Construction
=============================================================================#

"""
    build_while_op(tree::ControlTree, ir::IRCode, ctx::StructurizationContext)
        -> Tuple{WhileOp, Vector{Int}, Vector{Any}}

Build a WhileOp from a REGION_WHILE_LOOP control tree.
Returns (while_op, phi_indices, phi_types) where:
- while_op: The constructed WhileOp with before/after regions
- phi_indices: SSA indices of the header phi nodes (for getfield generation)
- phi_types: Julia types of the header phi nodes

The WhileOp structure:
- before: header statements + ConditionOp(condition, carried_args)
- after: body statements + YieldOp(carried_values)
"""
function build_while_op(tree::ControlTree, ir::IRCode, ctx::StructurizationContext)
    stmts = ir.stmts.stmt
    types = ir.stmts.type
    header_idx = node_index(tree)
    loop_blocks = get_loop_blocks(tree, ir)
    nblocks = length(ir.cfg.blocks)

    @assert 1 <= header_idx <= nblocks "Invalid header_idx from control tree: $header_idx"
    header_bb = ir.cfg.blocks[header_idx]
    header_range = first(header_bb.stmts):last(header_bb.stmts)

    # Extract phi node information: init_values (from outside loop) and carried_values (from inside loop)
    init_values = IRValue[]
    carried_values = IRValue[]
    phi_indices = Int[]
    phi_types = Any[]

    for si in header_range
        stmt = stmts[si]
        if stmt isa PhiNode
            push!(phi_indices, si)
            push!(phi_types, types[si])
            phi = stmt

            entry_val = nothing
            carried_val = nothing

            for (edge_idx, edge) in enumerate(phi.edges)
                if isassigned(phi.values, edge_idx)
                    val = phi.values[edge_idx]
                    # In IRCode, edge IS the block index directly
                    if edge ∈ loop_blocks
                        carried_val = val
                    else
                        entry_val = convert_phi_value(val)
                    end
                end
            end

            entry_val !== nothing && push!(init_values, entry_val)
            carried_val !== nothing && push!(carried_values, carried_val)
        end
    end

    # Find the condition for loop exit
    condition = nothing
    for si in header_range
        stmt = stmts[si]
        if stmt isa GotoIfNot
            condition = stmt.cond
            break
        end
    end

    # Build "before" region: header statements + ConditionOp
    before = Block()
    for si in header_range
        stmt = stmts[si]
        if !(stmt isa PhiNode || stmt isa GotoNode || stmt isa GotoIfNot || stmt isa ReturnNode)
            push!(before, si, stmt, types[si])
        end
    end

    # ConditionOp terminates the before block
    # The condition args are SSAValues of the phi nodes (will be substituted to BlockArgs later)
    condition_args = IRValue[SSAValue(idx) for idx in phi_indices]
    cond_value = condition !== nothing ? convert_phi_value(condition) : true
    before.terminator = ConditionOp(cond_value, condition_args)

    # Build "after" region: body statements + YieldOp
    after = Block()

    # Process loop body blocks (excluding header)
    for child in children(tree)
        child_idx = node_index(child)
        if child_idx != header_idx
            handle_block_region!(after, child, ir, ctx)
        end
    end

    # YieldOp carries values back to the loop header
    after.terminator = YieldOp(copy(carried_values))

    # Create BlockArgs and apply substitutions immediately
    subs = Substitutions()
    for (i, (phi_idx, phi_type)) in enumerate(zip(phi_indices, phi_types))
        arg = BlockArg(i, phi_type)
        push!(before.args, arg)
        push!(after.args, BlockArg(i, phi_type))  # Matching BlockArg for after region
        subs[phi_idx] = arg
    end
    apply_substitutions!(before, subs)
    apply_substitutions!(after, subs)

    while_op = WhileOp(before, after, init_values)
    return while_op, phi_indices, phi_types
end

"""
    build_loop_op(tree::ControlTree, ir::IRCode, ctx::StructurizationContext)
        -> Tuple{LoopOp, Vector{Int}, Vector{Any}}

Build a LoopOp from a control tree and return phi node information.
Returns (loop_op, phi_indices, phi_types) where:
- loop_op: The constructed LoopOp
- phi_indices: SSA indices of the header phi nodes (for getfield generation)
- phi_types: Julia types of the header phi nodes
"""
function build_loop_op(tree::ControlTree, ir::IRCode, ctx::StructurizationContext)
    stmts = ir.stmts.stmt
    types = ir.stmts.type
    header_idx = node_index(tree)
    loop_blocks = get_loop_blocks(tree, ir)
    nblocks = length(ir.cfg.blocks)

    @assert 1 <= header_idx <= nblocks "Invalid header_idx from control tree: $header_idx"
    header_bb = ir.cfg.blocks[header_idx]
    header_range = first(header_bb.stmts):last(header_bb.stmts)

    init_values = IRValue[]
    carried_values = IRValue[]
    phi_indices = Int[]
    phi_types = Any[]

    for si in header_range
        stmt = stmts[si]
        if stmt isa PhiNode
            push!(phi_indices, si)
            push!(phi_types, types[si])
            phi = stmt

            entry_val = nothing
            carried_val = nothing

            for (edge_idx, edge) in enumerate(phi.edges)
                if isassigned(phi.values, edge_idx)
                    val = phi.values[edge_idx]
                    # In IRCode, edge IS the block index directly
                    if edge ∈ loop_blocks
                        carried_val = val
                    else
                        entry_val = convert_phi_value(val)
                    end
                end
            end

            entry_val !== nothing && push!(init_values, entry_val)
            carried_val !== nothing && push!(carried_values, carried_val)
        end
    end

    body = Block()

    # Find the condition for loop exit and its SSA index
    condition = nothing
    condition_idx = nothing
    for si in header_range
        stmt = stmts[si]
        if stmt isa GotoIfNot
            condition = stmt.cond
            condition_idx = si
            break
        end
    end

    # Collect header statements (excluding phi nodes and control flow)
    for si in header_range
        stmt = stmts[si]
        if !(stmt isa PhiNode || stmt isa GotoNode || stmt isa GotoIfNot || stmt isa ReturnNode)
            push!(body, si, stmt, types[si])
        end
    end

    # Create the conditional structure inside the loop body
    if condition !== nothing
        cond_value = convert_phi_value(condition)

        then_blk = Block()

        # Process loop body blocks (excluding header)
        for child in children(tree)
            child_idx = node_index(child)
            if child_idx != header_idx
                handle_block_region!(then_blk, child, ir, ctx)
            end
        end
        then_blk.terminator = ContinueOp(copy(carried_values))

        else_blk = Block()
        else_blk.terminator = BreakOp(IRValue[SSAValue(idx) for idx in phi_indices])

        if_op = IfOp(cond_value, then_blk, else_blk)
        push!(body, condition_idx, if_op, Nothing)
    else
        # No condition - process children directly
        for child in children(tree)
            child_idx = node_index(child)
            if child_idx != header_idx
                handle_block_region!(body, child, ir, ctx)
            end
        end
        body.terminator = ContinueOp(copy(carried_values))
    end

    # Create BlockArgs and apply substitutions immediately
    subs = Substitutions()
    for (i, (phi_idx, phi_type)) in enumerate(zip(phi_indices, phi_types))
        arg = BlockArg(i, phi_type)
        push!(body.args, arg)
        subs[phi_idx] = arg
    end
    apply_substitutions!(body, subs)

    # Create loop op with init_values
    loop_op = LoopOp(body, init_values)
    return loop_op, phi_indices, phi_types
end

"""
    build_for_op(tree::ControlTree, ir::IRCode, ctx::StructurizationContext)
        -> Tuple{ForOp, Vector{Int}, Vector{Any}}

Build a ForOp directly from a REGION_FOR_LOOP control tree using metadata from CFG analysis.
Returns (for_op, phi_indices, phi_types) where:
- for_op: The constructed ForOp with bounds, step, IV, and body
- phi_indices: SSA indices of the non-IV phi nodes (for getfield generation)
- phi_types: Julia types of the non-IV phi nodes

The ForOp structure:
- lower: Lower bound from ForLoopInfo
- upper: Upper bound (adjusted +1 if is_inclusive)
- step: Step value from ForLoopInfo
- iv_arg: BlockArg for the induction variable
- body: Loop body statements + ContinueOp with carried values
- init_values: Non-IV loop-carried values
"""
function build_for_op(tree::ControlTree, ir::IRCode, ctx::StructurizationContext)
    stmts = ir.stmts.stmt
    types = ir.stmts.type
    header_idx = node_index(tree)
    loop_blocks = get_loop_blocks(tree, ir)
    for_info = metadata(tree)::ForLoopInfo
    nblocks = length(ir.cfg.blocks)

    @assert 1 <= header_idx <= nblocks "Invalid header_idx from control tree: $header_idx"
    header_bb = ir.cfg.blocks[header_idx]
    header_range = first(header_bb.stmts):last(header_bb.stmts)

    # Extract phi info: separate IV from other loop-carried variables
    iv_phi_idx = for_info.iv_phi_idx
    all_phi_indices = Int[]
    all_phi_types = Any[]
    init_values = IRValue[]
    carried_values = IRValue[]

    for si in header_range
        stmt = stmts[si]
        if stmt isa PhiNode
            push!(all_phi_indices, si)
            push!(all_phi_types, types[si])
            phi = stmt

            entry_val = nothing
            carried_val = nothing

            for (edge_idx, edge) in enumerate(phi.edges)
                if isassigned(phi.values, edge_idx)
                    val = phi.values[edge_idx]
                    # In IRCode, edge IS the block index directly
                    if edge ∈ loop_blocks
                        carried_val = val
                    else
                        entry_val = convert_phi_value(val)
                    end
                end
            end

            # Skip IV for init_values and carried_values, handle separately
            if si != iv_phi_idx
                entry_val !== nothing && push!(init_values, entry_val)
                carried_val !== nothing && push!(carried_values, carried_val)
            end
        end
    end

    # Build result phi_indices and phi_types (excluding IV)
    phi_indices = Int[]
    phi_types = Any[]
    for (idx, typ) in zip(all_phi_indices, all_phi_types)
        if idx != iv_phi_idx
            push!(phi_indices, idx)
            push!(phi_types, typ)
        end
    end

    # Get IV type
    iv_type = types[iv_phi_idx]

    # Create BlockArg for IV (id=1, first in ForOp's block args)
    iv_arg = BlockArg(1, iv_type)

    # Build the body block
    body = Block()

    # Find the condition SSA from GotoIfNot and compute the condition chain to exclude
    cond_ssa = nothing
    for si in header_range
        stmt = stmts[si]
        if stmt isa GotoIfNot && stmt.cond isa SSAValue
            cond_ssa = stmt.cond
            break
        end
    end
    excluded = cond_ssa !== nothing ?
        find_condition_chain(stmts, header_range, cond_ssa) : Set{Int}()

    # Collect header statements (excluding phi nodes, control flow, and condition chain)
    for si in header_range
        stmt = stmts[si]
        if si ∉ excluded && !(stmt isa PhiNode || stmt isa GotoNode || stmt isa GotoIfNot || stmt isa ReturnNode)
            push!(body, si, stmt, types[si])
        end
    end

    # Process loop body blocks (excluding header)
    for child in children(tree)
        child_idx = node_index(child)
        if child_idx != header_idx
            handle_block_region!(body, child, ir, ctx)
        end
    end

    # ContinueOp with non-IV carried values
    body.terminator = ContinueOp(copy(carried_values))

    # Build ForOp with bounds from ForLoopInfo
    lower = convert_phi_value(for_info.lower)
    upper = convert_phi_value(for_info.upper)
    step = convert_phi_value(for_info.step)

    # If is_inclusive (Julia's 1:n), we need upper+1 for exclusive semantics
    # This is handled by inserting add_int in the parent block during handle_loop!
    if for_info.is_inclusive
        # Allocate a synthesized SSA for the adjusted upper bound
        adj_ssa_idx = ctx.next_ssa_idx
        ctx.next_ssa_idx += 1
        upper = SSAValue(adj_ssa_idx)
    end

    # Create BlockArgs and apply substitutions immediately
    subs = Substitutions()
    subs[iv_phi_idx] = iv_arg  # IV at index 1

    # Non-IV loop-carried values at indices 2, 3, ...
    for (i, (phi_idx, phi_type)) in enumerate(zip(phi_indices, phi_types))
        arg = BlockArg(i + 1, phi_type)
        push!(body.args, arg)
        subs[phi_idx] = arg
    end

    apply_substitutions!(body, subs)

    for_op = ForOp(lower, for_info.is_inclusive ? for_info.upper : upper, step, iv_arg, body, init_values)

    return for_op, phi_indices, phi_types, for_info.is_inclusive, for_info.upper
end
