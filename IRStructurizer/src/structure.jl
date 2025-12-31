# Control Tree to Structured IR

using AbstractTrees: PreOrderDFS

#=============================================================================
 Helpers
=============================================================================#

"""
    get_loop_blocks(tree::ControlTree, blocks::Vector{BlockInfo}) -> Set{Int}

Get all block indices contained in a loop control tree.
"""
function get_loop_blocks(tree::ControlTree, blocks::Vector{BlockInfo})
    loop_blocks = Set{Int}()
    for subtree in PreOrderDFS(tree)
        idx = node_index(subtree)
        if 1 <= idx <= length(blocks)
            push!(loop_blocks, idx)
        end
    end
    return loop_blocks
end

"""
    convert_phi_value(val) -> IRValue

Convert a phi node value to an IRValue.
"""
function convert_phi_value(val)
    if val isa SSAValue
        return val
    elseif val isa Argument
        return val
    elseif val isa Integer
        return val
    elseif val isa QuoteNode
        return val.value
    else
        error("Unexpected phi value type: $(typeof(val))")
    end
end

#=============================================================================
 Control Tree to Structured IR
=============================================================================#

"""
    control_tree_to_structured_ir(ctree::ControlTree, code::CodeInfo, blocks::Vector{BlockInfo}, ctx::StructurizationContext) -> Block

Convert a control tree to structured IR entry block.
All loops become LoopOp (no pattern matching yet, no substitutions).
"""
function control_tree_to_structured_ir(ctree::ControlTree, code::CodeInfo, blocks::Vector{BlockInfo},
                                       ctx::StructurizationContext)
    return tree_to_block(ctree, code, blocks, ctx)
end

"""
    tree_to_block(tree::ControlTree, code::CodeInfo, blocks::Vector{BlockInfo}, ctx::StructurizationContext) -> Block

Convert a control tree node to a Block with raw expressions (no substitutions).
"""
function tree_to_block(tree::ControlTree, code::CodeInfo, blocks::Vector{BlockInfo},
                       ctx::StructurizationContext)
    idx = node_index(tree)
    rtype = region_type(tree)
    block = Block()

    if rtype == REGION_BLOCK
        handle_block_region!(block, tree, code, blocks, ctx)
    elseif rtype == REGION_IF_THEN_ELSE
        handle_if_then_else!(block, tree, code, blocks, ctx)
    elseif rtype == REGION_IF_THEN
        handle_if_then!(block, tree, code, blocks, ctx)
    elseif rtype == REGION_TERMINATION
        handle_termination!(block, tree, code, blocks, ctx)
    elseif rtype == REGION_FOR_LOOP || rtype == REGION_WHILE_LOOP || rtype == REGION_NATURAL_LOOP
        handle_loop!(block, tree, code, blocks, ctx)
    elseif rtype == REGION_SWITCH
        handle_switch!(block, tree, code, blocks, ctx)
    else
        error("Unknown region type: $rtype")
    end

    # Set terminator if not already set
    set_block_terminator!(block, code, blocks)

    return block
end

#=============================================================================
 Region Handlers
=============================================================================#

"""
    handle_block_region!(block::Block, tree::ControlTree, code::CodeInfo, blocks::Vector{BlockInfo}, ctx::StructurizationContext)

Handle REGION_BLOCK - a linear sequence of blocks.
"""
function handle_block_region!(block::Block, tree::ControlTree, code::CodeInfo, blocks::Vector{BlockInfo},
                              ctx::StructurizationContext)
    if isempty(children(tree))
        # Leaf node - collect statements from the block
        idx = node_index(tree)
        if 1 <= idx <= length(blocks)
            collect_block_statements!(block, blocks[idx], code)
        end
    else
        # Non-leaf - process children in order
        for child in children(tree)
            child_rtype = region_type(child)
            if child_rtype == REGION_BLOCK
                handle_block_region!(block, child, code, blocks, ctx)
            else
                # Nested control flow - create appropriate op
                handle_nested_region!(block, child, code, blocks, ctx)
            end
        end
    end
end

"""
    handle_nested_region!(block::Block, tree::ControlTree, code::CodeInfo, blocks::Vector{BlockInfo}, ctx::StructurizationContext)

Handle a nested control flow region.
"""
function handle_nested_region!(block::Block, tree::ControlTree, code::CodeInfo, blocks::Vector{BlockInfo},
                               ctx::StructurizationContext)
    rtype = region_type(tree)

    if rtype == REGION_IF_THEN_ELSE
        handle_if_then_else!(block, tree, code, blocks, ctx)
    elseif rtype == REGION_IF_THEN
        handle_if_then!(block, tree, code, blocks, ctx)
    elseif rtype == REGION_TERMINATION
        handle_termination!(block, tree, code, blocks, ctx)
    elseif rtype == REGION_FOR_LOOP || rtype == REGION_WHILE_LOOP || rtype == REGION_NATURAL_LOOP
        handle_loop!(block, tree, code, blocks, ctx)
    elseif rtype == REGION_SWITCH
        handle_switch!(block, tree, code, blocks, ctx)
    else
        error("Unknown region type in nested region: $rtype")
    end
end

"""
    handle_if_then_else!(block::Block, tree::ControlTree, code::CodeInfo, blocks::Vector{BlockInfo}, ctx::StructurizationContext)

Handle REGION_IF_THEN_ELSE.
"""
function handle_if_then_else!(block::Block, tree::ControlTree, code::CodeInfo, blocks::Vector{BlockInfo},
                              ctx::StructurizationContext)
    tree_children = children(tree)
    length(tree_children) >= 3 || return handle_block_region!(block, tree, code, blocks, ctx)

    # First child is the condition block
    cond_tree = tree_children[1]
    cond_idx = node_index(cond_tree)

    # Find the GotoIfNot's SSA index for keying (fallback if no merge phi)
    gotoifnot_idx = nothing
    if 1 <= cond_idx <= length(blocks)
        cond_block = blocks[cond_idx]
        for si in cond_block.range
            stmt = code.code[si]
            if stmt isa GotoIfNot
                gotoifnot_idx = si
            end
            if !(stmt isa GotoNode || stmt isa GotoIfNot || stmt isa ReturnNode || stmt isa PhiNode)
                push!(block, si, stmt, code.ssavaluetypes[si])
            end
        end
    end

    cond_value = find_condition_value(cond_idx, code, blocks)

    # Then and else blocks
    then_tree = tree_children[2]
    else_tree = tree_children[3]

    then_blk = tree_to_block(then_tree, code, blocks, ctx)
    else_blk = tree_to_block(else_tree, code, blocks, ctx)

    # Find merge block and detect merge phis
    then_block_idx = node_index(then_tree)
    else_block_idx = node_index(else_tree)
    merge_phis = find_merge_phis(code, blocks, then_block_idx, else_block_idx)

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
    find_merge_phis(code, blocks, then_block_idx, else_block_idx)

Find phis in the merge block (common successor of then and else blocks)
that receive values from both branches.

Returns a vector of NamedTuples: (ssa_idx, then_val, else_val)
"""
function find_merge_phis(code::CodeInfo, blocks::Vector{BlockInfo},
                         then_block_idx::Int, else_block_idx::Int)
    merge_phis = NamedTuple{(:ssa_idx, :then_val, :else_val), Tuple{Int, Any, Any}}[]

    # Find common successor (merge block)
    then_succs = 1 <= then_block_idx <= length(blocks) ? blocks[then_block_idx].succs : Int[]
    else_succs = 1 <= else_block_idx <= length(blocks) ? blocks[else_block_idx].succs : Int[]
    merge_blocks = intersect(then_succs, else_succs)
    isempty(merge_blocks) && return merge_phis

    merge_block_idx = first(merge_blocks)
    1 <= merge_block_idx <= length(blocks) || return merge_phis
    merge_block = blocks[merge_block_idx]

    then_range = blocks[then_block_idx].range
    else_range = blocks[else_block_idx].range

    # Look for phis that have edges from both then and else blocks
    for si in merge_block.range
        stmt = code.code[si]
        stmt isa PhiNode || continue

        then_val = nothing
        else_val = nothing
        for (edge_idx, edge) in enumerate(stmt.edges)
            if edge in then_range
                then_val = stmt.values[edge_idx]
            elseif edge in else_range
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
    handle_if_then!(block::Block, tree::ControlTree, code::CodeInfo, blocks::Vector{BlockInfo}, ctx::StructurizationContext)

Handle REGION_IF_THEN.
"""
function handle_if_then!(block::Block, tree::ControlTree, code::CodeInfo, blocks::Vector{BlockInfo},
                         ctx::StructurizationContext)
    tree_children = children(tree)
    length(tree_children) >= 2 || return handle_block_region!(block, tree, code, blocks, ctx)

    # First child is the condition block
    cond_tree = tree_children[1]
    cond_idx = node_index(cond_tree)

    # Find the GotoIfNot's SSA index for keying
    gotoifnot_idx = nothing
    if 1 <= cond_idx <= length(blocks)
        cond_block = blocks[cond_idx]
        for si in cond_block.range
            stmt = code.code[si]
            if stmt isa GotoIfNot
                gotoifnot_idx = si
            end
            if !(stmt isa GotoNode || stmt isa GotoIfNot || stmt isa ReturnNode || stmt isa PhiNode)
                push!(block, si, stmt, code.ssavaluetypes[si])
            end
        end
    end

    cond_value = find_condition_value(cond_idx, code, blocks)

    # Then block
    then_tree = tree_children[2]
    then_blk = tree_to_block(then_tree, code, blocks, ctx)

    # Empty else block
    else_blk = Block()

    if_op = IfOp(cond_value, then_blk, else_blk)
    result_idx = gotoifnot_idx !== nothing ? gotoifnot_idx : last(blocks[cond_idx].range)
    push!(block, result_idx, if_op, Nothing)
end

"""
    handle_termination!(block::Block, tree::ControlTree, code::CodeInfo, blocks::Vector{BlockInfo}, ctx::StructurizationContext)

Handle REGION_TERMINATION - branches where some paths terminate.
"""
function handle_termination!(block::Block, tree::ControlTree, code::CodeInfo, blocks::Vector{BlockInfo},
                             ctx::StructurizationContext)
    tree_children = children(tree)
    isempty(tree_children) && return handle_block_region!(block, tree, code, blocks, ctx)

    # First child is the condition block
    cond_tree = tree_children[1]
    cond_idx = node_index(cond_tree)

    # Find the GotoIfNot's SSA index for keying
    gotoifnot_idx = nothing
    if 1 <= cond_idx <= length(blocks)
        cond_block = blocks[cond_idx]
        for si in cond_block.range
            stmt = code.code[si]
            if stmt isa GotoIfNot
                gotoifnot_idx = si
            end
            if !(stmt isa GotoNode || stmt isa GotoIfNot || stmt isa ReturnNode || stmt isa PhiNode)
                push!(block, si, stmt, code.ssavaluetypes[si])
            end
        end
    end

    cond_value = find_condition_value(cond_idx, code, blocks)
    result_idx = gotoifnot_idx !== nothing ? gotoifnot_idx : last(blocks[cond_idx].range)

    # Build then and else blocks from remaining children
    if length(tree_children) >= 3
        then_tree = tree_children[2]
        else_tree = tree_children[3]
        then_blk = tree_to_block(then_tree, code, blocks, ctx)
        else_blk = tree_to_block(else_tree, code, blocks, ctx)
        if_op = IfOp(cond_value, then_blk, else_blk)
        push!(block, result_idx, if_op, Nothing)
    elseif length(tree_children) == 2
        then_tree = tree_children[2]
        then_blk = tree_to_block(then_tree, code, blocks, ctx)
        else_blk = Block()
        if_op = IfOp(cond_value, then_blk, else_blk)
        push!(block, result_idx, if_op, Nothing)
    end
end

"""
    handle_loop!(block::Block, tree::ControlTree, code::CodeInfo, blocks::Vector{BlockInfo}, ctx::StructurizationContext)

Handle REGION_FOR_LOOP, REGION_WHILE_LOOP, and REGION_NATURAL_LOOP.

For REGION_FOR_LOOP: Creates ForOp directly using metadata from CFG analysis.
For REGION_WHILE_LOOP: Creates WhileOp directly with before/after regions.
For REGION_NATURAL_LOOP: Creates LoopOp with internal IfOp (fallback for complex loops).

The loop is keyed at a synthesized SSA index, and getfield statements are generated
at the original phi node indices. This ensures that references like `return %2`
continue to work because getfield is placed at %2.
"""
function handle_loop!(block::Block, tree::ControlTree, code::CodeInfo, blocks::Vector{BlockInfo},
                      ctx::StructurizationContext)
    rtype = region_type(tree)

    # Dispatch based on region type
    if rtype == REGION_FOR_LOOP
        for_op, phi_indices, phi_types, is_inclusive, original_upper = build_for_op(tree, code, blocks, ctx)

        # Handle inclusive bounds (Julia's for i in 1:n) by inserting add_int(upper, 1)
        if is_inclusive
            # Allocate SSA for adjusted upper bound
            adj_ssa_idx = ctx.next_ssa_idx
            ctx.next_ssa_idx += 1

            # Insert add_int expression before the loop
            adj_upper = convert_phi_value(original_upper)
            add_int_expr = Expr(:call, GlobalRef(Base, :add_int), adj_upper, 1)
            push!(block, adj_ssa_idx, add_int_expr, Int64)

            # Update ForOp's upper bound to the adjusted value
            for_op.upper = SSAValue(adj_ssa_idx)
        end

        loop_op = for_op
    elseif rtype == REGION_WHILE_LOOP
        loop_op, phi_indices, phi_types = build_while_op(tree, code, blocks, ctx)
    else  # REGION_NATURAL_LOOP or other cyclic regions
        loop_op, phi_indices, phi_types = build_loop_op(tree, code, blocks, ctx)
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
    handle_switch!(block::Block, tree::ControlTree, code::CodeInfo, blocks::Vector{BlockInfo}, ctx::StructurizationContext)

Handle REGION_SWITCH.
"""
function handle_switch!(block::Block, tree::ControlTree, code::CodeInfo, blocks::Vector{BlockInfo},
                        ctx::StructurizationContext)
    # For now, handle as a nested if-else chain
    # TODO: Implement proper switch handling if needed
    handle_block_region!(block, tree, code, blocks, ctx)
end

#=============================================================================
 Statement Collection Helpers
=============================================================================#

"""
    collect_block_statements!(block::Block, info::BlockInfo, code::CodeInfo)

Collect statements from a BlockInfo into a Block, excluding control flow.
Stores raw expressions (no substitutions) with their SSA indices.
"""
function collect_block_statements!(block::Block, info::BlockInfo, code::CodeInfo)
    stmts = code.code
    types = code.ssavaluetypes
    for si in info.range
        stmt = stmts[si]
        if stmt isa ReturnNode
            block.terminator = stmt
        elseif !(stmt isa GotoNode || stmt isa GotoIfNot || stmt isa PhiNode)
            push!(block, si, stmt, types[si])
        end
    end
end

"""
    find_condition_value(block_idx::Int, code::CodeInfo, blocks::Vector{BlockInfo}) -> IRValue

Find the condition value for a GotoIfNot in the given block.
"""
function find_condition_value(block_idx::Int, code::CodeInfo, blocks::Vector{BlockInfo})
    @assert 1 <= block_idx <= length(blocks) "Invalid block index: $block_idx"

    block = blocks[block_idx]
    for si in block.range
        stmt = code.code[si]
        if stmt isa GotoIfNot
            cond = stmt.cond
            @assert cond isa SSAValue || cond isa SlotNumber || cond isa Argument "Unexpected condition type: $(typeof(cond))"
            return cond
        end
    end

    error("No GotoIfNot found in block $block_idx")
end

"""
    set_block_terminator!(block::Block, code::CodeInfo, blocks::Vector{BlockInfo})

Set the block terminator based on statements.
"""
function set_block_terminator!(block::Block, code::CodeInfo, blocks::Vector{BlockInfo})
    block.terminator !== nothing && return

    last_idx = nothing
    for (idx, entry) in block.body
        if !(entry.stmt isa ControlFlowOp)
            if last_idx === nothing || idx > last_idx
                last_idx = idx
            end
        end
    end
    if last_idx !== nothing && last_idx < length(code.code)
        next_stmt = code.code[last_idx + 1]
        if next_stmt isa ReturnNode
            block.terminator = next_stmt
        end
    end
end

#=============================================================================
 Loop Construction
=============================================================================#

"""
    build_while_op(tree::ControlTree, code::CodeInfo, blocks::Vector{BlockInfo}, ctx::StructurizationContext)
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
function build_while_op(tree::ControlTree, code::CodeInfo, blocks::Vector{BlockInfo},
                        ctx::StructurizationContext)
    stmts = code.code
    types = code.ssavaluetypes
    header_idx = node_index(tree)
    loop_blocks = get_loop_blocks(tree, blocks)

    @assert 1 <= header_idx <= length(blocks) "Invalid header_idx from control tree: $header_idx"
    header_block = blocks[header_idx]
    stmt_to_blk = stmt_to_block_map(blocks, length(stmts))

    # Extract phi node information: init_values (from outside loop) and carried_values (from inside loop)
    init_values = IRValue[]
    carried_values = IRValue[]
    phi_indices = Int[]
    phi_types = Any[]

    for si in header_block.range
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
                    # Check where control flow comes from, not where value is defined
                    edge_block = stmt_to_blk[edge]
                    if edge_block ∈ loop_blocks
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
    for si in header_block.range
        stmt = stmts[si]
        if stmt isa GotoIfNot
            condition = stmt.cond
            break
        end
    end

    # Build "before" region: header statements + ConditionOp
    before = Block()
    for si in header_block.range
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
            handle_block_region!(after, child, code, blocks, ctx)
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
    build_loop_op(tree::ControlTree, code::CodeInfo, blocks::Vector{BlockInfo}, ctx::StructurizationContext)
        -> Tuple{LoopOp, Vector{Int}, Vector{Any}}

Build a LoopOp from a control tree and return phi node information.
Returns (loop_op, phi_indices, phi_types) where:
- loop_op: The constructed LoopOp
- phi_indices: SSA indices of the header phi nodes (for getfield generation)
- phi_types: Julia types of the header phi nodes
"""
function build_loop_op(tree::ControlTree, code::CodeInfo, blocks::Vector{BlockInfo},
                                     ctx::StructurizationContext)
    stmts = code.code
    types = code.ssavaluetypes
    header_idx = node_index(tree)
    loop_blocks = get_loop_blocks(tree, blocks)

    @assert 1 <= header_idx <= length(blocks) "Invalid header_idx from control tree: $header_idx"
    header_block = blocks[header_idx]
    stmt_to_blk = stmt_to_block_map(blocks, length(stmts))

    init_values = IRValue[]
    carried_values = IRValue[]
    phi_indices = Int[]
    phi_types = Any[]

    for si in header_block.range
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
                    # Check where control flow comes from, not where value is defined
                    edge_block = stmt_to_blk[edge]
                    if edge_block ∈ loop_blocks
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
    for si in header_block.range
        stmt = stmts[si]
        if stmt isa GotoIfNot
            condition = stmt.cond
            condition_idx = si
            break
        end
    end

    # Collect header statements (excluding phi nodes and control flow)
    for si in header_block.range
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
                handle_block_region!(then_blk, child, code, blocks, ctx)
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
                handle_block_region!(body, child, code, blocks, ctx)
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
    build_for_op(tree::ControlTree, code::CodeInfo, blocks::Vector{BlockInfo}, ctx::StructurizationContext)
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
function build_for_op(tree::ControlTree, code::CodeInfo, blocks::Vector{BlockInfo},
                             ctx::StructurizationContext)
    stmts = code.code
    types = code.ssavaluetypes
    header_idx = node_index(tree)
    loop_blocks = get_loop_blocks(tree, blocks)
    for_info = metadata(tree)::ForLoopInfo

    @assert 1 <= header_idx <= length(blocks) "Invalid header_idx from control tree: $header_idx"
    header_block = blocks[header_idx]
    stmt_to_blk = stmt_to_block_map(blocks, length(stmts))

    # Extract phi info: separate IV from other loop-carried variables
    iv_phi_idx = for_info.iv_phi_idx
    all_phi_indices = Int[]
    all_phi_types = Any[]
    init_values = IRValue[]
    carried_values = IRValue[]

    for si in header_block.range
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
                    # Check where control flow comes from, not where value is defined
                    edge_block = stmt_to_blk[edge]
                    if edge_block ∈ loop_blocks
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

    # Collect header statements (excluding phi nodes and control flow)
    for si in header_block.range
        stmt = stmts[si]
        if !(stmt isa PhiNode || stmt isa GotoNode || stmt isa GotoIfNot || stmt isa ReturnNode)
            push!(body, si, stmt, types[si])
        end
    end

    # Process loop body blocks (excluding header)
    for child in children(tree)
        child_idx = node_index(child)
        if child_idx != header_idx
            handle_block_region!(body, child, code, blocks, ctx)
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
    # For now, store the upper bound as-is; the adjustment will be made in
    # apply_loop_patterns! replacement or a post-processing step

    # Actually, since we're building directly, let's handle inclusive bounds here
    # by allocating an SSA for add_int(upper, 1) if needed
    if for_info.is_inclusive
        # Allocate a synthesized SSA for the adjusted upper bound
        adj_ssa_idx = ctx.next_ssa_idx
        ctx.next_ssa_idx += 1

        # The actual add_int expression will be inserted before ForOp
        # For now, use SSAValue reference to the adjustment
        upper = SSAValue(adj_ssa_idx)

        # Store info for later (we'll handle this in handle_loop! by special-casing)
        # Actually, let's just return a marker struct or modify the flow...

        # Simpler approach: store the adjustment info and let handle_loop! insert it
        # For now, we'll store the original upper and mark is_inclusive
        # The adjustment can be done in handle_loop! before pushing the ForOp
        # Let me reconsider...
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

