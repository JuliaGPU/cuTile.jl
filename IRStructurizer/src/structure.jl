# Phase 1: Control Tree to Structured IR
#
# Converts a ControlTree (from graph contraction) to structured IR with Block
# and ControlFlowOp objects. All loops become LoopOp in this phase.
# Pattern matching (ForOp/WhileOp) and substitutions happen in later phases.

using AbstractTrees: PreOrderDFS

#=============================================================================
 Phase 1 Helpers
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
        return 0  # Fallback
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
    elseif rtype == REGION_WHILE_LOOP || rtype == REGION_NATURAL_LOOP
        handle_loop!(block, tree, code, blocks, ctx)
    elseif rtype == REGION_SELF_LOOP
        handle_self_loop!(block, tree, code, blocks, ctx)
    elseif rtype == REGION_PROPER
        handle_proper_region!(block, tree, code, blocks, ctx)
    elseif rtype == REGION_SWITCH
        handle_switch!(block, tree, code, blocks, ctx)
    else
        # Fallback: collect statements
        handle_block_region!(block, tree, code, blocks, ctx)
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
    elseif rtype == REGION_WHILE_LOOP || rtype == REGION_NATURAL_LOOP
        handle_loop!(block, tree, code, blocks, ctx)
    elseif rtype == REGION_SELF_LOOP
        handle_self_loop!(block, tree, code, blocks, ctx)
    elseif rtype == REGION_PROPER
        handle_proper_region!(block, tree, code, blocks, ctx)
    elseif rtype == REGION_SWITCH
        handle_switch!(block, tree, code, blocks, ctx)
    else
        handle_block_region!(block, tree, code, blocks, ctx)
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

    # Create IfOp - no outer capture yet, Phase 2 will handle it
    if_op = IfOp(cond_value, then_blk, else_blk)

    # Key by first merge phi's SSA index if available, else by GotoIfNot
    # Also get the result type(s) from the merge phis
    if !isempty(merge_phis)
        result_idx = merge_phis[1].ssa_idx
        # Get types from all merge phis
        result_types = [ctx.ssavaluetypes[phi.ssa_idx] for phi in merge_phis]
        if length(result_types) == 1
            result_type = result_types[1]
        else
            result_type = Tuple{result_types...}
        end
    else
        result_idx = gotoifnot_idx !== nothing ? gotoifnot_idx : last(blocks[cond_idx].range)
        result_type = Nothing
    end
    push!(block, result_idx, if_op, result_type)
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

    # Get statement ranges for then/else blocks to match against phi edges
    # PhiNode edges are statement indices, not block indices
    then_range = blocks[then_block_idx].range
    else_range = blocks[else_block_idx].range

    # Look for phis that have edges from both then and else blocks
    for si in merge_block.range
        stmt = code.code[si]
        stmt isa PhiNode || continue

        # Find values for then and else edges
        # Phi edges are statement indices - check if they fall within block ranges
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

    # Create IfOp - no outer capture yet, Phase 2 will handle it
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

Handle REGION_WHILE_LOOP and REGION_NATURAL_LOOP.
Phase 1: Always creates LoopOp with metadata. Pattern matching happens in Phase 3.
"""
function handle_loop!(block::Block, tree::ControlTree, code::CodeInfo, blocks::Vector{BlockInfo},
                      ctx::StructurizationContext)
    loop_op = build_loop_op_phase1(tree, code, blocks, ctx)
    results = derive_result_vars(loop_op)
    if !isempty(results)
        # Key by first result phi's SSA index
        push!(block, results[1].id, loop_op, Nothing)
    else
        # Loops with no results - use fallback
        header_idx = node_index(tree)
        header_block = blocks[header_idx]
        push!(block, last(header_block.range), loop_op, Nothing)
    end
end

"""
    handle_self_loop!(block::Block, tree::ControlTree, code::CodeInfo, blocks::Vector{BlockInfo}, ctx::StructurizationContext)

Handle REGION_SELF_LOOP.
"""
function handle_self_loop!(block::Block, tree::ControlTree, code::CodeInfo, blocks::Vector{BlockInfo},
                           ctx::StructurizationContext)
    idx = node_index(tree)

    body_blk = Block()

    if 1 <= idx <= length(blocks)
        collect_block_statements!(body_blk, blocks[idx], code)
    end

    loop_op = LoopOp(body_blk, IRValue[])
    # Self-loops typically don't have phi nodes - use block's last SSA index
    if 1 <= idx <= length(blocks)
        push!(block, last(blocks[idx].range), loop_op, Nothing)
    else
        push!(block, idx, loop_op, Nothing)  # Fallback
    end
end

"""
    handle_proper_region!(block::Block, tree::ControlTree, code::CodeInfo, blocks::Vector{BlockInfo}, ctx::StructurizationContext)

Handle REGION_PROPER - acyclic region not matching other patterns.
"""
function handle_proper_region!(block::Block, tree::ControlTree, code::CodeInfo, blocks::Vector{BlockInfo},
                               ctx::StructurizationContext)
    # Process as a sequence of blocks
    handle_block_region!(block, tree, code, blocks, ctx)
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
    block_idx < 1 || block_idx > length(blocks) && return SSAValue(1)

    block = blocks[block_idx]
    for si in block.range
        stmt = code.code[si]
        if stmt isa GotoIfNot
            cond = stmt.cond
            if cond isa SSAValue || cond isa SlotNumber || cond isa Argument
                return cond
            else
                return SSAValue(max(1, si - 1))
            end
        end
    end

    return SSAValue(max(1, first(block.range)))
end

"""
    set_block_terminator!(block::Block, code::CodeInfo, blocks::Vector{BlockInfo})

Set the block terminator based on statements.
"""
function set_block_terminator!(block::Block, code::CodeInfo, blocks::Vector{BlockInfo})
    block.terminator !== nothing && return

    # Find the last statement SSA index in body (largest positive key)
    last_idx = nothing
    for (idx, entry) in block.body
        if !(entry.stmt isa ControlFlowOp)  # Statement
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
    build_loop_op_phase1(tree::ControlTree, code::CodeInfo, blocks::Vector{BlockInfo}, ctx::StructurizationContext) -> LoopOp

Build a LoopOp for Phase 1. Pure structure building - no BlockArgs or substitutions.
BlockArg creation and SSA→BlockArg substitution happens in Phase 2 (apply_block_args!).
"""
function build_loop_op_phase1(tree::ControlTree, code::CodeInfo, blocks::Vector{BlockInfo},
                              ctx::StructurizationContext)
    stmts = code.code
    types = code.ssavaluetypes
    header_idx = node_index(tree)
    loop_blocks = get_loop_blocks(tree, blocks)

    @assert 1 <= header_idx <= length(blocks) "Invalid header_idx from control tree: $header_idx"
    header_block = blocks[header_idx]
    stmt_to_blk = stmt_to_block_map(blocks, length(stmts))

    # Find phi nodes in header - these become loop-carried values (iter_args)
    iter_args = IRValue[]        # Entry values for each phi (becomes iter_args)
    carried_values = IRValue[]   # Loop-back values for each phi (SSAValues)
    result_ssa_indices = SSAValue[]  # SSA indices of phi nodes (for BreakOp)

    for si in header_block.range
        stmt = stmts[si]
        if stmt isa PhiNode
            push!(result_ssa_indices, SSAValue(si))
            phi = stmt

            entry_val = nothing
            carried_val = nothing

            for (edge_idx, _) in enumerate(phi.edges)
                if isassigned(phi.values, edge_idx)
                    val = phi.values[edge_idx]

                    if val isa SSAValue
                        val_stmt = val.id
                        if val_stmt > 0 && val_stmt <= length(stmts)
                            val_block = stmt_to_blk[val_stmt]
                            if val_block ∈ loop_blocks
                                carried_val = val
                            else
                                entry_val = convert_phi_value(val)
                            end
                        else
                            entry_val = convert_phi_value(val)
                        end
                    else
                        entry_val = convert_phi_value(val)
                    end
                end
            end

            entry_val !== nothing && push!(iter_args, entry_val)
            carried_val !== nothing && push!(carried_values, carried_val)
        end
    end

    # Build loop body block (no BlockArgs yet - Phase 2 will add them)
    body = Block()
    # body.args stays empty - Phase 2 will populate it

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
        # ContinueOp with raw carried_values (SSAValues) - Phase 2 will substitute
        then_blk.terminator = ContinueOp(copy(carried_values))

        else_blk = Block()
        # BreakOp with SSAValues - kept as-is for derive_result_vars
        else_blk.terminator = BreakOp(IRValue[rv for rv in result_ssa_indices])

        if_op = IfOp(cond_value, then_blk, else_blk)
        # Key by the GotoIfNot's SSA index
        push!(body, condition_idx, if_op, Nothing)
    else
        # No condition - process children directly
        for child in children(tree)
            child_idx = node_index(child)
            if child_idx != header_idx
                handle_block_region!(body, child, code, blocks, ctx)
            end
        end
        # ContinueOp with raw carried_values (SSAValues) - Phase 2 will substitute
        body.terminator = ContinueOp(copy(carried_values))
    end

    # Create loop op with iter_args
    loop_op = LoopOp(body, iter_args)
    return loop_op
end

"""
    collect_defined_ssas!(defined::Set{Int}, block::Block, ctx::StructurizationContext)

Collect all SSA indices defined by statements in the block (recursively).
Also includes results from control flow ops (phi nodes define SSAValues).
"""
function collect_defined_ssas!(defined::Set{Int}, block::Block, ctx::StructurizationContext)
    for (idx, entry) in block.body
        if entry.stmt isa ControlFlowOp
            for rv in derive_result_vars(entry.stmt)
                push!(defined, rv.id)
            end
            # Recurse into regions based on op type
            if entry.stmt isa LoopOp
                collect_defined_ssas!(defined, entry.stmt.body, ctx)
            elseif entry.stmt isa IfOp
                collect_defined_ssas!(defined, entry.stmt.then_region, ctx)
                collect_defined_ssas!(defined, entry.stmt.else_region, ctx)
            elseif entry.stmt isa WhileOp
                collect_defined_ssas!(defined, entry.stmt.before, ctx)
                collect_defined_ssas!(defined, entry.stmt.after, ctx)
            elseif entry.stmt isa ForOp
                collect_defined_ssas!(defined, entry.stmt.body, ctx)
            end
        else  # Statement
            push!(defined, idx)
        end
    end
end

#=============================================================================
 Phase 2: Apply Block Arguments
=============================================================================#

"""
    apply_block_args!(block::Block, ctx::StructurizationContext, defined::Set{Int}=Set{Int}(), parent_subs::Substitutions=Substitutions())

Single pass that creates BlockArgs and substitutes SSAValue references.

Phase 2 of structurization - called after control_tree_to_structured_ir.
For each :loop op: creates BlockArgs for phi nodes (iter_args).
For :if ops: no BlockArgs needed (outer refs are accessed directly).
Substitutes phi refs → BlockArg references throughout.

The parent_subs parameter carries substitutions from outer scopes, so nested
control flow ops can convert phi refs to the correct BlockArgs.
"""
function apply_block_args!(block::Block, ctx::StructurizationContext,
                           defined::Set{Int}=Set{Int}(), parent_subs::Substitutions=Substitutions())
    # Track what's defined at this level
    defined = copy(defined)
    for (idx, entry) in block.body
        if !(entry.stmt isa ControlFlowOp)  # Statement
            push!(defined, idx)
        end
    end

    # Process each control flow op
    for stmt in statements(block.body)
        if stmt isa LoopOp
            process_loop_block_args!(stmt, ctx, defined, parent_subs)
        elseif stmt isa IfOp
            process_if_block_args!(stmt, ctx, defined, parent_subs)
        end
    end
end

"""
    process_loop_block_args!(loop::LoopOp, ctx::StructurizationContext, parent_defined::Set{Int}, parent_subs::Substitutions)

Create BlockArgs for a LoopOp and substitute SSAValue references.

1. Create BlockArgs for phi nodes (iter_args / carries)
2. Apply parent substitutions to iter_args
3. Substitute phi refs → BlockArg in body
4. Recurse into nested blocks

Outer scope SSA values are referenced directly (like MLIR), no captures needed.
"""
function process_loop_block_args!(loop::LoopOp, ctx::StructurizationContext,
                                  parent_defined::Set{Int}, parent_subs::Substitutions)
    body = loop.body::Block
    subs = Substitutions()
    result_vars = derive_result_vars(loop)

    # 1. Create BlockArgs for phi nodes (these are the carries)
    for (i, result_var) in enumerate(result_vars)
        phi_type = ctx.ssavaluetypes[result_var.id]
        new_arg = BlockArg(i, phi_type)
        push!(body.args, new_arg)
        subs[result_var.id] = new_arg
    end

    # 2. Apply parent substitutions to iter_args
    # This converts SSAValues to parent's BlockArgs for nested control flow
    for (j, v) in enumerate(loop.iter_args)
        loop.iter_args[j] = substitute_ssa(v, parent_subs)
    end

    # 3. Substitute phi refs → BlockArg in body (shallow - don't recurse into nested ops)
    substitute_block_shallow!(body, subs)

    # 4. Recurse into nested blocks, passing merged substitutions
    # Merge parent subs with this loop's subs so nested ops can access both
    merged_subs = merge(parent_subs, subs)
    nested_defined = Set{Int}(rv.id for rv in result_vars)
    collect_defined_ssas!(nested_defined, body, ctx)
    apply_block_args!(body, ctx, nested_defined, merged_subs)
end

"""
    process_if_block_args!(if_op::IfOp, ctx::StructurizationContext, parent_defined::Set{Int}, parent_subs::Substitutions)

Process an IfOp. For IfOp, no BlockArgs needed (no iteration).
Just recurse into nested blocks with parent substitutions.

Outer scope SSA values are referenced directly (like MLIR), no captures needed.
"""
function process_if_block_args!(if_op::IfOp, ctx::StructurizationContext,
                                parent_defined::Set{Int}, parent_subs::Substitutions)
    then_blk = if_op.then_region::Block
    else_blk = if_op.else_region::Block

    # Build combined defined set from parent + what's defined in each branch
    then_defined = copy(parent_defined)
    else_defined = copy(parent_defined)
    collect_defined_ssas!(then_defined, then_blk, ctx)
    collect_defined_ssas!(else_defined, else_blk, ctx)

    # Recurse into nested blocks, passing parent substitutions
    apply_block_args!(then_blk, ctx, then_defined, parent_subs)
    apply_block_args!(else_blk, ctx, else_defined, parent_subs)
end
