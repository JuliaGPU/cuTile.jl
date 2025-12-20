# Phase 1: Control Tree to Structured IR
#
# Converts a ControlTree (from graph contraction) to structured IR with Block,
# Statement, and ControlFlowOp objects. All loops become LoopOp in this phase.
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

"""
    find_exit_block(tree::ControlTree, blocks::Vector{BlockInfo}) -> Int

Find the exit block of a region (a block that has successors outside the region).
Returns the block index, or -1 if not found.
"""
function find_exit_block(tree::ControlTree, blocks::Vector{BlockInfo})
    region_blocks = get_loop_blocks(tree, blocks)

    # Find blocks whose successors include blocks outside the region
    for blk_idx in region_blocks
        if 1 <= blk_idx <= length(blocks)
            for succ in blocks[blk_idx].succs
                if !(succ in region_blocks)
                    return blk_idx
                end
            end
        end
    end

    # Fallback: return the last block in the region by traversal order
    last_idx = -1
    for subtree in PreOrderDFS(tree)
        idx = node_index(subtree)
        if 1 <= idx <= length(blocks)
            last_idx = idx
        end
    end
    return last_idx
end

"""
    find_merge_block(then_exit::Int, else_exit::Int, blocks::Vector{BlockInfo}) -> Union{Int, Nothing}

Find the merge block (common successor) of then and else branches.
Returns the block index, or nothing if not found.
"""
function find_merge_block(then_exit::Int, else_exit::Int, blocks::Vector{BlockInfo})
    (1 <= then_exit <= length(blocks) && 1 <= else_exit <= length(blocks)) || return nothing

    then_succs = Set(blocks[then_exit].succs)
    else_succs = Set(blocks[else_exit].succs)
    merge_candidates = intersect(then_succs, else_succs)

    isempty(merge_candidates) && return nothing
    return first(merge_candidates)
end

"""
    build_block_label_to_idx(blocks::Vector{BlockInfo}) -> Dict{Int, Int}

Build a mapping from statement indices to BlockInfo indices.
PhiNode edges in Julia IR are the statement index of the last statement in the predecessor block
(the transfer point), so we need to map all statement indices in each block to the block index.
"""
function build_block_label_to_idx(blocks::Vector{BlockInfo})
    label_to_idx = Dict{Int, Int}()
    for (idx, block) in enumerate(blocks)
        for si in block.range
            label_to_idx[si] = idx
        end
    end
    return label_to_idx
end

"""
    find_if_merge_phis(merge_idx::Int, then_blocks::Set{Int}, else_blocks::Set{Int},
                       code::CodeInfo, blocks::Vector{BlockInfo}) -> Vector{Int}

Find PhiNodes at the merge block that have edges from both then and else branches.
Returns the SSA indices of the PhiNodes.
"""
function find_if_merge_phis(merge_idx::Int, then_blocks::Set{Int}, else_blocks::Set{Int},
                            code::CodeInfo, blocks::Vector{BlockInfo})
    (1 <= merge_idx <= length(blocks)) || return Int[]

    phi_indices = Int[]
    merge_block = blocks[merge_idx]
    stmts = code.code

    # Build mapping from block labels (stmt indices) to BlockInfo indices
    label_to_idx = build_block_label_to_idx(blocks)

    for si in merge_block.range
        stmt = stmts[si]
        if stmt isa PhiNode
            # Check if phi has edges from both then and else branches
            has_then = false
            has_else = false

            for edge in stmt.edges
                # edge is a block label (first statement index of predecessor block)
                block_idx = get(label_to_idx, edge, 0)
                if block_idx in then_blocks
                    has_then = true
                elseif block_idx in else_blocks
                    has_else = true
                end
            end

            if has_then && has_else
                push!(phi_indices, si)
            end
        end
    end
    return phi_indices
end

"""
    get_phi_branch_value(phi::PhiNode, branch_blocks::Set{Int}, label_to_idx::Dict{Int, Int}) -> Union{IRValue, Nothing}

Get the value from a PhiNode that comes from the given branch blocks.
"""
function get_phi_branch_value(phi::PhiNode, branch_blocks::Set{Int}, label_to_idx::Dict{Int, Int})
    for (i, edge) in enumerate(phi.edges)
        block_idx = get(label_to_idx, edge, 0)
        if block_idx in branch_blocks && isassigned(phi.values, i)
            return convert_phi_value(phi.values[i])
        end
    end
    return nothing
end

#=============================================================================
 Control Tree to Structured IR
=============================================================================#

"""
    control_tree_to_structured_ir(ctree::ControlTree, code::CodeInfo, blocks::Vector{BlockInfo}) -> Block

Convert a control tree to structured IR entry block.
All loops become LoopOp (no pattern matching yet, no substitutions).
"""
function control_tree_to_structured_ir(ctree::ControlTree, code::CodeInfo, blocks::Vector{BlockInfo})
    block_id = Ref(1)
    entry_block = tree_to_block(ctree, code, blocks, block_id)
    return entry_block
end

"""
    tree_to_block(tree::ControlTree, code::CodeInfo, blocks::Vector{BlockInfo}, block_id::Ref{Int}) -> Block

Convert a control tree node to a Block. Creates Statement objects with raw expressions (no substitutions).
"""
function tree_to_block(tree::ControlTree, code::CodeInfo, blocks::Vector{BlockInfo}, block_id::Ref{Int})
    idx = node_index(tree)
    rtype = region_type(tree)
    id = block_id[]
    block_id[] += 1

    block = Block(id)

    if rtype == REGION_BLOCK
        handle_block_region!(block, tree, code, blocks, block_id)
    elseif rtype == REGION_IF_THEN_ELSE
        handle_if_then_else!(block, tree, code, blocks, block_id)
    elseif rtype == REGION_IF_THEN
        handle_if_then!(block, tree, code, blocks, block_id)
    elseif rtype == REGION_TERMINATION
        handle_termination!(block, tree, code, blocks, block_id)
    elseif rtype == REGION_WHILE_LOOP || rtype == REGION_NATURAL_LOOP
        handle_loop!(block, tree, code, blocks, block_id)
    elseif rtype == REGION_SELF_LOOP
        handle_self_loop!(block, tree, code, blocks, block_id)
    elseif rtype == REGION_PROPER
        handle_proper_region!(block, tree, code, blocks, block_id)
    elseif rtype == REGION_SWITCH
        handle_switch!(block, tree, code, blocks, block_id)
    else
        # Fallback: collect statements
        handle_block_region!(block, tree, code, blocks, block_id)
    end

    # Set terminator if not already set
    set_block_terminator!(block, code, blocks)

    return block
end

#=============================================================================
 Region Handlers
=============================================================================#

"""
    handle_block_region!(block::Block, tree::ControlTree, code::CodeInfo, blocks::Vector{BlockInfo}, block_id::Ref{Int})

Handle REGION_BLOCK - a linear sequence of blocks.
"""
function handle_block_region!(block::Block, tree::ControlTree, code::CodeInfo, blocks::Vector{BlockInfo}, block_id::Ref{Int})
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
                handle_block_region!(block, child, code, blocks, block_id)
            else
                # Nested control flow - create appropriate op
                handle_nested_region!(block, child, code, blocks, block_id)
            end
        end
    end
end

"""
    handle_nested_region!(block::Block, tree::ControlTree, code::CodeInfo, blocks::Vector{BlockInfo}, block_id::Ref{Int})

Handle a nested control flow region.
"""
function handle_nested_region!(block::Block, tree::ControlTree, code::CodeInfo, blocks::Vector{BlockInfo}, block_id::Ref{Int})
    rtype = region_type(tree)

    if rtype == REGION_IF_THEN_ELSE
        handle_if_then_else!(block, tree, code, blocks, block_id)
    elseif rtype == REGION_IF_THEN
        handle_if_then!(block, tree, code, blocks, block_id)
    elseif rtype == REGION_TERMINATION
        handle_termination!(block, tree, code, blocks, block_id)
    elseif rtype == REGION_WHILE_LOOP || rtype == REGION_NATURAL_LOOP
        handle_loop!(block, tree, code, blocks, block_id)
    elseif rtype == REGION_SELF_LOOP
        handle_self_loop!(block, tree, code, blocks, block_id)
    elseif rtype == REGION_PROPER
        handle_proper_region!(block, tree, code, blocks, block_id)
    elseif rtype == REGION_SWITCH
        handle_switch!(block, tree, code, blocks, block_id)
    else
        handle_block_region!(block, tree, code, blocks, block_id)
    end
end

"""
    handle_if_then_else!(block::Block, tree::ControlTree, code::CodeInfo, blocks::Vector{BlockInfo}, block_id::Ref{Int})

Handle REGION_IF_THEN_ELSE.
Also handles PhiNode results at the merge block by adding extraction statements.
"""
function handle_if_then_else!(block::Block, tree::ControlTree, code::CodeInfo, blocks::Vector{BlockInfo}, block_id::Ref{Int})
    tree_children = children(tree)
    length(tree_children) >= 3 || return handle_block_region!(block, tree, code, blocks, block_id)

    # First child is the condition block
    cond_tree = tree_children[1]
    cond_idx = node_index(cond_tree)

    # Collect condition block statements and find condition
    if 1 <= cond_idx <= length(blocks)
        cond_block = blocks[cond_idx]
        for si in cond_block.range
            stmt = code.code[si]
            if !(stmt isa GotoNode || stmt isa GotoIfNot || stmt isa ReturnNode || stmt isa PhiNode)
                push_op!(block, si, stmt, code.ssavaluetypes[si])
            end
        end
    end

    cond_value = find_condition_value(cond_idx, code, blocks)

    # Then and else blocks
    then_tree = tree_children[2]
    else_tree = tree_children[3]

    # Get all blocks in each branch (including the condition block for "then")
    then_blocks = get_loop_blocks(then_tree, blocks)
    else_blocks = get_loop_blocks(else_tree, blocks)
    # Include condition block in both sets for edge detection
    push!(then_blocks, cond_idx)
    push!(else_blocks, cond_idx)

    # Find merge block and PhiNodes
    then_exit = find_exit_block(then_tree, blocks)
    else_exit = find_exit_block(else_tree, blocks)
    merge_idx = find_merge_block(then_exit, else_exit, blocks)

    phi_ssa_indices = Int[]
    if merge_idx !== nothing
        phi_ssa_indices = find_if_merge_phis(merge_idx, then_blocks, else_blocks, code, blocks)
    end

    then_block = tree_to_block(then_tree, code, blocks, block_id)
    else_block = tree_to_block(else_tree, code, blocks, block_id)

    # Set YieldOp terminators if there are PhiNode results
    if !isempty(phi_ssa_indices)
        then_yields = IRValue[]
        else_yields = IRValue[]
        stmts = code.code
        label_to_idx = build_block_label_to_idx(blocks)

        for phi_idx in phi_ssa_indices
            phi = stmts[phi_idx]::PhiNode
            then_val = get_phi_branch_value(phi, then_blocks, label_to_idx)
            else_val = get_phi_branch_value(phi, else_blocks, label_to_idx)
            if then_val !== nothing
                push!(then_yields, then_val)
            end
            if else_val !== nothing
                push!(else_yields, else_val)
            end
        end

        # Only set YieldOp if both branches have values and no existing terminator
        # Convert SSAValue refs to LocalSSA using the block's ssa_map
        if length(then_yields) == length(phi_ssa_indices) && then_block.terminator === nothing
            then_block.terminator = convert_terminator_to_local_ssa(
                YieldOp(then_yields), then_block.ssa_map)
        end
        if length(else_yields) == length(phi_ssa_indices) && else_block.terminator === nothing
            else_block.terminator = convert_terminator_to_local_ssa(
                YieldOp(else_yields), else_block.ssa_map)
        end
    end

    # Compute result type
    types = code.ssavaluetypes
    result_type = if isempty(phi_ssa_indices)
        Nothing
    elseif length(phi_ssa_indices) == 1
        types[phi_ssa_indices[1]]
    else
        Tuple{(types[idx] for idx in phi_ssa_indices)...}
    end

    # Create IfOp with result type
    if_op = IfOp(cond_value, then_block, else_block, result_type)
    if_pos = push_cfop!(block, if_op)

    # Add extraction statements for each PhiNode result
    for (element_idx, phi_idx) in enumerate(phi_ssa_indices)
        phi_type = types[phi_idx]
        elem_idx = length(phi_ssa_indices) == 1 ? 0 : element_idx
        push_extraction!(block, phi_idx, if_pos, elem_idx, phi_type)
    end
end

"""
    handle_if_then!(block::Block, tree::ControlTree, code::CodeInfo, blocks::Vector{BlockInfo}, block_id::Ref{Int})

Handle REGION_IF_THEN.
Also handles PhiNode results at the merge block.
"""
function handle_if_then!(block::Block, tree::ControlTree, code::CodeInfo, blocks::Vector{BlockInfo}, block_id::Ref{Int})
    tree_children = children(tree)
    length(tree_children) >= 2 || return handle_block_region!(block, tree, code, blocks, block_id)

    # First child is the condition block
    cond_tree = tree_children[1]
    cond_idx = node_index(cond_tree)

    # Collect condition block statements
    if 1 <= cond_idx <= length(blocks)
        cond_block = blocks[cond_idx]
        for si in cond_block.range
            stmt = code.code[si]
            if !(stmt isa GotoNode || stmt isa GotoIfNot || stmt isa ReturnNode || stmt isa PhiNode)
                push_op!(block, si, stmt, code.ssavaluetypes[si])
            end
        end
    end

    cond_value = find_condition_value(cond_idx, code, blocks)

    # Then block
    then_tree = tree_children[2]

    # Get all blocks in the then branch (including condition block)
    then_blocks = get_loop_blocks(then_tree, blocks)
    push!(then_blocks, cond_idx)
    # For if-then, the else "branch" is just the condition block fallthrough
    else_blocks = Set{Int}([cond_idx])

    # Find merge block and PhiNodes
    then_exit = find_exit_block(then_tree, blocks)
    # For if-then, the else exit is the condition block itself
    merge_idx = nothing
    if 1 <= then_exit <= length(blocks) && 1 <= cond_idx <= length(blocks)
        then_succs = Set(blocks[then_exit].succs)
        cond_succs = Set(blocks[cond_idx].succs)
        merge_candidates = intersect(then_succs, cond_succs)
        if !isempty(merge_candidates)
            merge_idx = first(merge_candidates)
        end
    end

    phi_ssa_indices = Int[]
    if merge_idx !== nothing
        phi_ssa_indices = find_if_merge_phis(merge_idx, then_blocks, else_blocks, code, blocks)
    end

    then_block = tree_to_block(then_tree, code, blocks, block_id)

    # Empty else block
    else_block = Block(block_id[])
    block_id[] += 1

    # Set YieldOp terminators if there are PhiNode results
    if !isempty(phi_ssa_indices)
        then_yields = IRValue[]
        else_yields = IRValue[]
        stmts = code.code
        label_to_idx = build_block_label_to_idx(blocks)

        for phi_idx in phi_ssa_indices
            phi = stmts[phi_idx]::PhiNode
            then_val = get_phi_branch_value(phi, then_blocks, label_to_idx)
            else_val = get_phi_branch_value(phi, else_blocks, label_to_idx)
            if then_val !== nothing
                push!(then_yields, then_val)
            end
            if else_val !== nothing
                push!(else_yields, else_val)
            end
        end

        # Convert SSAValue refs to LocalSSA using the block's ssa_map
        if length(then_yields) == length(phi_ssa_indices) && then_block.terminator === nothing
            then_block.terminator = convert_terminator_to_local_ssa(
                YieldOp(then_yields), then_block.ssa_map)
        end
        if length(else_yields) == length(phi_ssa_indices) && else_block.terminator === nothing
            else_block.terminator = convert_terminator_to_local_ssa(
                YieldOp(else_yields), else_block.ssa_map)
        end
    end

    # Compute result type
    types = code.ssavaluetypes
    result_type = if isempty(phi_ssa_indices)
        Nothing
    elseif length(phi_ssa_indices) == 1
        types[phi_ssa_indices[1]]
    else
        Tuple{(types[idx] for idx in phi_ssa_indices)...}
    end

    # Create IfOp with result type
    if_op = IfOp(cond_value, then_block, else_block, result_type)
    if_pos = push_cfop!(block, if_op)

    # Add extraction statements for each PhiNode result
    for (element_idx, phi_idx) in enumerate(phi_ssa_indices)
        phi_type = types[phi_idx]
        elem_idx = length(phi_ssa_indices) == 1 ? 0 : element_idx
        push_extraction!(block, phi_idx, if_pos, elem_idx, phi_type)
    end
end

"""
    handle_termination!(block::Block, tree::ControlTree, code::CodeInfo, blocks::Vector{BlockInfo}, block_id::Ref{Int})

Handle REGION_TERMINATION - branches where some paths terminate.
Also handles PhiNode results if there's a merge block.
"""
function handle_termination!(block::Block, tree::ControlTree, code::CodeInfo, blocks::Vector{BlockInfo}, block_id::Ref{Int})
    tree_children = children(tree)
    isempty(tree_children) && return handle_block_region!(block, tree, code, blocks, block_id)

    # First child is the condition block
    cond_tree = tree_children[1]
    cond_idx = node_index(cond_tree)

    # Collect condition block statements
    if 1 <= cond_idx <= length(blocks)
        cond_block = blocks[cond_idx]
        for si in cond_block.range
            stmt = code.code[si]
            if !(stmt isa GotoNode || stmt isa GotoIfNot || stmt isa ReturnNode || stmt isa PhiNode)
                push_op!(block, si, stmt, code.ssavaluetypes[si])
            end
        end
    end

    cond_value = find_condition_value(cond_idx, code, blocks)

    # Build then and else blocks from remaining children
    if length(tree_children) >= 3
        then_tree = tree_children[2]
        else_tree = tree_children[3]

        # Get all blocks in each branch
        then_blocks = get_loop_blocks(then_tree, blocks)
        else_blocks = get_loop_blocks(else_tree, blocks)
        push!(then_blocks, cond_idx)
        push!(else_blocks, cond_idx)

        # Find merge block and PhiNodes
        then_exit = find_exit_block(then_tree, blocks)
        else_exit = find_exit_block(else_tree, blocks)
        merge_idx = find_merge_block(then_exit, else_exit, blocks)

        phi_ssa_indices = Int[]
        if merge_idx !== nothing
            phi_ssa_indices = find_if_merge_phis(merge_idx, then_blocks, else_blocks, code, blocks)
        end

        then_block = tree_to_block(then_tree, code, blocks, block_id)
        else_block = tree_to_block(else_tree, code, blocks, block_id)

        # Set YieldOp terminators if there are PhiNode results
        if !isempty(phi_ssa_indices)
            then_yields = IRValue[]
            else_yields = IRValue[]
            stmts = code.code
            label_to_idx = build_block_label_to_idx(blocks)

            for phi_idx in phi_ssa_indices
                phi = stmts[phi_idx]::PhiNode
                then_val = get_phi_branch_value(phi, then_blocks, label_to_idx)
                else_val = get_phi_branch_value(phi, else_blocks, label_to_idx)
                if then_val !== nothing
                    push!(then_yields, then_val)
                end
                if else_val !== nothing
                    push!(else_yields, else_val)
                end
            end

            # Convert SSAValue refs to LocalSSA using the block's ssa_map
            if length(then_yields) == length(phi_ssa_indices) && then_block.terminator === nothing
                then_block.terminator = convert_terminator_to_local_ssa(
                    YieldOp(then_yields), then_block.ssa_map)
            end
            if length(else_yields) == length(phi_ssa_indices) && else_block.terminator === nothing
                else_block.terminator = convert_terminator_to_local_ssa(
                    YieldOp(else_yields), else_block.ssa_map)
            end
        end

        # Compute result type
        types = code.ssavaluetypes
        result_type = if isempty(phi_ssa_indices)
            Nothing
        elseif length(phi_ssa_indices) == 1
            types[phi_ssa_indices[1]]
        else
            Tuple{(types[idx] for idx in phi_ssa_indices)...}
        end

        # Create IfOp with result type
        if_op = IfOp(cond_value, then_block, else_block, result_type)
        if_pos = push_cfop!(block, if_op)

        # Add extraction statements for each PhiNode result
        for (element_idx, phi_idx) in enumerate(phi_ssa_indices)
            phi_type = types[phi_idx]
            elem_idx = length(phi_ssa_indices) == 1 ? 0 : element_idx
            push_extraction!(block, phi_idx, if_pos, elem_idx, phi_type)
        end
    elseif length(tree_children) == 2
        then_tree = tree_children[2]
        then_block = tree_to_block(then_tree, code, blocks, block_id)
        else_block = Block(block_id[])
        block_id[] += 1
        if_op = IfOp(cond_value, then_block, else_block, Nothing)
        push_cfop!(block, if_op)
    end
end

"""
    handle_loop!(block::Block, tree::ControlTree, code::CodeInfo, blocks::Vector{BlockInfo}, block_id::Ref{Int})

Handle REGION_WHILE_LOOP and REGION_NATURAL_LOOP.
Phase 1: Always creates LoopOp with metadata. Pattern matching happens in Phase 2.
Also adds extraction statements for loop results so PhiNode references are resolved.
"""
function handle_loop!(block::Block, tree::ControlTree, code::CodeInfo, blocks::Vector{BlockInfo}, block_id::Ref{Int})
    loop_op, phi_ssa_indices = build_loop_op_phase1(tree, code, blocks, block_id)

    # Push the LoopOp
    loop_pos = push_cfop!(block, loop_op)

    # Add extraction statements for each PhiNode result
    types = code.ssavaluetypes
    for (element_idx, phi_idx) in enumerate(phi_ssa_indices)
        phi_type = types[phi_idx]
        elem_idx = length(phi_ssa_indices) == 1 ? 0 : element_idx
        push_extraction!(block, phi_idx, loop_pos, elem_idx, phi_type)
    end
end

"""
    handle_self_loop!(block::Block, tree::ControlTree, code::CodeInfo, blocks::Vector{BlockInfo}, block_id::Ref{Int})

Handle REGION_SELF_LOOP.
"""
function handle_self_loop!(block::Block, tree::ControlTree, code::CodeInfo, blocks::Vector{BlockInfo}, block_id::Ref{Int})
    idx = node_index(tree)
    stmts = code.code
    types = code.ssavaluetypes

    body_block = Block(block_id[])
    block_id[] += 1

    if 1 <= idx <= length(blocks)
        collect_block_statements!(body_block, blocks[idx], code)
    end

    # Capture outer scope SSAValue references
    outer_init_values, _ = capture_outer_refs!(body_block, stmts, types)

    # Compute result type from block args
    result_type = compute_result_type(body_block.args)

    loop_op = LoopOp(outer_init_values, body_block, result_type)
    push_cfop!(block, loop_op)
end

"""
    handle_proper_region!(block::Block, tree::ControlTree, code::CodeInfo, blocks::Vector{BlockInfo}, block_id::Ref{Int})

Handle REGION_PROPER - acyclic region not matching other patterns.
"""
function handle_proper_region!(block::Block, tree::ControlTree, code::CodeInfo, blocks::Vector{BlockInfo}, block_id::Ref{Int})
    # Process as a sequence of blocks
    handle_block_region!(block, tree, code, blocks, block_id)
end

"""
    handle_switch!(block::Block, tree::ControlTree, code::CodeInfo, blocks::Vector{BlockInfo}, block_id::Ref{Int})

Handle REGION_SWITCH.
"""
function handle_switch!(block::Block, tree::ControlTree, code::CodeInfo, blocks::Vector{BlockInfo}, block_id::Ref{Int})
    # For now, handle as a nested if-else chain
    # TODO: Implement proper switch handling if needed
    handle_block_region!(block, tree, code, blocks, block_id)
end

#=============================================================================
 Statement Collection Helpers
=============================================================================#

"""
    collect_block_statements!(block::Block, info::BlockInfo, code::CodeInfo)

Collect statements from a BlockInfo into a Block, excluding control flow.
Creates Statement objects with raw expressions (no substitutions).
"""
function collect_block_statements!(block::Block, info::BlockInfo, code::CodeInfo)
    stmts = code.code
    types = code.ssavaluetypes
    for si in info.range
        stmt = stmts[si]
        if stmt isa ReturnNode
            block.terminator = stmt
        elseif !(stmt isa GotoNode || stmt isa GotoIfNot || stmt isa PhiNode)
            push_op!(block, si, stmt, types[si])
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

    # Find the last original SSA index from the ssa_map
    last_idx = nothing
    for (ssa_idx, _) in block.ssa_map
        if last_idx === nothing || ssa_idx > last_idx
            last_idx = ssa_idx
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
    build_loop_op_phase1(tree::ControlTree, code::CodeInfo, blocks::Vector{BlockInfo}, block_id::Ref{Int}) -> Tuple{LoopOp, Vector{Int}}

Build a LoopOp with substitutions applied inline.
Phi node SSA references inside the loop body are replaced with BlockArgs.
Returns the LoopOp and the original PhiNode SSA indices it replaces (for result extraction).
"""
function build_loop_op_phase1(tree::ControlTree, code::CodeInfo, blocks::Vector{BlockInfo}, block_id::Ref{Int})
    stmts = code.code
    types = code.ssavaluetypes
    header_idx = node_index(tree)
    loop_blocks = get_loop_blocks(tree, blocks)

    @assert 1 <= header_idx <= length(blocks) "Invalid header_idx from control tree: $header_idx"
    header_block = blocks[header_idx]
    stmt_to_blk = stmt_to_block_map(blocks, length(stmts))

    # Find phi nodes in header - these become loop-carried values and block args
    # Also build substitution map: SSA index -> BlockArg
    # Track PhiNode indices for result extraction statements
    init_values = IRValue[]
    carried_values = IRValue[]
    block_args = BlockArg[]
    phi_ssa_indices = Int[]  # Track which PhiNodes this loop replaces
    subs = Substitutions()

    for si in header_block.range
        stmt = stmts[si]
        if stmt isa PhiNode
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

            # Only create a loop block arg if this is a LOOP PhiNode
            # (has at least one edge from inside the loop).
            # Non-loop PhiNodes (e.g., if/else merges before the loop) are
            # substituted with their entry value directly.
            if carried_val !== nothing
                entry_val !== nothing && push!(init_values, entry_val)
                push!(carried_values, carried_val)

                phi_type = types[si]
                block_arg = BlockArg(length(block_args) + 1, phi_type)
                push!(block_args, block_arg)
                push!(phi_ssa_indices, si)  # Track this PhiNode's SSA index
                # Map this phi's SSA index to its block arg
                subs[si] = block_arg
            else
                # Non-loop PhiNode: substitute with its entry value
                if entry_val !== nothing
                    subs[si] = entry_val
                end
            end
        end
    end

    # Build loop body block
    body = Block(block_id[])
    block_id[] += 1
    body.args = block_args

    # Find the condition for loop exit
    condition = nothing
    for si in header_block.range
        stmt = stmts[si]
        if stmt isa GotoIfNot
            condition = stmt.cond
            break
        end
    end

    # Collect header statements (excluding phi nodes and control flow)
    for si in header_block.range
        stmt = stmts[si]
        if !(stmt isa PhiNode || stmt isa GotoNode || stmt isa GotoIfNot || stmt isa ReturnNode)
            push_op!(body, si, stmt, types[si])
        end
    end

    # Create the conditional structure inside the loop body
    if condition !== nothing
        cond_value = convert_phi_value(condition)

        then_block = Block(block_id[])
        block_id[] += 1

        # Process loop body blocks (excluding header)
        for child in children(tree)
            child_idx = node_index(child)
            if child_idx != header_idx
                handle_block_region!(then_block, child, code, blocks, block_id)
            end
        end
        # Set terminator and convert SSAValue refs to LocalSSA using the block's ssa_map
        then_block.terminator = convert_terminator_to_local_ssa(
            ContinueOp(carried_values), then_block.ssa_map)

        else_block = Block(block_id[])
        block_id[] += 1
        # Block args are the references for break (already LocalSSA-compatible)
        result_values = IRValue[]
        for arg in block_args
            push!(result_values, arg)
        end
        else_block.terminator = BreakOp(result_values)

        if_op = IfOp(cond_value, then_block, else_block, Nothing)
        push_cfop!(body, if_op)
    else
        # No condition - process children directly
        for child in children(tree)
            child_idx = node_index(child)
            if child_idx != header_idx
                handle_block_region!(body, child, code, blocks, block_id)
            end
        end
        # Set terminator and convert SSAValue refs to LocalSSA
        body.terminator = convert_terminator_to_local_ssa(
            ContinueOp(carried_values), body.ssa_map)
    end

    # Apply substitutions to the loop body (phi SSA refs -> block args)
    substitute_block!(body, subs)

    # Capture outer scope SSAValue references and convert to block args
    n_loop_args = length(body.args)  # Number of PhiNode loop args
    outer_init_values, _ = capture_outer_refs!(body, stmts, types)
    append!(init_values, outer_init_values)

    # Update terminators to include captured args (passed through unchanged).
    # ContinueOp: extend with captured block args
    # BreakOp: extend with captured block args
    n_captured = length(body.args) - n_loop_args
    if n_captured > 0
        captured_args = body.args[n_loop_args+1:end]
        extend_loop_terminators!(body, carried_values, captured_args)
    end

    # Compute result type from block args
    result_type = compute_result_type(body.args)

    return LoopOp(init_values, body, result_type), phi_ssa_indices
end

"""
    extend_loop_terminators!(body::Block, carried_values::Vector, captured_args::Vector{BlockArg})

Extend ContinueOp and BreakOp terminators in a LoopOp body to include captured block args.
The captured args are passed through unchanged in both continue and break paths.
"""
function extend_loop_terminators!(body::Block, carried_values::Vector, captured_args::Vector{BlockArg})
    for (i, op) in enumerate(body.ops)
        if op.expr isa IfOp
            # Extend ContinueOp in then_block
            then_term = op.expr.then_block.terminator
            if then_term isa ContinueOp
                # Add captured args to continue values
                new_values = vcat(then_term.values, captured_args)
                op.expr.then_block.terminator = ContinueOp(new_values)
            end
            # Extend BreakOp in else_block
            else_term = op.expr.else_block.terminator
            if else_term isa BreakOp
                # Add captured args to break values
                new_values = vcat(else_term.values, captured_args)
                op.expr.else_block.terminator = BreakOp(new_values)
            end
        end
    end
    # Also extend the body terminator if it's a ContinueOp
    if body.terminator isa ContinueOp
        new_values = vcat(body.terminator.values, captured_args)
        body.terminator = ContinueOp(new_values)
    end
end

"""
    compute_result_type(block_args::Vector{BlockArg}) -> Type

Compute the result type for a control flow op based on its block arguments.
- 0 args: Nothing
- 1 arg: the single type
- 2+ args: Tuple{types...}
"""
function compute_result_type(block_args::Vector{BlockArg})
    if isempty(block_args)
        return Nothing
    elseif length(block_args) == 1
        return block_args[1].type
    else
        return Tuple{(arg.type for arg in block_args)...}
    end
end

