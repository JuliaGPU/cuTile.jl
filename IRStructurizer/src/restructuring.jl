# control tree to structured IR conversion
#
# Converts ControlTree (from analysis.jl) into structured IR (Block, IfOp, etc.)
# by dispatching on region types and calling pattern-specific builders.

export StructuredCodeInfo, structurize!

using Graphs: SimpleDiGraph, add_edge!, vertices, edges, nv, ne,
              inneighbors, outneighbors, Edge

#=============================================================================
 Control Tree to Structured IR Conversion
=============================================================================#

"""
    control_tree_to_structured_ir(ctree::ControlTree, code::CodeInfo, blocks::Vector{BlockInfo}; loop_patterning=true) -> Block

Convert a control tree to structured IR entry block.
"""
function control_tree_to_structured_ir(ctree::ControlTree, code::CodeInfo, blocks::Vector{BlockInfo}; loop_patterning::Bool=true)
    block_id = Ref(1)
    entry_block = tree_to_block(ctree, code, blocks, block_id; loop_patterning)
    return entry_block
end

"""
    tree_to_block(tree::ControlTree, code::CodeInfo, blocks::Vector{BlockInfo}, block_id::Ref{Int}; loop_patterning=true) -> Block

Convert a control tree node to a Block.
"""
function tree_to_block(tree::ControlTree, code::CodeInfo, blocks::Vector{BlockInfo}, block_id::Ref{Int}; loop_patterning::Bool=true)
    idx = node_index(tree)
    rtype = region_type(tree)
    id = block_id[]
    block_id[] += 1

    block = Block(id)

    if rtype == REGION_BLOCK
        handle_block_region!(block, tree, code, blocks, block_id; loop_patterning)
    elseif rtype == REGION_IF_THEN_ELSE
        handle_if_then_else!(block, tree, code, blocks, block_id; loop_patterning)
    elseif rtype == REGION_IF_THEN
        handle_if_then!(block, tree, code, blocks, block_id; loop_patterning)
    elseif rtype == REGION_TERMINATION
        handle_termination!(block, tree, code, blocks, block_id; loop_patterning)
    elseif rtype == REGION_WHILE_LOOP || rtype == REGION_NATURAL_LOOP
        handle_loop!(block, tree, code, blocks, block_id; loop_patterning)
    elseif rtype == REGION_SELF_LOOP
        handle_self_loop!(block, tree, code, blocks, block_id)
    elseif rtype == REGION_PROPER
        handle_proper_region!(block, tree, code, blocks, block_id; loop_patterning)
    elseif rtype == REGION_SWITCH
        handle_switch!(block, tree, code, blocks, block_id; loop_patterning)
    else
        # Fallback: collect statements
        handle_block_region!(block, tree, code, blocks, block_id; loop_patterning)
    end

    # Set terminator if not already set
    set_block_terminator!(block, code, blocks)

    return block
end

"""
    handle_block_region!(block::Block, tree::ControlTree, code::CodeInfo, blocks::Vector{BlockInfo}, block_id::Ref{Int}; loop_patterning=true)

Handle REGION_BLOCK - a linear sequence of blocks.
"""
function handle_block_region!(block::Block, tree::ControlTree, code::CodeInfo, blocks::Vector{BlockInfo}, block_id::Ref{Int}; loop_patterning::Bool=true)
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
                handle_block_region!(block, child, code, blocks, block_id; loop_patterning)
            else
                # Nested control flow - create appropriate op
                handle_nested_region!(block, child, code, blocks, block_id; loop_patterning)
            end
        end
    end
end

"""
    handle_nested_region!(block::Block, tree::ControlTree, code::CodeInfo, blocks::Vector{BlockInfo}, block_id::Ref{Int}; loop_patterning=true)

Handle a nested control flow region.
"""
function handle_nested_region!(block::Block, tree::ControlTree, code::CodeInfo, blocks::Vector{BlockInfo}, block_id::Ref{Int}; loop_patterning::Bool=true)
    rtype = region_type(tree)

    if rtype == REGION_IF_THEN_ELSE
        handle_if_then_else!(block, tree, code, blocks, block_id; loop_patterning)
    elseif rtype == REGION_IF_THEN
        handle_if_then!(block, tree, code, blocks, block_id; loop_patterning)
    elseif rtype == REGION_TERMINATION
        handle_termination!(block, tree, code, blocks, block_id; loop_patterning)
    elseif rtype == REGION_WHILE_LOOP || rtype == REGION_NATURAL_LOOP
        handle_loop!(block, tree, code, blocks, block_id; loop_patterning)
    elseif rtype == REGION_SELF_LOOP
        handle_self_loop!(block, tree, code, blocks, block_id)
    elseif rtype == REGION_PROPER
        handle_proper_region!(block, tree, code, blocks, block_id; loop_patterning)
    elseif rtype == REGION_SWITCH
        handle_switch!(block, tree, code, blocks, block_id; loop_patterning)
    else
        handle_block_region!(block, tree, code, blocks, block_id; loop_patterning)
    end
end

"""
    handle_if_then_else!(block::Block, tree::ControlTree, code::CodeInfo, blocks::Vector{BlockInfo}, block_id::Ref{Int}; loop_patterning=true)

Handle REGION_IF_THEN_ELSE.
"""
function handle_if_then_else!(block::Block, tree::ControlTree, code::CodeInfo, blocks::Vector{BlockInfo}, block_id::Ref{Int}; loop_patterning::Bool=true)
    tree_children = children(tree)
    length(tree_children) >= 3 || return handle_block_region!(block, tree, code, blocks, block_id; loop_patterning)

    # First child is the condition block
    cond_tree = tree_children[1]
    cond_idx = node_index(cond_tree)

    # Collect condition block statements and find condition
    if 1 <= cond_idx <= length(blocks)
        cond_block = blocks[cond_idx]
        for si in cond_block.range
            stmt = code.code[si]
            if !(stmt isa GotoNode || stmt isa GotoIfNot || stmt isa ReturnNode || stmt isa PhiNode)
                push!(block.body, si)
            end
        end
    end

    cond_value = find_condition_value(cond_idx, code, blocks)

    # Then and else blocks
    then_tree = tree_children[2]
    else_tree = tree_children[3]

    then_block = tree_to_block(then_tree, code, blocks, block_id; loop_patterning)
    else_block = tree_to_block(else_tree, code, blocks, block_id; loop_patterning)

    # Create IfOp
    if_op = IfOp(cond_value, then_block, else_block, SSAValue[])
    push!(block.body, if_op)
end

"""
    handle_if_then!(block::Block, tree::ControlTree, code::CodeInfo, blocks::Vector{BlockInfo}, block_id::Ref{Int}; loop_patterning=true)

Handle REGION_IF_THEN.
"""
function handle_if_then!(block::Block, tree::ControlTree, code::CodeInfo, blocks::Vector{BlockInfo}, block_id::Ref{Int}; loop_patterning::Bool=true)
    tree_children = children(tree)
    length(tree_children) >= 2 || return handle_block_region!(block, tree, code, blocks, block_id; loop_patterning)

    # First child is the condition block
    cond_tree = tree_children[1]
    cond_idx = node_index(cond_tree)

    # Collect condition block statements
    if 1 <= cond_idx <= length(blocks)
        cond_block = blocks[cond_idx]
        for si in cond_block.range
            stmt = code.code[si]
            if !(stmt isa GotoNode || stmt isa GotoIfNot || stmt isa ReturnNode || stmt isa PhiNode)
                push!(block.body, si)
            end
        end
    end

    cond_value = find_condition_value(cond_idx, code, blocks)

    # Then block
    then_tree = tree_children[2]
    then_block = tree_to_block(then_tree, code, blocks, block_id; loop_patterning)

    # Empty else block
    else_block = Block(block_id[])
    block_id[] += 1

    if_op = IfOp(cond_value, then_block, else_block, SSAValue[])
    push!(block.body, if_op)
end

"""
    handle_termination!(block::Block, tree::ControlTree, code::CodeInfo, blocks::Vector{BlockInfo}, block_id::Ref{Int}; loop_patterning=true)

Handle REGION_TERMINATION - branches where some paths terminate.
"""
function handle_termination!(block::Block, tree::ControlTree, code::CodeInfo, blocks::Vector{BlockInfo}, block_id::Ref{Int}; loop_patterning::Bool=true)
    tree_children = children(tree)
    isempty(tree_children) && return handle_block_region!(block, tree, code, blocks, block_id; loop_patterning)

    # First child is the condition block
    cond_tree = tree_children[1]
    cond_idx = node_index(cond_tree)

    # Collect condition block statements
    if 1 <= cond_idx <= length(blocks)
        cond_block = blocks[cond_idx]
        for si in cond_block.range
            stmt = code.code[si]
            if !(stmt isa GotoNode || stmt isa GotoIfNot || stmt isa ReturnNode || stmt isa PhiNode)
                push!(block.body, si)
            end
        end
    end

    cond_value = find_condition_value(cond_idx, code, blocks)

    # Build then and else blocks from remaining children
    if length(tree_children) >= 3
        then_tree = tree_children[2]
        else_tree = tree_children[3]
        then_block = tree_to_block(then_tree, code, blocks, block_id; loop_patterning)
        else_block = tree_to_block(else_tree, code, blocks, block_id; loop_patterning)
        if_op = IfOp(cond_value, then_block, else_block, SSAValue[])
        push!(block.body, if_op)
    elseif length(tree_children) == 2
        then_tree = tree_children[2]
        then_block = tree_to_block(then_tree, code, blocks, block_id; loop_patterning)
        else_block = Block(block_id[])
        block_id[] += 1
        if_op = IfOp(cond_value, then_block, else_block, SSAValue[])
        push!(block.body, if_op)
    end
end

"""
    handle_loop!(block::Block, tree::ControlTree, code::CodeInfo, blocks::Vector{BlockInfo}, block_id::Ref{Int}; loop_patterning=true)

Handle REGION_WHILE_LOOP and REGION_NATURAL_LOOP.
When loop_patterning=true, tries loop patterns in order: ForOp > WhileOp > LoopOp.
When loop_patterning=false, uses LoopOp directly.
"""
function handle_loop!(block::Block, tree::ControlTree, code::CodeInfo, blocks::Vector{BlockInfo}, block_id::Ref{Int}; loop_patterning::Bool=true)
    if loop_patterning
        # Try to detect for-loop pattern first (most structured)
        for_pattern = detect_for_loop_pattern(tree, code, blocks)
        if for_pattern !== nothing
            for_op = build_for_op(tree, code, blocks, for_pattern, block_id; loop_patterning)
            push!(block.body, for_op)
            return
        end

        # Try to detect while-loop pattern (condition at header)
        while_pattern = detect_while_loop_pattern(tree, code, blocks)
        if while_pattern !== nothing
            while_op = build_while_op(tree, code, blocks, while_pattern, block_id; loop_patterning)
            push!(block.body, while_op)
            return
        end
    end

    # Fallback (or when loop_patterning=false): Build general LoopOp
    loop_op = build_loop_op(tree, code, blocks, block_id; loop_patterning)
    push!(block.body, loop_op)
end

"""
    handle_self_loop!(block::Block, tree::ControlTree, code::CodeInfo, blocks::Vector{BlockInfo}, block_id::Ref{Int})

Handle REGION_SELF_LOOP.
"""
function handle_self_loop!(block::Block, tree::ControlTree, code::CodeInfo, blocks::Vector{BlockInfo}, block_id::Ref{Int})
    idx = node_index(tree)

    body_block = Block(block_id[])
    block_id[] += 1

    if 1 <= idx <= length(blocks)
        collect_block_statements!(body_block, blocks[idx], code)
    end

    loop_op = LoopOp(IRValue[], body_block, SSAValue[])
    push!(block.body, loop_op)
end

"""
    handle_proper_region!(block::Block, tree::ControlTree, code::CodeInfo, blocks::Vector{BlockInfo}, block_id::Ref{Int}; loop_patterning=true)

Handle REGION_PROPER - acyclic region not matching other patterns.
"""
function handle_proper_region!(block::Block, tree::ControlTree, code::CodeInfo, blocks::Vector{BlockInfo}, block_id::Ref{Int}; loop_patterning::Bool=true)
    # Process as a sequence of blocks
    handle_block_region!(block, tree, code, blocks, block_id; loop_patterning)
end

"""
    handle_switch!(block::Block, tree::ControlTree, code::CodeInfo, blocks::Vector{BlockInfo}, block_id::Ref{Int}; loop_patterning=true)

Handle REGION_SWITCH.
"""
function handle_switch!(block::Block, tree::ControlTree, code::CodeInfo, blocks::Vector{BlockInfo}, block_id::Ref{Int}; loop_patterning::Bool=true)
    # For now, handle as a nested if-else chain
    # TODO: Implement proper switch handling if needed
    handle_block_region!(block, tree, code, blocks, block_id; loop_patterning)
end

"""
    collect_block_statements!(block::Block, info::BlockInfo, code::CodeInfo)

Collect statements from a BlockInfo into a Block, excluding control flow.
"""
function collect_block_statements!(block::Block, info::BlockInfo, code::CodeInfo)
    stmts = code.code
    for si in info.range
        stmt = stmts[si]
        if stmt isa ReturnNode
            block.terminator = stmt
        elseif !(stmt isa GotoNode || stmt isa GotoIfNot || stmt isa PhiNode)
            push!(block.body, si)
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

    # Find the last statement index in body
    last_idx = nothing
    for item in reverse(block.body)
        if item isa Int
            last_idx = item
            break
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
 Public API
=============================================================================#

"""
    structurize!(sci::StructuredCodeInfo; loop_patterning=true) -> StructuredCodeInfo

Convert unstructured control flow in `sci` to structured control flow operations
(IfOp, ForOp, WhileOp, LoopOp) in-place.

This transforms GotoNode and GotoIfNot statements into nested structured ops
that can be traversed hierarchically.

When `loop_patterning=true` (default), loops are classified as ForOp (bounded counters)
or WhileOp (condition-based). When `false`, all loops become LoopOp.

Returns `sci` for convenience (allows chaining).
"""
function structurize!(sci::StructuredCodeInfo; loop_patterning::Bool=true)
    code = sci.code
    stmts = code.code
    n = length(stmts)

    n == 0 && return sci

    # Check if the code is straight-line (no control flow)
    has_control_flow = any(s -> s isa GotoNode || s isa GotoIfNot, stmts)

    if !has_control_flow
        # Straight-line code
        new_entry = Block(1)
        for i in 1:n
            stmt = stmts[i]
            if stmt isa ReturnNode
                new_entry.terminator = stmt
            elseif !(stmt isa GotoNode || stmt isa GotoIfNot)
                push!(new_entry.body, i)
            end
        end
        sci.entry = new_entry
        return sci
    end

    # Build block-level CFG
    blocks, cfg = build_block_cfg(code)

    # Build control tree using SPIRV.jl-style graph contraction
    ctree = ControlTree(cfg)

    # Convert control tree to structured IR
    sci.entry = control_tree_to_structured_ir(ctree, code, blocks; loop_patterning)

    return sci
end
