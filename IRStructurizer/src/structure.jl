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
                push!(block.body, Statement(si, stmt, code.ssavaluetypes[si]))
            end
        end
    end

    cond_value = find_condition_value(cond_idx, code, blocks)

    # Then and else blocks
    then_tree = tree_children[2]
    else_tree = tree_children[3]

    then_block = tree_to_block(then_tree, code, blocks, block_id)
    else_block = tree_to_block(else_tree, code, blocks, block_id)

    # Create IfOp
    if_op = IfOp(cond_value, then_block, else_block, SSAValue[])
    push!(block.body, if_op)
end

"""
    handle_if_then!(block::Block, tree::ControlTree, code::CodeInfo, blocks::Vector{BlockInfo}, block_id::Ref{Int})

Handle REGION_IF_THEN.
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
                push!(block.body, Statement(si, stmt, code.ssavaluetypes[si]))
            end
        end
    end

    cond_value = find_condition_value(cond_idx, code, blocks)

    # Then block
    then_tree = tree_children[2]
    then_block = tree_to_block(then_tree, code, blocks, block_id)

    # Empty else block
    else_block = Block(block_id[])
    block_id[] += 1

    if_op = IfOp(cond_value, then_block, else_block, SSAValue[])
    push!(block.body, if_op)
end

"""
    handle_termination!(block::Block, tree::ControlTree, code::CodeInfo, blocks::Vector{BlockInfo}, block_id::Ref{Int})

Handle REGION_TERMINATION - branches where some paths terminate.
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
                push!(block.body, Statement(si, stmt, code.ssavaluetypes[si]))
            end
        end
    end

    cond_value = find_condition_value(cond_idx, code, blocks)

    # Build then and else blocks from remaining children
    if length(tree_children) >= 3
        then_tree = tree_children[2]
        else_tree = tree_children[3]
        then_block = tree_to_block(then_tree, code, blocks, block_id)
        else_block = tree_to_block(else_tree, code, blocks, block_id)
        if_op = IfOp(cond_value, then_block, else_block, SSAValue[])
        push!(block.body, if_op)
    elseif length(tree_children) == 2
        then_tree = tree_children[2]
        then_block = tree_to_block(then_tree, code, blocks, block_id)
        else_block = Block(block_id[])
        block_id[] += 1
        if_op = IfOp(cond_value, then_block, else_block, SSAValue[])
        push!(block.body, if_op)
    end
end

"""
    handle_loop!(block::Block, tree::ControlTree, code::CodeInfo, blocks::Vector{BlockInfo}, block_id::Ref{Int})

Handle REGION_WHILE_LOOP and REGION_NATURAL_LOOP.
Phase 1: Always creates LoopOp with metadata. Pattern matching happens in Phase 2.
"""
function handle_loop!(block::Block, tree::ControlTree, code::CodeInfo, blocks::Vector{BlockInfo}, block_id::Ref{Int})
    loop_op = build_loop_op_phase1(tree, code, blocks, block_id)
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
            push!(block.body, Statement(si, stmt, types[si]))
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
        if item isa Statement
            last_idx = item.idx
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
 Loop Construction
=============================================================================#

"""
    build_loop_op_phase1(tree::ControlTree, code::CodeInfo, blocks::Vector{BlockInfo}, block_id::Ref{Int}) -> LoopOp

Build a LoopOp for Phase 1. No substitutions applied yet.
Pattern detection and substitution happens in Phase 2.
"""
function build_loop_op_phase1(tree::ControlTree, code::CodeInfo, blocks::Vector{BlockInfo}, block_id::Ref{Int})
    stmts = code.code
    types = code.ssavaluetypes
    header_idx = node_index(tree)
    loop_blocks = get_loop_blocks(tree, blocks)

    @assert 1 <= header_idx <= length(blocks) "Invalid header_idx from control tree: $header_idx"
    header_block = blocks[header_idx]
    stmt_to_blk = stmt_to_block_map(blocks, length(stmts))

    # Find phi nodes in header - these become loop-carried values and results
    init_values = IRValue[]
    carried_values = IRValue[]
    block_args = BlockArg[]
    result_vars = SSAValue[]

    for si in header_block.range
        stmt = stmts[si]
        if stmt isa PhiNode
            push!(result_vars, SSAValue(si))
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

            entry_val !== nothing && push!(init_values, entry_val)
            carried_val !== nothing && push!(carried_values, carried_val)

            phi_type = types[si]
            push!(block_args, BlockArg(length(block_args) + 1, phi_type))
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

    # Collect header statements (excluding phi nodes and control flow) - NO SUBSTITUTION
    for si in header_block.range
        stmt = stmts[si]
        if !(stmt isa PhiNode || stmt isa GotoNode || stmt isa GotoIfNot || stmt isa ReturnNode)
            push!(body.body, Statement(si, stmt, types[si]))
        end
    end

    # IMPORTANT: Compute phi→BlockArg substitutions BEFORE creating ContinueOp.
    # This fixes the phi-referencing-phi bug where carried_values contain SSAValues
    # that reference other phi nodes (which should be BlockArgs at this point).
    subs = Substitutions()
    for (i, result_var) in enumerate(result_vars)
        subs[result_var.id] = block_args[i]
    end

    # Apply substitutions to carried values
    substituted_carried = [substitute_ssa(v, subs) for v in carried_values]

    # Apply substitutions to header statements already in body
    substitute_block!(body, subs)

    # Create the conditional structure inside the loop body
    if condition !== nothing
        cond_value = convert_phi_value(condition)
        # Also substitute the condition if it references a phi
        cond_value = substitute_ssa(cond_value, subs)

        then_block = Block(block_id[])
        block_id[] += 1

        # Process loop body blocks (excluding header)
        for child in children(tree)
            child_idx = node_index(child)
            if child_idx != header_idx
                handle_block_region!(then_block, child, code, blocks, block_id)
            end
        end
        # Apply substitutions to then_block (body statements might reference phi nodes)
        substitute_block!(then_block, subs)
        # Substituted carried values (phi refs → BlockArg refs)
        then_block.terminator = ContinueOp(substituted_carried)

        else_block = Block(block_id[])
        block_id[] += 1
        # Block args are the references for break
        result_values = IRValue[]
        for arg in block_args
            push!(result_values, arg)
        end
        else_block.terminator = BreakOp(result_values)

        if_op = IfOp(cond_value, then_block, else_block, SSAValue[])
        push!(body.body, if_op)
    else
        # No condition - process children directly
        for child in children(tree)
            child_idx = node_index(child)
            if child_idx != header_idx
                handle_block_region!(body, child, code, blocks, block_id)
            end
        end
        # Apply substitutions to body (child statements might reference phi nodes)
        # Note: substitute_block! was already called on header statements, but
        # this call will handle the children that were added after
        substitute_block!(body, subs)
        body.terminator = ContinueOp(substituted_carried)
    end

    return LoopOp(init_values, body, result_vars)
end

#=============================================================================
 Phase 1 Completion: Apply SSA Substitutions (outer → inner)
=============================================================================#

"""
    apply_loop_substitutions!(block::Block)

Apply SSA→BlockArg substitutions to all loops, processing outer→inner.
Must be called after Phase 1 to complete the structured IR.
"""
function apply_loop_substitutions!(block::Block)
    for item in block.body
        if item isa LoopOp
            subs = compute_loop_subs(item)
            substitute_block!(item.body, subs)
            apply_loop_substitutions!(item.body)
        elseif item isa IfOp
            apply_loop_substitutions!(item.then_block)
            apply_loop_substitutions!(item.else_block)
        end
    end
end

"""
    compute_loop_subs(loop::LoopOp) -> Substitutions

Compute the SSA→BlockArg substitutions for a loop.
Maps each phi node SSA index to its corresponding block argument.
"""
function compute_loop_subs(loop::LoopOp)
    @assert length(loop.result_vars) == length(loop.body.args) "Mismatch between result_vars and body.args"
    subs = Substitutions()
    for (i, result_var) in enumerate(loop.result_vars)
        subs[result_var.id] = loop.body.args[i]
    end
    return subs
end
