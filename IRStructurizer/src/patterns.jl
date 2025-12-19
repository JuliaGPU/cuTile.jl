# loop classification through IR patterning
#
# This file contains pattern detection for specialized loop constructs:
# - ForOp: Counted for-loop with known bounds and step
# - WhileOp: Simple while-loop with explicit condition at loop header
# - LoopOp: General loop with dynamic exit (fallback)
#
# The detection follows a priority order: ForOp > WhileOp > LoopOp

#=============================================================================
 For-Loop Pattern Detection
=============================================================================#

"""
    ForLoopPattern

Detected for-loop pattern with all information needed to build a ForOp.
Contains bounds, induction variable info, and pre-extracted carried values.
"""
struct ForLoopPattern
    # Induction variable
    lower::IRValue           # Initial value of induction variable
    upper::IRValue           # Upper bound (exclusive)
    step::IRValue            # Step value
    induction_phi_idx::Int   # SSA index of induction variable phi
    iv_incr_idx::Int         # Statement index of add_int for induction var

    # Loop structure
    header_block::Int        # Block index of loop header
    loop_blocks::Set{Int}    # All block indices in the loop

    # Carried values (other phi nodes besides induction var)
    carried_phi_indices::Vector{Int}   # SSA indices of carried phi nodes
    init_values::Vector{IRValue}       # Initial values (from outside loop)
    yield_values::Vector{IRValue}      # Yielded values (from inside loop)
end

"""
    detect_for_loop_pattern(tree::ControlTree, code::CodeInfo, blocks::Vector{BlockInfo}) -> Union{ForLoopPattern, Nothing}

Detect if a loop is a simple counted for-loop with pattern:
- Header has phi for induction var: φ(init, next)
- Condition is slt_int(iv, upper) or similar
- Body has iv_next = add_int(iv, step)

Returns a complete ForLoopPattern with all information needed to build a ForOp,
or nothing if the pattern doesn't match.
"""
function detect_for_loop_pattern(tree::ControlTree, code::CodeInfo, blocks::Vector{BlockInfo})
    stmts = code.code
    header_idx = node_index(tree)
    header_idx < 1 || header_idx > length(blocks) && return nothing
    header = blocks[header_idx]

    # Find the GotoIfNot condition in header
    cond_var = nothing
    for si in header.range
        stmt = stmts[si]
        if stmt isa GotoIfNot
            cond_var = stmt.cond
            break
        end
    end
    cond_var === nothing && return nothing
    cond_var isa SSAValue || return nothing

    # Trace to comparison: slt_int(%iv, %upper)
    cond_var.id < 1 || cond_var.id > length(stmts) && return nothing
    cond_stmt = stmts[cond_var.id]
    cond_stmt isa Expr || return nothing
    cond_stmt.head === :call || return nothing

    func = cond_stmt.args[1]
    is_less_than = func isa GlobalRef && func.name in (:slt_int, :ult_int)
    is_less_than || return nothing

    iv_candidate = cond_stmt.args[2]  # induction variable
    upper_bound_raw = cond_stmt.args[3]   # upper bound

    # The iv_candidate should be a phi node in the header
    iv_candidate isa SSAValue || return nothing
    iv_phi_idx = iv_candidate.id
    iv_phi_idx < 1 || iv_phi_idx > length(stmts) && return nothing
    iv_phi_stmt = stmts[iv_phi_idx]
    iv_phi_stmt isa PhiNode || return nothing

    # Verify the phi is in the header
    iv_phi_idx in header.range || return nothing

    # Get all loop blocks from tree
    loop_blocks = get_loop_blocks(tree, blocks)
    stmt_to_blk = stmt_to_block_map(blocks, length(stmts))

    # Extract lower bound (init value from outside loop)
    lower_bound = nothing
    for (edge_idx, _) in enumerate(iv_phi_stmt.edges)
        if isassigned(iv_phi_stmt.values, edge_idx)
            val = iv_phi_stmt.values[edge_idx]
            if !is_value_in_loop(val, stmts, stmt_to_blk, loop_blocks)
                lower_bound = convert_phi_value(val)
                break
            end
        end
    end
    lower_bound === nothing && return nothing

    # Find step and iv_incr_idx by looking for add_int(iv_phi, step) in loop body
    step_raw = nothing
    iv_incr_idx = 0
    for bi in loop_blocks
        bi < 1 || bi > length(blocks) && continue
        for si in blocks[bi].range
            stmt = stmts[si]
            if stmt isa Expr && stmt.head === :call
                func = stmt.args[1]
                if func isa GlobalRef && func.name === :add_int
                    if length(stmt.args) >= 2 && stmt.args[2] isa SSAValue && stmt.args[2].id == iv_phi_idx
                        step_raw = stmt.args[3]
                        iv_incr_idx = si
                        break
                    end
                end
            end
        end
        step_raw !== nothing && break
    end
    step_raw === nothing && return nothing

    # Verify step and upper bound are loop-invariant (not modified inside the loop)
    # A valid for-loop requires: step and upper bound are constants, Arguments, or SSAValues defined outside the loop
    if is_value_in_loop(step_raw, stmts, stmt_to_blk, loop_blocks)
        return nothing  # Step changes inside loop, not a valid for-loop
    end
    if is_value_in_loop(upper_bound_raw, stmts, stmt_to_blk, loop_blocks)
        return nothing  # Upper bound changes inside loop, not a valid for-loop
    end
    step = convert_phi_value(step_raw)
    upper_bound = convert_phi_value(upper_bound_raw)

    # Extract carried phi nodes (other phis in header besides induction var)
    carried_phi_indices = Int[]
    init_values = IRValue[]
    yield_values = IRValue[]

    for si in header.range
        stmt = stmts[si]
        if stmt isa PhiNode && si != iv_phi_idx
            push!(carried_phi_indices, si)

            # Extract init value (from outside loop) and yield value (from inside loop)
            for (edge_idx, _) in enumerate(stmt.edges)
                if isassigned(stmt.values, edge_idx)
                    val = stmt.values[edge_idx]
                    if is_value_in_loop(val, stmts, stmt_to_blk, loop_blocks)
                        push!(yield_values, convert_phi_value(val))
                    else
                        push!(init_values, convert_phi_value(val))
                    end
                end
            end
        end
    end

    return ForLoopPattern(
        lower_bound, upper_bound, step,
        iv_phi_idx, iv_incr_idx, header_idx, loop_blocks,
        carried_phi_indices, init_values, yield_values
    )
end

#=============================================================================
 While-Loop Pattern Detection
=============================================================================#

"""
    WhileLoopPattern

Detected while-loop pattern with explicit condition.
Simpler than LoopOp - the condition is at the loop header and determines
whether to continue or exit.
"""
struct WhileLoopPattern
    # Condition
    condition::IRValue       # The condition SSA value
    condition_idx::Int       # Statement index of the condition

    # Loop structure
    header_block::Int        # Block index of loop header
    loop_blocks::Set{Int}    # All block indices in the loop

    # Carried values (phi nodes in header)
    carried_phi_indices::Vector{Int}   # SSA indices of carried phi nodes
    init_values::Vector{IRValue}       # Initial values (from outside loop)
    yield_values::Vector{IRValue}      # Yielded values (from inside loop)
end

"""
    detect_while_loop_pattern(tree::ControlTree, code::CodeInfo, blocks::Vector{BlockInfo}) -> Union{WhileLoopPattern, Nothing}

Detect if a loop is a simple while-loop with pattern:
- Header ends with GotoIfNot (condition check)
- Condition is a single SSA value
- False branch goes outside loop (exit)
- True branch continues to body then back to header

Returns a WhileLoopPattern if detected, or nothing if the pattern doesn't match.
"""
function detect_while_loop_pattern(tree::ControlTree, code::CodeInfo, blocks::Vector{BlockInfo})
    stmts = code.code
    header_idx = node_index(tree)
    header_idx < 1 || header_idx > length(blocks) && return nothing
    header = blocks[header_idx]

    # Find the GotoIfNot in header - this is the while condition check
    cond_var = nothing
    cond_idx = 0
    for si in header.range
        stmt = stmts[si]
        if stmt isa GotoIfNot
            cond_var = stmt.cond
            # The condition is the SSAValue used in GotoIfNot
            if cond_var isa SSAValue
                cond_idx = cond_var.id
            end
            break
        end
    end
    cond_var === nothing && return nothing

    # The condition must be a simple SSAValue (not complex control flow)
    cond_var isa SSAValue || return nothing

    # Get all loop blocks from tree
    loop_blocks = get_loop_blocks(tree, blocks)
    stmt_to_blk = stmt_to_block_map(blocks, length(stmts))

    # Extract carried phi nodes from header
    carried_phi_indices = Int[]
    init_values = IRValue[]
    yield_values = IRValue[]

    for si in header.range
        stmt = stmts[si]
        if stmt isa PhiNode
            push!(carried_phi_indices, si)

            # Extract init value (from outside loop) and yield value (from inside loop)
            entry_val = nothing
            carried_val = nothing
            for (edge_idx, _) in enumerate(stmt.edges)
                if isassigned(stmt.values, edge_idx)
                    val = stmt.values[edge_idx]
                    if is_value_in_loop(val, stmts, stmt_to_blk, loop_blocks)
                        carried_val = convert_phi_value(val)
                    else
                        entry_val = convert_phi_value(val)
                    end
                end
            end
            entry_val !== nothing && push!(init_values, entry_val)
            carried_val !== nothing && push!(yield_values, carried_val)
        end
    end

    return WhileLoopPattern(
        cond_var, cond_idx, header_idx, loop_blocks,
        carried_phi_indices, init_values, yield_values
    )
end

#=============================================================================
 Loop Building Functions
=============================================================================#

"""
    build_for_op(tree::ControlTree, code::CodeInfo, blocks::Vector{BlockInfo},
                 pattern::ForLoopPattern, block_id::Ref{Int}; loop_patterning=true) -> ForOp

Build a ForOp from a pre-analyzed ForLoopPattern.
Uses the pre-computed values from pattern detection to avoid redundant IR traversals.
"""
function build_for_op(tree::ControlTree, code::CodeInfo, blocks::Vector{BlockInfo},
                      pattern::ForLoopPattern, block_id::Ref{Int}; loop_patterning::Bool=true)
    # Build block args from pattern
    block_args = BlockArg[]

    # Induction variable is first block arg
    iv_type = code.ssavaluetypes[pattern.induction_phi_idx]
    push!(block_args, BlockArg(1, iv_type))

    # Other carried phis become additional block args
    result_vars = SSAValue[]
    for phi_idx in pattern.carried_phi_indices
        push!(result_vars, SSAValue(phi_idx))
        phi_type = code.ssavaluetypes[phi_idx]
        push!(block_args, BlockArg(length(block_args) + 1, phi_type))
    end

    # Build body block
    body = Block(block_id[])
    block_id[] += 1
    body.args = block_args

    # Process body using tree structure to handle nested loops
    header_idx = node_index(tree)
    for child in children(tree)
        child_idx = node_index(child)
        child_idx == header_idx && continue  # Skip header

        child_rtype = region_type(child)
        if child_rtype == REGION_WHILE_LOOP || child_rtype == REGION_NATURAL_LOOP
            handle_loop!(body, child, code, blocks, block_id; loop_patterning)
        elseif child_rtype == REGION_IF_THEN || child_rtype == REGION_IF_THEN_ELSE
            handle_nested_region!(body, child, code, blocks, block_id; loop_patterning)
        elseif child_rtype == REGION_BLOCK
            process_for_body_region!(body, child, code, blocks, block_id, pattern.iv_incr_idx; loop_patterning)
        else
            handle_nested_region!(body, child, code, blocks, block_id; loop_patterning)
        end
    end

    # Use pre-computed yield values
    body.terminator = ContinueOp(pattern.yield_values)

    iv_ssa = SSAValue(pattern.induction_phi_idx)
    return ForOp(pattern.lower, pattern.upper, pattern.step, iv_ssa,
                 pattern.init_values, body, result_vars)
end

"""
    process_for_body_region!(block::Block, tree::ControlTree, code::CodeInfo, blocks::Vector{BlockInfo}, block_id::Ref{Int}, iv_incr_idx; loop_patterning=true)

Process a REGION_BLOCK inside a ForOp body, handling nested loops and skipping induction variable operations.
"""
function process_for_body_region!(block::Block, tree::ControlTree, code::CodeInfo, blocks::Vector{BlockInfo}, block_id::Ref{Int}, iv_incr_idx; loop_patterning::Bool=true)
    stmts = code.code

    if isempty(children(tree))
        # Leaf node - collect statements from the block
        idx = node_index(tree)
        if 1 <= idx <= length(blocks)
            info = blocks[idx]
            for si in info.range
                stmt = stmts[si]
                if stmt isa ReturnNode
                    block.terminator = stmt
                elseif stmt isa PhiNode || stmt isa GotoNode || stmt isa GotoIfNot
                    continue
                elseif si == iv_incr_idx
                    continue
                elseif stmt isa Expr && stmt.head === :call
                    func = stmt.args[1]
                    if func isa GlobalRef && func.name in (:slt_int, :ult_int)
                        continue
                    end
                    push!(block.body, si)
                else
                    push!(block.body, si)
                end
            end
        end
    else
        # Non-leaf - process children in order
        for child in children(tree)
            child_rtype = region_type(child)
            if child_rtype == REGION_WHILE_LOOP || child_rtype == REGION_NATURAL_LOOP
                handle_loop!(block, child, code, blocks, block_id; loop_patterning)
            elseif child_rtype == REGION_BLOCK
                process_for_body_region!(block, child, code, blocks, block_id, iv_incr_idx; loop_patterning)
            else
                handle_nested_region!(block, child, code, blocks, block_id; loop_patterning)
            end
        end
    end
end

"""
    build_while_op(tree::ControlTree, code::CodeInfo, blocks::Vector{BlockInfo},
                   pattern::WhileLoopPattern, block_id::Ref{Int}; loop_patterning=true) -> WhileOp

Build a WhileOp from a pre-analyzed WhileLoopPattern.
"""
function build_while_op(tree::ControlTree, code::CodeInfo, blocks::Vector{BlockInfo},
                        pattern::WhileLoopPattern, block_id::Ref{Int}; loop_patterning::Bool=true)
    stmts = code.code

    # Build block args from carried phi nodes
    block_args = BlockArg[]
    result_vars = SSAValue[]

    for phi_idx in pattern.carried_phi_indices
        push!(result_vars, SSAValue(phi_idx))
        phi_type = code.ssavaluetypes[phi_idx]
        push!(block_args, BlockArg(length(block_args) + 1, phi_type))
    end

    # Build body block
    body = Block(block_id[])
    block_id[] += 1
    body.args = block_args

    # Process loop body blocks (excluding header condition check and phi nodes)
    header_idx = node_index(tree)
    header = blocks[header_idx]

    # Collect header statements (excluding phi nodes and control flow)
    # Include the condition computation so it's emitted before the condition check
    for si in header.range
        stmt = stmts[si]
        if !(stmt isa PhiNode || stmt isa GotoNode || stmt isa GotoIfNot || stmt isa ReturnNode)
            push!(body.body, si)
        end
    end

    # Process child regions (loop body blocks)
    for child in children(tree)
        child_idx = node_index(child)
        child_idx == header_idx && continue  # Skip header

        child_rtype = region_type(child)
        if child_rtype == REGION_WHILE_LOOP || child_rtype == REGION_NATURAL_LOOP
            handle_loop!(body, child, code, blocks, block_id; loop_patterning)
        elseif child_rtype == REGION_BLOCK
            handle_block_region!(body, child, code, blocks, block_id; loop_patterning)
        else
            handle_nested_region!(body, child, code, blocks, block_id; loop_patterning)
        end
    end

    # Set terminator with yield values
    body.terminator = ContinueOp(pattern.yield_values)

    return WhileOp(pattern.condition, pattern.init_values, body, result_vars)
end

"""
    build_loop_op(tree::ControlTree, code::CodeInfo, blocks::Vector{BlockInfo}, block_id::Ref{Int}; loop_patterning=true) -> LoopOp

Build a general LoopOp from a loop control tree.
This is the fallback when ForOp and WhileOp patterns don't match.
"""
function build_loop_op(tree::ControlTree, code::CodeInfo, blocks::Vector{BlockInfo}, block_id::Ref{Int}; loop_patterning::Bool=true)
    stmts = code.code
    header_idx = node_index(tree)
    loop_blocks = get_loop_blocks(tree, blocks)

    header_idx < 1 || header_idx > length(blocks) && return LoopOp(IRValue[], Block(block_id[]), SSAValue[])
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

            phi_type = code.ssavaluetypes[si]
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

    # Collect header statements (excluding phi nodes and control flow)
    for si in header_block.range
        stmt = stmts[si]
        if !(stmt isa PhiNode || stmt isa GotoNode || stmt isa GotoIfNot || stmt isa ReturnNode)
            push!(body.body, si)
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
                handle_block_region!(then_block, child, code, blocks, block_id; loop_patterning)
            end
        end
        then_block.terminator = ContinueOp(carried_values)

        else_block = Block(block_id[])
        block_id[] += 1
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
                handle_block_region!(body, child, code, blocks, block_id; loop_patterning)
            end
        end
        body.terminator = ContinueOp(carried_values)
    end

    return LoopOp(init_values, body, result_vars)
end

#=============================================================================
 Helper Functions
=============================================================================#

"""
    is_value_in_loop(val, stmts, stmt_to_blk, loop_blocks) -> Bool

Check if a value originates from inside the loop.
"""
function is_value_in_loop(val, stmts, stmt_to_blk, loop_blocks)
    if val isa SSAValue && val.id > 0 && val.id <= length(stmts)
        return stmt_to_blk[val.id] ∈ loop_blocks
    end
    return false
end

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
