export StructuredCodeInfo, structurize!

using Graphs: SimpleDiGraph, add_edge!, vertices, edges, nv, ne,
              inneighbors, outneighbors, Edge

#=============================================================================
 Block-Level CFG from Julia IR
=============================================================================#

"""
    BlockInfo

Information about a basic block for block-level CFG analysis.
"""
struct BlockInfo
    index::Int                # Block index (1-based)
    range::UnitRange{Int}     # Statement indices in this block
    preds::Vector{Int}        # Predecessor block indices
    succs::Vector{Int}        # Successor block indices
end

"""
    find_basic_blocks(code::CodeInfo) -> Vector{UnitRange{Int}}

Find basic block boundaries in Julia IR.
Returns ranges of statement indices for each basic block.
"""
function find_basic_blocks(code::CodeInfo)
    stmts = code.code
    n = length(stmts)
    n == 0 && return UnitRange{Int}[]

    # Find block starts (targets of jumps and start)
    is_block_start = falses(n)
    is_block_start[1] = true

    for (i, stmt) in enumerate(stmts)
        if stmt isa GotoNode
            is_block_start[stmt.label] = true
        elseif stmt isa GotoIfNot
            is_block_start[stmt.dest] = true
            if i + 1 <= n
                is_block_start[i + 1] = true
            end
        end
    end

    # Also mark after GotoNode/ReturnNode as block starts
    for (i, stmt) in enumerate(stmts)
        if (stmt isa GotoNode || stmt isa ReturnNode) && i + 1 <= n
            is_block_start[i + 1] = true
        end
    end

    # Build block ranges
    blocks = UnitRange{Int}[]
    block_start = 1
    for i in 2:n
        if is_block_start[i]
            push!(blocks, block_start:i-1)
            block_start = i
        end
    end
    push!(blocks, block_start:n)

    return blocks
end

"""
    build_block_cfg(code::CodeInfo) -> (Vector{BlockInfo}, SimpleDiGraph)

Build a block-level CFG from Julia CodeInfo.
Returns a vector of BlockInfo and a SimpleDiGraph representing the CFG.
"""
function build_block_cfg(code::CodeInfo)
    stmts = code.code
    n = length(stmts)
    n == 0 && return BlockInfo[], SimpleDiGraph(0)

    # Get basic block ranges
    block_ranges = find_basic_blocks(code)
    nblocks = length(block_ranges)
    nblocks == 0 && return BlockInfo[], SimpleDiGraph(0)

    # Create mapping from statement index to block index
    stmt_to_block = zeros(Int, n)
    for (bi, range) in enumerate(block_ranges)
        for si in range
            stmt_to_block[si] = bi
        end
    end

    # Initialize block info with empty pred/succ lists
    blocks = [BlockInfo(i, block_ranges[i], Int[], Int[]) for i in 1:nblocks]

    # Build CFG
    cfg = SimpleDiGraph(nblocks)

    # Build edges based on control flow
    for (bi, range) in enumerate(block_ranges)
        last_stmt_idx = last(range)
        last_stmt = stmts[last_stmt_idx]

        if last_stmt isa GotoNode
            target_block = stmt_to_block[last_stmt.label]
            push!(blocks[bi].succs, target_block)
            push!(blocks[target_block].preds, bi)
            add_edge!(cfg, bi, target_block)
        elseif last_stmt isa GotoIfNot
            # Fallthrough edge
            if last_stmt_idx + 1 <= n
                fallthrough_block = stmt_to_block[last_stmt_idx + 1]
                push!(blocks[bi].succs, fallthrough_block)
                push!(blocks[fallthrough_block].preds, bi)
                add_edge!(cfg, bi, fallthrough_block)
            end
            # Branch edge
            target_block = stmt_to_block[last_stmt.dest]
            if target_block ∉ blocks[bi].succs  # Avoid duplicates
                push!(blocks[bi].succs, target_block)
                push!(blocks[target_block].preds, bi)
                add_edge!(cfg, bi, target_block)
            end
        elseif last_stmt isa ReturnNode
            # No successors
        else
            # Fallthrough to next block
            if bi < nblocks
                push!(blocks[bi].succs, bi + 1)
                push!(blocks[bi + 1].preds, bi)
                add_edge!(cfg, bi, bi + 1)
            end
        end
    end

    return blocks, cfg
end

"""
    stmt_to_block_map(blocks::Vector{BlockInfo}, n_stmts::Int) -> Vector{Int}

Create a mapping from statement index to block index.
"""
function stmt_to_block_map(blocks::Vector{BlockInfo}, n_stmts::Int)
    mapping = zeros(Int, n_stmts)
    for block in blocks
        for si in block.range
            mapping[si] = block.index
        end
    end
    return mapping
end

#=============================================================================
 Control Tree to Structured IR Conversion
=============================================================================#

"""
    control_tree_to_structured_ir(ctree::ControlTree, code::CodeInfo, blocks::Vector{BlockInfo}) -> Block

Convert a control tree to structured IR entry block.
"""
function control_tree_to_structured_ir(ctree::ControlTree, code::CodeInfo, blocks::Vector{BlockInfo})
    block_id = Ref(1)
    entry_block = tree_to_block(ctree, code, blocks, block_id)
    return entry_block
end

"""
    tree_to_block(tree::ControlTree, code::CodeInfo, blocks::Vector{BlockInfo}, block_id::Ref{Int}) -> Block

Convert a control tree node to a Block.
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
                push!(block.body, si)
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
                push!(block.body, si)
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
                push!(block.body, si)
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
"""
function handle_loop!(block::Block, tree::ControlTree, code::CodeInfo, blocks::Vector{BlockInfo}, block_id::Ref{Int})
    tree_children = children(tree)

    # Try to detect for-loop pattern first
    pattern = detect_for_loop_pattern(tree, code, blocks)
    if pattern !== nothing
        for_op = build_for_op(tree, code, blocks, pattern, block_id)
        push!(block.body, for_op)
        return
    end

    # Build general LoopOp
    loop_op = build_loop_op(tree, code, blocks, block_id)
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
 For-Loop Pattern Detection
=============================================================================#

"""
    ForLoopPattern

Detected for-loop pattern with bounds and induction variable info.
"""
struct ForLoopPattern
    lower::IRValue           # Initial value of induction variable
    upper::IRValue           # Upper bound (exclusive)
    step::IRValue            # Step value
    induction_phi_idx::Int   # SSA index of induction variable phi
    header_block::Int        # Block index of loop header
end

"""
    detect_for_loop_pattern(tree::ControlTree, code::CodeInfo, blocks::Vector{BlockInfo}) -> Union{ForLoopPattern, Nothing}

Detect if a loop is a simple counted for-loop with pattern:
- Header has phi for induction var: φ(init, next)
- Condition is slt_int(iv, upper) or similar
- Body has iv_next = add_int(iv, step)
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
    upper_bound = cond_stmt.args[3]   # upper bound

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

    # Extract lower bound (init value from outside loop)
    stmt_to_blk = stmt_to_block_map(blocks, length(stmts))
    lower_bound = nothing

    for (edge_idx, _) in enumerate(iv_phi_stmt.edges)
        if isassigned(iv_phi_stmt.values, edge_idx)
            val = iv_phi_stmt.values[edge_idx]
            val_in_loop = false
            if val isa SSAValue && val.id > 0 && val.id <= length(stmts)
                val_in_loop = stmt_to_blk[val.id] ∈ loop_blocks
            end
            if !val_in_loop
                lower_bound = convert_phi_value(val)
                break
            end
        end
    end
    lower_bound === nothing && return nothing

    # Find step by looking for add_int(iv_phi, step) in loop body
    step = nothing
    for bi in loop_blocks
        bi < 1 || bi > length(blocks) && continue
        for si in blocks[bi].range
            stmt = stmts[si]
            if stmt isa Expr && stmt.head === :call
                func = stmt.args[1]
                if func isa GlobalRef && func.name === :add_int
                    # Check if it's iv + constant
                    if stmt.args[2] isa SSAValue && stmt.args[2].id == iv_phi_idx
                        step = convert_phi_value(stmt.args[3])
                        break
                    end
                end
            end
        end
        step !== nothing && break
    end
    step === nothing && return nothing

    return ForLoopPattern(lower_bound, convert_phi_value(upper_bound), step, iv_phi_idx, header_idx)
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
    build_for_op(tree::ControlTree, code::CodeInfo, blocks::Vector{BlockInfo},
                 pattern::ForLoopPattern, block_id::Ref{Int}) -> ForOp

Build a ForOp from detected for-loop pattern.
"""
function build_for_op(tree::ControlTree, code::CodeInfo, blocks::Vector{BlockInfo},
                      pattern::ForLoopPattern, block_id::Ref{Int})
    stmts = code.code
    header_block = blocks[pattern.header_block]
    stmt_to_blk = stmt_to_block_map(blocks, length(stmts))
    loop_blocks = get_loop_blocks(tree, blocks)

    # Collect block args and init values
    block_args = BlockArg[]
    init_values = IRValue[]
    result_vars = SSAValue[]
    carried_phi_indices = Int[]

    # Induction variable is first block arg
    iv_type = code.ssavaluetypes[pattern.induction_phi_idx]
    push!(block_args, BlockArg(1, iv_type))

    # Other phis become carried values
    for si in header_block.range
        stmt = stmts[si]
        if stmt isa PhiNode && si != pattern.induction_phi_idx
            push!(result_vars, SSAValue(si))
            push!(carried_phi_indices, si)
            phi_type = code.ssavaluetypes[si]
            push!(block_args, BlockArg(length(block_args) + 1, phi_type))

            # Extract init value (from outside loop)
            for (edge_idx, _) in enumerate(stmt.edges)
                if isassigned(stmt.values, edge_idx)
                    val = stmt.values[edge_idx]
                    val_in_loop = false
                    if val isa SSAValue && val.id > 0 && val.id <= length(stmts)
                        val_in_loop = stmt_to_blk[val.id] ∈ loop_blocks
                    end
                    if !val_in_loop
                        push!(init_values, convert_phi_value(val))
                        break
                    end
                end
            end
        end
    end

    # Build body block
    body = Block(block_id[])
    block_id[] += 1
    body.args = block_args

    # Find the induction increment statement to exclude
    iv_incr_idx = find_induction_increment(stmts, blocks, loop_blocks, pattern.induction_phi_idx)

    # Process body using tree structure to handle nested loops
    header_idx = node_index(tree)
    for child in children(tree)
        child_idx = node_index(child)
        child_idx == header_idx && continue  # Skip header

        child_rtype = region_type(child)
        if child_rtype == REGION_WHILE_LOOP || child_rtype == REGION_NATURAL_LOOP
            # Nested loop - create nested op
            handle_loop!(body, child, code, blocks, block_id)
        elseif child_rtype == REGION_IF_THEN || child_rtype == REGION_IF_THEN_ELSE
            handle_nested_region!(body, child, code, blocks, block_id)
        elseif child_rtype == REGION_BLOCK
            # Process block region, which may contain nested loops
            process_for_body_region!(body, child, code, blocks, block_id, iv_incr_idx)
        else
            handle_nested_region!(body, child, code, blocks, block_id)
        end
    end

    # Find yield values (carried values from inside loop)
    yield_values = IRValue[]
    for phi_idx in carried_phi_indices
        phi = stmts[phi_idx]
        for (edge_idx, _) in enumerate(phi.edges)
            if isassigned(phi.values, edge_idx)
                val = phi.values[edge_idx]
                val_in_loop = false
                if val isa SSAValue && val.id > 0 && val.id <= length(stmts)
                    val_in_loop = stmt_to_blk[val.id] ∈ loop_blocks
                end
                if val_in_loop
                    push!(yield_values, convert_phi_value(val))
                    break
                end
            end
        end
    end
    body.terminator = ContinueOp(yield_values)

    iv_ssa = SSAValue(pattern.induction_phi_idx)
    return ForOp(pattern.lower, pattern.upper, pattern.step, iv_ssa, init_values, body, result_vars)
end

"""
    find_induction_increment(stmts, blocks, loop_blocks, iv_phi_idx) -> Union{Int, Nothing}

Find the statement index of the induction variable increment.
"""
function find_induction_increment(stmts, blocks::Vector{BlockInfo}, loop_blocks::Set{Int}, iv_phi_idx::Int)
    for bi in loop_blocks
        bi < 1 || bi > length(blocks) && continue
        for si in blocks[bi].range
            stmt = stmts[si]
            if stmt isa Expr && stmt.head === :call
                func = stmt.args[1]
                if func isa GlobalRef && func.name === :add_int
                    if length(stmt.args) >= 2 && stmt.args[2] isa SSAValue && stmt.args[2].id == iv_phi_idx
                        return si
                    end
                end
            end
        end
    end
    return nothing
end

"""
    process_for_body_region!(block::Block, tree::ControlTree, code::CodeInfo, blocks::Vector{BlockInfo}, block_id::Ref{Int}, iv_incr_idx)

Process a REGION_BLOCK inside a ForOp body, handling nested loops and skipping induction variable operations.
"""
function process_for_body_region!(block::Block, tree::ControlTree, code::CodeInfo, blocks::Vector{BlockInfo}, block_id::Ref{Int}, iv_incr_idx)
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
                handle_loop!(block, child, code, blocks, block_id)
            elseif child_rtype == REGION_BLOCK
                process_for_body_region!(block, child, code, blocks, block_id, iv_incr_idx)
            else
                handle_nested_region!(block, child, code, blocks, block_id)
            end
        end
    end
end

"""
    build_loop_op(tree::ControlTree, code::CodeInfo, blocks::Vector{BlockInfo}, block_id::Ref{Int}) -> LoopOp

Build a general LoopOp from a loop control tree.
"""
function build_loop_op(tree::ControlTree, code::CodeInfo, blocks::Vector{BlockInfo}, block_id::Ref{Int})
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
                handle_block_region!(then_block, child, code, blocks, block_id)
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
                handle_block_region!(body, child, code, blocks, block_id)
            end
        end
        body.terminator = ContinueOp(carried_values)
    end

    return LoopOp(init_values, body, result_vars)
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
 Public API
=============================================================================#

"""
    structurize!(sci::StructuredCodeInfo) -> StructuredCodeInfo

Convert unstructured control flow in `sci` to structured control flow operations
(IfOp, LoopOp, ForOp) in-place.

This transforms GotoNode and GotoIfNot statements into nested structured ops
that can be traversed hierarchically.

Returns `sci` for convenience (allows chaining).
"""
function structurize!(sci::StructuredCodeInfo)
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
    sci.entry = control_tree_to_structured_ir(ctree, code, blocks)

    return sci
end
