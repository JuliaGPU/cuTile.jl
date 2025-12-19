# block-level CFG from Julia IR

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
