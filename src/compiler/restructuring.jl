# Control flow restructuring - converts unstructured Julia IR to nested structured IR
#
# Adapted from SPIRV.jl's structural_analysis.jl

using Graphs: SimpleDiGraph, add_edge!, vertices, edges, nv, ne,
              inneighbors, outneighbors, Edge, strongly_connected_components

#=============================================================================
 Region Types
=============================================================================#

"""
Single-entry control-flow structure classified according to well-specified patterns.
"""
@enum RegionType begin
    REGION_BLOCK           # Sequence of blocks
    REGION_IF_THEN         # Conditional with one branch
    REGION_IF_THEN_ELSE    # Conditional with two branches merging
    REGION_IF_THEN_ELSE_TERMINATING  # Conditional with both branches returning
    REGION_SWITCH          # Multi-way branch
    REGION_TERMINATION     # Block ending with function termination
    REGION_PROPER          # Acyclic region not matching other patterns
    REGION_SELF_LOOP       # Single block with self-edge
    REGION_WHILE_LOOP      # Simple while loop pattern
    REGION_NATURAL_LOOP    # General single-entry loop
    REGION_IMPROPER        # Multiple-entry cyclic region
end

#=============================================================================
 Control Flow Graph from Julia IR
=============================================================================#

"""
    julia_cfg(code::CodeInfo) -> SimpleDiGraph

Build a CFG from Julia CodeInfo.
"""
function julia_cfg(code::CodeInfo)
    stmts = code.code
    n = length(stmts)
    cfg = SimpleDiGraph(n)

    for (i, stmt) in enumerate(stmts)
        if stmt isa GotoNode
            add_edge!(cfg, i, stmt.label)
        elseif stmt isa GotoIfNot
            # Fallthrough edge
            add_edge!(cfg, i, i + 1)
            # Branch edge
            add_edge!(cfg, i, stmt.dest)
        elseif stmt isa ReturnNode
            # No outgoing edges
        else
            # Fallthrough to next statement
            if i < n
                add_edge!(cfg, i, i + 1)
            end
        end
    end

    return cfg
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

#=============================================================================
 Block-Level CFG for Loop Detection
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
    build_block_cfg(code::CodeInfo) -> Vector{BlockInfo}

Build a block-level CFG from Julia CodeInfo.
Returns a vector of BlockInfo where each entry represents a basic block
with its predecessor and successor blocks.
"""
function build_block_cfg(code::CodeInfo)
    stmts = code.code
    n = length(stmts)
    n == 0 && return BlockInfo[]

    # Get basic block ranges
    block_ranges = find_basic_blocks(code)
    nblocks = length(block_ranges)
    nblocks == 0 && return BlockInfo[]

    # Create mapping from statement index to block index
    stmt_to_block = zeros(Int, n)
    for (bi, range) in enumerate(block_ranges)
        for si in range
            stmt_to_block[si] = bi
        end
    end

    # Initialize block info with empty pred/succ lists
    blocks = [BlockInfo(i, block_ranges[i], Int[], Int[]) for i in 1:nblocks]

    # Build edges based on control flow
    for (bi, range) in enumerate(block_ranges)
        last_stmt_idx = last(range)
        last_stmt = stmts[last_stmt_idx]

        if last_stmt isa GotoNode
            target_block = stmt_to_block[last_stmt.label]
            push!(blocks[bi].succs, target_block)
            push!(blocks[target_block].preds, bi)
        elseif last_stmt isa GotoIfNot
            # Fallthrough edge
            if last_stmt_idx + 1 <= n
                fallthrough_block = stmt_to_block[last_stmt_idx + 1]
                push!(blocks[bi].succs, fallthrough_block)
                push!(blocks[fallthrough_block].preds, bi)
            end
            # Branch edge
            target_block = stmt_to_block[last_stmt.dest]
            if target_block ∉ blocks[bi].succs  # Avoid duplicates
                push!(blocks[bi].succs, target_block)
                push!(blocks[target_block].preds, bi)
            end
        elseif last_stmt isa ReturnNode
            # No successors
        else
            # Fallthrough to next block
            if bi < nblocks
                push!(blocks[bi].succs, bi + 1)
                push!(blocks[bi + 1].preds, bi)
            end
        end
    end

    return blocks
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
 Dominator Analysis using Julia's Core.Compiler
=============================================================================#

"""
    compute_dominators(cfg::SimpleDiGraph) -> Dict{Int, Set{Int}}

Compute dominator sets for each vertex.
"""
function compute_dominators(cfg::SimpleDiGraph)
    n = nv(cfg)
    n == 0 && return Dict{Int, Set{Int}}()

    # Initialize: entry dominated only by itself, others by all
    doms = Dict{Int, Set{Int}}()
    entry = 1  # Julia IR always starts at statement 1
    all_verts = Set(1:n)

    for v in 1:n
        if v == entry
            doms[v] = Set([entry])
        else
            doms[v] = copy(all_verts)
        end
    end

    # Iterate until fixed point
    changed = true
    while changed
        changed = false
        for v in 2:n
            preds = inneighbors(cfg, v)
            if isempty(preds)
                continue
            end

            # dom(v) = {v} ∪ ∩{dom(p) : p ∈ preds(v)}
            new_doms = copy(doms[first(preds)])
            for p in preds[2:end]
                intersect!(new_doms, doms[p])
            end
            push!(new_doms, v)

            if new_doms != doms[v]
                doms[v] = new_doms
                changed = true
            end
        end
    end

    return doms
end

"""
    compute_immediate_dominators(doms::Dict{Int, Set{Int}}) -> Dict{Int, Int}

Compute immediate dominator for each vertex.
"""
function compute_immediate_dominators(doms::Dict{Int, Set{Int}})
    idoms = Dict{Int, Int}()

    for (v, dom_set) in doms
        # Entry node has no idom
        if length(dom_set) == 1
            continue
        end

        # Find the immediate dominator: the dominator that is dominated by all others
        candidates = setdiff(dom_set, Set([v]))
        for c in candidates
            # c is idom if all other dominators (except v) dominate c
            is_idom = true
            for other in candidates
                if other != c && !(other in doms[c])
                    is_idom = false
                    break
                end
            end
            if is_idom
                idoms[v] = c
                break
            end
        end
    end

    return idoms
end

#=============================================================================
 Back Edge Detection
=============================================================================#

"""
    find_back_edges(cfg::SimpleDiGraph, doms::Dict{Int, Set{Int}}) -> Set{Edge{Int}}

Find back edges in the CFG (edges where target dominates source).
"""
function find_back_edges(cfg::SimpleDiGraph, doms::Dict{Int, Set{Int}})
    back_edges = Set{Edge{Int}}()

    for e in edges(cfg)
        src, dst = e.src, e.dst
        # Back edge if dst dominates src
        if haskey(doms, src) && dst in doms[src]
            push!(back_edges, e)
        end
    end

    return back_edges
end

#=============================================================================
 Block-Level Loop Detection
=============================================================================#

"""
    compute_block_dominators(blocks::Vector{BlockInfo}) -> Dict{Int, Set{Int}}

Compute dominator sets for each block in the block-level CFG.
"""
function compute_block_dominators(blocks::Vector{BlockInfo})
    n = length(blocks)
    n == 0 && return Dict{Int, Set{Int}}()

    # Initialize: entry dominated only by itself, others by all
    doms = Dict{Int, Set{Int}}()
    entry = 1  # First block is entry
    all_blocks = Set(1:n)

    for bi in 1:n
        if bi == entry
            doms[bi] = Set([entry])
        else
            doms[bi] = copy(all_blocks)
        end
    end

    # Iterate until fixed point
    changed = true
    while changed
        changed = false
        for bi in 2:n
            preds = blocks[bi].preds
            if isempty(preds)
                continue
            end

            # dom(bi) = {bi} ∪ ∩{dom(p) : p ∈ preds(bi)}
            new_doms = copy(doms[first(preds)])
            for p in preds[2:end]
                intersect!(new_doms, doms[p])
            end
            push!(new_doms, bi)

            if new_doms != doms[bi]
                doms[bi] = new_doms
                changed = true
            end
        end
    end

    return doms
end

"""
    dominates(doms::Dict{Int, Set{Int}}, a::Int, b::Int) -> Bool

Check if block `a` dominates block `b`.
"""
dominates(doms::Dict{Int, Set{Int}}, a::Int, b::Int) = haskey(doms, b) && a in doms[b]

"""
    find_block_back_edges(blocks::Vector{BlockInfo}, doms::Dict{Int, Set{Int}}) -> Vector{Pair{Int,Int}}

Find back edges in the block-level CFG (edges where target dominates source).
Returns pairs of (source_block, target_block).
"""
function find_block_back_edges(blocks::Vector{BlockInfo}, doms::Dict{Int, Set{Int}})
    back_edges = Pair{Int,Int}[]

    for block in blocks
        for succ in block.succs
            # Back edge if succ dominates block.index
            if dominates(doms, succ, block.index)
                push!(back_edges, block.index => succ)
            end
        end
    end

    return back_edges
end

"""
    LoopInfo

Information about a detected loop.
"""
struct LoopInfo
    header::Int               # Header block index (loop entry point)
    latches::Vector{Int}      # Blocks with back-edges to header
    blocks::Set{Int}          # All blocks in the loop (including header)
    exit_blocks::Set{Int}     # Blocks outside loop that are successors of loop blocks
    # For future nested loop support:
    parent::Union{Nothing, Int}  # Parent loop header (for nested loops)
    children::Vector{Int}        # Child loop headers (for nested loops)
end

LoopInfo(header::Int, latches::Vector{Int}, blocks::Set{Int}, exit_blocks::Set{Int}) =
    LoopInfo(header, latches, blocks, exit_blocks, nothing, Int[])

"""
    compute_natural_loop(blocks::Vector{BlockInfo}, header::Int, latch::Int) -> Set{Int}

Compute the natural loop for a back-edge from latch to header.
The natural loop is all blocks from which the latch is reachable without
passing through the header.
"""
function compute_natural_loop(blocks::Vector{BlockInfo}, header::Int, latch::Int)
    # Start with header and latch
    loop_blocks = Set{Int}([header, latch])

    # Worklist of blocks to process (predecessors of blocks already in loop)
    worklist = copy(blocks[latch].preds)

    while !isempty(worklist)
        bi = pop!(worklist)

        # Skip if already in loop or is header (don't go past header)
        bi ∈ loop_blocks && continue

        # Add to loop
        push!(loop_blocks, bi)

        # Add predecessors to worklist
        append!(worklist, blocks[bi].preds)
    end

    return loop_blocks
end

"""
    find_loop_exits(blocks::Vector{BlockInfo}, loop_blocks::Set{Int}) -> Set{Int}

Find blocks outside the loop that are successors of blocks inside the loop.
"""
function find_loop_exits(blocks::Vector{BlockInfo}, loop_blocks::Set{Int})
    exit_blocks = Set{Int}()

    for bi in loop_blocks
        for succ in blocks[bi].succs
            if succ ∉ loop_blocks
                push!(exit_blocks, succ)
            end
        end
    end

    return exit_blocks
end

"""
    find_loops(blocks::Vector{BlockInfo}) -> Vector{LoopInfo}

Detect all loops in the block-level CFG.
"""
function find_loops(blocks::Vector{BlockInfo})
    isempty(blocks) && return LoopInfo[]

    # Compute dominators
    doms = compute_block_dominators(blocks)

    # Find back edges
    back_edges = find_block_back_edges(blocks, doms)

    # Group back edges by header (multiple latches can target same header)
    header_to_latches = Dict{Int, Vector{Int}}()
    for (latch, header) in back_edges
        if !haskey(header_to_latches, header)
            header_to_latches[header] = Int[]
        end
        push!(header_to_latches[header], latch)
    end

    # Build loop info for each header
    loops = LoopInfo[]
    for (header, latches) in header_to_latches
        # Compute natural loop as union of all back-edge natural loops
        loop_blocks = Set{Int}()
        for latch in latches
            union!(loop_blocks, compute_natural_loop(blocks, header, latch))
        end

        # Find exit blocks
        exit_blocks = find_loop_exits(blocks, loop_blocks)

        push!(loops, LoopInfo(header, latches, loop_blocks, exit_blocks))
    end

    return loops
end

#=============================================================================
 Control Tree Node
=============================================================================#

"""
    ControlNode

A node in the control tree, containing block index and region type.
"""
struct ControlNode
    index::Int           # Starting block/statement index
    region_type::RegionType
end

"""
    ControlTree

Tree structure representing hierarchical control flow regions.
Leaves are REGION_BLOCK nodes.
"""
mutable struct ControlTree
    node::ControlNode
    parent::Union{ControlTree, Nothing}
    children::Vector{ControlTree}
end

ControlTree(index::Int, region_type::RegionType) =
    ControlTree(ControlNode(index, region_type), nothing, ControlTree[])

node_index(tree::ControlTree) = tree.node.index
region_type(tree::ControlTree) = tree.node.region_type

function Base.show(io::IO, tree::ControlTree)
    print(io, "ControlTree(", tree.node.index, ", ", tree.node.region_type, ")")
end

#=============================================================================
 Helper Functions
=============================================================================#

"""
    is_single_entry_single_exit(cfg, v)

Check if vertex v has exactly one incoming and one outgoing edge.
"""
is_single_entry_single_exit(cfg, v) =
    length(inneighbors(cfg, v)) == 1 && length(outneighbors(cfg, v)) == 1

#=============================================================================
 Region Pattern Matching
=============================================================================#

"""
    detect_block_region(cfg, v, visited, consumed)

Detect a linear sequence of blocks starting at v.
Returns the sequence of vertices if found, nothing otherwise.
"""
function detect_block_region(cfg, v, visited, consumed)
    # Must have at most 1 in-edge and 1 out-edge to start
    length(outneighbors(cfg, v)) > 1 && return nothing
    length(inneighbors(cfg, v)) > 1 && return nothing

    # Walk forward
    vs = [v]
    curr = v
    while length(outneighbors(cfg, curr)) == 1
        next = only(outneighbors(cfg, curr))
        length(inneighbors(cfg, next)) == 1 || break
        length(outneighbors(cfg, next)) > 1 && break
        next in vs && break  # Avoid cycles
        next in visited && break
        next in consumed && break  # Don't include already consumed vertices
        push!(vs, next)
        curr = next
    end

    length(vs) == 1 && return nothing
    return vs
end

"""
    detect_if_then(cfg, v, consumed)

Detect if-then pattern at v (conditional with one branch).
Returns (condition_block, then_block, merge_block) or nothing.
"""
function detect_if_then(cfg, v, consumed)
    outs = outneighbors(cfg, v)
    length(outs) == 2 || return nothing

    a, b = outs

    # Check if a is the "then" block going to b (merge)
    if !(a in consumed) &&
       is_single_entry_single_exit(cfg, a) &&
       only(inneighbors(cfg, a)) == v &&
       only(outneighbors(cfg, a)) == b
        return (v, a, b)
    end

    # Check if b is the "then" block going to a (merge)
    if !(b in consumed) &&
       is_single_entry_single_exit(cfg, b) &&
       only(inneighbors(cfg, b)) == v &&
       only(outneighbors(cfg, b)) == a
        return (v, b, a)
    end

    return nothing
end

"""
    detect_if_then_else(cfg, v, consumed)

Detect if-then-else pattern at v.
Returns (condition_block, then_block, else_block, merge_block) or nothing.
"""
function detect_if_then_else(cfg, v, consumed)
    outs = outneighbors(cfg, v)
    length(outs) == 2 || return nothing

    a, b = outs

    # Neither branch can be consumed
    a in consumed && return nothing
    b in consumed && return nothing

    # Both branches must be SESE
    is_single_entry_single_exit(cfg, a) || return nothing
    is_single_entry_single_exit(cfg, b) || return nothing

    # Both must come from v
    only(inneighbors(cfg, a)) == v || return nothing
    only(inneighbors(cfg, b)) == v || return nothing

    # Both must go to same merge point
    merge_a = only(outneighbors(cfg, a))
    merge_b = only(outneighbors(cfg, b))
    merge_a == merge_b || return nothing

    # Merge can't be v (would be a loop)
    merge_a == v && return nothing

    return (v, a, b, merge_a)
end

"""
    detect_if_then_else_terminating(cfg, v, code, consumed)

Detect if-then-else pattern where both branches terminate (e.g., return statements).
Handles both single-statement returns and multi-statement branches ending in returns.
Returns (condition_block, then_stmts, else_stmts) or nothing,
where then_stmts/else_stmts are vectors of statement indices in the branch.
"""
function detect_if_then_else_terminating(cfg, v, code, consumed)
    outs = outneighbors(cfg, v)
    length(outs) == 2 || return nothing

    a, b = outs

    # Neither branch start can be consumed
    a in consumed && return nothing
    b in consumed && return nothing

    # Both branches must have exactly one in-edge (from v)
    length(inneighbors(cfg, a)) == 1 || return nothing
    length(inneighbors(cfg, b)) == 1 || return nothing
    only(inneighbors(cfg, a)) == v || return nothing
    only(inneighbors(cfg, b)) == v || return nothing

    # Follow fallthrough chains to find terminating returns
    then_chain = follow_to_terminating_return(cfg, a, code, consumed)
    else_chain = follow_to_terminating_return(cfg, b, code, consumed)

    # Both branches must terminate with returns
    then_chain === nothing && return nothing
    else_chain === nothing && return nothing

    # Make sure chains don't overlap
    then_set = Set(then_chain)
    for stmt in else_chain
        stmt in then_set && return nothing
    end

    return (v, then_chain, else_chain)
end

"""
    follow_to_terminating_return(cfg, start, code, consumed) -> Vector{Int} or nothing

Follow a single-successor chain from `start` until reaching a return statement.
Returns the list of statement indices in the chain, or nothing if the chain
branches, loops back, or doesn't end in a return.
"""
function follow_to_terminating_return(cfg, start::Int, code::CodeInfo, consumed::Set{Int})
    chain = Int[]
    current = start

    while true
        # Don't follow through consumed statements (except start)
        if current != start && current in consumed
            return nothing
        end

        push!(chain, current)

        # Check if this is a return
        if 1 <= current <= length(code.code) && code.code[current] isa ReturnNode
            return chain
        end

        # Must have exactly one outgoing edge to continue
        successors = outneighbors(cfg, current)
        length(successors) == 1 || return nothing

        next = only(successors)

        # Next must have exactly one incoming edge (from current)
        length(inneighbors(cfg, next)) == 1 || return nothing

        # Avoid cycles
        next in chain && return nothing

        current = next
    end
end

"""
    detect_while_loop(cfg, v, back_edges, consumed)

Detect while loop pattern at v.
Returns (header, body, exit) or nothing.
"""
function detect_while_loop(cfg, v, back_edges, consumed)
    ins = inneighbors(cfg, v)
    outs = outneighbors(cfg, v)

    # Header needs 2 in-edges (entry + back) and 2 out-edges (body + exit)
    length(ins) == 2 || return nothing
    length(outs) == 2 || return nothing

    a, b = outs

    # Check if a is body looping back to v
    if !(a in consumed) &&
       is_single_entry_single_exit(cfg, a) &&
       only(inneighbors(cfg, a)) == v &&
       only(outneighbors(cfg, a)) == v &&
       Edge(a, v) in back_edges
        return (v, a, b)
    end

    # Check if b is body looping back to v
    if !(b in consumed) &&
       is_single_entry_single_exit(cfg, b) &&
       only(inneighbors(cfg, b)) == v &&
       only(outneighbors(cfg, b)) == v &&
       Edge(b, v) in back_edges
        return (v, b, a)
    end

    return nothing
end

"""
    detect_self_loop(cfg, v, back_edges)

Detect self-loop at v.
"""
function detect_self_loop(cfg, v, back_edges)
    return Edge(v, v) in back_edges
end

#=============================================================================
 Control Tree Construction
=============================================================================#

"""
    build_control_tree(cfg::SimpleDiGraph, code::CodeInfo; max_iterations=10000) -> ControlTree

Build a control tree from the CFG.
"""
function build_control_tree(cfg::SimpleDiGraph, code::CodeInfo; max_iterations::Int=10000)
    n = nv(cfg)
    n == 0 && return ControlTree(0, REGION_BLOCK)

    doms = compute_dominators(cfg)
    back_edges = find_back_edges(cfg, doms)

    # Initialize each vertex as a block region
    trees = Dict(v => ControlTree(v, REGION_BLOCK) for v in 1:n)
    visited = Set{Int}()
    consumed = Set{Int}()  # Vertices that have been merged into a region

    # Process in reverse post-order (approximated by forward order for Julia IR)
    worklist = collect(1:n)

    iterations = 0
    while !isempty(worklist)
        iterations += 1
        if iterations > max_iterations
            @warn "Control tree construction exceeded iteration limit ($max_iterations)"
            break
        end

        v = popfirst!(worklist)
        haskey(trees, v) || continue
        v in visited && continue

        # Try to match region patterns
        region = try_match_region(cfg, v, trees, back_edges, visited, consumed, code)

        if region !== nothing
            (region_type, included_verts) = region

            # Check if any vertex (other than v) has already been consumed
            non_v_verts = filter(w -> w != v, included_verts)
            if any(w -> w in consumed, non_v_verts)
                # Already consumed, mark as visited and move on
                push!(visited, v)
                continue
            end

            # Create new tree node
            children = [trees[w] for w in included_verts if haskey(trees, w)]
            new_tree = ControlTree(v, region_type)
            new_tree.children = children
            for child in children
                child.parent = new_tree
            end

            # Mark merged vertices as consumed
            for w in included_verts
                if w != v
                    push!(consumed, w)
                    delete!(trees, w)
                end
            end
            trees[v] = new_tree

            # Re-process this vertex for possible further reduction
            # but only if we actually made progress (merged something)
            if !isempty(non_v_verts)
                pushfirst!(worklist, v)
            else
                push!(visited, v)
            end
        else
            push!(visited, v)
        end
    end

    # Combine remaining trees into a sequence if needed
    # This handles cases like: stmt1 -> if-then-else where stmt1 is separate
    remaining_roots = sort(collect(keys(trees)))

    if length(remaining_roots) == 1
        return trees[only(remaining_roots)]
    elseif length(remaining_roots) > 1
        # Create a sequence containing all remaining trees
        # Use the first vertex as the root with REGION_BLOCK type
        root_v = remaining_roots[1]
        root_tree = ControlTree(root_v, REGION_BLOCK)
        root_tree.children = [trees[v] for v in remaining_roots]
        for child in root_tree.children
            child.parent = root_tree
        end
        return root_tree
    else
        # No trees at all - create empty block
        return ControlTree(0, REGION_BLOCK)
    end
end

"""
    try_match_region(cfg, v, trees, back_edges, visited, consumed, code)

Try to match a region pattern at vertex v.
Returns (RegionType, included_vertices) or nothing.
"""
function try_match_region(cfg, v, trees, back_edges, visited, consumed, code)
    # Try acyclic patterns first
    result = detect_block_region(cfg, v, visited, consumed)
    if result !== nothing
        return (REGION_BLOCK, result)
    end

    result = detect_if_then_else(cfg, v, consumed)
    if result !== nothing
        (_, then_v, else_v, _) = result
        return (REGION_IF_THEN_ELSE, [v, then_v, else_v])
    end

    # Try terminating if-then-else (both branches return)
    result = detect_if_then_else_terminating(cfg, v, code, consumed)
    if result !== nothing
        (_, then_chain, else_chain) = result
        # Include condition vertex and all statements in both chains
        all_verts = [v; then_chain; else_chain]
        return (REGION_IF_THEN_ELSE_TERMINATING, all_verts)
    end

    result = detect_if_then(cfg, v, consumed)
    if result !== nothing
        (_, then_v, _) = result
        return (REGION_IF_THEN, [v, then_v])
    end

    # Try cyclic patterns
    if detect_self_loop(cfg, v, back_edges)
        return (REGION_SELF_LOOP, [v])
    end

    result = detect_while_loop(cfg, v, back_edges, consumed)
    if result !== nothing
        (_, body_v, _) = result
        return (REGION_WHILE_LOOP, [v, body_v])
    end

    return nothing
end

#=============================================================================
 Control Tree to Structured IR Conversion
=============================================================================#

"""
    control_tree_to_structured_ir(tree::ControlTree, code::CodeInfo) -> StructuredCodeInfo

Convert a control tree to structured IR.
"""
function control_tree_to_structured_ir(tree::ControlTree, code::CodeInfo)
    entry_block = tree_to_block(tree, code, Ref(1))
    return StructuredCodeInfo(code, entry_block)
end

"""
    tree_to_block(tree::ControlTree, code::CodeInfo, block_id::Ref{Int}) -> Block

Convert a control tree node to a Block.
"""
function tree_to_block(tree::ControlTree, code::CodeInfo, block_id::Ref{Int})
    idx = node_index(tree)
    rtype = region_type(tree)
    id = block_id[]
    block_id[] += 1

    block = Block(id)

    if rtype == REGION_BLOCK
        # Collect all statement indices from this block and its children
        collect_statements!(block, tree, code, block_id)
    elseif rtype == REGION_IF_THEN_ELSE
        # First child is condition block, then then_block, else_block
        handle_if_then_else!(block, tree, code, block_id)
    elseif rtype == REGION_IF_THEN_ELSE_TERMINATING
        # Both branches terminate (return) - no merge point
        handle_if_then_else_terminating!(block, tree, code, block_id)
    elseif rtype == REGION_IF_THEN
        # Condition block followed by then_block
        handle_if_then!(block, tree, code, block_id)
    elseif rtype == REGION_WHILE_LOOP || rtype == REGION_NATURAL_LOOP
        handle_loop!(block, tree, code, block_id)
    elseif rtype == REGION_SELF_LOOP
        handle_self_loop!(block, tree, code, block_id)
    else
        # For unsupported regions, collect statements sequentially
        collect_statements!(block, tree, code, block_id)
    end

    # Set terminator based on last statement
    set_terminator!(block, code)

    return block
end

"""
    collect_statements!(block::Block, tree::ControlTree, code::CodeInfo, block_id::Ref{Int})

Collect statement indices from a control tree into a block.
For REGION_BLOCK children, collects statements; for other region types, creates nested ops.
"""
function collect_statements!(block::Block, tree::ControlTree, code::CodeInfo, block_id::Ref{Int})
    idx = node_index(tree)
    rtype = region_type(tree)

    if rtype == REGION_BLOCK && !isempty(tree.children)
        # REGION_BLOCK with children: process children in order
        for child in tree.children
            child_rtype = region_type(child)
            if child_rtype == REGION_BLOCK
                # Recursively collect statements from block children
                collect_statements!(block, child, code, block_id)
            else
                # Other region types need to be converted to nested ops
                handle_nested_region!(block, child, code, block_id)
            end
        end
    else
        # Leaf node or single statement
        if 1 <= idx <= length(code.code)
            stmt = code.code[idx]
            # Don't include control flow terminators as regular statements
            if !(stmt isa GotoNode || stmt isa GotoIfNot || stmt isa ReturnNode)
                push!(block.body, idx)
            elseif stmt isa ReturnNode
                # Mark as terminator, don't include in body
                block.terminator = stmt
            end
        end
    end
end

"""
    handle_nested_region!(block::Block, tree::ControlTree, code::CodeInfo, block_id::Ref{Int})

Handle a nested control flow region by creating the appropriate op and adding to block.body.
"""
function handle_nested_region!(block::Block, tree::ControlTree, code::CodeInfo, block_id::Ref{Int})
    rtype = region_type(tree)

    if rtype == REGION_IF_THEN_ELSE
        handle_if_then_else!(block, tree, code, block_id)
    elseif rtype == REGION_IF_THEN_ELSE_TERMINATING
        handle_if_then_else_terminating!(block, tree, code, block_id)
    elseif rtype == REGION_IF_THEN
        handle_if_then!(block, tree, code, block_id)
    elseif rtype == REGION_WHILE_LOOP || rtype == REGION_NATURAL_LOOP
        handle_loop!(block, tree, code, block_id)
    elseif rtype == REGION_SELF_LOOP
        handle_self_loop!(block, tree, code, block_id)
    else
        # For unsupported nested regions, just collect statements
        collect_statements!(block, tree, code, block_id)
    end
end

"""
    handle_if_then_else!(block::Block, tree::ControlTree, code::CodeInfo, block_id::Ref{Int})

Handle if-then-else region.
"""
function handle_if_then_else!(block::Block, tree::ControlTree, code::CodeInfo, block_id::Ref{Int})
    @assert length(tree.children) >= 3 "if-then-else needs at least 3 children"

    # First child is the condition block
    cond_tree = tree.children[1]
    cond_idx = node_index(cond_tree)

    # Collect condition block statements (up to but not including GotoIfNot)
    if 1 <= cond_idx <= length(code.code)
        stmt = code.code[cond_idx]
        if !(stmt isa GotoNode || stmt isa GotoIfNot || stmt isa ReturnNode)
            push!(block.body, cond_idx)
        end
    end

    # Find the GotoIfNot that determines the condition
    cond_value = find_condition_value(cond_idx, code)

    # Then and else blocks
    then_tree = tree.children[2]
    else_tree = tree.children[3]

    then_block = tree_to_block(then_tree, code, block_id)
    else_block = tree_to_block(else_tree, code, block_id)

    # Create IfOp
    if_op = IfOp(cond_value, then_block, else_block, SSAValue[])
    push!(block.body, if_op)
end

"""
    handle_if_then_else_terminating!(block::Block, tree::ControlTree, code::CodeInfo, block_id::Ref{Int})

Handle if-then-else region where both branches terminate (return).
Reconstructs the branch structure from the GotoIfNot at the condition vertex.
"""
function handle_if_then_else_terminating!(block::Block, tree::ControlTree, code::CodeInfo, block_id::Ref{Int})
    cond_idx = node_index(tree)

    # Find the GotoIfNot to get condition and branch targets
    gotoifnot_idx = cond_idx
    for i in cond_idx:length(code.code)
        if code.code[i] isa GotoIfNot
            gotoifnot_idx = i
            break
        end
    end

    gotoifnot = code.code[gotoifnot_idx]
    @assert gotoifnot isa GotoIfNot "Expected GotoIfNot at condition"

    # Get condition value
    cond_value = find_condition_value(cond_idx, code)

    # Collect any statements before the GotoIfNot in the condition block
    for i in cond_idx:gotoifnot_idx-1
        stmt = code.code[i]
        if !(stmt isa GotoNode || stmt isa GotoIfNot || stmt isa ReturnNode)
            push!(block.body, i)
        end
    end

    # The GotoIfNot branches: false -> dest, true -> fallthrough (next stmt)
    else_start = gotoifnot.dest
    then_start = gotoifnot_idx + 1

    # Build then block - follow statements until return
    then_blk = Block(block_id[])
    block_id[] += 1
    collect_branch_statements!(then_blk, then_start, code)

    # Build else block - follow statements until return
    else_blk = Block(block_id[])
    block_id[] += 1
    collect_branch_statements!(else_blk, else_start, code)

    # Create IfOp - no result vars since both branches terminate
    if_op = IfOp(cond_value, then_blk, else_blk, SSAValue[])
    push!(block.body, if_op)
end

"""
    collect_branch_statements!(block::Block, start_idx::Int, code::CodeInfo)

Collect statements from start_idx until hitting a return.
Adds non-control-flow statements to block.body and sets the return as terminator.
"""
function collect_branch_statements!(block::Block, start_idx::Int, code::CodeInfo)
    i = start_idx
    while i <= length(code.code)
        stmt = code.code[i]
        if stmt isa ReturnNode
            block.terminator = stmt
            break
        elseif stmt isa GotoNode || stmt isa GotoIfNot
            # Unexpected control flow - stop
            break
        else
            push!(block.body, i)
        end
        i += 1
    end
end

"""
    handle_if_then!(block::Block, tree::ControlTree, code::CodeInfo, block_id::Ref{Int})

Handle if-then region (no else branch).
"""
function handle_if_then!(block::Block, tree::ControlTree, code::CodeInfo, block_id::Ref{Int})
    @assert length(tree.children) >= 2 "if-then needs at least 2 children"

    # First child is the condition block
    cond_tree = tree.children[1]
    cond_idx = node_index(cond_tree)

    # Collect condition block statements
    if 1 <= cond_idx <= length(code.code)
        stmt = code.code[cond_idx]
        if !(stmt isa GotoNode || stmt isa GotoIfNot || stmt isa ReturnNode)
            push!(block.body, cond_idx)
        end
    end

    cond_value = find_condition_value(cond_idx, code)

    # Then block
    then_tree = tree.children[2]
    then_block = tree_to_block(then_tree, code, block_id)

    # Empty else block
    else_block = Block(block_id[])
    block_id[] += 1

    # Create IfOp
    if_op = IfOp(cond_value, then_block, else_block, SSAValue[])
    push!(block.body, if_op)
end

"""
    handle_loop!(block::Block, tree::ControlTree, code::CodeInfo, block_id::Ref{Int})

Handle loop region.
"""
function handle_loop!(block::Block, tree::ControlTree, code::CodeInfo, block_id::Ref{Int})
    # First child is header, second is body
    if isempty(tree.children)
        # Single-block loop
        idx = node_index(tree)
        body_block = Block(block_id[])
        block_id[] += 1
        if 1 <= idx <= length(code.code)
            stmt = code.code[idx]
            if !(stmt isa GotoNode || stmt isa GotoIfNot || stmt isa ReturnNode)
                push!(body_block.body, idx)
            end
        end
        loop_op = LoopOp(IRValue[], body_block, SSAValue[])
        push!(block.body, loop_op)
    else
        # Header + body
        header_tree = tree.children[1]
        header_idx = node_index(header_tree)

        # Collect header statements
        if 1 <= header_idx <= length(code.code)
            stmt = code.code[header_idx]
            if !(stmt isa GotoNode || stmt isa GotoIfNot || stmt isa ReturnNode)
                push!(block.body, header_idx)
            end
        end

        # Body block
        if length(tree.children) >= 2
            body_tree = tree.children[2]
            body_block = tree_to_block(body_tree, code, block_id)
        else
            body_block = Block(block_id[])
            block_id[] += 1
        end

        loop_op = LoopOp(IRValue[], body_block, SSAValue[])
        push!(block.body, loop_op)
    end
end

"""
    handle_self_loop!(block::Block, tree::ControlTree, code::CodeInfo, block_id::Ref{Int})

Handle self-loop region.
"""
function handle_self_loop!(block::Block, tree::ControlTree, code::CodeInfo, block_id::Ref{Int})
    idx = node_index(tree)

    body_block = Block(block_id[])
    block_id[] += 1

    if 1 <= idx <= length(code.code)
        stmt = code.code[idx]
        if !(stmt isa GotoNode || stmt isa GotoIfNot || stmt isa ReturnNode)
            push!(body_block.body, idx)
        end
    end

    loop_op = LoopOp(IRValue[], body_block, SSAValue[])
    push!(block.body, loop_op)
end

"""
    find_condition_value(stmt_idx::Int, code::CodeInfo) -> IRValue

Find the condition value for a GotoIfNot at or near stmt_idx.
"""
function find_condition_value(stmt_idx::Int, code::CodeInfo)
    # Look for GotoIfNot at or after this statement
    for i in stmt_idx:length(code.code)
        stmt = code.code[i]
        if stmt isa GotoIfNot
            cond = stmt.cond
            if cond isa SSAValue
                return cond
            elseif cond isa SlotNumber
                return cond
            elseif cond isa Argument
                return cond
            else
                # Fallback: condition must be evaluated before
                return SSAValue(i - 1)
            end
        end
        # Stop if we hit another control flow
        if stmt isa GotoNode || stmt isa ReturnNode
            break
        end
    end

    # Fallback: use previous SSA value as condition
    return SSAValue(max(1, stmt_idx - 1))
end

"""
    set_terminator!(block::Block, code::CodeInfo)

Set the block terminator based on statements.
"""
function set_terminator!(block::Block, code::CodeInfo)
    if block.terminator !== nothing
        return
    end

    # Check if the last collected statement leads to a return
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
 In-Place Structurization
=============================================================================#

"""
    structurize!(sci::StructuredCodeInfo) -> StructuredCodeInfo

Convert unstructured control flow in `sci` to structured control flow operations
(IfOp, LoopOp, ForOp) in-place.

This transforms GotoNode and GotoIfNot statements into nested structured ops
that can be traversed hierarchically.

Returns `sci` for convenience (allows chaining).

# Example
```julia
ci, _ = get_typed_ir(f, Tuple{Int})
sci = StructuredCodeInfo(ci)  # flat view with raw control flow
structurize!(sci)              # convert to structured ops
validate_structured_control_flow(sci)  # verify no unstructured control flow remains
```
"""
function structurize!(sci::StructuredCodeInfo)
    code = sci.code
    stmts = code.code
    n = length(stmts)

    n == 0 && return sci

    # Check if the code is straight-line (no control flow)
    has_control_flow = any(s -> s isa GotoNode || s isa GotoIfNot, stmts)

    if !has_control_flow
        # Straight-line code: just filter out the control flow from body
        # (there shouldn't be any, but be safe)
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

    # Has control flow - use block-level analysis
    blocks = build_block_cfg(code)
    loops = find_loops(blocks)

    if !isempty(loops)
        # Use block-level structuring with loop detection
        sci.entry = structurize_with_loops(code, blocks, loops)
    else
        # No loops - use control tree for if-then-else patterns
        cfg = julia_cfg(code)
        control_tree = build_control_tree(cfg, code)
        sci.entry = control_tree_to_structured_ir(control_tree, code).entry
    end

    return sci
end

"""
    structurize_with_loops(code::CodeInfo, blocks::Vector{BlockInfo}, loops::Vector{LoopInfo}) -> Block

Structurize code that contains loops using block-level analysis.
Returns the entry block with structured control flow.
"""
function structurize_with_loops(code::CodeInfo, blocks::Vector{BlockInfo}, loops::Vector{LoopInfo})
    stmts = code.code

    # For now, handle the common case: single loop with optional pre/post code
    @assert length(loops) == 1 "Multiple loops not yet supported"
    loop = loops[1]

    entry = Block(1)
    block_id = Ref(2)

    # Process blocks before the loop (entry blocks)
    entry_blocks = filter(bi -> bi ∉ loop.blocks, 1:loop.header-1)
    for bi in entry_blocks
        collect_block_stmts!(entry, blocks[bi], code)
    end

    # Create the LoopOp
    loop_op = build_loop_op(code, blocks, loop, block_id)
    push!(entry.body, loop_op)

    # Process blocks after the loop (exit blocks) - add to body after the loop
    for exit_bi in sort(collect(loop.exit_blocks))
        block = blocks[exit_bi]
        for si in block.range
            stmt = stmts[si]
            if stmt isa ReturnNode
                entry.terminator = stmt
            elseif !(stmt isa GotoNode || stmt isa GotoIfNot || stmt isa PhiNode)
                push!(entry.body, si)
            end
        end
    end

    return entry
end

"""
    collect_block_stmts!(block::Block, info::BlockInfo, code::CodeInfo)

Collect statements from a BlockInfo into a Block, excluding control flow.
"""
function collect_block_stmts!(block::Block, info::BlockInfo, code::CodeInfo)
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
end

"""
    detect_for_loop_pattern(code, blocks, loop) -> Union{ForLoopPattern, Nothing}

Detect if a loop is a simple counted for-loop with pattern:
- Header has phi for induction var: φ(init, next)
- Condition is slt_int(iv, upper) or similar
- Body has iv_next = add_int(iv, step)
"""
function detect_for_loop_pattern(code::CodeInfo, blocks::Vector{BlockInfo}, loop::LoopInfo)
    stmts = code.code
    header = blocks[loop.header]

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
    iv_phi_stmt = stmts[iv_phi_idx]
    iv_phi_stmt isa PhiNode || return nothing

    # Verify the phi is in the header
    iv_phi_idx in header.range || return nothing

    # Extract lower bound (init value from outside loop)
    stmt_to_blk = stmt_to_block_map(blocks, length(stmts))
    lower_bound = nothing

    for (edge_idx, _) in enumerate(iv_phi_stmt.edges)
        if isassigned(iv_phi_stmt.values, edge_idx)
            val = iv_phi_stmt.values[edge_idx]
            val_in_loop = false
            if val isa SSAValue && val.id > 0 && val.id <= length(stmts)
                val_in_loop = stmt_to_blk[val.id] ∈ loop.blocks
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
    for bi in loop.blocks
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

    return ForLoopPattern(lower_bound, convert_phi_value(upper_bound), step, iv_phi_idx)
end

"""
    build_for_op(code, blocks, loop, pattern, block_id) -> ForOp

Build a ForOp from detected for-loop pattern.
"""
function build_for_op(code::CodeInfo, blocks::Vector{BlockInfo}, loop::LoopInfo,
                      pattern::ForLoopPattern, block_id::Ref{Int})
    stmts = code.code
    header_block = blocks[loop.header]
    stmt_to_blk = stmt_to_block_map(blocks, length(stmts))

    # Collect block args and init values
    # First arg is induction variable, rest are carried values
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
                        val_in_loop = stmt_to_blk[val.id] ∈ loop.blocks
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
    iv_incr_idx = nothing
    for bi in loop.blocks
        for si in blocks[bi].range
            stmt = stmts[si]
            if stmt isa Expr && stmt.head === :call
                func = stmt.args[1]
                if func isa GlobalRef && func.name === :add_int
                    if stmt.args[2] isa SSAValue && stmt.args[2].id == pattern.induction_phi_idx
                        iv_incr_idx = si
                        break
                    end
                end
            end
        end
        iv_incr_idx !== nothing && break
    end

    # Collect body statements from all loop blocks, excluding:
    # - phi nodes
    # - control flow (goto, gotoifnot)
    # - the induction increment
    # - the condition comparison
    for bi in sort(collect(loop.blocks))
        block = blocks[bi]
        for si in block.range
            stmt = stmts[si]
            if stmt isa PhiNode || stmt isa GotoNode || stmt isa GotoIfNot || stmt isa ReturnNode
                continue
            end
            if si == iv_incr_idx
                continue
            end
            # Also skip the condition comparison
            if stmt isa Expr && stmt.head === :call
                func = stmt.args[1]
                if func isa GlobalRef && func.name in (:slt_int, :ult_int)
                    continue
                end
            end
            push!(body.body, si)
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
                    val_in_loop = stmt_to_blk[val.id] ∈ loop.blocks
                end
                if val_in_loop
                    push!(yield_values, convert_phi_value(val))
                    break
                end
            end
        end
    end
    # ForOp bodies use ContinueOp (not YieldOp) to continue iteration with updated values
    body.terminator = ContinueOp(yield_values)

    iv_ssa = SSAValue(pattern.induction_phi_idx)
    return ForOp(pattern.lower, pattern.upper, pattern.step, iv_ssa, init_values, body, result_vars)
end

"""
    build_loop_op(code::CodeInfo, blocks::Vector{BlockInfo}, loop::LoopInfo, block_id::Ref{Int}) -> Union{ForOp, LoopOp}

Build a ForOp or LoopOp from loop information.
Tries to detect for-loop pattern first, falls back to LoopOp.
"""
function build_loop_op(code::CodeInfo, blocks::Vector{BlockInfo}, loop::LoopInfo, block_id::Ref{Int})
    # Try to detect for-loop pattern first
    pattern = detect_for_loop_pattern(code, blocks, loop)
    if pattern !== nothing
        return build_for_op(code, blocks, loop, pattern, block_id)
    end

    # Fall back to LoopOp
    stmts = code.code
    header_block = blocks[loop.header]

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
            edges = phi.edges
            values = phi.values

            entry_preds = filter(p -> p ∉ loop.blocks, header_block.preds)
            latch_preds = filter(p -> p ∈ loop.blocks, header_block.preds)

            entry_val = nothing
            carried_val = nothing

            for (edge_idx, _edge) in enumerate(edges)
                if isassigned(values, edge_idx)
                    val = values[edge_idx]

                    if val isa SSAValue
                        val_stmt = val.id
                        stmt_to_blk = stmt_to_block_map(blocks, length(stmts))
                        if val_stmt > 0 && val_stmt <= length(stmts)
                            val_block = stmt_to_blk[val_stmt]
                            if val_block ∈ loop.blocks
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

            if entry_val !== nothing
                push!(init_values, entry_val)
            end
            if carried_val !== nothing
                push!(carried_values, carried_val)
            end

            phi_type = code.ssavaluetypes[si]
            arg = BlockArg(length(block_args) + 1, phi_type)
            push!(block_args, arg)
        end
    end

    # Build loop body block
    body = Block(block_id[])
    block_id[] += 1
    body.args = block_args

    # Collect header statements (excluding phi nodes and control flow)
    for si in header_block.range
        stmt = stmts[si]
        if !(stmt isa PhiNode || stmt isa GotoNode || stmt isa GotoIfNot || stmt isa ReturnNode)
            push!(body.body, si)
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

    # Collect body block statements (from latch blocks)
    for latch_bi in loop.latches
        if latch_bi != loop.header
            latch_block = blocks[latch_bi]
            for si in latch_block.range
                stmt = stmts[si]
                if !(stmt isa GotoNode || stmt isa GotoIfNot || stmt isa ReturnNode)
                    push!(body.body, si)
                end
            end
        end
    end

    # Create the conditional structure inside the loop body
    if condition !== nothing
        cond_value = convert_phi_value(condition)

        then_block = Block(block_id[])
        block_id[] += 1
        then_block.terminator = ContinueOp(carried_values)

        else_block = Block(block_id[])
        block_id[] += 1
        result_values = IRValue[]
        for (i, arg) in enumerate(block_args)
            push!(result_values, arg)
        end
        else_block.terminator = BreakOp(result_values)

        if_op = IfOp(cond_value, then_block, else_block, SSAValue[])
        push!(body.body, if_op)
    else
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
        return Literal(val)
    elseif val isa QuoteNode
        inner = val.value
        if inner isa Integer
            return Literal(inner)
        end
        return Literal(inner)
    else
        return Literal(0)  # Fallback
    end
end

#=============================================================================
 Structured Control Flow Validation
=============================================================================#

"""
    UnstructuredControlFlowError <: Exception

Exception thrown when unstructured control flow is detected in structured IR.
"""
struct UnstructuredControlFlowError <: Exception
    stmt_indices::Vector{Int}
end

function Base.showerror(io::IO, e::UnstructuredControlFlowError)
    print(io, "UnstructuredControlFlowError: unstructured control flow at statement(s): ",
          join(e.stmt_indices, ", "))
end

"""
    validate_scf(sci::StructuredCodeInfo) -> Bool

Validate that all control flow in the original CodeInfo has been properly
converted to structured control flow operations (IfOp, LoopOp, ForOp).

Throws `UnstructuredControlFlowError` if unstructured control flow remains.
Returns `true` if all control flow is properly structured.

The invariant is simple: no statement index in any `block.body` should point
to a `GotoNode` or `GotoIfNot` - those should have been replaced by structured
ops that the visitor descends into.
"""
function validate_scf(sci::StructuredCodeInfo)
    stmts = sci.code.code
    unstructured = Int[]

    # Walk all blocks and check that no statement is unstructured control flow
    each_stmt(sci.entry) do idx
        stmt = stmts[idx]
        if stmt isa GotoNode || stmt isa GotoIfNot
            push!(unstructured, idx)
        end
    end

    if !isempty(unstructured)
        throw(UnstructuredControlFlowError(sort!(unstructured)))
    end

    return true
end
