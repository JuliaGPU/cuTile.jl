# CFG analysis: backedge detection, edge classification, traversal

using Graphs
using Graphs: AbstractGraph, SimpleDiGraph, Edge, add_edge!, rem_edge!,
              vertices, edges, nv, ne, inneighbors, outneighbors, edgetype, is_cyclic

# Re-export from Core.Compiler (imported in IRStructurizer.jl)
# construct_domtree, DomTree, dominates, bb_unreachable

"""
    is_backedge(domtree::DomTree, from::Int, to::Int) -> Bool

Check if an edge from→to is a backedge (target dominates source).
"""
is_backedge(domtree::DomTree, from::Int, to::Int) = dominates(domtree, to, from)

"""
    backedges(ir::IRCode) -> Vector{Tuple{Int,Int}}

Find all backedges in the CFG.
A backedge is an edge where the target dominates the source.
"""
function backedges(ir::IRCode)
    domtree = construct_domtree(ir)
    result = Tuple{Int,Int}[]
    for (i, bb) in enumerate(ir.cfg.blocks)
        for succ in bb.succs
            if dominates(domtree, succ, i)
                push!(result, (i, succ))
            end
        end
    end
    result
end

"""
    backedges(cfg::SimpleDiGraph, domtree::DomTree) -> Set{Edge{Int}}

Find all backedges in the CFG using pre-computed dominator tree.
"""
function backedges(cfg::SimpleDiGraph, domtree::DomTree)
    result = Set{Edge{Int}}()
    for v in vertices(cfg)
        for succ in outneighbors(cfg, v)
            if dominates(domtree, succ, v)
                push!(result, Edge(v, succ))
            end
        end
    end
    result
end

"""
    EdgeClassification

Simple edge classification struct for control tree construction.
Only tracks retreating_edges since that's all that's used.
"""
struct EdgeClassification
    retreating_edges::Set{Edge{Int}}
end

"""
    EdgeClassification(cfg::SimpleDiGraph)

Build edge classification from a CFG.
Retreating edges are back-edges (where target dominates source).
"""
function EdgeClassification(cfg::SimpleDiGraph)
    nblocks = nv(cfg)
    nblocks == 0 && return EdgeClassification(Set{Edge{Int}}())

    # Build BasicBlocks for construct_domtree
    blocks = Vector{BasicBlock}(undef, nblocks)
    for v in 1:nblocks
        preds = collect(Int, inneighbors(cfg, v))
        succs = collect(Int, outneighbors(cfg, v))
        blocks[v] = BasicBlock(StmtRange(v, v), preds, succs)
    end

    domtree = construct_domtree(blocks)
    retreating = backedges(cfg, domtree)
    EdgeClassification(retreating)
end

# Minimal graph utilities needed by control_tree.jl

function entry_node(g::AbstractGraph)
    for v in vertices(g)
        if isempty(inneighbors(g, v))
            return v
        end
    end
    # Fallback for graphs with back edges to entry
    1 in vertices(g) ? 1 : error("No entry node found")
end

sinks(g::AbstractGraph) = [v for v in vertices(g) if isempty(outneighbors(g, v))]

"""
    traverse(cfg::SimpleDiGraph) -> Vector{Int}

Return vertices in reverse post-order (topological order for acyclic graphs).
"""
function traverse(cfg::SimpleDiGraph)
    n = nv(cfg)
    visited = falses(n)
    post_order = Int[]

    function dfs(v)
        visited[v] = true
        for w in outneighbors(cfg, v)
            if !visited[w]
                dfs(w)
            end
        end
        push!(post_order, v)
    end

    # Start from entry (vertex 1)
    if n > 0
        dfs(1)
    end

    # Visit any remaining unvisited vertices
    for v in 1:n
        if !visited[v]
            dfs(v)
        end
    end

    reverse(post_order)
end
