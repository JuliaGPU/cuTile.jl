# directed graph implementation with stable vertex indices after deletions

using Graphs
using Graphs: AbstractGraph, Edge, SimpleDiGraph, add_edge!, rem_edge!,
              vertices, edges, nv, ne, inneighbors, outneighbors,
              has_vertex, is_cyclic, is_weakly_connected, edgetype

"""
Graph whose vertices and edges remain identical after deletion of other vertices.
"""
struct DeltaGraph{T} <: AbstractGraph{T}
    vertices::Vector{T}
    fadjlist::Vector{Vector{T}}
    badjlist::Vector{Vector{T}}
end

Base.broadcastable(dg::DeltaGraph) = Ref(dg)

DeltaGraph{T}() where {T} = DeltaGraph{T}(T[], Vector{T}[], Vector{T}[])
DeltaGraph(n::T=0) where {T<:Integer} = DeltaGraph{T}(n)
function DeltaGraph(n::Integer, edges::AbstractVector)
    dg = DeltaGraph(n)
    for e in edges
        e isa Pair && (e = Edge(e...))
        add_edge!(dg, e)
    end
    dg
end
DeltaGraph(n::Integer, edges...) = DeltaGraph(n, collect(edges))
function DeltaGraph{T}(n::Integer) where {T}
    dg = DeltaGraph{T}()
    add_vertices!(dg, n)
    dg
end
DeltaGraph(edges...) = DeltaGraph(collect(edges))
DeltaGraph(edges::AbstractVector) = DeltaGraph(maximum(x -> max(x.first, x.second), edges), edges)

function vertex_index(dg::DeltaGraph, v)
    idx = findfirst(==(v), dg.vertices)
    isnothing(idx) && error("Vertex $v not found in $dg.")
    idx
end

Base.eltype(dg::DeltaGraph{T}) where {T} = T
Base.zero(dg::DeltaGraph) = zero(typeof(dg))
Base.zero(T::Type{DeltaGraph}) = T()
Graphs.edgetype(dg::DeltaGraph) = Edge{eltype(dg)}
Graphs.has_vertex(dg::DeltaGraph, v) = v in vertices(dg)
Graphs.vertices(dg::DeltaGraph) = dg.vertices

function Base.empty!(dg::DeltaGraph)
    empty!(dg.vertices)
    empty!(dg.badjlist)
    empty!(dg.fadjlist)
    dg
end

is_single_node(dg::DeltaGraph) = isempty(edges(dg)) && nv(dg) == 1

Base.isempty(dg::DeltaGraph) = isempty(dg.vertices)

Graphs.is_directed(dg::DeltaGraph) = is_directed(typeof(dg))
Graphs.is_directed(::Type{<:DeltaGraph}) = true

Graphs.has_edge(dg::DeltaGraph, e::Edge) = has_edge(dg, e.src, e.dst)
Graphs.has_edge(dg::DeltaGraph, src, dst) = has_vertex(dg, src) && has_vertex(dg, dst) && dst in outneighbors(dg, src)

function Graphs.edges(dg::DeltaGraph)
    edge_lists = map(enumerate(dg.fadjlist)) do (i, fadj)
        Edge.(dg.vertices[i], fadj)
    end
    edgetype(dg)[edge_lists...;]
end

Graphs.inneighbors(dg::DeltaGraph, v) = dg.badjlist[vertex_index(dg, v)]
Graphs.outneighbors(dg::DeltaGraph, v) = dg.fadjlist[vertex_index(dg, v)]

Graphs.ne(dg::DeltaGraph) = sum(length, dg.fadjlist; init=0)
Graphs.nv(dg::DeltaGraph) = length(vertices(dg))

function Graphs.add_vertex!(dg::DeltaGraph; fill_holes=true)
    idx = fill_holes ? findfirst(((i, x),) -> i ≠ x, collect(enumerate(dg.vertices))) : length(dg.vertices) + 1
    idx = something(idx, length(dg.vertices) + 1)
    T = eltype(dg)
    insert!(dg.vertices, idx, idx)
    insert!(dg.fadjlist, idx, T[])
    insert!(dg.badjlist, idx, T[])
    idx
end

function Graphs.add_vertices!(dg::DeltaGraph, n::Integer)
    [add_vertex!(dg, fill_holes=n < 1000) for _ = 1:n]
    nothing
end

Graphs.add_edge!(dg::DeltaGraph, e::Edge) = add_edge!(dg, e.src, e.dst)
function Graphs.add_edge!(dg::DeltaGraph, src, dst)
    src_idx = vertex_index(dg, src)
    if dst ∉ dg.fadjlist[src_idx]
        push!(dg.fadjlist[src_idx], dst)
    end
    dst_idx = vertex_index(dg, dst)
    if src ∉ dg.badjlist[dst_idx]
        push!(dg.badjlist[dst_idx], src)
    end
    nothing
end

"""
Remove all edges from or to vertex `v`.
"""
function rem_edges!(g::AbstractGraph, v::Integer)
    for i in outneighbors(g, v)
        rem_edge!(g, v, i)
    end
    for i in inneighbors(g, v)
        rem_edge!(g, i, v)
    end
end
rem_edges!(g::AbstractGraph, edges) = foreach(Base.Fix1(rem_edge!, g), edges)

function Graphs.rem_vertex!(dg::DeltaGraph, v)
    idx = vertex_index(dg, v)
    isnothing(idx) && return false
    rem_edges!(dg, v)
    deleteat!(dg.vertices, idx)
    deleteat!(dg.fadjlist, idx)
    deleteat!(dg.badjlist, idx)

    # Clean up all edges to avoid having invalid edges hanging around.
    for fadj in dg.fadjlist
        setdiff!(fadj, v)
    end
    for badj in dg.badjlist
        setdiff!(badj, v)
    end
    true
end

Graphs.rem_vertices!(dg::DeltaGraph, vs) = foreach(Base.Fix1(rem_vertex!, dg), vs)
Graphs.rem_vertices!(dg::DeltaGraph, v::Integer, vs::Integer...) = rem_vertices!(dg, (v, vs...))

add_edges!(g::AbstractGraph, edges) = foreach(Base.Fix1(add_edge!, g), edges)
Graphs.rem_edge!(dg::DeltaGraph, e::Edge) = rem_edge!(dg, e.src, e.dst)

function Graphs.rem_edge!(dg::DeltaGraph, src, dst)
    has_edge(dg, src, dst) || return false
    outs = outneighbors(dg, src)
    deleteat!(outs, findfirst(==(dst), outs))
    ins = inneighbors(dg, dst)
    deleteat!(ins, findfirst(==(src), ins))
    true
end

"""
Merge vertices `vs` into the first one.
"""
Graphs.merge_vertices!(dg::DeltaGraph, vs::AbstractVector) = foldl((x, y) -> merge_vertices!(dg, x, y), vs)
Graphs.merge_vertices!(dg::DeltaGraph, origin, merged...) = merge_vertices!(dg, [origin; collect(merged)])

function Graphs.merge_vertices!(dg::DeltaGraph, origin, merged)
    copy_edges!(dg, merged, origin)
    rem_vertex!(dg, merged)
    origin
end

function copy_edges!(dg::DeltaGraph, from, to)
    for i in outneighbors(dg, from)
        add_edge!(dg, to, i)
    end
    for i in inneighbors(dg, from)
        add_edge!(dg, i, to)
    end
end

function compact(dg::DeltaGraph)
    g = SimpleDiGraph(dg)
    dg = typeof(dg)()
    append!(dg.vertices, 1:nv(g))
    append!(dg.badjlist, g.badjlist)
    append!(dg.fadjlist, g.fadjlist)
    dg
end

function Graphs.SimpleDiGraph(dg::DeltaGraph)
    perm = sortperm(dg.vertices)
    verts_sorted = dg.vertices[perm]
    fadjs = dg.fadjlist[perm]
    badjs = dg.badjlist[perm]
    g = SimpleDiGraph{eltype(dg)}(length(dg.vertices))
    for (v, (fadj, badj)) in enumerate(zip(fadjs, badjs))
        for i in fadj
            add_edge!(g, v, findfirst(==(i), verts_sorted))
        end
        for i in badj
            add_edge!(g, findfirst(==(i), verts_sorted), v)
        end
    end
    g
end

function DeltaGraph(g::AbstractGraph{T}) where {T}
    is_directed(typeof(g)) || error("Delta graphs can only be used from directed graphs.")
    dg = DeltaGraph{T}(nv(g))
    add_edges!(dg, edges(g))
    dg
end

# Re-implementation of functions from Graphs.jl to be compatible with non-contiguous vertex storage.

"""
    has_path(g::AbstractGraph, u, v; exclude_vertices=Vector())

Return `true` if there is a path from `u` to `v` in `g` (while avoiding vertices in
`exclude_vertices`) or `u == v`. Return false if there is no such path or if `u` or `v`
is in `excluded_vertices`.

Re-implemented from Graphs.jl.
"""
function Graphs.has_path(g::DeltaGraph{T}, u::Integer, v::Integer;
        exclude_vertices::AbstractVector=Vector{T}()) where {T}
    seen = Set{T}(exclude_vertices)
    (in(u, seen) || in(v, seen)) && return false
    u == v && return true
    next = T[]
    push!(next, u)
    push!(seen, u)
    while !isempty(next)
        src = popfirst!(next)
        for vertex in outneighbors(g, src)
            vertex == v && return true
            if !in(vertex, seen)
                push!(next, vertex)
                push!(seen, vertex)
            end
        end
    end
    false
end

"""
Return the set of all vertices that one must go through to reach `v` starting from `u`, endpoints included.

`v` should be a post-dominator of `u` for this function to make sense.
"""
function vertices_between(g::AbstractGraph{T}, u::Integer, v::Union{Nothing,Integer}) where {T}
    collected = T[u]
    next = copy(outneighbors(g, u))
    while !isempty(next)
        w = popfirst!(next)
        in(w, collected) && continue
        push!(collected, w)
        (isnothing(v) || w ≠ v) && append!(next, outneighbors(g, w))
    end
    !isnothing(v) && !in(v, collected) && error("`v` could not be reached from `u`.")
    collected
end
