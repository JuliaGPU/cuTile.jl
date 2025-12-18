# Adapted from SPIRV.jl/src/analysis/control_flow.jl

using Graphs
using Graphs: AbstractGraph, SimpleDiGraph, Edge, add_edge!, rem_edge!,
              vertices, edges, nv, ne, inneighbors, outneighbors, edgetype, is_cyclic
using AbstractTrees
using AbstractTrees: AbstractTrees, PreOrderDFS, PostOrderDFS

function entry_node(g::AbstractGraph)
    vs = sources(g)
    isempty(vs) && error("No entry node was found.")
    length(vs) > 1 && error("Multiple entry nodes were found.")
    first(vs)
end

sinks(g::AbstractGraph) = vertices(g)[findall(isempty ∘ Base.Fix1(outneighbors, g), vertices(g))]
sources(g::AbstractGraph) = vertices(g)[findall(isempty ∘ Base.Fix1(inneighbors, g), vertices(g))]

mutable struct SimpleTree{T}
    data::T
    parent::Union{Nothing,SimpleTree{T}}
    children::Vector{SimpleTree{T}}
    function SimpleTree{T}(data::T, parent, children) where {T}
        tree = new{T}(data, parent, SimpleTree{T}[])
        # Make sure that all the children mark this new tree as parent.
        for c in children
            child_copy = SimpleTree{T}(c.data, tree, c.children)
            push!(tree.children, child_copy)
        end
        tree
    end
end
SimpleTree{T}(data::T, children=SimpleTree{T}[]) where {T} = SimpleTree{T}(data, nothing, children)
SimpleTree(data::T, parent, children) where {T} = SimpleTree{T}(data, parent, children)
SimpleTree(data::T, children=SimpleTree{T}[]) where {T} = SimpleTree{T}(data, children)

"""
Equality is defined for `SimpleTree`s over data and children. The equality of
parents is not tested to avoid infinite recursion, and only the presence of
parents is tested instead.
"""
Base.:(==)(x::SimpleTree{T}, y::SimpleTree{T}) where {T} = x.data == y.data && x.children == y.children && isnothing(x.parent) == isnothing(y.parent)

function Base.show(io::IO, ::MIME"text/plain", tree::SimpleTree)
    if isempty(children(tree))
        print(io, typeof(tree), "(", tree.data, ", [])")
    else
        print(io, sprint(print_tree, tree; context=io))
    end
end
Base.show(io::IO, tree::SimpleTree) = print(io, typeof(tree), "(", nodevalue(tree), isroot(tree) ? "" : string(", parent = ", nodevalue(parent(tree))), ", children = [", join(nodevalue.(children(tree)), ", "), "])")

Base.getindex(tree::SimpleTree, index) = children(tree)[index]
Base.firstindex(tree::SimpleTree) = firstindex(children(tree))
Base.lastindex(tree::SimpleTree) = lastindex(children(tree))

# AbstractTrees interface
AbstractTrees.nodetype(T::Type{<:SimpleTree}) = T
AbstractTrees.NodeType(::Type{SimpleTree{T}}) where {T} = AbstractTrees.HasNodeType()
AbstractTrees.nodevalue(tree::SimpleTree) = tree.data
AbstractTrees.ChildIndexing(::Type{<:SimpleTree}) = AbstractTrees.IndexedChildren()
AbstractTrees.ParentLinks(::Type{<:SimpleTree}) = AbstractTrees.StoredParents()
AbstractTrees.parent(tree::SimpleTree) = tree.parent
Base.parent(tree::SimpleTree) = tree.parent
AbstractTrees.children(tree::SimpleTree) = tree.children
AbstractTrees.childrentype(::Type{T}) where {T<:SimpleTree} = T

isroot(tree::SimpleTree) = isnothing(tree.parent)
nodevalue(tree::SimpleTree) = tree.data
children(tree::SimpleTree) = tree.children
parent(tree::SimpleTree) = tree.parent

# print_tree helper using AbstractTrees
function print_tree(io::IO, tree::SimpleTree; maxdepth=10)
    AbstractTrees.print_tree(io, tree; maxdepth)
end

struct SpanningTreeDFS{G<:AbstractGraph}
    tree::G
    discovery_times::Vector{Int}
    finish_times::Vector{Int}
end

function SpanningTreeDFS(g::AbstractGraph{T}, source=1) where {T}
    tree = SimpleDiGraph{T}(nv(g))
    dfst = SpanningTreeDFS(tree, zeros(Int, nv(g)), zeros(Int, nv(g)))
    build!(dfst, [source], zeros(Bool, nv(g)), g)
    dfst
end

function build!(dfst::SpanningTreeDFS, next, visited, g::AbstractGraph, time=0)
    v = pop!(next)
    visited[v] = true
    dfst.discovery_times[v] = (time += 1)
    for w in outneighbors(g, v)
        if !visited[w]
            add_edge!(dfst.tree, v, w)
            push!(next, w)
            time = build!(dfst, next, visited, g, time)
        end
    end
    dfst.finish_times[v] = (time += 1)
    time
end

pre_ordering(dfst::SpanningTreeDFS) = sortperm(dfst.discovery_times)
post_ordering(dfst::SpanningTreeDFS) = sortperm(dfst.finish_times)

struct EdgeClassification{E<:AbstractEdge}
    tree_edges::Set{E}
    forward_edges::Set{E}
    retreating_edges::Set{E}
    cross_edges::Set{E}
end

EdgeClassification{E}() where {E} = EdgeClassification(Set{E}(), Set{E}(), Set{E}(), Set{E}())

function SimpleTree(dfst::SpanningTreeDFS, parent::Union{Nothing,SimpleTree{T}}, v::T) where {T}
    tree = SimpleTree(v, parent, SimpleTree{T}[])
    for w in outneighbors(dfst.tree, v)
        push!(tree.children, SimpleTree(dfst, tree, w))
    end
    tree
end

SimpleTree(dfst::SpanningTreeDFS) = SimpleTree(dfst, nothing, entry_node(dfst.tree))

EdgeClassification(g::AbstractGraph, dfst::SpanningTreeDFS=SpanningTreeDFS(g)) = EdgeClassification(g, SimpleTree(dfst))

function EdgeClassification(g::AbstractGraph{T}, tree::SimpleTree{T}) where {T}
    E = edgetype(g)
    ec = EdgeClassification{E}()
    for subtree in PreOrderDFS(tree)
        # Traverse the tree and classify edges based on ancestor information.
        # Outgoing edges are used to find retreating edges (if pointing to an ancestor).
        # Incoming edges are used to find tree edges (if coming from parent) and forward edges (if pointing to an ancestor that is not the parent).
        # Other edges are cross-edges.
        v = nodevalue(subtree)
        p = parent(subtree)

        for u in inneighbors(g, v)
            e = E(u, v)
            if !isnothing(p) && u == nodevalue(p)
                push!(ec.tree_edges, e)
            elseif !isnothing(find_parent(==(u) ∘ nodevalue, subtree))
                push!(ec.forward_edges, e)
            end
        end

        for w in outneighbors(g, v)
            e = E(v, w)
            !isnothing(find_parent(==(w) ∘ nodevalue, subtree)) && push!(ec.retreating_edges, e)
        end
    end

    for e in edges(g)
        !in(e, ec.tree_edges) && !in(e, ec.forward_edges) && !in(e, ec.retreating_edges) && push!(ec.cross_edges, e)
    end

    ec
end

struct ControlFlowGraph{E<:AbstractEdge,T,G<:AbstractGraph{T}} <: AbstractGraph{T}
    g::G
    dfst::SpanningTreeDFS{G}
    ec::EdgeClassification{E}
    is_reducible::Bool
    is_structured::Bool
end

Base.broadcastable(cfg::ControlFlowGraph) = Ref(cfg)

# Forward methods to underlying graph
Graphs.vertices(cfg::ControlFlowGraph) = vertices(cfg.g)
Graphs.edges(cfg::ControlFlowGraph) = edges(cfg.g)
Graphs.add_edge!(cfg::ControlFlowGraph, e) = add_edge!(cfg.g, e)
Base.eltype(cfg::ControlFlowGraph) = eltype(cfg.g)
Graphs.edgetype(cfg::ControlFlowGraph) = edgetype(cfg.g)
Graphs.add_vertex!(cfg::ControlFlowGraph, v) = add_vertex!(cfg.g, v)
Graphs.rem_edge!(cfg::ControlFlowGraph, e) = rem_edge!(cfg.g, e)
Graphs.rem_vertex!(cfg::ControlFlowGraph, v) = rem_vertex!(cfg.g, v)
Graphs.rem_vertices!(cfg::ControlFlowGraph, vs) = rem_vertices!(cfg.g, vs)
Graphs.inneighbors(cfg::ControlFlowGraph, v) = inneighbors(cfg.g, v)
Graphs.outneighbors(cfg::ControlFlowGraph, v) = outneighbors(cfg.g, v)
Graphs.nv(cfg::ControlFlowGraph) = nv(cfg.g)
Graphs.ne(cfg::ControlFlowGraph) = ne(cfg.g)
dominators(cfg::ControlFlowGraph) = dominators(cfg.g)

Graphs.is_directed(::Type{<:ControlFlowGraph}) = true

Base.reverse(cfg::ControlFlowGraph) = ControlFlowGraph(reverse(cfg.g))

is_reducible(cfg::ControlFlowGraph) = cfg.is_reducible
is_structured(cfg::ControlFlowGraph) = cfg.is_structured

function ControlFlowGraph(cfg::AbstractGraph)
    dfst = SpanningTreeDFS(cfg)
    ec = EdgeClassification(cfg, dfst)

    analysis_cfg = deepcopy(cfg)
    rem_edges!(analysis_cfg, backedges(cfg, ec))
    is_reducible = !is_cyclic(analysis_cfg)

    # TODO: actually test whether CFG is structured or not.
    is_structured = is_reducible
    ControlFlowGraph(cfg, dfst, ec, is_reducible, is_structured)
end

dominators(g::AbstractGraph{T}) where {T} = dominators(g, vertices(g), entry_node(g))
function dominators(g::AbstractGraph{T}, vs, source) where {T}
    doms = Dict(v => Set{T}() for v in vs)
    push!(doms[source], source)
    vs_excluding_source = filter(≠(source), vs)
    for v in vs_excluding_source
        union!(doms[v], vs)
    end
    vs_set = Set(vs)

    converged = false
    while !converged
        converged = true
        for v in vs_excluding_source
            h = hash(doms[v])
            preds = [u for u in inneighbors(g, v) if in(u, vs_set)]
            if !isempty(preds)
                set = intersect((doms[u] for u in preds)...)
            else
                set = Set{T}()
            end
            doms[v] = set
            push!(set, v)
            h ≠ hash(set) && (converged &= false)
        end
    end

    doms
end

function backedges(cfg::ControlFlowGraph)
    is_reducible(cfg) && return copy(cfg.ec.retreating_edges)
    backedges(cfg.g, cfg.ec)
end

function backedges(g::AbstractGraph{T}, ec::EdgeClassification=EdgeClassification(g), domsets::Dict{T,Set{T}}=dominators(g)) where {T}
    filter(ec.retreating_edges) do e
        in(e.dst, domsets[e.src])
    end
end

function remove_backedges(cfg::ControlFlowGraph)
    g = deepcopy(cfg.g)
    rem_edges!(g, backedges(cfg))
    ControlFlowGraph(g)
end

traverse(cfg::ControlFlowGraph) = reverse(post_ordering(cfg.dfst))
traverse(cfg::AbstractGraph) = reverse(post_ordering(SpanningTreeDFS(cfg)))

struct DominatorNode
    index::Int
end

const DominatorTree = SimpleTree{DominatorNode}

node_index(tree::DominatorTree) = nodevalue(tree).index

immediate_postdominators(tree::DominatorTree) = node_index.(children(tree))
immediate_dominator(tree::DominatorTree) = node_index(@something(parent(tree), return))

DominatorTree(cfg::AbstractGraph) = DominatorTree(dominators(cfg))
DominatorTree(node::Integer, children=DominatorNode[]) = DominatorTree(DominatorNode(node), children)

function DominatorTree(domsets::Dict{T,Set{T}}) where {T}
    root = nothing
    idoms = Dict{T,T}()

    # Compute immediate dominators from the dominator sets.
    # One node's only immediate dominator is going to be its parent
    # in the tree representation.
    for (v, domset) in pairs(domsets)
        if length(domset) == 1
            isnothing(root) || error("Found multiple root dominators.")
            root = v
            continue
        end

        candidates = copy(domset)
        delete!(candidates, v)
        for p in candidates
            for dom in domsets[p]
                dom == p && continue
                in(dom, candidates) && delete!(candidates, dom)
            end
        end
        idom = only(candidates)
        idoms[v] = idom
    end

    # Attach all the subtrees together and to the root tree.
    root_tree = DominatorTree(DominatorNode(root))
    trees = Dict(v => DominatorTree(DominatorNode(v)) for v in keys(idoms))
    for (v, tree) in pairs(trees)
        # Skip trees which have already been attached.
        isroot(tree) || continue
        idom = idoms[v]
        p = get(trees, idom, root_tree)
        tree.parent = p
        push!(p.children, tree)
    end
    root_tree
end

common_ancestor(trees) = common_ancestor(Iterators.peel(trees)...)
function common_ancestor(tree, trees)
    common_anc = tree
    parent_chain = parents(common_anc)
    for candidate in trees
        common_anc = in(candidate, parent_chain) ? candidate : find_parent(in(parent_chain), candidate)
        parent_chain = parents(common_anc)
        isnothing(common_anc) && return nothing
    end
    common_anc
end

is_ancestor(candidate, tree) = !isnothing(find_parent(==(candidate), tree))

function parents(tree)
    res = [tree]
    while true
        isroot(tree) && break
        tree = parent(tree)
        push!(res, tree)
    end
    res
end

function find_parent(f, tree)
    original = tree
    while true
        f(tree) === true && tree !== original && return tree
        isroot(tree) && break
        tree = parent(tree)
    end
end

function find_subtree(f, tree::SimpleTree)
    original = tree
    for tree in PreOrderDFS(tree)
        f(tree) === true && tree !== original && return tree
    end
    nothing
end
