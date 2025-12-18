# Adapted from SPIRV.jl/src/analysis/structural_analysis.jl

using MLStyle: @active, @match
using Graphs
using Graphs: AbstractGraph, SimpleDiGraph, Edge, vertices, edges, nv, ne,
              inneighbors, outneighbors, strongly_connected_components
using AbstractTrees: PreOrderDFS, PostOrderDFS

"""
Single-entry control-flow structure which is classified according to well-specified pattern structures.
"""
@enum RegionType begin
    REGION_BLOCK
    REGION_IF_THEN
    REGION_IF_THEN_ELSE
    REGION_SWITCH
    REGION_TERMINATION
    REGION_PROPER
    REGION_SELF_LOOP
    REGION_WHILE_LOOP
    REGION_NATURAL_LOOP
    REGION_IMPROPER
end

"Sequence of blocks `u` ─→ [`v`, `vs...`] ─→ `w`"
REGION_BLOCK
"Conditional with one branch `u` ─→ `v` and one merge block reachable by `u` ─→ `w` or `v` ─→ `w`."
REGION_IF_THEN
"Conditional with two symmetric branches `u` ─→ `v` and `u` ─→ `w` and a single merge block reachable by `v` ─→ x or `w` ─→ x."
REGION_IF_THEN_ELSE
"Conditional with any number of branches [`u` ─→ `vᵢ`, `u` ─→ `vᵢ₊₁`, ...] and a single merge block reachable by [`vᵢ` ─→ `w`, `vᵢ₊₁` ─→ `w`, ...]."
REGION_SWITCH
"""
Acyclic region which contains a block `v` with multiple branches, including one or multiple branches to blocks `wᵢ` which end with a function termination instruction.
The region is composed of `v` and all the `wᵢ`.
"""
REGION_TERMINATION
"Acyclic region which does not match any other acyclic patterns."
REGION_PROPER
"Single node region which exhibits a self-loop."
REGION_SELF_LOOP
"Simple cycling region made of a condition block `u`, a loop body block `v` and a merge block `w` such that `v` ⇆ `u` ─→ `w`."
REGION_WHILE_LOOP
"Single-entry cyclic region with varying complexity such that the entry point dominates all nodes in the cyclic structure."
REGION_NATURAL_LOOP
"Single-entry region containing a multiple-entry cyclic region, such that the single entry is the least common dominator of all cycle entry nodes."
REGION_IMPROPER

# Define active patterns for use in pattern matching with MLStyle.

@active block_region(args) begin
    (g, v) = args
    start = v
    length(outneighbors(g, start)) < 2 || return nothing
    length(inneighbors(g, start)) < 2 || return nothing
    # Look ahead for a chain.
    vs = [start]
    while length(outneighbors(g, v)) == 1
        v = only(outneighbors(g, v))
        length(inneighbors(g, v)) == 1 || break
        length(outneighbors(g, v)) < 2 || break
        all(!in(vs), outneighbors(g, v)) || break
        in(v, vs) && break
        push!(vs, v)
    end
    # Look behind for a chain.
    v = start
    while length(inneighbors(g, v)) == 1
        v = only(inneighbors(g, v))
        length(outneighbors(g, v)) == 1 || break
        length(inneighbors(g, v)) < 2 || break
        all(!in(vs), inneighbors(g, v)) || break
        in(v, vs) && break
        pushfirst!(vs, v)
    end
    length(vs) == 1 && return nothing
    Some(vs)
end

@active self_loop(args) begin
    (g, v) = args
    in(v, outneighbors(g, v))
end

@active if_then_region(args) begin
    (g, v) = args
    if length(outneighbors(g, v)) == 2
        b, c = outneighbors(g, v)
        if is_single_entry_single_exit(g, b)
            only(outneighbors(g, b)) == c && return (v, b, c)
        elseif is_single_entry_single_exit(g, c)
            only(outneighbors(g, c)) == b && return (v, c, b)
        end
    end
    nothing
end

@active if_then_else_region(args) begin
    (g, v) = args
    if length(outneighbors(g, v)) == 2
        b, c = outneighbors(g, v)
        if is_single_entry_single_exit(g, b) && is_single_entry_single_exit(g, c)
            d = only(outneighbors(g, b))
            d == v && return nothing
            d == only(outneighbors(g, c)) || return nothing
            return (v, b, c, d)
        end
    end
    nothing
end

@active switch_region(args) begin
    (g, v) = args
    ws = outneighbors(g, v)
    length(ws) ≥ 2 || return nothing
    break_candidate = nothing
    for w in ws
        us = inneighbors(g, w)
        length(us) ≤ 2 || return nothing
        for u in us
            u == v || in(u, ws) && u ≠ w || return nothing
        end
        out = outneighbors(g, w)
        length(out) ≤ 2 || return nothing
        # Consider that a termination region instead.
        length(ws) == 2 && isempty(out) && return nothing
        for w′ in out
            in(w′, ws) && continue
            if isnothing(break_candidate)
                w′ == v && return nothing
                break_candidate = w′
            end
            w′ == break_candidate || return nothing
        end
    end
    Some(ws)
end

@active termination_region(args) begin
    (g, v, backedges) = args
    length(outneighbors(g, v)) ≥ 2 || return nothing
    all(!in(Edge(v, w), backedges) for w in outneighbors(g, v)) || return nothing
    termination_blocks = filter(w -> isempty(outneighbors(g, w)) && length(inneighbors(g, w)) == 1, outneighbors(g, v))
    isempty(termination_blocks) && return nothing
    Some(termination_blocks)
end

function acyclic_region(g, v, ec, doms, domtrees, backedges)
    @match (g, v) begin
        block_region(vs) => return (REGION_BLOCK, vs)
        if_then_region(v, t, m) => return (REGION_IF_THEN, [v, t])
        if_then_else_region(v, t, e, m) => return (REGION_IF_THEN_ELSE, [v, t, e])
        switch_region(branches) => return (REGION_SWITCH, [v; branches])
        _ => nothing
    end
    @match (g, v, backedges) begin
        termination_region(termination_blocks) => return (REGION_TERMINATION, [v; termination_blocks])
        _ => nothing
    end

    # Possibly a proper region.

    # Test that we don't have a loop or improper region.
    any(u -> in(Edge(u, v), backedges), inneighbors(g, v)) && return nothing

    domtree = domtrees[v]
    pdom_indices = findall(children(domtree)) do tree
        w = node_index(tree)
        in(w, vertices(g)) && !in(w, outneighbors(g, v))
    end
    length(pdom_indices) ≥ 1 || return nothing
    vs = Int64[]
    ws = Int64[]
    for i in pdom_indices
        pdomtree = domtree[i]
        w = node_index(pdomtree)
        append!(vs, vertices_between(g, v, w))
        push!(ws, w)
    end
    sort!(vs)
    unique!(vs)
    setdiff!(vs, [v; ws])
    pushfirst!(vs, v)
    append!(vs, ws)
    (REGION_PROPER, vs)
end

@active while_loop(args) begin
    (g, v) = args
    length(inneighbors(g, v)) ≠ 2 && return nothing
    length(outneighbors(g, v)) ≠ 2 && return nothing
    a, b = outneighbors(g, v)
    perm = if is_single_entry_single_exit(g, a) && only(inneighbors(g, a)) == only(outneighbors(g, a)) == v
        false
    elseif is_single_entry_single_exit(g, b) && only(inneighbors(g, b)) == only(outneighbors(g, b)) == v
        true
    else
        nothing
    end
    isnothing(perm) && return nothing
    # The returned result is of the form (loop condition, loop body, loop merge)
    perm ? (v, b, a) : (v, a, b)
end

function minimal_cyclic_component(g, v, backedges)
    vs = [v]
    for w in vertices(g)
        w == v && continue
        for e in backedges
            e.dst == v || continue
            if has_path(g, w, e.src; exclude_vertices=[v])
                push!(vs, w)
                break
            end
        end
    end
    vs
end

function cyclic_region!(sccs, g, v, ec, doms, domtrees, backedges)
    ret = @match (g, v) begin
        self_loop() => (REGION_SELF_LOOP, eltype(g)[])
        while_loop(cond, body, merge) => (REGION_WHILE_LOOP, [cond, body])
        _ => nothing
    end
    !isnothing(ret) && return ret

    scc = sccs[findfirst(Base.Fix1(in, v), sccs)]
    filter!(in(vertices(g)), scc)

    length(scc) == 1 && return nothing

    any(u -> in(Edge(u, v), ec.retreating_edges), inneighbors(g, v)) || return nothing
    cycle = minimal_cyclic_component(g, v, backedges)
    entry_edges = filter(e -> in(e.dst, cycle) && !in(e.src, cycle), edges(g))

    if any(u -> in(Edge(u, v), backedges), inneighbors(g, v))
        # Natural loop.
        # XXX: Are vertices returned in reverse post-order? The SSA condition on block labels should ensure
        # that iterating over `vertices(g)` already yields a topologically sorted graph.
        # Does it apply to Julia IR?
        all(==(v) ∘ (e -> e.dst), entry_edges) && return (REGION_NATURAL_LOOP, cycle)
    end

    # Improper region.
    length(entry_edges) < 2 && return nothing
    entry_points = unique!(map(e -> e.src, collect(entry_edges)))
    entry = common_ancestor(domtrees[v], [domtrees[ep] for ep in entry_points])
    @assert !isnothing(entry) "Multiple-entry cyclic region encountered with no single dominator"
    entry_node_v = node_index(entry)
    vs = [entry_node_v]
    exclude_vertices = [entry_node_v]
    mec_entries = unique!(map(e -> e.dst, collect(entry_edges)))
    for v in vertices(g)
        v == entry_node_v && continue
        !has_path(g, entry_node_v, v) && continue
        any(has_path(g, v, ep; exclude_vertices) for ep in mec_entries) && push!(vs, v)
    end
    # XXX: Might be that vertices are not topologically sorted.
    # If so, we may need to come up with one possible post-order traversal.
    (REGION_IMPROPER, vs)
end

function acyclic_region(g, v)
    dfst = SpanningTreeDFS(g)
    ec = EdgeClassification(g, dfst)
    doms = dominators(g)
    bedges = backedges(g, ec, doms)
    domtree = DominatorTree(doms)
    domtrees = sort(collect(PostOrderDFS(domtree)); by=x -> node_index(x))
    domtrees_dict = Dict(node_index(dt) => dt for dt in domtrees)
    acyclic_region(g, v, ec, doms, domtrees_dict, bedges)
end

function cyclic_region(g, v)
    dfst = SpanningTreeDFS(g)
    ec = EdgeClassification(g, dfst)
    doms = dominators(g)
    bedges = backedges(g, ec, doms)
    domtree = DominatorTree(doms)
    domtrees = sort(collect(PostOrderDFS(domtree)); by=x -> node_index(x))
    domtrees_dict = Dict(node_index(dt) => dt for dt in domtrees)
    sccs = strongly_connected_components(g)
    cyclic_region!(sccs, g, v, ec, doms, domtrees_dict, bedges)
end

struct ControlNode
    index::Int
    region_type::RegionType
end

"""
Control tree.

The leaves are labeled as [`REGION_BLOCK`](@ref) regions, with the distinguishing property that they have no children.

Children nodes of any given subtree are in reverse postorder according to the
original control-flow graph.
"""
const ControlTree = SimpleTree{ControlNode}

# Structures are constructed via pattern matching on the graph.

"Get the node index of the control tree."
node_index(tree::ControlTree) = nodevalue(tree).index
region_type(tree::ControlTree) = nodevalue(tree).region_type
ControlTree(v::Integer, region_type::RegionType, children=ControlTree[]) = ControlTree(ControlNode(v, region_type), children)

is_loop(ctree::ControlTree) = in(region_type(ctree), (REGION_NATURAL_LOOP, REGION_WHILE_LOOP))
is_selection(ctree::ControlTree) = in(region_type(ctree), (REGION_IF_THEN, REGION_IF_THEN_ELSE, REGION_SWITCH, REGION_TERMINATION))
is_block(ctree::ControlTree) = region_type(ctree) == REGION_BLOCK
is_proper_region(ctree::ControlTree) = region_type(ctree) == REGION_PROPER
is_switch(ctree::ControlTree) = region_type(ctree) == REGION_SWITCH

is_single_entry_single_exit(g::AbstractGraph, v) = length(inneighbors(g, v)) == 1 && length(outneighbors(g, v)) == 1
is_single_entry_single_exit(g::AbstractGraph) = is_weakly_connected(g) && length(sinks(g)) == length(sources(g)) == 1

function ControlTree(cfg::AbstractGraph{T}) where {T}
    dfst = SpanningTreeDFS(cfg)
    abstract_graph = DeltaGraph(cfg)
    ec = EdgeClassification(cfg, dfst)
    doms = dominators(cfg)
    bedges = backedges(cfg, ec, doms)
    domtree = DominatorTree(doms)
    domtrees = sort(collect(PostOrderDFS(domtree)); by=x -> node_index(x))
    domtrees_dict = Dict(node_index(dt) => dt for dt in domtrees)
    sccs = strongly_connected_components(cfg)

    control_trees = Dict{T,ControlTree}(v => ControlTree(ControlNode(v, REGION_BLOCK)) for v in vertices(cfg))
    next = post_ordering(dfst)

    while !isempty(next)
        start = popfirst!(next)
        haskey(control_trees, start) || continue
        # `v` can change if we follow a chain of blocks in a block region which starts before `v`, or if we need to find a dominator to an improper region.
        v = start
        ret = acyclic_region(abstract_graph, v, ec, doms, domtrees_dict, bedges)
        ret = if !isnothing(ret)
            (region_type, (v, ws...)) = ret
            (region_type, ws)
        else
            ret = cyclic_region!(sccs, abstract_graph, v, ec, doms, domtrees_dict, bedges)
            if !isnothing(ret)
                (region_type, (v, ws...)) = ret
                (region_type, ws)
            end
        end
        isnothing(ret) && continue
        if region_type == REGION_TERMINATION
            cyclic = cyclic_region!(sccs, abstract_graph, v, ec, doms, domtrees_dict, bedges)
            # Between a cyclic region and a termination region, choose
            # the cylic region; termination on the cycle's entry node
            # is not really a termination, just the natural program flow.
            if !isnothing(cyclic)
                (region_type, (v, ws...)) = cyclic
                ret = (region_type, ws)
            end
        end
        update_control_tree!(control_trees, v, ws, region_type)

        # Merge region vertices.
        if region_type ≠ REGION_SELF_LOOP
            for w in ws
                delete!(control_trees, w)

                # Adjust back-edges and retreating edges.
                for w′ in outneighbors(abstract_graph, w)
                    e = Edge(w, w′)
                    if in(e, bedges)
                        delete!(bedges, e)
                        push!(bedges, Edge(v, w′))
                    end
                    if in(e, ec.retreating_edges)
                        delete!(ec.retreating_edges, e)
                        push!(ec.retreating_edges, Edge(v, w′))
                    end
                end

                merge_vertices!(abstract_graph, v, w)
                rem_edge!(abstract_graph, v, v)
            end
        else
            rem_edge!(abstract_graph, v, v)
        end

        # Process transformed node next for eventual successive transformations.
        pushfirst!(next, v)
    end

    @assert nv(abstract_graph) == 1 string("Expected to contract the CFG into a single vertex, got ", nv(abstract_graph), " vertices instead.")
    only(values(control_trees))
end

function update_control_tree!(control_trees, v, ws, region)
    # `ws` must be in reverse post-order.
    ctree = compact_blocks(control_trees, v, ws, region)
    control_trees[v] = ctree
end

function compact_blocks(control_trees, v, ws, region)
    !in(region, (REGION_BLOCK, REGION_NATURAL_LOOP, REGION_WHILE_LOOP)) && return ControlTree(ControlNode(v, region), [control_trees[w] for w in [v; ws]])
    nodes = ControlTree[]
    for w in ws
        cctree = control_trees[w]
        if region_type(cctree) == REGION_BLOCK
            if all(region_type(x) == REGION_BLOCK for x in children(cctree))
                isempty(children(cctree)) ? push!(nodes, cctree) : append!(nodes, children(cctree))
                continue
            end
        end
        push!(nodes, cctree)
    end
    pushfirst!(nodes, control_trees[v])
    ControlTree(ControlNode(v, region), nodes)
end

function is_structured(ctree::ControlTree)
    all(PreOrderDFS(ctree)) do tree
        node = nodevalue(tree)
        !in(node.region_type, (REGION_PROPER, REGION_IMPROPER, REGION_SELF_LOOP))
    end
end

function outermost_tree(ctree::ControlTree, v::Integer)
    for subtree in PreOrderDFS(ctree)
        node_index(subtree) == v && return subtree
    end
end
