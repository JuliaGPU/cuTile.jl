# graph-level CFG pattern detection
#
# inspects the control-flow graph to build a control tree

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
    REGION_FOR_LOOP
    REGION_WHILE_LOOP
    REGION_NATURAL_LOOP
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
"Simple cycling region made of a condition block `u`, a loop body block `v` and a merge block `w` such that `v` ⇆ `u` ─→ `w`."
REGION_WHILE_LOOP
"Single-entry cyclic region with varying complexity such that the entry point dominates all nodes in the cyclic structure."
REGION_NATURAL_LOOP

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

#=============================================================================
 For-Loop Detection During CFG Analysis
=============================================================================#

"""
    try_detect_for_loop_while(header_idx::Int, body_idx::Int, code::CodeInfo, blocks)

Try to detect a for-loop pattern in a while-loop structure.
Returns ForLoopInfo if found, nothing otherwise.

Matches: while iv < bound; ...; iv = iv + step; end
- Condition: slt_int(iv, bound) or ult_int(iv, bound)
- Step: add_int(iv, step) in body
"""
function try_detect_for_loop_while(header_idx::Int, body_idx::Int, code::CodeInfo, blocks)
    stmts = code.code

    # Check bounds
    (1 <= header_idx <= length(blocks) && 1 <= body_idx <= length(blocks)) || return nothing

    header_block = blocks[header_idx]
    body_block = blocks[body_idx]

    # Find phi nodes in header (loop-carried variables)
    # Use SSA index position to distinguish entry vs carried values:
    # - Entry values come from before the header (lower SSA indices)
    # - Carried values come from within the loop body (higher SSA indices)
    phi_info = Dict{Int, NamedTuple{(:entry_val, :carried_val), Tuple{Any, Any}}}()
    for si in header_block.range
        stmt = stmts[si]
        if stmt isa PhiNode
            entry_val = nothing
            carried_val = nothing
            for (edge_idx, _) in enumerate(stmt.edges)
                if isassigned(stmt.values, edge_idx)
                    val = stmt.values[edge_idx]
                    # Carried value comes from after header, entry from before
                    if val isa SSAValue && val.id > header_block.range.stop
                        carried_val = val
                    elseif !(val isa SSAValue) || val.id < header_block.range.start
                        entry_val = val
                    end
                end
            end
            if entry_val !== nothing && carried_val !== nothing
                phi_info[si] = (entry_val=entry_val, carried_val=carried_val)
            end
        end
    end

    isempty(phi_info) && return nothing

    # Find the condition: GotoIfNot with slt_int/ult_int
    condition_ssa = nothing
    for si in header_block.range
        stmt = stmts[si]
        if stmt isa GotoIfNot
            condition_ssa = stmt.cond
            break
        end
    end
    condition_ssa isa SSAValue || return nothing

    # Get the condition expression
    cond_stmt = stmts[condition_ssa.id]
    cond_stmt isa Expr || return nothing
    cond_stmt.head === :call || return nothing
    length(cond_stmt.args) >= 3 || return nothing

    func = cond_stmt.args[1]
    func isa GlobalRef || return nothing
    func.name in (:slt_int, :ult_int) || return nothing

    # Extract IV candidate and upper bound from condition
    iv_candidate = cond_stmt.args[2]
    upper_bound = cond_stmt.args[3]

    # IV must be an SSAValue pointing to a phi node in header
    iv_candidate isa SSAValue || return nothing
    iv_phi_idx = iv_candidate.id
    haskey(phi_info, iv_phi_idx) || return nothing

    # Find step expression by tracing data flow from carried value
    # (handles nested control flow where add_int may not be in immediate body block)
    step = find_step_from_carried_value(stmts, phi_info[iv_phi_idx].carried_val, iv_candidate)
    step === nothing && return nothing

    # Get lower bound from phi entry value
    lower_bound = phi_info[iv_phi_idx].entry_val

    # Check if upper bound is loop-invariant (should be defined before header)
    if upper_bound isa SSAValue && upper_bound.id >= header_block.range.start
        return nothing
    end

    return ForLoopInfo(iv_phi_idx, lower_bound, upper_bound, step, false)
end

"""
    try_detect_for_loop_natural(header_idx::Int, cycle::Vector{Int}, code::CodeInfo, blocks)

Try to detect a for-loop pattern in a natural loop (Julia's `for i in 1:n`).
Returns ForLoopInfo if found, nothing otherwise.

Matches: for i in 1:n ... end
- Condition: ===(iv, bound) (equality check at end of range)
- Step: add_int(iv, 1) typically

Note: Uses data-flow tracing rather than cycle blocks to find the step expression,
because inner control flow may be collapsed before loop detection.
"""
function try_detect_for_loop_natural(header_idx::Int, cycle::Vector{Int}, code::CodeInfo, blocks)
    stmts = code.code

    (1 <= header_idx <= length(blocks)) || return nothing
    header_block = blocks[header_idx]

    # Find phi nodes in header by tracing carried values through data flow
    # We can't rely on cycle blocks because inner control flow may be collapsed
    phi_info = Dict{Int, NamedTuple{(:entry_val, :carried_val), Tuple{Any, Any}}}()

    for si in header_block.range
        stmt = stmts[si]
        if stmt isa PhiNode
            entry_val = nothing
            carried_val = nothing
            for (edge_idx, _) in enumerate(stmt.edges)
                if isassigned(stmt.values, edge_idx)
                    val = stmt.values[edge_idx]
                    # Carried value comes from after header, entry from before
                    if val isa SSAValue && val.id > header_block.range.stop
                        carried_val = val
                    elseif !(val isa SSAValue) || val.id < header_block.range.start
                        entry_val = val
                    end
                end
            end
            if entry_val !== nothing && carried_val !== nothing
                phi_info[si] = (entry_val=entry_val, carried_val=carried_val)
            end
        end
    end

    isempty(phi_info) && return nothing

    # Find the condition: GotoIfNot with === (equality check)
    condition_ssa = nothing
    for si in header_block.range
        stmt = stmts[si]
        if stmt isa GotoIfNot
            condition_ssa = stmt.cond
            break
        end
    end
    condition_ssa isa SSAValue || return nothing

    # Get the condition expression
    cond_stmt = stmts[condition_ssa.id]
    cond_stmt isa Expr || return nothing
    cond_stmt.head === :call || return nothing
    length(cond_stmt.args) >= 3 || return nothing

    func = cond_stmt.args[1]
    # Match === (either as GlobalRef or direct symbol)
    is_equality = if func isa GlobalRef
        func.name === :(===)
    elseif func === :(===)
        true
    else
        false
    end
    is_equality || return nothing

    # Extract IV candidate and upper bound
    iv_candidate = cond_stmt.args[2]
    upper_bound = cond_stmt.args[3]

    iv_candidate isa SSAValue || return nothing
    iv_phi_idx = iv_candidate.id
    haskey(phi_info, iv_phi_idx) || return nothing

    # Find step expression by following data flow from carried value
    # Trace: iv_phi -> carried_val -> (possibly through phis) -> add_int
    step = find_step_from_carried_value(stmts, phi_info[iv_phi_idx].carried_val, iv_candidate)

    step === nothing && return nothing

    lower_bound = phi_info[iv_phi_idx].entry_val

    # Check loop invariance of upper bound (should be defined before header)
    if upper_bound isa SSAValue && upper_bound.id >= header_block.range.start
        return nothing
    end

    # is_inclusive=true for Julia's 1:n pattern (needs upper+1 adjustment)
    return ForLoopInfo(iv_phi_idx, lower_bound, upper_bound, step, true)
end

"""
    find_step_from_carried_value(stmts, carried_val, iv_candidate) -> Union{Any, Nothing}

Follow data flow from a phi's carried value to find an add_int step expression.
Handles Julia's pattern where carried value goes through intermediate phi nodes.
"""
function find_step_from_carried_value(stmts, carried_val, iv_candidate, depth=0)
    # Prevent infinite recursion
    depth > 5 && return nothing
    carried_val isa SSAValue || return nothing

    stmt = stmts[carried_val.id]

    # Direct case: carried_val is the result of add_int
    if stmt isa Expr && stmt.head === :call && length(stmt.args) >= 3
        func = stmt.args[1]
        if func isa GlobalRef && func.name === :add_int
            arg1 = stmt.args[2]
            if arg1 == iv_candidate
                return stmt.args[3]
            end
        end
    end

    # Indirect case: carried_val is a phi that merges values including the step
    if stmt isa PhiNode
        for (edge_idx, _) in enumerate(stmt.edges)
            if isassigned(stmt.values, edge_idx)
                val = stmt.values[edge_idx]
                result = find_step_from_carried_value(stmts, val, iv_candidate, depth + 1)
                if result !== nothing
                    return result
                end
            end
        end
    end

    return nothing
end

#=============================================================================
 Cyclic Region Detection
=============================================================================#

function cyclic_region!(sccs, g, v, ec, doms, domtrees, backedges, code, blocks)
    # Try to match while-loop pattern first
    matched = @match (g, v) begin
        while_loop(cond, body, merge) => begin
            # Try for-loop detection on while-loop structure
            for_info = try_detect_for_loop_while(cond, body, code, blocks)
            if for_info !== nothing
                (REGION_FOR_LOOP, [cond, body], for_info)
            else
                (REGION_WHILE_LOOP, [cond, body], nothing)
            end
        end
        _ => nothing
    end
    !isnothing(matched) && return matched

    scc = sccs[findfirst(Base.Fix1(in, v), sccs)]
    filter!(in(vertices(g)), scc)

    length(scc) == 1 && return nothing

    any(u -> in(Edge(u, v), ec.retreating_edges), inneighbors(g, v)) || return nothing
    cycle = minimal_cyclic_component(g, v, backedges)
    entry_edges = filter(e -> in(e.dst, cycle) && !in(e.src, cycle), edges(g))

    if any(u -> in(Edge(u, v), backedges), inneighbors(g, v))
        # Natural loop - try for-loop detection
        if all(==(v) ∘ (e -> e.dst), entry_edges)
            for_info = try_detect_for_loop_natural(v, cycle, code, blocks)
            if for_info !== nothing
                return (REGION_FOR_LOOP, cycle, for_info)
            end
            return (REGION_NATURAL_LOOP, cycle, nothing)
        end
    end

    # Irreducible control flow - not supported
    return nothing
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

"""
    ForLoopInfo

Metadata for REGION_FOR_LOOP detected during CFG analysis.
Contains information needed to construct a ForOp during structurization.
"""
struct ForLoopInfo
    iv_phi_idx::Int           # SSA index of the induction variable phi node
    lower::Any                # Lower bound (SSAValue, Argument, or constant)
    upper::Any                # Upper bound
    step::Any                 # Step value
    is_inclusive::Bool        # True for Julia's 1:n (needs upper+1 adjustment)
end

struct ControlNode
    index::Int
    region_type::RegionType
    metadata::Union{Nothing, ForLoopInfo}
end

ControlNode(idx::Integer, region_type::RegionType) = ControlNode(idx, region_type, nothing)

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
metadata(tree::ControlTree) = nodevalue(tree).metadata
ControlTree(v::Integer, region_type::RegionType, children=ControlTree[]) = ControlTree(ControlNode(v, region_type), children)
ControlTree(v::Integer, region_type::RegionType, meta::ForLoopInfo, children=ControlTree[]) = ControlTree(ControlNode(v, region_type, meta), children)

is_loop(ctree::ControlTree) = in(region_type(ctree), (REGION_FOR_LOOP, REGION_NATURAL_LOOP, REGION_WHILE_LOOP))
is_selection(ctree::ControlTree) = in(region_type(ctree), (REGION_IF_THEN, REGION_IF_THEN_ELSE, REGION_SWITCH, REGION_TERMINATION))
is_block(ctree::ControlTree) = region_type(ctree) == REGION_BLOCK
is_proper_region(ctree::ControlTree) = region_type(ctree) == REGION_PROPER
is_switch(ctree::ControlTree) = region_type(ctree) == REGION_SWITCH

is_single_entry_single_exit(g::AbstractGraph, v) = length(inneighbors(g, v)) == 1 && length(outneighbors(g, v)) == 1
is_single_entry_single_exit(g::AbstractGraph) = is_weakly_connected(g) && length(sinks(g)) == length(sources(g)) == 1

function ControlTree(cfg::AbstractGraph{T}, code::CodeInfo, blocks) where {T}
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
        region_meta = nothing
        ret = if !isnothing(ret)
            (region_type, (v, ws...)) = ret
            (region_type, ws)
        else
            cyclic_ret = cyclic_region!(sccs, abstract_graph, v, ec, doms, domtrees_dict, bedges, code, blocks)
            if !isnothing(cyclic_ret)
                (region_type, (v, ws...), region_meta) = cyclic_ret
                (region_type, ws)
            else
                nothing
            end
        end
        isnothing(ret) && continue
        if region_type == REGION_TERMINATION
            cyclic = cyclic_region!(sccs, abstract_graph, v, ec, doms, domtrees_dict, bedges, code, blocks)
            # Between a cyclic region and a termination region, choose
            # the cylic region; termination on the cycle's entry node
            # is not really a termination, just the natural program flow.
            if !isnothing(cyclic)
                (region_type, (v, ws...), region_meta) = cyclic
                ret = (region_type, ws)
            end
        end
        update_control_tree!(control_trees, v, ws, region_type, region_meta)

        # Merge region vertices.
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

        # Process transformed node next for eventual successive transformations.
        pushfirst!(next, v)
    end

    @assert nv(abstract_graph) == 1 string("Expected to contract the CFG into a single vertex, got ", nv(abstract_graph), " vertices instead.")
    only(values(control_trees))
end

function update_control_tree!(control_trees, v, ws, region, region_meta=nothing)
    # `ws` must be in reverse post-order.
    ctree = compact_blocks(control_trees, v, ws, region, region_meta)
    control_trees[v] = ctree
end

function compact_blocks(control_trees, v, ws, region, region_meta=nothing)
    # For most regions, create a simple tree with children
    if !in(region, (REGION_BLOCK, REGION_FOR_LOOP, REGION_NATURAL_LOOP, REGION_WHILE_LOOP))
        return ControlTree(ControlNode(v, region, nothing), [control_trees[w] for w in [v; ws]])
    end

    # For loops and blocks, compact nested block regions
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

    # Preserve metadata for REGION_FOR_LOOP
    ControlTree(ControlNode(v, region, region_meta), nodes)
end

"All remaining region types are structured (unstructured types were removed)."
is_structured(::ControlTree) = true

function outermost_tree(ctree::ControlTree, v::Integer)
    for subtree in PreOrderDFS(ctree)
        node_index(subtree) == v && return subtree
    end
end
