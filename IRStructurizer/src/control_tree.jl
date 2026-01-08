# control tree construction via CFG pattern matching

using MLStyle: @active, @match
using Graphs
using Graphs: AbstractGraph, SimpleDiGraph, Edge, vertices, edges, nv, ne,
              inneighbors, outneighbors, strongly_connected_components
using AbstractTrees: PreOrderDFS, PostOrderDFS

"""
    RegionType

Single-entry control-flow region classification for pattern-based structurization.

Acyclic regions:
- `REGION_BLOCK`: Linear sequence of blocks `u` ─→ [`v`, `vs...`] ─→ `w`
- `REGION_IF_THEN`: Conditional with one branch `u` ─→ `v` and one merge block reachable by `u` ─→ `w` or `v` ─→ `w`
- `REGION_IF_THEN_ELSE`: Conditional with two symmetric branches `u` ─→ `v` and `u` ─→ `w` and a single merge block reachable by `v` ─→ x or `w` ─→ x
- `REGION_SWITCH`: Conditional with any number of branches [`u` ─→ `vᵢ`, `u` ─→ `vᵢ₊₁`, ...] and a single merge block reachable by [`vᵢ` ─→ `w`, `vᵢ₊₁` ─→ `w`, ...]
- `REGION_TERMINATION`: Acyclic region which contains a block `v` with multiple branches, including one or multiple branches to blocks `wᵢ` which end with a function termination instruction. The region is composed of `v` and all the `wᵢ`.

Cyclic regions:
- `REGION_WHILE_LOOP`: Simple cyclic region made of a condition block `u`, a loop body block `v` and a merge block `w` such that `v` ⇆ `u` ─→ `w`
- `REGION_FOR_LOOP`: While loop with detected counting pattern
- `REGION_NATURAL_LOOP`: Single-entry cyclic region with varying complexity such that the entry point dominates all nodes in the cyclic structure.
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

function acyclic_region(g, v, ec, doms, domtree, backedges)
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

    # Use Julia's DomTree directly - children are block indices
    dom_children = domtree.nodes[v].children
    pdom_indices = findall(dom_children) do w
        in(w, vertices(g)) && !in(w, outneighbors(g, v))
    end
    length(pdom_indices) ≥ 1 || return nothing
    vs = Int64[]
    ws = Int64[]
    for i in pdom_indices
        w = dom_children[i]
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

#=
Julia's iteration protocol for ranges (after type inference) produces:
- Phi node with entry value (start) and carried value (next iteration)
- Equality check `===(i, stop)` for termination (canonical pattern)
- Increment via `add_int(i, step)`

The detection below uses a unified approach with pluggable condition patterns
to handle both Julia's canonical `for i in 1:n` lowering and explicit `while i < n`.
=#

"""
    PhiAnalysisResult

Result of analyzing a phi node in a loop header.
"""
struct PhiAnalysisResult
    phi_idx::Int        # SSA index of the phi node
    entry_val::Any      # Value from loop entry edge
    carried_val::Any    # Value from back-edge (loop body)
end

"""
    analyze_header_phis(header_idx, ir) -> Dict{Int, PhiAnalysisResult}

Analyze phi nodes in a loop header block, extracting entry and carried values.

In IRCode, PhiNode edges are BLOCK indices (not statement indices).
- Entry edges: come from blocks before the header block index
- Carried edges: come from blocks at or after the header block index (back-edge)
"""
function analyze_header_phis(header_idx::Int, ir::IRCode)
    phi_info = Dict{Int, PhiAnalysisResult}()
    stmts = ir.stmts.stmt
    bb = ir.cfg.blocks[header_idx]

    for si in first(bb.stmts):last(bb.stmts)
        stmt = stmts[si]
        stmt isa PhiNode || continue

        entry_val = nothing
        carried_val = nothing

        for (edge_idx, edge) in enumerate(stmt.edges)
            isassigned(stmt.values, edge_idx) || continue
            val = stmt.values[edge_idx]

            # In IRCode, edge is a block index
            # Carried values come from blocks >= header (back-edges)
            # Entry values come from blocks < header
            if edge >= header_idx
                carried_val = val
            else
                entry_val = val
            end
        end

        if entry_val !== nothing && carried_val !== nothing
            phi_info[si] = PhiAnalysisResult(si, entry_val, carried_val)
        end
    end

    return phi_info
end

"""
    extract_loop_condition(header_idx, ir) -> (SSAValue, Expr) | nothing

Extract the loop condition from a GotoIfNot in the header block.
"""
function extract_loop_condition(header_idx::Int, ir::IRCode)
    stmts = ir.stmts.stmt
    bb = ir.cfg.blocks[header_idx]
    for si in first(bb.stmts):last(bb.stmts)
        stmt = stmts[si]
        if stmt isa GotoIfNot && stmt.cond isa SSAValue
            cond_stmt = stmts[stmt.cond.id]
            if cond_stmt isa Expr && cond_stmt.head === :call
                return (stmt.cond, cond_stmt)
            end
        end
    end
    return nothing
end

"""
    is_loop_invariant(value, header_idx, ir) -> Bool

Check if a value is loop-invariant (defined before the loop header).
"""
function is_loop_invariant(value, header_idx::Int, ir::IRCode)
    if value isa SSAValue
        bb = ir.cfg.blocks[header_idx]
        return value.id < first(bb.stmts)
    end
    # Arguments, constants, etc. are loop-invariant
    return true
end

#=============================================================================
 Condition Pattern Matching
=============================================================================#

"""
    ConditionMatch

Result of matching a loop condition pattern.
"""
struct ConditionMatch
    iv_phi_idx::Int     # SSA index of induction variable phi
    bound::Any          # Loop bound
    is_inclusive::Bool  # Whether bound is inclusive (true for `===` pattern)
end

"""
Abstract type for loop condition patterns.
Implementations define how to recognize specific loop condition forms.
"""
abstract type ConditionPattern end

"""
    match_condition(pattern, cond_expr, phi_info) -> ConditionMatch | nothing

Try to match a condition expression against a pattern.
"""
function match_condition end

"""
Matches Julia's iteration protocol equality check: `===(iv, bound)`
This is the canonical pattern from `for i in 1:n` after type inference.
"""
struct EqualityPattern <: ConditionPattern end

function match_condition(::EqualityPattern, cond_expr::Expr, phi_info::Dict)
    length(cond_expr.args) >= 3 || return nothing

    func = cond_expr.args[1]
    is_equality = (func isa GlobalRef && func.name === :(===)) || func === :(===)
    is_equality || return nothing

    iv_candidate = cond_expr.args[2]
    bound = cond_expr.args[3]

    iv_candidate isa SSAValue || return nothing
    haskey(phi_info, iv_candidate.id) || return nothing

    return ConditionMatch(iv_candidate.id, bound, true)  # inclusive bound
end

"""
Matches while-style less-than: `slt_int(iv, bound)` or `ult_int(iv, bound)`
From explicit `while i < n` code.
"""
struct LessThanPattern <: ConditionPattern end

function match_condition(::LessThanPattern, cond_expr::Expr, phi_info::Dict)
    length(cond_expr.args) >= 3 || return nothing

    func = cond_expr.args[1]
    func isa GlobalRef || return nothing
    func.name in (:slt_int, :ult_int) || return nothing

    iv_candidate = cond_expr.args[2]
    bound = cond_expr.args[3]

    iv_candidate isa SSAValue || return nothing
    haskey(phi_info, iv_candidate.id) || return nothing

    return ConditionMatch(iv_candidate.id, bound, false)  # exclusive bound
end

"""
Matches while-style less-equal: `sle_int(iv, bound)`
From explicit `while i <= n` code.
"""
struct LessEqualPattern <: ConditionPattern end

function match_condition(::LessEqualPattern, cond_expr::Expr, phi_info::Dict)
    length(cond_expr.args) >= 3 || return nothing

    func = cond_expr.args[1]
    func isa GlobalRef || return nothing
    func.name === :sle_int || return nothing

    iv_candidate = cond_expr.args[2]
    bound = cond_expr.args[3]

    iv_candidate isa SSAValue || return nothing
    haskey(phi_info, iv_candidate.id) || return nothing

    return ConditionMatch(iv_candidate.id, bound, true)  # inclusive bound
end

# Default patterns to try, in order
const DEFAULT_CONDITION_PATTERNS = (
    EqualityPattern(),      # === (Julia's for i in 1:n)
    LessThanPattern(),      # slt_int/ult_int (while i < n)
    LessEqualPattern(),     # sle_int (while i <= n)
)

#=============================================================================
 Unified For-Loop Detection
=============================================================================#

"""
    find_step_from_carried_value(stmts, carried_val, iv_candidate, depth=0) -> Any | nothing

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

"""
    try_detect_for_loop(header_idx, ir; patterns=DEFAULT_CONDITION_PATTERNS)

Unified for-loop detection using pluggable condition patterns.

Analyzes the loop header to detect counting for-loop patterns:
1. Extract phi node information (entry/carried values)
2. Extract loop condition expression
3. Try each condition pattern until one matches
4. Verify step expression (add_int pattern)
5. Check loop-invariance of bounds

Returns ForLoopInfo if a for-loop pattern is detected, nothing otherwise.

NOTE: Assumes SESE structure (single GotoIfNot in header). For non-SESE loops
with multiple GotoIfNots, `extract_loop_condition` returns the first one found,
which may be an internal branch rather than the loop exit condition.
"""
function try_detect_for_loop(header_idx::Int, ir::IRCode;
                             patterns=DEFAULT_CONDITION_PATTERNS)
    stmts = ir.stmts.stmt

    # Bounds check
    (1 <= header_idx <= length(ir.cfg.blocks)) || return nothing

    # Step 1: Analyze phi nodes
    phi_info = analyze_header_phis(header_idx, ir)
    isempty(phi_info) && return nothing

    # Step 2: Extract condition
    cond_result = extract_loop_condition(header_idx, ir)
    cond_result === nothing && return nothing
    (_, cond_expr) = cond_result

    # Step 3: Try condition patterns
    match = nothing
    for pattern in patterns
        match = match_condition(pattern, cond_expr, phi_info)
        match !== nothing && break
    end
    match === nothing && return nothing

    # Step 4: Find step expression
    phi = phi_info[match.iv_phi_idx]
    step = find_step_from_carried_value(stmts, phi.carried_val, SSAValue(match.iv_phi_idx))
    step === nothing && return nothing

    # Step 5: Check loop-invariance of bound
    is_loop_invariant(match.bound, header_idx, ir) || return nothing

    # Success: construct ForLoopInfo
    return ForLoopInfo(match.iv_phi_idx, phi.entry_val, match.bound, step, match.is_inclusive)
end

#=============================================================================
 Cyclic Region Detection
=============================================================================#

function cyclic_region!(sccs, g, v, ec, doms, domtree, backedges, ir)
    # Try to match while-loop pattern first
    matched = @match (g, v) begin
        while_loop(cond, body, merge) => begin
            # Try unified for-loop detection
            for_info = try_detect_for_loop(cond, ir)
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
        # Natural loop - keep as LoopOp (don't try to promote to ForOp)
        # For-loop detection only works on SESE while-loops.
        # Complex loops with internal control flow stay as LoopOp.
        if all(==(v) ∘ (e -> e.dst), entry_edges)
            return (REGION_NATURAL_LOOP, cycle, nothing)
        end
    end

    # Irreducible control flow - not supported
    return nothing
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

"""
    build_cfg_graph(ir::IRCode) -> SimpleDiGraph

Build a SimpleDiGraph from IRCode's CFG for use in control tree construction.
"""
function build_cfg_graph(ir::IRCode)
    nblocks = length(ir.cfg.blocks)
    g = SimpleDiGraph(nblocks)
    for (i, bb) in enumerate(ir.cfg.blocks)
        for succ in bb.succs
            add_edge!(g, i, succ)
        end
    end
    g
end

function ControlTree(ir::IRCode)
    # Build SimpleDiGraph from IRCode's CFG
    cfg = build_cfg_graph(ir)
    T = eltype(cfg)

    # Use Julia's dominator tree directly
    domtree = construct_domtree(ir)

    abstract_graph = DeltaGraph(cfg)

    # Build edge classification for retreating edges
    ec = EdgeClassification(cfg)

    # Build backedges using dominance
    bedges = backedges(cfg, domtree)

    # For acyclic_region compatibility - build doms dict from domtree
    doms = Dict{Int, Set{Int}}()
    for bb in 1:length(ir.cfg.blocks)
        doms[bb] = Set{Int}([bb])
        idom = bb
        while true
            idom == 1 && break
            idom = domtree.idoms_bb[idom]
            idom == 0 && break
            push!(doms[bb], idom)
        end
    end

    sccs = strongly_connected_components(cfg)

    control_trees = Dict{T,ControlTree}(v => ControlTree(ControlNode(v, REGION_BLOCK)) for v in vertices(cfg))
    # Use reverse post-order traversal
    next = reverse(traverse(cfg))

    while !isempty(next)
        start = popfirst!(next)
        haskey(control_trees, start) || continue
        # `v` can change if we follow a chain of blocks in a block region which starts before `v`, or if we need to find a dominator to an improper region.
        v = start
        ret = acyclic_region(abstract_graph, v, ec, doms, domtree, bedges)
        region_meta = nothing
        ret = if !isnothing(ret)
            (region_type, (v, ws...)) = ret
            (region_type, ws)
        else
            cyclic_ret = cyclic_region!(sccs, abstract_graph, v, ec, doms, domtree, bedges, ir)
            if !isnothing(cyclic_ret)
                (region_type, (v, ws...), region_meta) = cyclic_ret
                (region_type, ws)
            else
                nothing
            end
        end
        isnothing(ret) && continue
        if region_type == REGION_TERMINATION
            cyclic = cyclic_region!(sccs, abstract_graph, v, ec, doms, domtree, bedges, ir)
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
