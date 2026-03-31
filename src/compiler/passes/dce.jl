# Dead Code Elimination for Structured IR
#
# General-purpose DCE using dependency graph + BFS reachability.
# Matches Python cuTile's `dead_code_elimination_pass` (dce.py),
# adapted for our positional SSA-based structured IR.
#
# Algorithm:
# 1. Build a dependency graph: each value → list of values it depends on
# 2. Seed live set from side-effectful operations (stores, atomics, returns)
# 3. BFS backward through dependencies to find all live values
# 4. Prune: remove dead instructions and dead loop carries
#
# Handles cycles naturally: dead token carries that form
# body_arg → JoinTokens → ContinueOp → body_arg are never reachable
# from any root, so they remain dead.

using Core: SSAValue, Argument

#=============================================================================
 CF pseudo-nodes
=============================================================================#

# Each ControlFlowOp gets a unique sentinel key in the dependency graph,
# matching Python's "$cf.<N>" naming (dce.py lines 32-37).
struct CFNode
    id::UInt64
end

Base.hash(n::CFNode, h::UInt) = hash(n.id, hash(:CFNode, h))
Base.:(==)(a::CFNode, b::CFNode) = a.id == b.id

cf_node(op) = CFNode(objectid(op))

#=============================================================================
 Build dependency graph
=============================================================================#

"""
    is_trackable_value(x) -> Bool

Check if `x` is a trackable value in the dependency graph.
"""
is_trackable_value(@nospecialize(x)) = x isa SSAValue || x isa BlockArgument || x isa Argument

"""
    get_stmt_operands(s) -> Vector{Any}

Extract operand values from a statement (Expr, JoinTokensNode, TokenResultNode, etc.).
Only returns values that are trackable in the dependency graph.
"""
function get_stmt_operands(@nospecialize(s))
    result = Any[]
    if s isa Expr
        start = s.head === :invoke ? 3 : 2
        for i in start:length(s.args)
            is_trackable_value(s.args[i]) && push!(result, s.args[i])
        end
    elseif s isa JoinTokensNode
        for t in s.tokens
            is_trackable_value(t) && push!(result, t)
        end
    elseif s isa TokenResultNode
        push!(result, SSAValue(s.mem_op_ssa))
    end
    # MakeTokenNode has no operands
    return result
end

"""
    must_keep(s) -> Bool

Check if a statement is side-effectful and must be kept as a root.

Uses the Julia effects system: each cuTile intrinsic has an `efunc` override
that specifies `effect_free=ALWAYS_FALSE` for side-effectful operations
(stores, atomics, assert). Intrinsics without an efunc override are pure.
Unknown calls are conservatively kept.

Mirrors Python cuTile's `_must_keep` (dce.py:205-206) and Julia's compiler
`stmt_effect_free` — both classify by per-instruction effect annotations.
"""
function must_keep(block::Block, @nospecialize(s))
    # Token bookkeeping: no side effects
    s isa JoinTokensNode && return false
    s isa TokenResultNode && return false
    s isa MakeTokenNode && return false

    # ReturnNode: always keep
    s isa ReturnNode && return true

    call = resolve_call(block, s)
    if call !== nothing
        resolved_func, _ = call
        # cuTile intrinsics: use the efunc effects system
        if resolved_func isa Function && parentmodule(resolved_func) === Intrinsics
            # Query the efunc override for this intrinsic
            override = efunc(resolved_func, CC.Effects())
            if override !== nothing
                # Has custom effects — keep if not effect-free
                return override.effect_free !== CC.ALWAYS_TRUE
            end
            # No efunc override → pure intrinsic, safe to remove
            return false
        end
        # getfield is pure (both Core.getfield GlobalRef and bare getfield)
        if s isa Expr
            func = s.args[1]
            if func === getfield || (func isa GlobalRef && func.mod === Core && func.name === :getfield)
                return false
            end
        end
    end

    # Unresolvable calls, unknown functions, non-call Exprs: keep conservatively
    return true
end

"""
    _add_dep!(graph, key, dep)

Add a dependency edge: `key` depends on `dep`.
"""
function _add_dep!(graph::Dict{Any, Vector{Any}}, @nospecialize(key), @nospecialize(dep))
    deps = get!(Vector{Any}, graph, key)
    push!(deps, dep)
end

"""
    _build_dataflow_graph!(graph, roots, op_to_cf, block, loop_op, loop_cf, cf_parent)

Recursively build the dependency graph for a block and all nested blocks.
Mirrors Python's `_build_dataflow_graph` (dce.py lines 87-203).
"""
function _build_dataflow_graph!(graph::Dict{Any, Vector{Any}},
                                  roots::Set{Any},
                                  op_to_cf::Dict{UInt64, CFNode},
                                  block::Block,
                                  innermost_loop_op,
                                  innermost_loop_cf::Union{CFNode, Nothing},
                                  innermost_cf::Union{CFNode, Nothing})
    for inst in instructions(block)
        s = stmt(inst)
        val = SSAValue(inst.ssa_idx)

        if s isa ForOp
            cf = cf_node(s)
            op_to_cf[objectid(s)] = cf
            graph[cf] = Any[]

            # CF_COND: ForOp depends on its bounds
            is_trackable_value(s.lower) && _add_dep!(graph, cf, s.lower)
            is_trackable_value(s.upper) && _add_dep!(graph, cf, s.upper)
            is_trackable_value(s.step) && _add_dep!(graph, cf, s.step)

            # CF_NESTED
            innermost_cf !== nothing && _add_dep!(graph, cf, innermost_cf)

            # CF_DEFINED_VARS: body_args and result getfields depend on CF node
            for i in 1:length(s.init_values)
                ba = s.body.args[i]
                graph[ba] = Any[s.init_values[i], cf]
            end

            # Recurse into body
            _build_dataflow_graph!(graph, roots, op_to_cf, s.body,
                                    s, cf, cf)

            # For for-loops: result getfields also depend on init_values
            # (zero-iteration path: loop may not execute at all)
            _build_loop_result_deps!(graph, block, inst, s, cf)

        elseif s isa LoopOp
            cf = cf_node(s)
            op_to_cf[objectid(s)] = cf
            graph[cf] = Any[]

            # CF_NESTED
            innermost_cf !== nothing && _add_dep!(graph, cf, innermost_cf)

            # CF_DEFINED_VARS
            for i in 1:length(s.init_values)
                ba = s.body.args[i]
                graph[ba] = Any[s.init_values[i], cf]
            end

            _build_dataflow_graph!(graph, roots, op_to_cf, s.body,
                                    s, cf, cf)

            _build_loop_result_deps!(graph, block, inst, s, cf)

        elseif s isa WhileOp
            cf = cf_node(s)
            op_to_cf[objectid(s)] = cf
            graph[cf] = Any[]

            # CF_NESTED
            innermost_cf !== nothing && _add_dep!(graph, cf, innermost_cf)

            # CF_DEFINED_VARS: before.args are carries
            for i in 1:length(s.init_values)
                ba = s.before.args[i]
                graph[ba] = Any[s.init_values[i], cf]
                # Also set up after.args to depend on before.args
                if i <= length(s.after.args)
                    graph[s.after.args[i]] = Any[ba, cf]
                end
            end

            _build_dataflow_graph!(graph, roots, op_to_cf, s.before,
                                    s, cf, cf)
            _build_dataflow_graph!(graph, roots, op_to_cf, s.after,
                                    s, cf, cf)

            _build_loop_result_deps!(graph, block, inst, s, cf)

        elseif s isa IfOp
            cf = cf_node(s)
            op_to_cf[objectid(s)] = cf

            # CF_COND
            deps = Any[]
            is_trackable_value(s.condition) && push!(deps, s.condition)
            innermost_cf !== nothing && push!(deps, innermost_cf)
            graph[cf] = deps

            _build_dataflow_graph!(graph, roots, op_to_cf, s.then_region,
                                    innermost_loop_op, innermost_loop_cf, cf)
            _build_dataflow_graph!(graph, roots, op_to_cf, s.else_region,
                                    innermost_loop_op, innermost_loop_cf, cf)

            # IfOp result getfields depend on CF node
            _build_if_result_deps!(graph, block, inst, s, cf)

        else
            # Regular instruction — skip if already handled by a CF result dep builder
            if !haskey(graph, val)
                operands = get_stmt_operands(s)
                deps = copy(operands)
                innermost_cf !== nothing && push!(deps, innermost_cf)
                graph[val] = deps
            end

            if must_keep(block, s)
                operands = get_stmt_operands(s)
                push!(roots, val)
                for op in operands
                    push!(roots, op)
                end
            end
        end
    end

    # Handle terminator
    term = terminator(block)
    _build_terminator_deps!(graph, roots, term, block,
                             innermost_loop_op, innermost_loop_cf, innermost_cf)
end

"""
    _build_terminator_deps!(graph, roots, term, ...)

Add dependency edges for terminators (ContinueOp, BreakOp, YieldOp, ConditionOp).
"""
function _build_terminator_deps!(graph, roots, term, block,
                                   innermost_loop_op, innermost_loop_cf, innermost_cf)
    term === nothing && return

    if term isa ContinueOp && innermost_loop_op !== nothing
        # CF_BREAK_CONTINUE: loop depends on innermost CF containing this Continue
        innermost_cf !== nothing && _add_dep!(graph, innermost_loop_cf, innermost_cf)

        # Continue values feed into body_args of next iteration
        body = if innermost_loop_op isa WhileOp
            innermost_loop_op.before
        else
            innermost_loop_op.body
        end
        n_carries = length(innermost_loop_op.init_values)
        for i in 1:min(n_carries, length(operands(term)))
            _add_dep!(graph, body.args[i], operands(term)[i])
        end

        # For for-loops, continue values also flow to results (may exit immediately)
        if innermost_loop_op isa ForOp
            _add_loop_continue_to_results!(graph, innermost_loop_op, term)
        end

    elseif term isa BreakOp && innermost_loop_op !== nothing
        # CF_BREAK_CONTINUE
        innermost_cf !== nothing && _add_dep!(graph, innermost_loop_cf, innermost_cf)

        # Break values flow to loop result getfields
        # These edges are added during _build_loop_result_deps! already
        # but we also need to add the break values as deps of the result getfield vars
        # This is handled by _build_loop_result_deps! scanning terminators

    elseif term isa ConditionOp && innermost_loop_op !== nothing
        # ConditionOp's condition is an operand of the WhileOp's CF node
        if innermost_loop_cf !== nothing && is_trackable_value(term.condition)
            _add_dep!(graph, innermost_loop_cf, term.condition)
        end
        # ConditionOp args carry values back to the loop (like ContinueOp)
        body = innermost_loop_op isa WhileOp ? innermost_loop_op.before : innermost_loop_op.body
        n_carries = length(innermost_loop_op.init_values)
        for i in 1:min(n_carries, length(operands(term)))
            _add_dep!(graph, body.args[i], operands(term)[i])
        end

    elseif term isa YieldOp
        # YieldOp in a WhileOp `after` block carries values back to the loop
        # (same semantics as ContinueOp). For IfOp branches, edges are handled
        # by _build_if_result_deps! scanning then/else YieldOps.
        if innermost_loop_op isa WhileOp
            body = innermost_loop_op.before
            n_carries = length(innermost_loop_op.init_values)
            for i in 1:min(n_carries, length(operands(term)))
                v = operands(term)[i]
                is_trackable_value(v) && _add_dep!(graph, body.args[i], v)
            end
        end

    elseif term isa ReturnNode
        if isdefined(term, :val) && is_trackable_value(term.val)
            push!(roots, term.val)
        end
    end
end

"""
    _build_loop_result_deps!(graph, parent_block, loop_inst, op, cf)

Add dependency edges for a loop's result extractions (getfield calls in parent).
"""
function _build_loop_result_deps!(graph, parent_block::Block, loop_inst::Instruction,
                                    op, cf::CFNode)
    loop_ssa = SSAValue(loop_inst.ssa_idx)
    n_carries = length(op.init_values)

    for inst in instructions(parent_block)
        is_getfield_of(stmt(inst), loop_ssa) || continue
        field_idx = stmt(inst).args[3]
        field_idx isa Int || continue
        gf_val = SSAValue(inst.ssa_idx)

        deps = Any[cf]
        # Result depends on init_value (for-loop zero-iteration path)
        if op isa ForOp && field_idx <= n_carries
            push!(deps, op.init_values[field_idx])
        end
        # Result depends on all terminator values at this position
        body = op isa WhileOp ? op.before : op.body
        for term in reachable_terminators(body)
            ops = operands(term)
            if field_idx <= length(ops) && is_trackable_value(ops[field_idx])
                push!(deps, ops[field_idx])
            end
        end
        # For WhileOp, also check after block terminators
        if op isa WhileOp
            for term in reachable_terminators(op.after)
                ops = operands(term)
                if field_idx <= length(ops) && is_trackable_value(ops[field_idx])
                    push!(deps, ops[field_idx])
                end
            end
        end
        graph[gf_val] = deps
    end
end

"""
Add continue-to-result edges for ForOp (zero-iteration path).
"""
function _add_loop_continue_to_results!(graph, loop_op::ForOp, term::ContinueOp)
    # We need to find the getfield extractions in the parent block.
    # These edges are already created in _build_loop_result_deps! by scanning
    # reachable terminators, so we just need the term_values → result deps.
    # _build_loop_result_deps! handles this by scanning all reachable terminators.
    nothing
end

"""
    _build_if_result_deps!(graph, parent_block, if_inst, op, cf)

Add dependency edges for IfOp result extractions.
"""
function _build_if_result_deps!(graph, parent_block::Block, if_inst::Instruction,
                                  op::IfOp, cf::CFNode)
    if_ssa = SSAValue(if_inst.ssa_idx)

    for inst in instructions(parent_block)
        is_getfield_of(stmt(inst), if_ssa) || continue
        field_idx = stmt(inst).args[3]
        field_idx isa Int || continue
        gf_val = SSAValue(inst.ssa_idx)

        deps = Any[cf]
        # Result depends on YieldOp values from both branches
        then_term = terminator(op.then_region)
        if then_term isa YieldOp && field_idx <= length(operands(then_term))
            v = operands(then_term)[field_idx]
            is_trackable_value(v) && push!(deps, v)
        end
        else_term = terminator(op.else_region)
        if else_term isa YieldOp && field_idx <= length(operands(else_term))
            v = operands(else_term)[field_idx]
            is_trackable_value(v) && push!(deps, v)
        end
        graph[gf_val] = deps
    end
end

"""
    is_getfield_of(s, ref::SSAValue) -> Bool

Check if `s` is a `getfield(ref, idx::Int)` expression.
Handles both `Core.getfield` (GlobalRef) and bare `getfield` (resolved function).
"""
function is_getfield_of(@nospecialize(s), ref::SSAValue)
    s isa Expr || return false
    s.head === :call || return false
    length(s.args) >= 3 || return false
    func = s.args[1]
    is_gf = if func isa GlobalRef
        func.mod === Core && func.name === :getfield
    else
        func === getfield
    end
    is_gf || return false
    s.args[2] == ref || return false
    s.args[3] isa Int || return false
    return true
end

#=============================================================================
 BFS liveness propagation
=============================================================================#

function _find_live_values!(graph::Dict{Any, Vector{Any}}, live::Set{Any})
    worklist = collect(live)
    while !isempty(worklist)
        val = pop!(worklist)
        for dep in get(graph, val, Any[])
            if dep ∉ live
                push!(live, dep)
                push!(worklist, dep)
            end
        end
    end
end

#=============================================================================
 Pruning
=============================================================================#

"""
    _prune_block!(block, live, op_to_cf, yield_mask)

Remove dead instructions and filter dead loop carries/yield values.
`yield_mask` is the keep-mask for the enclosing IfOp's yield values (or nothing).
Loop carry pruning is handled by `filter!(carries(op))` in `_prune_loop!`.
"""
function _prune_block!(block::Block, live::Set{Any}, op_to_cf::Dict{UInt64, CFNode},
                         yield_mask)
    changed = false
    to_delete = Instruction[]

    for inst in instructions(block)
        s = stmt(inst)
        val = SSAValue(inst.ssa_idx)

        if s isa ForOp || s isa LoopOp || s isa WhileOp
            cf = get(op_to_cf, objectid(s), nothing)
            if cf !== nothing && cf ∉ live
                # Entire loop is dead
                push!(to_delete, inst)
                changed = true
            else
                # Loop is live — prune its carries and recurse
                changed |= _prune_loop!(block, inst, s, live, op_to_cf)
            end

        elseif s isa IfOp
            cf = get(op_to_cf, objectid(s), nothing)
            if cf !== nothing && cf ∉ live
                push!(to_delete, inst)
                changed = true
            else
                changed |= _prune_if!(block, inst, s, live, op_to_cf)
            end

        else
            # Regular instruction: dead if not live and not must-keep
            if val ∉ live && !must_keep(block, s)
                push!(to_delete, inst)
                changed = true
            end
        end
    end

    for inst in to_delete
        delete!(block, inst)
    end

    # Prune IfOp yield values only (loop terminators handled by filter!(carries))
    changed |= _prune_terminator!(block, live, yield_mask)

    return changed
end

"""
    _prune_loop!(parent_block, inst, op, live, op_to_cf) -> Bool

Filter dead carries from a loop and recurse into its body.
"""
function _prune_loop!(parent_block::Block, inst::Instruction,
                        op::Union{ForOp, LoopOp, WhileOp},
                        live::Set{Any}, op_to_cf::Dict{UInt64, CFNode})
    changed = false
    n_carries = length(op.init_values)
    body = op isa WhileOp ? op.before : op.body

    # Build carry keep mask: keep if body_arg is live OR any result getfield is live
    carry_live = BitVector(false for _ in 1:n_carries)
    for i in 1:n_carries
        carry_live[i] = body.args[i] ∈ live
    end
    # Also check result getfield liveness
    loop_ssa = SSAValue(inst.ssa_idx)
    for pinst in instructions(parent_block)
        is_getfield_of(stmt(pinst), loop_ssa) || continue
        field_idx = stmt(pinst).args[3]
        field_idx isa Int || continue
        if field_idx <= n_carries && SSAValue(pinst.ssa_idx) ∈ live
            carry_live[field_idx] = true
        end
    end

    if !all(carry_live)
        lc = carries(op)
        old_to_new = filter!(lc) do cr
            carry_live[cr.index]
        end

        # Renumber getfield extractions in parent
        _renumber_getfields!(parent_block, loop_ssa, old_to_new)

        # Update loop result type
        _update_cf_result_type!(parent_block, inst, body)

        changed = true
    end

    # Recurse into body with the carry mask
    if op isa WhileOp
        changed |= _prune_block!(op.before, live, op_to_cf, nothing)
        changed |= _prune_block!(op.after, live, op_to_cf, nothing)
    else
        changed |= _prune_block!(op.body, live, op_to_cf, nothing)
    end

    return changed
end

"""
    _prune_if!(parent_block, inst, op, live, op_to_cf, parent_carry_mask) -> Bool

Filter dead results from an IfOp and recurse into its branches.
"""
function _prune_if!(parent_block::Block, inst::Instruction, op::IfOp,
                      live::Set{Any}, op_to_cf::Dict{UInt64, CFNode})
    changed = false

    # Determine which IfOp results are live
    if_ssa = SSAValue(inst.ssa_idx)
    result_type = value_type(inst)
    n_results = if result_type === Nothing
        0
    elseif result_type <: Tuple
        length(result_type.parameters)
    else
        1
    end

    if n_results > 0
        result_live = BitVector(false for _ in 1:n_results)
        for pinst in instructions(parent_block)
            is_getfield_of(stmt(pinst), if_ssa) || continue
            field_idx = stmt(pinst).args[3]
            field_idx isa Int || continue
            if field_idx <= n_results && SSAValue(pinst.ssa_idx) ∈ live
                result_live[field_idx] = true
            end
        end

        if !all(result_live)
            # Build old→new mapping and remove dead yield values + getfields
            old_to_new = Dict{Int, Int}()
            new_idx = 0
            for i in 1:n_results
                if result_live[i]
                    new_idx += 1
                    old_to_new[i] = new_idx
                end
            end

            _renumber_getfields!(parent_block, if_ssa, old_to_new)

            # Update IfOp result type
            kept_types = Type[]
            if result_type <: Tuple
                for i in 1:n_results
                    result_live[i] && push!(kept_types, result_type.parameters[i])
                end
            elseif result_live[1]
                push!(kept_types, result_type)
            end
            new_type = isempty(kept_types) ? Nothing : Tuple{kept_types...}
            update_type!(parent_block, inst, new_type)

            yield_mask = result_live
            changed = true
        else
            yield_mask = nothing
        end
    else
        yield_mask = nothing
    end

    # Recurse into branches with the yield mask
    changed |= _prune_block!(op.then_region, live, op_to_cf, yield_mask)
    changed |= _prune_block!(op.else_region, live, op_to_cf, yield_mask)

    return changed
end

"""
    _prune_terminator!(block, live, yield_mask) -> Bool

Filter dead values from IfOp YieldOp terminators only. Loop terminators
(ContinueOp, BreakOp, ConditionOp) are handled by `filter!(carries(op))`
in `_prune_loop!` and must NOT be modified here to avoid double-removal.
"""
function _prune_terminator!(block::Block, live::Set{Any}, yield_mask)
    term = terminator(block)
    term === nothing && return false

    if term isa YieldOp && yield_mask !== nothing
        ops = operands(term)
        n = min(length(ops), length(yield_mask))
        changed = false
        for i in n:-1:1
            if !yield_mask[i]
                deleteat!(ops, i)
                changed = true
            end
        end
        return changed
    end

    return false
end

#=============================================================================
 Getfield renumbering
=============================================================================#

"""
    _renumber_getfields!(block, cf_ssa, old_to_new)

Update or remove getfield extractions for a CF op after carry/result removal.
"""
function _renumber_getfields!(block::Block, cf_ssa::SSAValue, old_to_new::Dict{Int, Int})
    to_delete = Instruction[]
    for inst in instructions(block)
        is_getfield_of(stmt(inst), cf_ssa) || continue
        field_idx = stmt(inst).args[3]::Int
        if haskey(old_to_new, field_idx)
            stmt(inst).args[3] = old_to_new[field_idx]
        else
            push!(to_delete, inst)
        end
    end
    for inst in to_delete
        delete!(block, inst)
    end
end

"""
    _update_cf_result_type!(block, inst, body_block)

Recompute a CF op's result type from its remaining body block args.
"""
function _update_cf_result_type!(block::Block, inst::Instruction, body_block::Block)
    types = Type[is_token_type(arg.type) ? TokenType : arg.type for arg in body_block.args]
    new_type = isempty(types) ? Nothing : Tuple{types...}
    update_type!(block, inst, new_type)
end

#=============================================================================
 Top-level API
=============================================================================#

"""
    dce_pass!(sci::StructuredIRCode)

Dead code elimination for structured IR. Removes dead instructions, dead loop
carries, and dead IfOp results using dependency graph reachability analysis.
"""
function dce_pass!(sci::StructuredIRCode)
    # 1. Build dependency graph
    graph = Dict{Any, Vector{Any}}()
    roots = Set{Any}()
    op_to_cf = Dict{UInt64, CFNode}()
    _build_dataflow_graph!(graph, roots, op_to_cf, sci.entry, nothing, nothing, nothing)

    # 2. BFS from roots to find all live values
    live = copy(roots)
    _find_live_values!(graph, live)

    # 3. Prune dead code
    _prune_block!(sci.entry, live, op_to_cf, nothing)
end
