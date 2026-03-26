# Declarative IR Rewrite Pattern Framework
#
# Inspired by MLIR's PDLL: patterns are expressed as IR-like expressions
# that compile into generic pattern/rewrite node trees. The framework handles
# matching (recursive SSA def-chain walking), use counting, and
# legality-checked rewrite application.
#
# Usage:
#   rules = RewriteRule[
#       @rewrite addf(one_use(mulf(~x, ~y)), ~z) => fma(~x, ~y, ~z)
#       @rewrite subf(one_use(mulf(~x, ~y)), ~z) => fma(~x, ~y, negf(~z))
#   ]
#   rewrite_patterns!(sci, rules)

using Core: SSAValue, Argument

#=============================================================================
 Pattern Nodes (the "pattern IR" — describes what to match)
=============================================================================#

abstract type PatternNode end

"""Match a call to `Intrinsics.<func_name>` with operand sub-patterns."""
struct PCall <: PatternNode
    func_name::Symbol             # e.g., :mulf → resolved lazily via Intrinsics
    operands::Vector{PatternNode}
end

"""Capture the matched value into a named binding."""
struct PBind <: PatternNode
    name::Symbol
end

"""Require the matched SSA value has exactly one use, then match inner pattern."""
struct POneUse <: PatternNode
    inner::PatternNode
end

#=============================================================================
 Rewrite Nodes (the "replacement IR" — describes what to produce)
=============================================================================#

abstract type RewriteNode end

"""Emit a new call to `Intrinsics.<func_name>` with operand sub-templates."""
struct RCall <: RewriteNode
    func_name::Symbol
    operands::Vector{RewriteNode}
end

"""Reference a value captured during matching."""
struct RBind <: RewriteNode
    name::Symbol
end

#=============================================================================
 Rewrite Rule
=============================================================================#

struct RewriteRule
    lhs::PCall         # root pattern (always a PCall)
    rhs::RewriteNode   # replacement template
end

"""Resolve the function object for a PCall's func_name."""
resolve_func(pat::PCall) = getfield(Intrinsics, pat.func_name)

"""Get the root function that this rule matches on (for dispatch)."""
root_func(rule::RewriteRule) = resolve_func(rule.lhs)

#=============================================================================
 @rewrite Macro
=============================================================================#

"""
    @rewrite lhs => rhs

Compile a declarative rewrite rule into a `RewriteRule`.

LHS syntax:
- `func(args...)` — match a call to `Intrinsics.func`
- `~x` — bind matched value to name `x`
- `one_use(pat)` — require single use, then match `pat`

RHS syntax:
- `func(args...)` — emit a call to `Intrinsics.func`
- `~x` — reference a value bound in the LHS
"""
macro rewrite(ex)
    if ex isa Expr && ex.head === :call && ex.args[1] === :(=>)
        lhs_expr = ex.args[2]
        rhs_expr = ex.args[3]
    else
        error("@rewrite expects: lhs => rhs")
    end
    lhs = _compile_lhs(lhs_expr)
    rhs = _compile_rhs(rhs_expr)
    return esc(:(RewriteRule($lhs, $rhs)))
end

function _compile_lhs(ex)
    if ex isa Expr && ex.head === :call
        func_name = ex.args[1]
        if func_name === :~ && length(ex.args) == 2
            # ~x → PBind(:x)
            return :(PBind($(QuoteNode(ex.args[2]))))
        elseif func_name === :one_use && length(ex.args) == 2
            inner = _compile_lhs(ex.args[2])
            return :(POneUse($inner))
        else
            # func(args...) → PCall(:func, [...])
            operands = [_compile_lhs(arg) for arg in ex.args[2:end]]
            return :(PCall($(QuoteNode(func_name)), PatternNode[$(operands...)]))
        end
    elseif ex isa Symbol
        error("@rewrite LHS: bare symbol `$ex` not allowed; use `~$ex` to bind")
    else
        error("@rewrite LHS: unexpected expression: $ex")
    end
end

function _compile_rhs(ex)
    if ex isa Expr && ex.head === :call
        func_name = ex.args[1]
        if func_name === :~ && length(ex.args) == 2
            return :(RBind($(QuoteNode(ex.args[2]))))
        else
            operands = [_compile_rhs(arg) for arg in ex.args[2:end]]
            return :(RCall($(QuoteNode(func_name)), RewriteNode[$(operands...)]))
        end
    elseif ex isa Symbol
        error("@rewrite RHS: bare symbol `$ex` not allowed; use `~$ex` to reference a binding")
    else
        error("@rewrite RHS: unexpected expression: $ex")
    end
end

#=============================================================================
 Match Result
=============================================================================#

struct MatchResult
    bindings::Dict{Symbol, Any}   # name → SSAValue/Argument/literal
    matched_ssas::Vector{Int}     # SSA indices consumed by the match (for dead code)
end

#=============================================================================
 Structured IR Navigation
=============================================================================#

"""Return nested blocks of a control-flow op for recursive traversal."""
_nested_blocks(op::IfOp) = (op.then_region, op.else_region)
_nested_blocks(op::ForOp) = (op.body,)
_nested_blocks(op::WhileOp) = (op.before, op.after)
_nested_blocks(op::LoopOp) = (op.body,)
_nested_blocks(::ControlFlowOp) = ()

#=============================================================================
 Matching Engine
=============================================================================#

struct MatchContext
    defs::Dict{Int, Tuple{Any, Vector{Any}}}   # ssa_idx => (func, operands)
    types::Dict{Int, Any}                       # ssa_idx => Julia type (for shape checks)
    use_counts::Dict{Int, Int}
end

function pattern_match(ctx::MatchContext, @nospecialize(val), pat::PCall)
    val isa SSAValue || return nothing
    def = get(ctx.defs, val.id, nothing)
    def === nothing && return nothing
    func, operands = def

    target = resolve_func(pat)

    # Direct match
    if func === target && length(operands) == length(pat.operands)
        result = MatchResult(Dict{Symbol, Any}(), Int[val.id])
        for (op, sub_pat) in zip(operands, pat.operands)
            sub = pattern_match(ctx, op, sub_pat)
            sub === nothing && return nothing
            merge!(result.bindings, sub.bindings)
            append!(result.matched_ssas, sub.matched_ssas)
        end
        return result
    end

    # Transparent ops: to_scalar, from_scalar are always no-ops at codegen.
    # broadcast is a no-op only when input/output shapes match.
    # Trace through single-use transparent ops to find the underlying operation.
    if _is_transparent(func) && !isempty(operands)
        get(ctx.use_counts, val.id, 0) == 1 || return nothing
        # For broadcast: verify it's a no-op (shapes match) before tracing through
        if func === Intrinsics.broadcast
            inner_val = operands[1]
            if inner_val isa SSAValue
                inner_type = get(ctx.types, inner_val.id, nothing)
                outer_type = get(ctx.types, val.id, nothing)
                if inner_type !== nothing && outer_type !== nothing
                    inner_type <: Tile && outer_type <: Tile || return nothing
                    size(inner_type) == size(outer_type) || return nothing
                end
            end
        end
        result = pattern_match(ctx, operands[1], pat)
        result === nothing && return nothing
        push!(result.matched_ssas, val.id)
        return result
    end

    return nothing
end

# to_scalar/from_scalar/broadcast are codegen no-ops (just CGVal type reinterpretation).
# Pattern matching traces through them transparently.
_is_transparent(func) = (func === Intrinsics.to_scalar ||
                          func === Intrinsics.from_scalar ||
                          func === Intrinsics.broadcast)

function pattern_match(ctx::MatchContext, @nospecialize(val), pat::PBind)
    return MatchResult(Dict{Symbol, Any}(pat.name => val), Int[])
end

function pattern_match(ctx::MatchContext, @nospecialize(val), pat::POneUse)
    val isa SSAValue || return nothing
    get(ctx.use_counts, val.id, 0) == 1 || return nothing
    return pattern_match(ctx, val, pat.inner)
end

#=============================================================================
 Rewrite Materialization
=============================================================================#

"""
    materialize_rhs(sci, rhs, bindings) -> Vector{Tuple{Int, Expr, Nothing}}

Produce new IR statements from a rewrite template and matched bindings.
Returns a list of (ssa_idx, stmt, type) tuples. The LAST entry reuses
ssa_idx = -1 as a sentinel (caller assigns the root SSA index).
"""
function materialize_rhs(sci::StructuredIRCode, rhs::RBind, bindings)
    # A bare binding reference — no new statement needed
    return Tuple{Int, Any, Nothing}[]
end

function materialize_rhs(sci::StructuredIRCode, rhs::RCall, bindings)
    # Materialize operands first (depth-first)
    stmts = Tuple{Int, Any, Nothing}[]
    operand_vals = Any[]
    for op in rhs.operands
        if op isa RBind
            push!(operand_vals, bindings[op.name])
        elseif op isa RCall
            # Nested call: materialize and use its SSA result
            sub_stmts = materialize_rhs(sci, op, bindings)
            append!(stmts, sub_stmts)
            if isempty(sub_stmts)
                error("@rewrite RHS: nested call produced no statements")
            end
            push!(operand_vals, SSAValue(sub_stmts[end][1]))
        end
    end

    ssa = new_ssa_idx!(sci)
    stmt = Expr(:call, GlobalRef(Intrinsics, rhs.func_name), operand_vals...)
    push!(stmts, (ssa, stmt, nothing))
    return stmts
end

#=============================================================================
 Driver: rewrite_patterns!
=============================================================================#

"""
    rewrite_patterns!(sci::StructuredIRCode, rules::Vector{RewriteRule}; dce_transparent=false)

Apply declarative rewrite rules to the structured IR.

When `dce_transparent=true`, run a fixpoint DCE pass after rewriting to eliminate
unused transparent ops (to_scalar, from_scalar, broadcast).
"""
function rewrite_patterns!(sci::StructuredIRCode, rules::Vector{RewriteRule};
                            dce_transparent::Bool=false)
    @debug "rewrite_patterns: collecting defs"
    defs = Dict{Int, Tuple{Any, Vector{Any}}}()
    types = Dict{Int, Any}()
    use_counts = Dict{Int, Int}()
    _rw_collect!(sci.entry, use_counts; defs, types)

    ctx = MatchContext(defs, types, use_counts)

    # Group rules by root function for dispatch
    dispatch = Dict{Any, Vector{RewriteRule}}()
    for rule in rules
        push!(get!(dispatch, root_func(rule), RewriteRule[]), rule)
    end

    @debug "rewrite_patterns: matching $(length(rules)) rules against $(length(defs)) defs"
    # Forward walk: try matching rules on each call statement.
    # NOTE: use counts are static (built once). After a rule consumes an SSA,
    # its operands' effective use counts drop, but the context retains the original
    # counts. This is safe because `one_use` constraints apply to the matched op
    # (e.g., mulf), not its operands. The `consumed` set prevents overlapping matches.
    consumed = Set{Int}()
    _rw_match_and_rewrite!(sci, sci.entry, ctx, dispatch, consumed)
    @debug "rewrite_patterns: done ($(length(consumed)) SSAs consumed)"

    dce_transparent && _rw_dce_transparent!(sci, defs, consumed)
end

function _rw_match_and_rewrite!(sci, block::Block, ctx, dispatch, consumed)
    # Recurse into nested control flow first (nested defs precede uses in forward order)
    for i in 1:length(block.body.ssa_idxes)
        stmt = block.body.stmts[i]
        stmt isa ControlFlowOp || continue
        for b in _nested_blocks(stmt)
            _rw_match_and_rewrite!(sci, b, ctx, dispatch, consumed)
        end
    end

    body = block.body

    for i in 1:length(body.ssa_idxes)
        stmt = body.stmts[i]
        stmt isa Expr || continue
        ssa_idx = body.ssa_idxes[i]
        ssa_idx in consumed && continue

        result = resolve_call(stmt)
        result === nothing && continue
        func, _ = result

        applicable_rules = get(dispatch, func, nothing)
        applicable_rules === nothing && continue

        for rule in applicable_rules
            m = pattern_match(ctx, SSAValue(ssa_idx), rule.lhs)
            m === nothing && continue

            # Legality: no consumed SSAs in the match
            any(s in consumed for s in m.matched_ssas) && continue

            _rw_apply!(sci, block, i, ssa_idx, rule, m, consumed)
            break  # first match wins
        end
    end
end

function _rw_apply!(sci, block, pos, root_ssa, rule, match, consumed)
    body = block.body

    # Materialize RHS into new statements
    new_stmts = materialize_rhs(sci, rule.rhs, match.bindings)

    push!(consumed, root_ssa)

    if isempty(new_stmts)
        # Forwarding rewrite: RHS is a bare binding (e.g., `~x`).
        # Replace all uses of root_ssa with the bound value, kill only the root.
        # Inner matched SSAs (from_scalar, broadcast) may have other uses —
        # the DCE fixpoint (dce_transparent=true) handles cleanup of unused ones.
        forwarded_val = match.bindings[rule.rhs::RBind |> x -> x.name]
        _rw_replace_ssa!(sci.entry, Dict{Int, Any}(root_ssa => forwarded_val))
        _rw_kill_ssa!(sci.entry, root_ssa)
        return
    end

    # Substitution rewrite: kill matched intermediates and replace root statement.
    for dead_ssa in match.matched_ssas
        dead_ssa == root_ssa && continue
        push!(consumed, dead_ssa)
        _rw_kill_ssa!(sci.entry, dead_ssa)
    end

    # Insert intermediate stmts (all except the last) before the root position
    for j in 1:(length(new_stmts) - 1)
        new_ssa, new_stmt, _ = new_stmts[j]
        insert_before!(body, root_ssa, new_ssa, new_stmt, body.types[pos])
    end

    # Replace the root statement in-place (reuses root_ssa for downstream uses)
    _, final_stmt, _ = new_stmts[end]
    body.stmts[findfirst(==(root_ssa), body.ssa_idxes)] = final_stmt
end

#=============================================================================
 IR Collection (def-map + use counts)
=============================================================================#

"""
    _rw_collect!(block, use_counts; defs=nothing, types=nothing)

Walk the structured IR, counting SSA uses. Optionally also build a def-map and
type-map when the corresponding keyword arguments are provided.
"""
function _rw_collect!(block::Block, use_counts::Dict{Int, Int};
                      defs::Union{Nothing, Dict{Int, Tuple{Any, Vector{Any}}}}=nothing,
                      types::Union{Nothing, Dict{Int, Any}}=nothing)
    for i in 1:length(block.body.ssa_idxes)
        stmt = block.body.stmts[i]
        if stmt isa ControlFlowOp
            _count_cf_refs!(stmt, use_counts)
            for b in _nested_blocks(stmt)
                _rw_collect!(b, use_counts; defs, types)
            end
        elseif stmt isa Expr
            if defs !== nothing
                ssa_idx = block.body.ssa_idxes[i]
                result = resolve_call(stmt)
                if result !== nothing
                    func, operands = result
                    defs[ssa_idx] = (func, collect(Any, operands))
                end
                types !== nothing && (types[ssa_idx] = block.body.types[i])
            end
            _rw_count_expr!(stmt, use_counts)
        elseif stmt isa JoinTokensNode
            for tok in stmt.tokens; _rw_count_ref!(tok, use_counts); end
        elseif stmt isa ReturnNode
            isdefined(stmt, :val) && _rw_count_ref!(stmt.val, use_counts)
        end
    end
    _rw_count_terminator!(block.terminator, use_counts)
end

"""Count SSA references in a control-flow op's own fields (not nested blocks)."""
_count_cf_refs!(op::IfOp, c) = _rw_count_ref!(op.condition, c)
function _count_cf_refs!(op::ForOp, c)
    _rw_count_ref!(op.lower, c); _rw_count_ref!(op.upper, c); _rw_count_ref!(op.step, c)
    for v in op.init_values; _rw_count_ref!(v, c); end
end
function _count_cf_refs!(op::WhileOp, c)
    for v in op.init_values; _rw_count_ref!(v, c); end
end
function _count_cf_refs!(op::LoopOp, c)
    for v in op.init_values; _rw_count_ref!(v, c); end
end
_count_cf_refs!(::ControlFlowOp, c) = nothing

function _rw_count_ref!(@nospecialize(val), counts)
    val isa SSAValue && (counts[val.id] = get(counts, val.id, 0) + 1)
end

function _rw_count_expr!(expr::Expr, counts)
    start = expr.head === :invoke ? 3 : 2
    for i in start:length(expr.args)
        _rw_count_ref!(expr.args[i], counts)
    end
end

function _rw_count_terminator!(term, counts)
    if term isa YieldOp || term isa ContinueOp || term isa BreakOp
        for v in term.values; _rw_count_ref!(v, counts); end
    elseif term isa ConditionOp
        _rw_count_ref!(term.condition, counts)
        for v in term.args; _rw_count_ref!(v, counts); end
    elseif term isa ReturnNode
        isdefined(term, :val) && _rw_count_ref!(term.val, counts)
    end
end

#=============================================================================
 SSA Replacement (for forwarding rewrites)
=============================================================================#

function _rw_sub(@nospecialize(val), replacements)
    val isa SSAValue && haskey(replacements, val.id) ? replacements[val.id] : val
end

"""Replace SSAValue references in the IR according to the replacement map."""
function _rw_replace_ssa!(block::Block, replacements)
    for i in 1:length(block.body.ssa_idxes)
        stmt = block.body.stmts[i]
        if stmt isa ControlFlowOp
            _replace_cf_refs!(stmt, replacements)
            for b in _nested_blocks(stmt)
                _rw_replace_ssa!(b, replacements)
            end
        elseif stmt isa Expr
            start = stmt.head === :invoke ? 3 : 2
            for j in start:length(stmt.args)
                stmt.args[j] = _rw_sub(stmt.args[j], replacements)
            end
        elseif stmt isa JoinTokensNode
            for j in 1:length(stmt.tokens)
                stmt.tokens[j] = _rw_sub(stmt.tokens[j], replacements)
            end
        elseif stmt isa ReturnNode
            if isdefined(stmt, :val)
                new_val = _rw_sub(stmt.val, replacements)
                new_val !== stmt.val && (block.body.stmts[i] = ReturnNode(new_val))
            end
        end
    end
    _rw_replace_terminator!(block, replacements)
end

"""Replace SSA references in a control-flow op's own fields."""
function _replace_cf_refs!(op::IfOp, r)
    op.condition = _rw_sub(op.condition, r)
end
function _replace_cf_refs!(op::ForOp, r)
    op.lower = _rw_sub(op.lower, r); op.upper = _rw_sub(op.upper, r); op.step = _rw_sub(op.step, r)
    for j in 1:length(op.init_values); op.init_values[j] = _rw_sub(op.init_values[j], r); end
end
function _replace_cf_refs!(op::WhileOp, r)
    for j in 1:length(op.init_values); op.init_values[j] = _rw_sub(op.init_values[j], r); end
end
function _replace_cf_refs!(op::LoopOp, r)
    for j in 1:length(op.init_values); op.init_values[j] = _rw_sub(op.init_values[j], r); end
end
_replace_cf_refs!(::ControlFlowOp, r) = nothing

function _rw_replace_terminator!(block::Block, r)
    term = block.terminator
    if term isa YieldOp || term isa ContinueOp || term isa BreakOp
        for j in 1:length(term.values); term.values[j] = _rw_sub(term.values[j], r); end
    elseif term isa ConditionOp
        new_cond = _rw_sub(term.condition, r)
        for j in 1:length(term.args); term.args[j] = _rw_sub(term.args[j], r); end
        new_cond !== term.condition && (block.terminator = ConditionOp(new_cond, term.args))
    elseif term isa ReturnNode
        if isdefined(term, :val)
            new_val = _rw_sub(term.val, r)
            new_val !== term.val && (block.terminator = ReturnNode(new_val))
        end
    end
end

#=============================================================================
 Kill SSA / DCE
=============================================================================#

"""Null out the statement for a given SSA index (searches all blocks)."""
function _rw_kill_ssa!(block::Block, ssa_idx::Int)
    for i in 1:length(block.body.ssa_idxes)
        if block.body.ssa_idxes[i] == ssa_idx
            block.body.stmts[i] = nothing
            return
        end
        stmt = block.body.stmts[i]
        stmt isa ControlFlowOp || continue
        for b in _nested_blocks(stmt)
            _rw_kill_ssa!(b, ssa_idx)
        end
    end
end

"""DCE fixpoint: iteratively kill unused transparent ops (to_scalar/from_scalar/broadcast)."""
function _rw_dce_transparent!(sci::StructuredIRCode, defs, already_dead::Set{Int})
    max_iter = 20
    for iter in 1:max_iter
        counts = Dict{Int, Int}()
        _rw_collect!(sci.entry, counts)
        killed = false
        for (ssa_idx, (func, _)) in defs
            ssa_idx in already_dead && continue
            _is_transparent(func) || continue
            if get(counts, ssa_idx, 0) == 0
                _rw_kill_ssa!(sci.entry, ssa_idx)
                push!(already_dead, ssa_idx)
                killed = true
            end
        end
        killed || break
        iter == max_iter && @warn "rewrite_patterns DCE: fixpoint not reached after $max_iter iterations"
    end
end
