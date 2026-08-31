# Declarative IR Rewrite Pattern Framework
#
# Worklist-based fixpoint driver inspired by MLIR's GreedyPatternRewriteDriver.
# Patterns compile into pattern/rewrite node trees. The driver processes a LIFO
# worklist until fixpoint: when a rewrite fires, affected instructions are
# re-added to the worklist for further matching. Dead code cleanup is delegated
# to the pipeline's dce_pass!.
#
# Usage:
#   rules = RewriteRule[
#       @rewrite Intrinsics.addf(one_use(Intrinsics.mulf(~x, ~y)), ~z) =>
#               Intrinsics.fma(~x, ~y, ~z)
#       @rewrite Core.Intrinsics.slt_int(~x, ~y) =>
#               Intrinsics.cmpi(~x, ~y, $(ComparisonPredicate.LessThan), $(Signedness.Signed))
#   ]
#   rewrite_patterns!(sci, rules)

using Core: SSAValue

#=============================================================================
 Pattern & Rewrite Nodes
=============================================================================#

abstract type PatternNode end
struct PCall <: PatternNode; func::Any; operands::Vector{PatternNode}; end
struct PBind <: PatternNode; name::Symbol; end
struct PTypedBind <: PatternNode; name::Symbol; type::Type; end
struct POneUse <: PatternNode; inner::PatternNode; end
struct PLiteral <: PatternNode; val::Any; end
struct PSplat <: PatternNode; name::Symbol; end  # ~x... — captures remaining operands

abstract type RewriteNode end
struct RCall <: RewriteNode; func::Any; operands::Vector{RewriteNode}; end
struct RBind <: RewriteNode; name::Symbol; end
struct RConst <: RewriteNode; val::Any; end
struct RSplat <: RewriteNode; name::Symbol; end  # ~x... — expands splat binding

"""
    RFunc(func)

Imperative rewrite node. The function is called with
`(sci, block, inst, match, driver)` and returns `true` if the rewrite was
applied, `false` to skip this rule and try the next one.
"""
struct RFunc <: RewriteNode; func::Function; end

struct RewriteRule
    lhs::PCall
    rhs::RewriteNode
    guard::Union{Function, Nothing}  # (match, driver) -> Bool, or nothing
end
RewriteRule(lhs::PCall, rhs::RewriteNode) = RewriteRule(lhs, rhs, nothing)

root_func(rule::RewriteRule) = rule.lhs.func

#=============================================================================
 @rewrite / @rewriter Macros
=============================================================================#

"""
    @rewrite lhs => rhs
    @rewrite(lhs => rhs, guard)

Compile a declarative rewrite rule. LHS: `func(args...)` matches calls,
`~x` binds (repeated names require equality), `~x::T` binds with type constraint,
`one_use(pat)` requires single use, `\$(expr)` matches literal values.
RHS: `func(args...)` emits calls, `~x` references bindings,
`\$(expr)` injects a literal constant.

Optional `guard` is a function `(match, driver) -> Bool` checked after pattern match.
"""
macro rewrite(ex, guard=nothing)
    ex isa Expr && ex.head === :call && ex.args[1] === :(=>) ||
        error("@rewrite expects: lhs => rhs")
    g = guard === nothing ? :nothing : guard
    esc(:(RewriteRule($(compile_lhs(ex.args[2])), $(compile_rhs(ex.args[3])), $g)))
end

"""
    @rewriter lhs => func

Declarative pattern with imperative rewrite. LHS uses the same pattern syntax as
`@rewrite`. RHS is a function `(sci, block, inst, match, driver) -> Bool` that
performs the rewrite and returns `true`, or returns `false` to skip and try the
next rule.
"""
macro rewriter(ex)
    ex isa Expr && ex.head === :call && ex.args[1] === :(=>) ||
        error("@rewriter expects: lhs => func")
    esc(:(RewriteRule($(compile_lhs(ex.args[2])), RFunc($(ex.args[3])))))
end

function compile_lhs(ex)
    # $(expr) on the LHS: match a literal value
    if ex isa Expr && ex.head === :$
        return :(PLiteral($(ex.args[1])))
    end
    # ~x... on the LHS: splat capture of remaining operands
    # Julia parses `~x...` as Expr(:..., Expr(:call, :~, :x))
    if ex isa Expr && ex.head === :... && length(ex.args) == 1
        inner = ex.args[1]
        if inner isa Expr && inner.head === :call && inner.args[1] === :~ && length(inner.args) == 2
            name = inner.args[2]
            return :(PSplat($(QuoteNode(name))))
        end
    end
    ex isa Expr && ex.head === :call || error("@rewrite LHS: expected call, got $ex")
    f = ex.args[1]
    if f === :~
        inner = ex.args[2]
        if inner isa Expr && inner.head === :(::)
            return :(PTypedBind($(QuoteNode(inner.args[1])), $(inner.args[2])))
        end
        return :(PBind($(QuoteNode(inner))))
    end
    f === :one_use && return :(POneUse($(compile_lhs(ex.args[2]))))
    :(PCall($f, PatternNode[$(compile_lhs.(ex.args[2:end])...)]))
end

function compile_rhs(ex)
    if ex isa Expr && ex.head === :$
        return :(RConst($(ex.args[1])))
    end
    # ~x... on the RHS: expand splat binding
    if ex isa Expr && ex.head === :... && length(ex.args) == 1
        inner = ex.args[1]
        if inner isa Expr && inner.head === :call && inner.args[1] === :~ && length(inner.args) == 2
            name = inner.args[2]
            return :(RSplat($(QuoteNode(name))))
        end
    end
    ex isa Expr && ex.head === :call || error("@rewrite RHS: expected call or \$const, got $ex")
    f = ex.args[1]
    f === :~ && return :(RBind($(QuoteNode(ex.args[2]))))
    :(RCall($f, RewriteNode[$(compile_rhs.(ex.args[2:end])...)]))
end

#=============================================================================
 Worklist
=============================================================================#

mutable struct Worklist
    list::Vector{SSAValue}            # entries (SSAValue(-1) = removed sentinel)
    member::Dict{SSAValue, Int}       # val -> position in list
end

const SENTINEL = SSAValue(-1)

Worklist() = Worklist(SSAValue[], Dict{SSAValue, Int}())

function Base.push!(wl::Worklist, val::SSAValue)
    haskey(wl.member, val) && return
    push!(wl.list, val)
    wl.member[val] = length(wl.list)
end

function Base.pop!(wl::Worklist)
    while !isempty(wl.list)
        val = pop!(wl.list)
        val == SENTINEL && continue
        delete!(wl.member, val)
        return val
    end
    return nothing
end

function remove!(wl::Worklist, val::SSAValue)
    pos = get(wl.member, val, 0)
    pos == 0 && return
    wl.list[pos] = SENTINEL
    delete!(wl.member, val)
end

Base.isempty(wl::Worklist) = isempty(wl.member)

#=============================================================================
 Driver State
=============================================================================#

struct DefEntry
    block::Block
    val::SSAValue
    func::Any
end

"""Operands of a DefEntry, read from the live IR."""
function def_operands(entry::DefEntry)
    haskey(entry.block, entry.val.id) || return Any[]
    call = resolve_call(entry.block, entry.block[entry.val.id][:stmt])
    call === nothing && return Any[]
    _, ops = call
    return ops
end

mutable struct RewriteDriver
    rewriter::Rewriter
    defs::Dict{SSAValue, DefEntry}
    dispatch::Dict{Any, Vector{RewriteRule}}
    worklist::Worklist
    constants::Union{Nothing, ConstantInfo}
    modified::Set{SSAValue}          # instructions whose operands were modified by forwarding
    max_rewrites::Int
end

#=============================================================================
 Rewriter listener: driver-side bookkeeping reacts to IR mutations
=============================================================================#

# The driver has its own state on top of the Rewriter's index: a `defs` map
# keyed by func, the `worklist`, and a `modified` set for cascading. These
# `notify_*` hooks let the Rewriter keep that state in sync, so the driver
# doesn't need to wrap every mutation callsite. Same pattern as MLIR's
# `RewriterBase::Listener::notifyOperation{Inserted,Modified,Erased}`.

function notify_inserted!(d::RewriteDriver, block::Block, inst::Instruction)
    stmt = inst[:stmt]
    call = resolve_call(block, stmt)
    call === nothing && return
    func, _ = call
    val = SSAValue(inst)
    d.defs[val] = DefEntry(block, val, func)
    push!(d.worklist, val)
end

function notify_modified!(d::RewriteDriver, block::Block, val::SSAValue,
                          @nospecialize(old_stmt), @nospecialize(new_stmt))
    # Refresh the def entry so worklist dispatch picks the new func.
    call = resolve_call(block, new_stmt)
    if call !== nothing
        func, _ = call
        d.defs[val] = DefEntry(block, val, func)
    end
    # Worklist cascading is done by the callers (`apply_rewrite!`,
    # `commute_arith_transparent`), which re-seed `val` and its users
    # themselves.
end

function notify_erased!(d::RewriteDriver, ::Block, val::SSAValue,
                        @nospecialize(old_stmt))
    # Operand-defs may now be dead; cascade them to the worklist.
    if old_stmt isa Expr
        for_expr_operands(old_stmt) do op
            op isa SSAValue || return
            haskey(d.defs, op) && push!(d.worklist, op)
        end
    end
    delete!(d.defs, val)
    remove!(d.worklist, val)
    delete!(d.modified, val)
end

#=============================================================================
 Driver-side query helpers (thin proxies to the Rewriter)
=============================================================================#

users_of(driver::RewriteDriver, val::SSAValue) = users(driver.rewriter, val)

use_count(driver::RewriteDriver, val::SSAValue) = use_count(driver.rewriter, val)

"""Add instructions that use `val` to the worklist (their operand changed)."""
function add_users_to_worklist!(driver::RewriteDriver, val::SSAValue)
    for u in users_of(driver, val)
        push!(driver.worklist, u)
    end
end

#=============================================================================
 Matching
=============================================================================#

struct MatchResult
    bindings::Dict{Symbol, Any}
    matched_ssas::Vector{SSAValue}
end

"""Merge bindings, requiring repeated names to bind the same value (=== equality)."""
function merge_bindings!(dest::Dict{Symbol,Any}, src::Dict{Symbol,Any})
    for (k, v) in src
        if haskey(dest, k)
            dest[k] === v || return false
        else
            dest[k] = v
        end
    end
    return true
end

function pattern_match(driver::RewriteDriver, @nospecialize(val), pat::PCall,
                       block::Block=driver.rewriter.sci.entry)
    val isa SSAValue || return nothing
    entry = get(driver.defs, val, nothing)
    entry === nothing && return nothing

    if entry.func === pat.func
        ops = def_operands(entry)
        has_splat = !isempty(pat.operands) && last(pat.operands) isa PSplat
        n_fixed = has_splat ? length(pat.operands) - 1 : length(pat.operands)

        if has_splat ? length(ops) >= n_fixed : length(ops) == n_fixed
            result = MatchResult(Dict{Symbol,Any}(), SSAValue[val])
            # Match fixed operands
            for i in 1:n_fixed
                m = pattern_match(driver, ops[i], pat.operands[i], entry.block)
                m === nothing && return nothing
                merge_bindings!(result.bindings, m.bindings) || return nothing
                append!(result.matched_ssas, m.matched_ssas)
            end
            # Capture remaining operands into the splat binding
            if has_splat
                splat_name = pat.operands[end]::PSplat
                result.bindings[splat_name.name] = ops[n_fixed+1:end]
            end
            return result
        end
    end

    return nothing
end

pattern_match(driver::RewriteDriver, @nospecialize(val), pat::PBind, block::Block=driver.rewriter.sci.entry) =
    MatchResult(Dict{Symbol,Any}(pat.name => val), SSAValue[])

function pattern_match(driver::RewriteDriver, @nospecialize(val), pat::PTypedBind,
                       block::Block=driver.rewriter.sci.entry)
    T = value_type(block, val)
    T === nothing && return nothing
    CC.widenconst(T) <: pat.type || return nothing
    MatchResult(Dict{Symbol,Any}(pat.name => val), SSAValue[])
end

function pattern_match(driver::RewriteDriver, @nospecialize(val), pat::POneUse,
                       block::Block=driver.rewriter.sci.entry)
    val isa SSAValue && use_count(driver, val) == 1 || return nothing
    pattern_match(driver, val, pat.inner, block)
end

# PLiteral: match if the operand equals the given value.
# For non-SSA operands (enum constants, predicates): checks ===.
# For SSA operands: routed through const_value on the ConstantAnalysis result.
function pattern_match(driver::RewriteDriver, @nospecialize(val), pat::PLiteral,
                       block::Block=driver.rewriter.sci.entry)
    val === pat.val && return MatchResult(Dict{Symbol,Any}(), SSAValue[])
    if val isa SSAValue
        c = const_value(driver.constants, val)
        c !== nothing && c == pat.val &&
            return MatchResult(Dict{Symbol,Any}(), SSAValue[])
    end
    return nothing
end

#=============================================================================
 Rewrite Application
=============================================================================#

"""Resolve an RHS operand, inserting sub-calls before `ref` as needed.
`root_typ` is the type of the original matched instruction — used only for the
outermost RCall (whose statement replaces the root). Intermediate RCalls infer
their type from the first value-like operand, since element-wise ops preserve type."""
resolve_rhs(driver, block, ref, op::RBind, bindings, root_typ) = bindings[op.name]
resolve_rhs(driver, block, ref, op::RConst, bindings, root_typ) = op.val
function resolve_rhs(driver::RewriteDriver, block, ref, op::RCall, bindings, root_typ)
    # Flatten RSplat nodes: each RSplat expands to multiple operands
    operands = Any[]
    for sub in op.operands
        if sub isa RSplat
            append!(operands, bindings[sub.name])
        else
            push!(operands, resolve_rhs(driver, block, ref, sub, bindings, root_typ))
        end
    end
    # Infer type from the first value-like operand (SSA, argument, or block
    # argument) — correct for element-wise ops (addi, subi, negf, etc.) whose
    # result type matches their operands. Kernel arguments still have their
    # source-level scalar type, so promote them to the canonical 0-D tile type.
    # Literal operands are skipped: their scalar type is not necessarily the
    # tile result type. Falls back to root_typ when no value operand is available.
    typ = root_typ
    for o in operands
        is_trackable_value(o) || continue
        t = value_type(block, o)
        t === nothing && continue
        typ = boundary_jltype(CC.widenconst(t))
        break
    end
    inst = insert_before!(driver.rewriter, block, ref, Expr(:call, op.func, operands...), typ;
                          flag=inferred_flags(op.func))
    SSAValue(inst)
end

function apply_rewrite!(driver::RewriteDriver, block, val::SSAValue, rule, match)
    entry = driver.defs[val]
    if rule.rhs isa RFunc
        # Look up live instruction for RFunc interface
        haskey(block, val.id) || return false
        inst = block[val.id]
        rule.rhs.func(driver.rewriter.sci, block, inst, match, driver) || return false
        return true
    elseif rule.rhs isa RBind
        # Forwarding: replace all uses of root with the bound value, delete root.
        # Mark immediate users as modified — their operands are about to change.
        # When these are later popped from the worklist without a match, the
        # driver propagates to THEIR users (see modified check in main loop).
        # This gives MLIR-style notifyOperationModified cascading.
        new_val = match.bindings[rule.rhs.name]
        for u in users_of(driver, val)
            push!(driver.modified, u)
            push!(driver.worklist, u)
        end
        replace_uses!(driver.rewriter, val, new_val)
        erase!(driver.rewriter, entry.block, val)
    else
        # Substitution: rewrite the root statement, clean up dead intermediates.
        # Only delete intermediates with no remaining uses — transparent-op
        # tracing may have added multi-use intermediates to matched_ssas.
        for dead_val in match.matched_ssas
            dead_val == val && continue
            dead_entry = get(driver.defs, dead_val, nothing)
            dead_entry === nothing && continue
            use_count(driver, dead_val) == 0 || continue
            erase!(driver.rewriter, dead_entry.block, dead_val)
        end
        typ = block[val.id][:type]
        # Build operands, flattening RSplat nodes into multiple operands
        operands = Any[]
        for op in rule.rhs.operands
            if op isa RSplat
                append!(operands, match.bindings[op.name])
            else
                push!(operands, resolve_rhs(driver, block, val, op, match.bindings, typ))
            end
        end
        # When the substituted func differs from the matched root, the
        # inferred IR_FLAG_* bits describe the OLD op; recompute from the new
        # func's `efunc` effects so downstream gates (CSE, LICM) see fresh,
        # correct information.
        new_stmt = Expr(:call, rule.rhs.func, operands...)
        flag = rule.rhs.func === driver.defs[val].func ? nothing : inferred_flags(rule.rhs.func)
        replace_stmt!(driver.rewriter, block, val, new_stmt; flag)
        push!(driver.worklist, val)
        add_users_to_worklist!(driver, val)
    end
end

#=============================================================================
 Driver
=============================================================================#

"""
    rewrite_patterns!(sci::StructuredIRCode, rules::Vector{RewriteRule}; max_rewrites=10_000)

Apply rewrite rules to the structured IR using a worklist-based fixpoint driver.
Rules are tried until no more matches fire or `max_rewrites` is reached.
Dead code left behind is cleaned up by the pipeline's `dce_pass!`.
"""
function rewrite_patterns!(sci::StructuredIRCode, rules::Vector{RewriteRule};
                           max_rewrites::Int=10_000,
                           constants=nothing)
    # Build dispatch table
    dispatch = Dict{Any, Vector{RewriteRule}}()
    for rule in rules
        push!(get!(dispatch, root_func(rule), RewriteRule[]), rule)
    end

    # Build defs index
    defs = Dict{SSAValue, DefEntry}()
    for block in eachblock(sci)
        for inst in instructions(block)
            call = resolve_call(block, inst)
            call === nothing && continue
            func, _ = call
            val = SSAValue(inst)
            defs[val] = DefEntry(block, val, func)
        end
    end

    # Seed worklist (forward order → reversed by LIFO → processes top-down)
    wl = Worklist()
    for block in eachblock(sci)
        for inst in instructions(block)
            val = SSAValue(inst)
            haskey(defs, val) && push!(wl, val)
        end
    end

    rewriter = Rewriter(sci)
    driver = RewriteDriver(rewriter, defs, dispatch, wl, constants, Set{SSAValue}(),
                           max_rewrites)
    rewriter.listener = driver

    num_rewrites = 0
    while !isempty(driver.worklist) && num_rewrites < driver.max_rewrites
        val = pop!(driver.worklist)::SSAValue
        entry = get(driver.defs, val, nothing)
        entry === nothing && continue

        # Verify instruction is still live in its block
        haskey(entry.block, val.id) || begin
            delete!(driver.defs, val)
            continue
        end

        # Trivial dead-op elimination: if this op has no uses and is pure,
        # erase it. This keeps use counts accurate for `one_use` patterns
        # (e.g., FMA fusion needs mulf's dead transparent-op users removed
        # so the mulf reads as single-use). Full DCE handles the rest.
        if use_count(driver, val) == 0
            stmt = entry.block[val.id][:stmt]
            if !must_keep(entry.block, stmt)
                erase!(driver.rewriter, entry.block, val)
                continue
            end
        end

        # Look up applicable rules by function
        applicable = get(driver.dispatch, entry.func, nothing)
        matched = false
        if applicable !== nothing
            for rule in applicable
                m = pattern_match(driver, val, rule.lhs)
                m === nothing && continue
                rule.guard !== nothing && !rule.guard(m, driver) && continue
                if apply_rewrite!(driver, entry.block, val, rule, m) === false
                    continue  # RFunc declined — try next rule
                end
                num_rewrites += 1
                matched = true
                break
            end
        end

        # Operand-modified propagation (MLIR notifyOperationModified equivalent):
        # if this instruction's operands were changed by a forwarding rewrite but
        # no rule fired here, propagate to users — the operand change may enable
        # new matches further up the use chain. Mark users as modified too so the
        # cascade continues through the fixpoint.
        if !matched && val in driver.modified
            delete!(driver.modified, val)
            for u in users_of(driver, val)
                push!(driver.modified, u)
                haskey(driver.defs, u) && push!(driver.worklist, u)
            end
        end
    end
end
