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
    inplace::Bool                    # true = modify matched ops in-place (no new instructions)
end
RewriteRule(lhs::PCall, rhs::RewriteNode) = RewriteRule(lhs, rhs, nothing, false)
RewriteRule(lhs::PCall, rhs::RewriteNode, guard) = RewriteRule(lhs, rhs, guard, false)

root_func(rule::RewriteRule) = rule.lhs.func

#=============================================================================
 @rewrite / @rewriter Macros
=============================================================================#

"""
    @rewrite lhs => rhs
    @rewrite(lhs => rhs, guard)
    @rewrite(inplace=true, lhs => rhs)
    @rewrite(inplace=true, lhs => rhs, guard)

Compile a declarative rewrite rule. LHS: `func(args...)` matches calls,
`~x` binds (repeated names require equality), `~x::T` binds with type constraint,
`one_use(pat)` requires single use, `\$(expr)` matches literal values.
RHS: `func(args...)` emits calls, `~x` references bindings,
`\$(expr)` injects a literal constant.

Optional `guard` is a function `(match, driver) -> Bool` checked after pattern match.

With `inplace=true`, the RHS describes modifications to the matched ops' operands
rather than creating new ops. The LHS and RHS trees are walked in parallel: where
they share the same function, the existing op is modified in-place; where the RHS
has a different binding or constant, the operand is replaced. This avoids the
worklist cascade that occurs when the standard mode creates new instructions.
"""
macro rewrite(args...)
    # Parse keyword arguments
    inplace = false
    positional = Any[]
    for arg in args
        if arg isa Expr && arg.head === :(=) && arg.args[1] === :inplace
            inplace = arg.args[2]::Bool
        else
            push!(positional, arg)
        end
    end
    length(positional) >= 1 || error("@rewrite expects: lhs => rhs")
    ex = positional[1]
    guard = length(positional) >= 2 ? positional[2] : nothing

    ex isa Expr && ex.head === :call && ex.args[1] === :(=>) ||
        error("@rewrite expects: lhs => rhs")
    g = guard === nothing ? :nothing : guard
    esc(:(RewriteRule($(compile_lhs(ex.args[2])), $(compile_rhs(ex.args[3])), $g, $inplace)))
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
    sci::StructuredIRCode
    defs::Dict{SSAValue, DefEntry}
    dispatch::Dict{Any, Vector{RewriteRule}}
    worklist::Worklist
    constants::Union{Nothing, ConstantInfo}
    modified::Set{SSAValue}          # instructions whose operands were modified by forwarding
    max_rewrites::Int
end

"""Compute fresh use count for an SSA value."""
use_count(driver::RewriteDriver, val::SSAValue) =
    length(uses(driver.sci.entry, val))

#=============================================================================
 Notifications
=============================================================================#

"""Add operand-producing instructions to the worklist (enables cascading)."""
function add_operands_to_worklist!(driver::RewriteDriver, entry::DefEntry)
    for op in def_operands(entry)
        op isa SSAValue || continue
        haskey(driver.defs, op) && push!(driver.worklist, op)
    end
end

"""Add instructions that use `val` to the worklist (their operand changed)."""
function add_users_to_worklist!(driver::RewriteDriver, val::SSAValue)
    for inst in users(driver.sci.entry, val)
        push!(driver.worklist, SSAValue(inst))
    end
end

"""Erase an instruction and notify the worklist."""
function erase_op!(driver::RewriteDriver, entry::DefEntry)
    add_operands_to_worklist!(driver, entry)
    if haskey(entry.block, entry.val.id)
        delete!(entry.block, entry.val.id)
    end
    delete!(driver.defs, entry.val)
    remove!(driver.worklist, entry.val)
end

"""Register a newly inserted instruction."""
function notify_insert!(driver::RewriteDriver, block::Block, inst::Instruction)
    val = SSAValue(inst)
    call = resolve_call(block, inst)
    call === nothing && return
    func, _ = call
    driver.defs[val] = DefEntry(block, val, func)
    push!(driver.worklist, val)
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
                       block::Block=driver.sci.entry)
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

pattern_match(driver::RewriteDriver, @nospecialize(val), pat::PBind, block::Block=driver.sci.entry) =
    MatchResult(Dict{Symbol,Any}(pat.name => val), SSAValue[])

function pattern_match(driver::RewriteDriver, @nospecialize(val), pat::PTypedBind,
                       block::Block=driver.sci.entry)
    T = value_type(block, val)
    T === nothing && return nothing
    CC.widenconst(T) <: pat.type || return nothing
    MatchResult(Dict{Symbol,Any}(pat.name => val), SSAValue[])
end

function pattern_match(driver::RewriteDriver, @nospecialize(val), pat::POneUse,
                       block::Block=driver.sci.entry)
    val isa SSAValue && use_count(driver, val) == 1 || return nothing
    pattern_match(driver, val, pat.inner, block)
end

# PLiteral: match if the operand equals the given value.
# For non-SSA operands (enum constants, predicates): checks ===.
# For SSA operands: routed through const_value on the ConstantAnalysis result.
function pattern_match(driver::RewriteDriver, @nospecialize(val), pat::PLiteral,
                       block::Block=driver.sci.entry)
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
outermost RCall (which replaces the root in-place). Intermediate RCalls infer
their type from the first SSA operand, since element-wise ops preserve type."""
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
    # Infer type from first SSA operand — correct for element-wise ops (addi,
    # subi, negf, etc.) whose result type matches their operands. Falls back to
    # root_typ when no SSA operand is available.
    typ = root_typ
    for o in operands
        o isa SSAValue || continue
        t = value_type(block, o)
        t === nothing && continue
        typ = CC.widenconst(t)
        break
    end
    inst = insert_before!(block, ref, Expr(:call, op.func, operands...), typ;
                          flag=inferred_flags(op.func))
    notify_insert!(driver, block, inst)
    SSAValue(inst)
end

"""
In-place rewrite: walk LHS/RHS trees in parallel, modifying matched ops' operands.
No new instructions are created — existing ops are modified in-place.
"""
function apply_inplace_rewrite!(driver::RewriteDriver, block, val::SSAValue, rule, match)
    haskey(block, val.id) || return false

    # Build new operands for the root op from the RHS.
    new_operands = Any[resolve_inplace_rhs(driver, match.bindings, op, lhs_op)
                       for (op, lhs_op) in zip(rule.rhs.operands, rule.lhs.operands)]
    new_stmt = Expr(:call, rule.rhs.func, new_operands...)
    # Same-func rewrites (most common: only operands change) preserve flag
    # via the partial-NamedTuple setindex. Different-func rewrites recompute
    # the flag from the new func's declared effects (`efunc` overrides),
    # mirroring inference's `flags_for_effects` — the inferred bits on the
    # old call describe the OLD op and don't carry over.
    if rule.rhs.func === driver.defs[val].func
        block[val.id] = (stmt=new_stmt,)
    else
        block[val.id] = (stmt=new_stmt, flag=inferred_flags(rule.rhs.func))
    end
    driver.defs[val] = DefEntry(block, val, rule.rhs.func)
    push!(driver.worklist, val)
    add_users_to_worklist!(driver, val)
    return true
end

"""Resolve an RHS operand for in-place mode. If the RHS sub-tree is an RCall
matching a PCall at the same position, modify the existing op in-place and
return its SSAValue. Otherwise fall back to bindings/constants."""
resolve_inplace_rhs(driver, bindings, op::RBind, @nospecialize(lhs_op)) = bindings[op.name]
resolve_inplace_rhs(driver, bindings, op::RConst, @nospecialize(lhs_op)) = op.val

function resolve_inplace_rhs(driver, bindings, op::RCall, lhs_op::PCall)
    # The LHS matched an op with this function. Find it via the match bindings
    # or the defs index and modify it in-place.
    op.func === lhs_op.func && length(op.operands) == length(lhs_op.operands) ||
        error("inplace rewrite: RHS sub-call $(op.func) doesn't match LHS structure")
    matched_ssa = @something find_matched_ssa(driver, lhs_op, bindings) error(
        "inplace rewrite: could not find matched SSA for $(lhs_op.func)")
    entry = @something get(driver.defs, matched_ssa, nothing) error(
        "inplace rewrite: no def entry for $matched_ssa")
    haskey(entry.block, matched_ssa.id) ||
        error("inplace rewrite: $matched_ssa not found in block")
    new_ops = Any[resolve_inplace_rhs(driver, bindings, sub_rhs, sub_lhs)
                  for (sub_rhs, sub_lhs) in zip(op.operands, lhs_op.operands)]
    # Sub-call func is enforced equal to the matched lhs func above
    # (`op.func === lhs_op.func`), so the flag is still valid — only operands
    # changed. Partial-NamedTuple setindex preserves type and flag.
    entry.block[matched_ssa.id] = (stmt=Expr(:call, op.func, new_ops...),)
    push!(driver.worklist, matched_ssa)
    return matched_ssa
end

# Fallback for mismatched LHS/RHS structure
function resolve_inplace_rhs(driver, bindings, op::RCall, @nospecialize(lhs_op))
    error("inplace rewrite: RHS has RCall but LHS has $(typeof(lhs_op)) at same position")
end

"""Find the SSA value that was matched by a PCall pattern node during matching.
The matched_ssas in MatchResult are ordered root-first, but we need to find
the specific SSA for a sub-pattern. We do this by looking up the first operand's
binding and finding the op that defines it."""
function find_matched_ssa(driver, pat::PCall, bindings)
    entry = driver.sci.entry
    for sub in pat.operands
        if sub isa PBind
            bound = get(bindings, sub.name, nothing)
            bound isa SSAValue || continue
            for inst in users(entry, bound)
                call = resolve_call(entry, inst)
                call === nothing && continue
                func, _ = call
                func === pat.func && return SSAValue(inst)
            end
        elseif sub isa PCall
            inner_ssa = find_matched_ssa(driver, sub, bindings)
            if inner_ssa !== nothing
                for inst in users(entry, inner_ssa)
                    call = resolve_call(entry, inst)
                    call === nothing && continue
                    func, _ = call
                    func === pat.func && return SSAValue(inst)
                end
            end
        end
    end
    return nothing
end

function apply_rewrite!(driver::RewriteDriver, block, val::SSAValue, rule, match)
    # In-place mode: modify matched ops' operands without creating new instructions
    if rule.inplace
        return apply_inplace_rewrite!(driver, block, val, rule, match)
    end

    entry = driver.defs[val]
    if rule.rhs isa RFunc
        # Look up live instruction for RFunc interface
        haskey(block, val.id) || return false
        inst = block[val.id]
        rule.rhs.func(driver.sci, block, inst, match, driver) || return false
        return true
    elseif rule.rhs isa RBind
        # Forwarding: replace all uses of root with the bound value, delete root.
        # Mark immediate users as modified — their operands are about to change.
        # When these are later popped from the worklist without a match, the
        # driver propagates to THEIR users (see modified check in main loop).
        # This gives MLIR-style notifyOperationModified cascading.
        for inst in users(driver.sci.entry, val)
            push!(driver.modified, SSAValue(inst))
        end
        add_users_to_worklist!(driver, val)
        replace_uses!(driver.sci.entry, val, match.bindings[rule.rhs.name])
        erase_op!(driver, entry)
    else
        # Substitution: replace root in-place, clean up dead intermediates.
        # Only delete intermediates with no remaining uses — transparent-op
        # tracing may have added multi-use intermediates to matched_ssas.
        for dead_val in match.matched_ssas
            dead_val == val && continue
            dead_entry = get(driver.defs, dead_val, nothing)
            dead_entry === nothing && continue
            use_count(driver, dead_val) == 0 || continue
            erase_op!(driver, dead_entry)
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
        # The substituted func almost always differs from the matched root
        # (otherwise this would be an inplace rule). Inferred IR_FLAG_* bits
        # describe the OLD op; recompute from the new func's `efunc` effects
        # so downstream gates (CSE, LICM) see fresh, correct information.
        new_stmt = Expr(:call, rule.rhs.func, operands...)
        if rule.rhs.func === driver.defs[val].func
            block[val.id] = (stmt=new_stmt,)
        else
            block[val.id] = (stmt=new_stmt, flag=inferred_flags(rule.rhs.func))
        end
        # Update defs, re-add self and users to worklist (statement changed)
        driver.defs[val] = DefEntry(block, val, rule.rhs.func)
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

    driver = RewriteDriver(sci, defs, dispatch, wl, constants, Set{SSAValue}(), max_rewrites)

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
                erase_op!(driver, entry)
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
            for inst in users(driver.sci.entry, val)
                uv = SSAValue(inst)
                push!(driver.modified, uv)
                haskey(driver.defs, uv) && push!(driver.worklist, uv)
            end
        end
    end
end
