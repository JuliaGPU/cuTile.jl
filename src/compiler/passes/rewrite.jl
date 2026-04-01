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

abstract type RewriteNode end
struct RCall <: RewriteNode; func::Any; operands::Vector{RewriteNode}; end
struct RBind <: RewriteNode; name::Symbol; end
struct RConst <: RewriteNode; val::Any; end

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
    pos = findfirst(==(entry.val.id), entry.block.body.ssa_idxes)
    pos === nothing && return Any[]
    call = resolve_call(entry.block, entry.block.body.stmts[pos])
    call === nothing && return Any[]
    _, ops = call
    return ops
end

mutable struct RewriteDriver
    sci::StructuredIRCode
    defs::Dict{SSAValue, DefEntry}
    dispatch::Dict{Any, Vector{RewriteRule}}
    worklist::Worklist
    constants::Dict{SSAValue, Any}   # SSA → constant value (from propagate_constants)
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
    pos = findfirst(==(entry.val.id), entry.block.body.ssa_idxes)
    if pos !== nothing
        deleteat!(entry.block.body.ssa_idxes, pos)
        deleteat!(entry.block.body.stmts, pos)
        deleteat!(entry.block.body.types, pos)
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
        if length(ops) == length(pat.operands)
            result = MatchResult(Dict{Symbol,Any}(), SSAValue[val])
            for (op, sub) in zip(ops, pat.operands)
                m = pattern_match(driver, op, sub, entry.block)
                m === nothing && return nothing
                merge_bindings!(result.bindings, m.bindings) || return nothing
                append!(result.matched_ssas, m.matched_ssas)
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
# For SSA operands: O(1) lookup in the constants map built by propagate_constants.
function pattern_match(driver::RewriteDriver, @nospecialize(val), pat::PLiteral,
                       block::Block=driver.sci.entry)
    val === pat.val && return MatchResult(Dict{Symbol,Any}(), SSAValue[])
    if val isa SSAValue
        c = get(driver.constants, val, nothing)
        if c isa AbstractArray
            all(==(pat.val), c) && return MatchResult(Dict{Symbol,Any}(), SSAValue[])
        elseif c !== nothing
            c == pat.val && return MatchResult(Dict{Symbol,Any}(), SSAValue[])
        end
    end
    return nothing
end

#=============================================================================
 Rewrite Application
=============================================================================#

"""Resolve an RHS operand, inserting sub-calls before `ref` as needed."""
resolve_rhs(driver, block, ref, op::RBind, bindings, typ) = bindings[op.name]
resolve_rhs(driver, block, ref, op::RConst, bindings, typ) = op.val
function resolve_rhs(driver::RewriteDriver, block, ref, op::RCall, bindings, typ)
    operands = Any[resolve_rhs(driver, block, ref, sub, bindings, typ) for sub in op.operands]
    inst = insert_before!(block, ref, Expr(:call, op.func, operands...), typ)
    notify_insert!(driver, block, inst)
    SSAValue(inst)
end

"""
In-place rewrite: walk LHS/RHS trees in parallel, modifying matched ops' operands.
No new instructions are created — existing ops are modified in-place.
"""
function apply_inplace_rewrite!(driver::RewriteDriver, block, val::SSAValue, rule, match)
    pos = findfirst(==(val.id), block.body.ssa_idxes)
    pos === nothing && return false

    # Build new operands for the root op from the RHS
    new_operands = Any[resolve_inplace_rhs(driver, match.bindings, op, lhs_op)
                       for (op, lhs_op) in zip(rule.rhs.operands, rule.lhs.operands)]
    block.body.stmts[pos] = Expr(:call, rule.rhs.func, new_operands...)
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
    epos = @something findfirst(==(matched_ssa.id), entry.block.body.ssa_idxes) error(
        "inplace rewrite: $matched_ssa not found in block")
    new_ops = Any[resolve_inplace_rhs(driver, bindings, sub_rhs, sub_lhs)
                  for (sub_rhs, sub_lhs) in zip(op.operands, lhs_op.operands)]
    entry.block.body.stmts[epos] = Expr(:call, op.func, new_ops...)
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
        pos = findfirst(==(val.id), block.body.ssa_idxes)
        pos === nothing && return false
        inst = Instruction(val.id, block.body.stmts[pos], block.body.types[pos])
        rule.rhs.func(driver.sci, block, inst, match, driver) || return false
        return true
    elseif rule.rhs isa RBind
        # Forwarding: replace all uses of root with the bound value, delete root.
        # Collect users BEFORE replace_uses! updates their operands.
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
        pos = findfirst(==(val.id), block.body.ssa_idxes)
        typ = block.body.types[pos]
        operands = Any[resolve_rhs(driver, block, val, op, match.bindings, typ)
                       for op in rule.rhs.operands]
        block.body.stmts[pos] = Expr(:call, rule.rhs.func, operands...)
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
                           constants::Dict{SSAValue, Any}=Dict{SSAValue, Any}())
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

    driver = RewriteDriver(sci, defs, dispatch, wl, constants, max_rewrites)

    num_rewrites = 0
    while !isempty(driver.worklist) && num_rewrites < driver.max_rewrites
        val = pop!(driver.worklist)::SSAValue
        entry = get(driver.defs, val, nothing)
        entry === nothing && continue

        # Verify instruction is still live in its block
        pos = findfirst(==(val.id), entry.block.body.ssa_idxes)
        pos === nothing && begin
            delete!(driver.defs, val)
            continue
        end

        # Trivial dead-op elimination: if this op has no uses and is pure,
        # erase it. This keeps use counts accurate for `one_use` patterns
        # (e.g., FMA fusion needs mulf's dead transparent-op users removed
        # so the mulf reads as single-use). Full DCE handles the rest.
        if use_count(driver, val) == 0
            stmt = entry.block.body.stmts[pos]
            if !must_keep(entry.block, stmt)
                erase_op!(driver, entry)
                continue
            end
        end

        # Look up applicable rules by function
        applicable = get(driver.dispatch, entry.func, nothing)
        applicable === nothing && continue

        for rule in applicable
            m = pattern_match(driver, val, rule.lhs)
            m === nothing && continue
            rule.guard !== nothing && !rule.guard(m, driver) && continue
            apply_rewrite!(driver, entry.block, val, rule, m)
            num_rewrites += 1
            break
        end
    end
end
