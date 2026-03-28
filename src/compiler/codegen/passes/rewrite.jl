# Declarative IR Rewrite Pattern Framework
#
# Inspired by MLIR's PDLL. Patterns compile into pattern/rewrite node trees.
# The framework handles matching (recursive SSA def-chain walking) and rewrite
# application. Cleanup of dead code is delegated to the pipeline's dce_pass!.
#
# Usage:
#   rules = RewriteRule[
#       @rewrite addf(one_use(mulf(~x, ~y)), ~z) => fma(~x, ~y, ~z)
#       @rewrite subf(one_use(mulf(~x, ~y)), ~z) => fma(~x, ~y, negf(~z))
#   ]
#   rewrite_patterns!(sci, rules)

using Core: SSAValue

#=============================================================================
 Pattern & Rewrite Nodes
=============================================================================#

abstract type PatternNode end
struct PCall <: PatternNode; func_name::Symbol; operands::Vector{PatternNode}; end
struct PBind <: PatternNode; name::Symbol; end
struct POneUse <: PatternNode; inner::PatternNode; end

abstract type RewriteNode end
struct RCall <: RewriteNode; func_name::Symbol; operands::Vector{RewriteNode}; end
struct RBind <: RewriteNode; name::Symbol; end

struct RewriteRule; lhs::PCall; rhs::RewriteNode; end

resolve_func(pat::PCall) = getfield(Intrinsics, pat.func_name)
root_func(rule::RewriteRule) = resolve_func(rule.lhs)

#=============================================================================
 @rewrite Macro
=============================================================================#

"""
    @rewrite lhs => rhs

Compile a declarative rewrite rule. LHS: `func(args...)` matches calls,
`~x` binds, `one_use(pat)` requires single use. RHS: `func(args...)` emits
calls, `~x` references bindings.
"""
macro rewrite(ex)
    ex isa Expr && ex.head === :call && ex.args[1] === :(=>) ||
        error("@rewrite expects: lhs => rhs")
    esc(:(RewriteRule($(_compile_lhs(ex.args[2])), $(_compile_rhs(ex.args[3])))))
end

function _compile_lhs(ex)
    ex isa Expr && ex.head === :call || error("@rewrite LHS: expected call, got $ex")
    f = ex.args[1]
    f === :~ && return :(PBind($(QuoteNode(ex.args[2]))))
    f === :one_use && return :(POneUse($(_compile_lhs(ex.args[2]))))
    :(PCall($(QuoteNode(f)), PatternNode[$(_compile_lhs.(ex.args[2:end])...)]))
end

function _compile_rhs(ex)
    ex isa Expr && ex.head === :call || error("@rewrite RHS: expected call, got $ex")
    f = ex.args[1]
    f === :~ && return :(RBind($(QuoteNode(ex.args[2]))))
    :(RCall($(QuoteNode(f)), RewriteNode[$(_compile_rhs.(ex.args[2:end])...)]))
end

#=============================================================================
 Matching
=============================================================================#

struct MatchResult
    bindings::Dict{Symbol, Any}
    matched_ssas::Vector{Int}
end

struct DefEntry
    block::Block
    inst::Instruction
    func::Any
    operands::Vector{Any}
end

struct MatchContext
    defs::Dict{Int, DefEntry}
    use_index  # UseIndex from IRStructurizer
end

function MatchContext(sci::StructuredIRCode)
    defs = Dict{Int, DefEntry}()
    for block in eachblock(sci)
        for inst in instructions(block)
            call = resolve_call(inst)
            call === nothing && continue
            func, operands = call
            defs[inst.ssa_idx] = DefEntry(block, inst, func, collect(Any, operands))
        end
    end
    MatchContext(defs, uses(sci.entry))
end

_use_count(ctx::MatchContext, val::SSAValue) =
    haskey(ctx.use_index, val) ? length(ctx.use_index[val]) : 0

# Codegen no-ops that pattern matching traces through transparently.
_is_transparent(func) = func === Intrinsics.to_scalar ||
                         func === Intrinsics.from_scalar ||
                         func === Intrinsics.broadcast

function pattern_match(ctx::MatchContext, @nospecialize(val), pat::PCall)
    val isa SSAValue || return nothing
    entry = get(ctx.defs, val.id, nothing)
    entry === nothing && return nothing

    target = resolve_func(pat)
    if entry.func === target && length(entry.operands) == length(pat.operands)
        result = MatchResult(Dict{Symbol,Any}(), Int[val.id])
        for (op, sub) in zip(entry.operands, pat.operands)
            m = pattern_match(ctx, op, sub)
            m === nothing && return nothing
            merge!(result.bindings, m.bindings)
            append!(result.matched_ssas, m.matched_ssas)
        end
        return result
    end

    # Trace through single-use transparent ops to find the underlying operation
    if _is_transparent(entry.func) && !isempty(entry.operands)
        _use_count(ctx, val) == 1 || return nothing
        if entry.func === Intrinsics.broadcast
            inner = entry.operands[1]
            if inner isa SSAValue
                inner_entry = get(ctx.defs, inner.id, nothing)
                if inner_entry !== nothing
                    it, ot = value_type(inner_entry.inst), value_type(entry.inst)
                    it <: Tile && ot <: Tile || return nothing
                    size(it) == size(ot) || return nothing
                end
            end
        end
        result = pattern_match(ctx, entry.operands[1], pat)
        result === nothing && return nothing
        push!(result.matched_ssas, val.id)
        return result
    end
    return nothing
end

pattern_match(ctx::MatchContext, @nospecialize(val), pat::PBind) =
    MatchResult(Dict{Symbol,Any}(pat.name => val), Int[])

function pattern_match(ctx::MatchContext, @nospecialize(val), pat::POneUse)
    val isa SSAValue && _use_count(ctx, val) == 1 || return nothing
    pattern_match(ctx, val, pat.inner)
end

#=============================================================================
 Rewrite Application
=============================================================================#

"""Resolve an RHS operand, inserting sub-calls before `ref` as needed."""
_resolve_rhs(block, ref, op::RBind, bindings, typ) = bindings[op.name]
function _resolve_rhs(block, ref, op::RCall, bindings, typ)
    operands = Any[_resolve_rhs(block, ref, sub, bindings, typ) for sub in op.operands]
    SSAValue(insert_before!(block, ref, Expr(:call, GlobalRef(Intrinsics, op.func_name),
                                              operands...), typ))
end

function _apply_rewrite!(sci, block, inst, rule, match, ctx, consumed)
    if rule.rhs isa RBind
        # Forwarding: replace all uses of root with the bound value, delete root
        replace_uses!(sci.entry, SSAValue(inst), match.bindings[rule.rhs.name])
        delete!(block, inst)
        _cleanup_dead_operands!(sci, inst.ssa_idx, ctx, consumed)
    else
        # Substitution: delete matched intermediates, replace root statement in-place.
        # No operand cleanup here — the new replacement may reference these operands.
        for dead_ssa in match.matched_ssas
            dead_ssa == inst.ssa_idx && continue
            entry = ctx.defs[dead_ssa]
            delete!(entry.block, entry.inst)
        end
        typ = value_type(inst)
        operands = Any[_resolve_rhs(block, SSAValue(inst), op, match.bindings, typ)
                       for op in rule.rhs.operands]
        pos = findfirst(==(inst.ssa_idx), block.body.ssa_idxes)
        block.body.stmts[pos] = Expr(:call, GlobalRef(Intrinsics, rule.rhs.func_name),
                                      operands...)
    end
end

"""Recursively erase dead pure operands after an instruction is deleted (MLIR-style)."""
function _cleanup_dead_operands!(sci, ssa_idx, ctx, consumed)
    entry = get(ctx.defs, ssa_idx, nothing)
    entry === nothing && return
    for op in entry.operands
        op isa SSAValue || continue
        op_entry = get(ctx.defs, op.id, nothing)
        op_entry === nothing && continue
        _is_transparent(op_entry.func) || continue
        op.id in consumed && continue
        isempty(uses(sci.entry, op)) || continue
        push!(consumed, op.id)
        delete!(op_entry.block, op_entry.inst)
        _cleanup_dead_operands!(sci, op.id, ctx, consumed)
    end
end

#=============================================================================
 Driver
=============================================================================#

"""
    rewrite_patterns!(sci::StructuredIRCode, rules::Vector{RewriteRule})

Apply declarative rewrite rules to the structured IR. Dead code left behind
is cleaned up by the pipeline's `dce_pass!`.
"""
function rewrite_patterns!(sci::StructuredIRCode, rules::Vector{RewriteRule})
    ctx = MatchContext(sci)
    dispatch = Dict{Any, Vector{RewriteRule}}()
    for rule in rules
        push!(get!(dispatch, root_func(rule), RewriteRule[]), rule)
    end

    consumed = Set{Int}()
    for block in eachblock(sci)
        for inst in collect(instructions(block))
            inst.ssa_idx in consumed && continue
            call = resolve_call(inst)
            call === nothing && continue
            applicable = get(dispatch, call[1], nothing)
            applicable === nothing && continue
            for rule in applicable
                m = pattern_match(ctx, SSAValue(inst), rule.lhs)
                m === nothing && continue
                any(s in consumed for s in m.matched_ssas) && continue
                _apply_rewrite!(sci, block, inst, rule, m, ctx, consumed)
                union!(consumed, m.matched_ssas)
                break
            end
        end
    end
end
