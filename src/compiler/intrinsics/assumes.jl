# Assume intrinsics
#
# IR-level annotation intrinsics whose emit_intrinsic! wraps the underlying
# Tile IR value with an `AssumeOp` (with a `DivBy` attribute). Inserted by the
# `insert_divby_assumes!` pass so that divisibility annotations live in the IR
# as first-class operations rather than as codegen-time side tables.
#
# Mirrors cuTile Python's `AssumeDivBy` op inserted by `propagate_divby.py`.

# Intrinsics.assume_div_by(x, divisor::Val{D}) -> typeof(x)
#
# Pass-through wrapper that annotates `x` as being divisible by `D`.
# At codegen, lowers to `encode_AssumeOp!(cb, t, x_val, DivBy(D))`.
# Inserted automatically by `insert_divby_assumes!`.
@intrinsic assume_div_by(x, divisor::Val)

function tfunc(𝕃, ::typeof(Intrinsics.assume_div_by),
               @nospecialize(x), @nospecialize(divisor))
    return CC.widenconst(x)
end

# assume_div_by is side-effect-free — enable DCE when unused.
efunc(::typeof(Intrinsics.assume_div_by), effects::CC.Effects) =
    CC.Effects(effects; effect_free=CC.ALWAYS_TRUE)

function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.assume_div_by), args)
    cb = ctx.cb

    length(args) >= 2 ||
        throw(IRError("assume_div_by: expected 2 args, got $(length(args))"))

    x_tv = emit_value!(ctx, args[1])
    x_tv === nothing && throw(IRError("assume_div_by: cannot resolve value"))

    d_const = get_constant(ctx, args[2])
    d_const === nothing && throw(IRError("assume_div_by: divisor must be Val{D}"))
    d = unwrap_val_int(d_const)
    d isa Integer || throw(IRError("assume_div_by: expected Val{Int}, got $(typeof(d_const))"))
    d > 0 || throw(IRError("assume_div_by: divisor must be positive, got $d"))

    wrapped = encode_AssumeOp!(cb, x_tv.type_id, x_tv.v, DivBy(Int(d)))
    return CGVal(wrapped, x_tv.type_id, x_tv.jltype, x_tv.shape,
                 x_tv.arg_ref, x_tv.constant, x_tv.tuple)
end

"""Unwrap a `Val{D}` constant or its type to the underlying integer D."""
function unwrap_val_int(@nospecialize(v))
    if v isa Val
        return first(typeof(v).parameters)
    elseif v isa Type && v <: Val
        return first(v.parameters)
    end
    return nothing
end

"""
    assume_divby(ssa) -> Int

Peek through a wrapping `Intrinsics.assume_div_by(ssa, Val(D))` to recover D.
Returns 1 if `ssa` is not wrapped. Used by emitters that need to preserve
divisibility on derived values (e.g. slice's internal OffsetOp/SubIOp results)
without depending on external analysis state.
"""
function assume_divby(ctx::CGCtx, @nospecialize(op))::Int
    op isa SSAValue || return 1
    # Look up the def in the sci IR.
    d = find_assume_div_by(ctx.sci, op)
    return d === nothing ? 1 : d
end

"""Find an Intrinsics.assume_div_by wrapping the given SSAValue; return D or nothing."""
function find_assume_div_by(sci::StructuredIRCode, ssa::SSAValue)
    return walk_find_divby(sci.entry, ssa.id)
end

function walk_find_divby(block::Block, target_id::Int)
    for inst in instructions(block)
        s = stmt(inst)
        if s isa ControlFlowOp
            for sub in blocks(s)
                d = walk_find_divby(sub, target_id)
                d === nothing || return d
            end
        elseif inst.ssa_idx == target_id
            call = resolve_call(block, inst)
            call === nothing && return nothing
            func, ops = call
            func === Intrinsics.assume_div_by || return nothing
            length(ops) >= 2 || return nothing
            d = unwrap_val_int(ops[2])
            return d isa Integer ? Int(d) : nothing
        end
    end
    return nothing
end
