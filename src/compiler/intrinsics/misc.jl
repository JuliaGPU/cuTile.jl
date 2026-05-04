# miscellaneous intrinsics

"""
    Intrinsics.assert(cond::Tile{Bool}, message::String) -> Nothing

Element-wise runtime assertion that every entry of `cond` is `true`;
lowers to `cuda_tile.assert`.

Also invocable with a scalar `Bool` `cond`, promoted to a 0-D tile before
codegen. `message` must be a compile-time constant. The op is elided when
`cond` folds to `true` at compile time.
"""
@intrinsic assert(cond::Bool, message::String)
tfunc(𝕃, ::typeof(Intrinsics.assert), @nospecialize(cond), @nospecialize(message)) = Nothing
efunc(::typeof(Intrinsics.assert), effects::CC.Effects) =
    CC.Effects(effects; effect_free=CC.ALWAYS_FALSE)
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.assert), args)
    # Elide the AssertOp when the condition folds to `true`
    get_constant(ctx, args[1]) === Some(true) && return nothing

    cond = @something emit_value!(ctx, args[1]) throw(IRError("assert: cannot resolve condition"))
    message = @something get_constant(ctx, args[2]) throw(IRError("assert: requires constant message"))
    encode_AssertOp!(ctx.cb, cond.v, message)
    nothing  # no result value
end

#=============================================================================
 cuda_tile.assume

 A single SCI intrinsic mirroring Tile IR's single `AssumeOp` opcode,
 polymorphic over the predicate kind (`DivBy`, `Bounded`,
 `SameElements`). Returns its input value — a pure-data annotation,
 eliminated if downstream uses vanish.

 Consumer-driven assumes (`make_tensor_view`, `load_ptr_tko`,
 `store_ptr_tko`) are emitted directly to bytecode at codegen time —
 the chain comes from `op_predicates` / `arg_chain` (analysis/assume.jl)
 and the wrapping happens via `wrap_for` (intrinsics/views.jl), so they
 never materialise as `Intrinsics.assume` SCI ops. This intrinsic exists
 for hand-written user annotations and as the lattice-level shape the
 dataflow analyses recognise (so a future pass that does insert
 SCI-level assumes still composes correctly with divisibility/bounds).

 cuTile Python uses one IR op class per predicate (`AssumeDivBy`,
 `AssumeBounded`, …); we collapse to a single polymorphic intrinsic
 because Julia's structural equality on the predicate type lifts
 cleanly into the SCI signature for CSE — two
 `Intrinsics.assume(x, DivBy(16))` calls dedup naturally.

 (Pointer alignment is *not* a separate predicate: it's encoded as
 `DivBy(alignment)` on the pointer, the same way Tile IR / Python
 represent it.)
=============================================================================#

"""
    Intrinsics.assume(x, predicate::AssumePredicate) -> typeof(x)

Annotate `x` with `predicate` (one of `DivBy`, `Bounded`,
`SameElements`); lowers to `cuda_tile.assume`. `predicate` must be a
compile-time constant.
"""
@intrinsic assume(x, predicate)
tfunc(𝕃, ::typeof(Intrinsics.assume), @nospecialize(x), @nospecialize(predicate)) =
    CC.widenconst(x)
efunc(::typeof(Intrinsics.assume), effects::CC.Effects) =
    CC.Effects(effects; effect_free=CC.ALWAYS_FALSE)
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.assume), args)
    x = @something emit_value!(ctx, args[1]) throw(IRError("assume: cannot resolve value"))
    x.v === nothing && throw(IRError("assume: value must be materialized"))
    pred_c = get_constant(ctx, args[2])
    pred_c === nothing && throw(IRError("assume: predicate must be a compile-time constant"))
    pred = pred_c.value
    pred isa AssumePredicate ||
        throw(IRError("assume: predicate must be an AssumePredicate, got $(typeof(pred))"))
    new_val = encode_AssumeOp!(ctx.cb, x.type_id, x.v, pred)
    return CGVal(new_val, x.type_id, x.jltype, x.shape)
end

# cuda_tile.print_tko

# Format specifier inference for print_tko
function infer_format_specifier(::Type{T}) where T
    if T === Bool
        return "%d"
    elseif T <: Signed
        return sizeof(T) == 8 ? "%lld" : "%d"
    elseif T <: Unsigned
        return sizeof(T) == 8 ? "%llu" : "%u"
    elseif T <: AbstractFloat  # Float16, BFloat16, Float32, TFloat32, Float64
        return sizeof(T) == 8 ? "%lf" : "%f"
    else
        throw(IRError("print: unsupported element type $T"))
    end
end

# Escape literal `%` as `%%` for C printf format strings
escape_printf(s::String) = replace(s, "%" => "%%")

"""
    Intrinsics.print_tko(xs...) -> Nothing

Token-ordered formatted print to the device console; lowers to
`cuda_tile.print_tko`.

Each argument is either a compile-time constant (folded into the format
string, with `%` escaped to `%%`) or a runtime tile (which receives an
inferred `printf` specifier such as `%d`/`%ld`/`%f` based on its element
type). The token argument is appended by `token_order_pass!` and is not
part of the user-visible signature; on Tile IR < 13.2 a fresh token is
synthesised since the op did not yet return one.
"""
@intrinsic print_tko(xs...)
tfunc(𝕃, ::typeof(Intrinsics.print_tko), @nospecialize(args...)) = Nothing
efunc(::typeof(Intrinsics.print_tko), effects::CC.Effects) =
    CC.Effects(effects; effect_free=CC.ALWAYS_FALSE)
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.print_tko), args)
    cb = ctx.cb
    tt = ctx.tt

    # Extract input token from last arg (added by token_order_pass!)
    input_token = extract_token_arg!(ctx, args)

    # Build format string and collect tile operands
    format_parts = String[]
    tile_args = Value[]

    for arg in args
        c = get_constant(ctx, arg)
        if c !== nothing
            val = something(c)
            if val isa String
                push!(format_parts, escape_printf(val))
            elseif val isa Number
                push!(format_parts, escape_printf(string(val)))
            else
                throw(IRError("print: unsupported constant type $(typeof(val))"))
            end
        else
            tv = emit_value!(ctx, arg)
            tv === nothing && throw(IRError("print: cannot resolve argument"))
            jltype = CC.widenconst(tv.jltype)
            elem_type = jltype <: Tile ? eltype(jltype) : jltype
            push!(format_parts, infer_format_specifier(elem_type))
            push!(tile_args, tv.v)
        end
    end

    format_string = join(format_parts)
    token_type = Token(tt)

    result = encode_PrintTkoOp!(cb, token_type, tile_args;
                                 token=input_token, format_string)

    # v13.2+ returns a post-op token; v13.1 returns nothing, so forward the
    # input token to keep prior happens-after edges instead of synthesising a
    # fresh root that would discard them.
    new_token = result isa Value ? result : input_token
    ctx.result_tokens[ctx.current_ssa_idx] = new_token

    nothing  # print returns Nothing
end

"""
    Intrinsics.format_string(xs...) -> String

Placeholder for string-interpolation fusion: every call must be fused
into a `print_tko` by the print-fusion pass before codegen runs.
Reaching `emit_intrinsic!` for this op signals an unsupported standalone
`string()` with tile arguments and raises an `IRError`.

There is no Tile IR opcode for this intrinsic.
"""
@intrinsic format_string(xs...)
tfunc(𝕃, ::typeof(Intrinsics.format_string), @nospecialize(args...)) = String
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.format_string), args)
    throw(IRError("format_string intrinsic should have been fused into print_tko by the print fusion pass. " *
                  "Standalone string() with Tile arguments is not supported in cuTile kernels."))
end
