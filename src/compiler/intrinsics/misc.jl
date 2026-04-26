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

# XXX: cuda_tile.assume
# make this a pass?
function emit_assume_ops!(ctx::CGCtx, array_val::Value, size_vals::Vector{Value},
                          stride_vals::Vector{Value}, array_spec::ArraySpec, dtype::TypeId, scalar_type::TypeId;
                          tv_strides::Union{Vector{Int64}, Nothing}=nothing)
    cb = ctx.cb
    tt = ctx.tt

    # Pointer alignment
    if array_spec.alignment > 0
        ptr_dtype = pointer_type!(tt, dtype)
        ptr_tile_type = tile_type!(tt, ptr_dtype, RowMajorShape(()))
        array_val = encode_AssumeOp!(cb, ptr_tile_type, array_val, DivBy(array_spec.alignment))
    end

    # Bounds assumes for sizes
    size_vals = Value[encode_AssumeOp!(cb, scalar_type, v, Bounded(0, nothing)) for v in size_vals]

    # Bounds assumes for strides - only for dynamic strides
    if tv_strides !== nothing
        stride_vals = Value[tv_strides[i] == DYNAMIC_SHAPE ?
                       encode_AssumeOp!(cb, scalar_type, v, Bounded(0, nothing)) : v
                       for (i, v) in enumerate(stride_vals)]
    else
        stride_vals = Value[encode_AssumeOp!(cb, scalar_type, v, Bounded(0, nothing)) for v in stride_vals]
    end

    # Divisibility assumes for sizes
    # ArraySpec fields are in Julia order; size_vals are in Tile IR order (reversed)
    ndim = length(size_vals)
    for (julia_i, div_by) in enumerate(array_spec.shape_div_by)
        tileir_i = ndim + 1 - julia_i  # Reverse index mapping
        if div_by > 0 && tileir_i <= length(size_vals)
            size_vals[tileir_i] = encode_AssumeOp!(cb, scalar_type, size_vals[tileir_i], DivBy(div_by))
        end
    end

    # Divisibility assumes for strides - only for dynamic strides
    for (julia_i, div_by) in enumerate(array_spec.stride_div_by)
        tileir_i = ndim + 1 - julia_i  # Reverse index mapping
        if div_by > 0 && tileir_i <= length(stride_vals)
            # Skip if this stride is static (not DYNAMIC_SHAPE)
            if tv_strides === nothing || tv_strides[tileir_i] == DYNAMIC_SHAPE
                stride_vals[tileir_i] = encode_AssumeOp!(cb, scalar_type, stride_vals[tileir_i], DivBy(div_by))
            end
        end
    end

    return array_val, size_vals, stride_vals
end

# cuda_tile.print_tko

# Format specifier inference for print_tko
function infer_format_specifier(::Type{T}) where T
    if T <: Union{Bool, Int8, Int16, Int32, UInt8, UInt16, UInt32}
        return "%d"
    elseif T <: Union{Int64, UInt64}
        return "%ld"
    elseif T <: AbstractFloat  # Float16, BFloat16, Float32, TFloat32, Float64
        return "%f"
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

    # Store result token for TokenResultNode
    # v13.2+ returns a token Value; v13.1 returns nothing (no token support)
    new_token = if result isa Value
        result
    else
        # Pre-13.2: create a fresh token to satisfy the token chain
        encode_MakeTokenOp!(cb, token_type)
    end
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
