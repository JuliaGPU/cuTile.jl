# Julia built-in intrinsics
#
# Handles Julia built-ins that survive into the StructuredIRCode and are NOT
# lowered by normalize_ir! (because they have no direct cuTile equivalent
# or are compile-time-only constructs).
#
# Julia Core.Intrinsics (add_int, sub_int, slt_int, etc.) and Core.ifelse /
# === are lowered to cuTile Intrinsics by normalize_ir! and should not
# appear here.

# built-in: tuple (ghost — no runtime representation)
emit_intrinsic!(ctx::CGCtx, ::typeof(Core.tuple), args) = nothing

# built-in: isa (compile-time type narrowing)
emit_intrinsic!(ctx::CGCtx, ::typeof(isa), args) = nothing

# built-in: donotdelete (keep-alive barrier — no Tile IR emission)
emit_intrinsic!(ctx::CGCtx, ::typeof(donotdelete), args) = nothing

# XXX: Tile constructor
function emit_intrinsic!(ctx::CGCtx, func::Type{<:Tile}, args)
    # Emit the scalar value
    source = emit_value!(ctx, args[1])

    # Get element type from the constructor type (Tile{T, S})
    # If func is fully parametric, extract T; otherwise infer from source
    elem_type = if func !== Tile && length(func.parameters) >= 1
        func.parameters[1]
    elseif source.constant !== nothing
        typeof(something(source.constant))
    else
        CC.widenconst(source.jltype)
    end

    # Return as 0D tile type with element type from the constructor
    result_jltype = Tile{elem_type, Tuple{}}
    CGVal(source.v, source.type_id, result_jltype, source.shape, nothing, source.constant, nothing)
end
