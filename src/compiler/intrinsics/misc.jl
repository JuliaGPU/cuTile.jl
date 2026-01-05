#=============================================================================
 Miscellaneous Intrinsics
 getfield/getindex for argument destructuring, scalar comparisons, no-ops
=============================================================================#

#-----------------------------------------------------------------------------
# No-op dispatches
#-----------------------------------------------------------------------------

# Skip tuple construction
emit_intrinsic!(ctx::CodegenContext, ::typeof(Core.tuple), args, @nospecialize(result_type)) = nothing

# Skip isa type assertions (inserted by Julia during inlining)
emit_intrinsic!(ctx::CodegenContext, ::typeof(isa), args, @nospecialize(result_type)) = nothing

#-----------------------------------------------------------------------------
# getfield for destructured arguments (lazy chain extension)
#-----------------------------------------------------------------------------

function emit_intrinsic!(ctx::CodegenContext, ::typeof(Base.getfield), args, @nospecialize(result_type))
    length(args) >= 2 || return nothing

    obj_arg = args[1]
    field_arg = args[2]

    # Extract field name or index
    field = get_constant(ctx, field_arg)

    # Try to get the object as a CGVal
    obj_tv = emit_value!(ctx, obj_arg)

    # If obj is a lazy arg_ref, extend the chain
    if obj_tv !== nothing && is_arg_ref(obj_tv)
        arg_idx, chain = obj_tv.arg_ref

        if field isa Symbol
            # Field access: extend chain with symbol
            new_chain = Union{Symbol, Int}[chain..., field]
            # Check if this resolves to a scalar field (auto-materialize leaf)
            # Don't auto-materialize tuple types - they need indexing first
            rt = unwrap_type(result_type)
            if !(rt <: Tuple)
                values = get_arg_flat_values(ctx, arg_idx, field)
                if values !== nothing && length(values) == 1
                    # Scalar field - materialize immediately
                    type_id = tile_type_for_julia!(ctx, rt)
                    return CGVal(values[1], type_id, rt)
                end
            end
            return arg_ref_value(arg_idx, new_chain, rt)
        elseif field isa Integer && !isempty(chain) && chain[end] isa Symbol
            # Tuple indexing: chain ends with field name, now indexing into it
            # This is a leaf - materialize immediately
            field_name = chain[end]
            values = get_arg_flat_values(ctx, arg_idx, field_name)
            if values !== nothing && 1 <= field <= length(values)
                type_id = tile_type_for_julia!(ctx, unwrap_type(result_type))
                return CGVal(values[field], type_id, unwrap_type(result_type))
            end
        end
    end

    nothing
end

#-----------------------------------------------------------------------------
# getindex for tuple field access (lazy chain extension)
#-----------------------------------------------------------------------------

function emit_intrinsic!(ctx::CodegenContext, ::typeof(Base.getindex), args, @nospecialize(result_type))
    length(args) >= 2 || return nothing

    obj_arg = args[1]
    index_arg = args[2]

    # Extract constant index
    index = get_constant(ctx, index_arg)
    index isa Integer || return nothing

    # Try to get the object as a CGVal
    obj_tv = emit_value!(ctx, obj_arg)
    obj_tv === nothing && return nothing

    # If obj is a lazy arg_ref, try to materialize or extend the chain
    if is_arg_ref(obj_tv)
        arg_idx, chain = obj_tv.arg_ref

        # If chain ends with a symbol (field name), we're indexing into a tuple field
        # Try to materialize immediately
        if !isempty(chain) && chain[end] isa Symbol
            field_name = chain[end]
            values = get_arg_flat_values(ctx, arg_idx, field_name)
            if values !== nothing && 1 <= index <= length(values)
                type_id = tile_type_for_julia!(ctx, unwrap_type(result_type))
                return CGVal(values[index], type_id, unwrap_type(result_type))
            end
        end

        # Otherwise extend the chain
        new_chain = Union{Symbol, Int}[chain..., Int(index)]
        return arg_ref_value(arg_idx, new_chain, unwrap_type(result_type))
    end

    # Not an arg_ref - not handled here
    nothing
end

#-----------------------------------------------------------------------------
# Scalar comparison operators
#-----------------------------------------------------------------------------

function emit_cmp!(ctx::CodegenContext, args, predicate::ComparisonPredicate)
    emit_int_cmp!(ctx, args, predicate, SignednessSigned)
end

function emit_int_cmp!(ctx::CodegenContext, args, predicate::ComparisonPredicate, signedness::Signedness)
    cb = ctx.cb
    tt = ctx.tt

    lhs = emit_value!(ctx, args[1])
    rhs = emit_value!(ctx, args[2])

    lhs === nothing && error("Cannot resolve LHS operand for comparison")
    rhs === nothing && error("Cannot resolve RHS operand for comparison")

    # Result type is a boolean (i1) scalar
    result_type = tile_type!(tt, I1(tt), Int[])

    lhs_v = lhs isa CGVal ? lhs.v : lhs
    rhs_v = rhs isa CGVal ? rhs.v : rhs

    result_v = encode_CmpIOp!(cb, result_type, lhs_v, rhs_v;
                              predicate=predicate, signedness=signedness)

    CGVal(result_v, result_type, Bool, Int[])
end

emit_intrinsic!(ctx::CodegenContext, ::typeof(Base.:(>)), args, @nospecialize(_)) =
    emit_cmp!(ctx, args, CmpGreaterThan)

emit_intrinsic!(ctx::CodegenContext, ::typeof(Base.:(<)), args, @nospecialize(_)) =
    emit_cmp!(ctx, args, CmpLessThan)

emit_intrinsic!(ctx::CodegenContext, ::typeof(Base.:(>=)), args, @nospecialize(_)) =
    emit_cmp!(ctx, args, CmpGreaterThanOrEqual)

emit_intrinsic!(ctx::CodegenContext, ::typeof(Base.:(<=)), args, @nospecialize(_)) =
    emit_cmp!(ctx, args, CmpLessThanOrEqual)

emit_intrinsic!(ctx::CodegenContext, ::typeof(===), args, @nospecialize(_)) =
    emit_cmp!(ctx, args, CmpEqual)

emit_intrinsic!(ctx::CodegenContext, ::typeof(Base.:(==)), args, @nospecialize(_)) =
    emit_cmp!(ctx, args, CmpEqual)

emit_intrinsic!(ctx::CodegenContext, ::typeof(Base.:(!=)), args, @nospecialize(_)) =
    emit_cmp!(ctx, args, CmpNotEqual)
