# Canonicalization Passes
#
# Imperative passes that canonicalize the SCI before optimization.

function canonicalize!(sci::StructuredIRCode)
    scalar_elim_pass!(sci)
    lower_intr_pass!(sci)
end


#=============================================================================
 Intrinsics lowering
=============================================================================#

# Lowers Julia Core intrinsics and builtins to cuTile Intrinsics.

const INTRINSIC_RULES = RewriteRule[
    # Integer arithmetic
    @rewrite Core.Intrinsics.add_int(~x, ~y) => Intrinsics.addi(~x, ~y)
    @rewrite Core.Intrinsics.sub_int(~x, ~y) => Intrinsics.subi(~x, ~y)
    @rewrite Core.Intrinsics.mul_int(~x, ~y) => Intrinsics.muli(~x, ~y)
    @rewrite Core.Intrinsics.neg_int(~x)     => Intrinsics.negi(~x)

    # Integer comparison
    @rewrite Core.Intrinsics.slt_int(~x, ~y) =>
            Intrinsics.cmpi(~x, ~y, $(ComparisonPredicate.LessThan), $(Signedness.Signed))
    @rewrite Core.Intrinsics.sle_int(~x, ~y) =>
            Intrinsics.cmpi(~x, ~y, $(ComparisonPredicate.LessThanOrEqual), $(Signedness.Signed))
    @rewrite Core.Intrinsics.ult_int(~x, ~y) =>
            Intrinsics.cmpi(~x, ~y, $(ComparisonPredicate.LessThan), $(Signedness.Unsigned))

    # Bitwise
    @rewrite Core.Intrinsics.and_int(~x, ~y) => Intrinsics.andi(~x, ~y)
    @rewrite Core.Intrinsics.or_int(~x, ~y)  => Intrinsics.ori(~x, ~y)
    @rewrite Core.Intrinsics.xor_int(~x, ~y) => Intrinsics.xori(~x, ~y)

    # not_int: xori with all-ones constant (type-dependent)
    @rewrite Core.Intrinsics.not_int(~x::Tile{Bool})   => Intrinsics.xori(~x, $(true))
    @rewrite Core.Intrinsics.not_int(~x::Tile{Int32})  => Intrinsics.xori(~x, $(Int32(-1)))
    @rewrite Core.Intrinsics.not_int(~x::Tile{Int64})  => Intrinsics.xori(~x, $(Int64(-1)))
    @rewrite Core.Intrinsics.not_int(~x::Tile{UInt32}) => Intrinsics.xori(~x, $(~UInt32(0)))
    @rewrite Core.Intrinsics.not_int(~x::Tile{UInt64}) => Intrinsics.xori(~x, $(~UInt64(0)))

    # Float arithmetic
    @rewrite Core.Intrinsics.add_float(~x, ~y) => Intrinsics.addf(~x, ~y)
    @rewrite Core.Intrinsics.sub_float(~x, ~y) => Intrinsics.subf(~x, ~y)
    @rewrite Core.Intrinsics.mul_float(~x, ~y) => Intrinsics.mulf(~x, ~y)
    @rewrite Core.Intrinsics.div_float(~x, ~y) => Intrinsics.divf(~x, ~y)
    @rewrite Core.Intrinsics.neg_float(~x)     => Intrinsics.negf(~x)

    # Float comparison
    @rewrite Core.Intrinsics.lt_float(~x, ~y) =>
            Intrinsics.cmpf(~x, ~y, $(ComparisonPredicate.LessThan))
    @rewrite Core.Intrinsics.le_float(~x, ~y) =>
            Intrinsics.cmpf(~x, ~y, $(ComparisonPredicate.LessThanOrEqual))
    @rewrite Core.Intrinsics.eq_float(~x, ~y) =>
            Intrinsics.cmpf(~x, ~y, $(ComparisonPredicate.Equal))
    @rewrite Core.Intrinsics.ne_float(~x, ~y) =>
            Intrinsics.cmpf(~x, ~y, $(ComparisonPredicate.NotEqual), $(ComparisonOrdering.Unordered))

    # Bitcast (reinterpret): Core.Intrinsics.bitcast(T, x) → Intrinsics.bitcast(x, T)
    @rewrite Core.Intrinsics.bitcast(~T, ~x) => Intrinsics.bitcast(~x, ~T)

    # Builtins
    @rewrite (===)(~x, ~y) =>
            Intrinsics.cmpi(~x, ~y, $(ComparisonPredicate.Equal), $(Signedness.Signed))
    @rewrite Core.ifelse(~c, ~x, ~y) => Intrinsics.select(~c, ~x, ~y)
]

lower_intr_pass!(sci::StructuredIRCode) = rewrite_patterns!(sci, INTRINSIC_RULES)


#=============================================================================
 Scalar Elimination
=============================================================================#

# Replaces scalar SSA values from the SCI, making the IR uniformly tile-typed.
# This also removes `to_scalar` and `from_scalar` intrinsics from the IR.
#
# For to_scalar(x): replaces all uses with x, updates types to x's tile type.
# For from_scalar(x, S): replaces all uses with x, updates types. Then deletes
# the instruction.
#
# This eliminates the scalar/tile duality that complicates codegen and pattern
# matching. Scalars now only can exist as literal values embedded in the IR.

function scalar_elim_pass!(sci::StructuredIRCode)
    scalar_elim_block!(sci.entry)
end

function scalar_elim_block!(block::Block)
    # Recurse into nested control flow first
    for inst in instructions(block)
        s = stmt(inst)
        if s isa ForOp
            scalar_elim_block!(s.body)
        elseif s isa IfOp
            scalar_elim_block!(s.then_region)
            scalar_elim_block!(s.else_region)
        elseif s isa WhileOp
            scalar_elim_block!(s.before)
            scalar_elim_block!(s.after)
        elseif s isa LoopOp
            scalar_elim_block!(s.body)
        end
    end

    # Phase 1: Eliminate to_scalar (forward tile-typed operand)
    to_delete = Instruction[]
    for inst in instructions(block)
        call = resolve_call(block, stmt(inst))
        call === nothing && continue
        func, ops = call
        func === Intrinsics.to_scalar || continue
        replace_uses!(block, SSAValue(inst), ops[1])
        push!(to_delete, inst)
    end

    # Phase 2: Propagate tile types from operands to results.
    # Instructions that consumed to_scalar results still have scalar type
    # annotations but now receive tile-typed operands — inherit their shape.
    for inst in instructions(block)
        call = resolve_call(block, stmt(inst))
        call === nothing && continue
        func, ops = call

        # isa is a scalar type check, not a tile operation — its result shape
        # should not inherit from operands. (On Julia nightly, InferenceCache
        # interactions can leave isa unresolved in the IR; see _combine_masks.)
        func === isa && continue

        current_type = value_type(inst)
        current_type === nothing && continue
        is_token_type(current_type) && continue
        T = CC.widenconst(current_type)
        T <: Tile && continue     # already tile-typed
        T <: Number || continue   # only promote scalar number types

        for op in ops
            op_type = value_type(block, op)
            op_type === nothing && continue
            is_token_type(op_type) && continue
            OT = CC.widenconst(op_type)
            OT <: Tile || continue
            S = OT.parameters[2]
            update_type!(block, inst, Tile{T, S})
            break
        end
    end

    # Phase 3: Eliminate from_scalar (operand now has correct tile type)
    for inst in instructions(block)
        call = resolve_call(block, stmt(inst))
        call === nothing && continue
        func, ops = call
        func === Intrinsics.from_scalar || continue
        replace_uses!(block, SSAValue(inst), ops[1])
        push!(to_delete, inst)
    end

    for inst in to_delete
        delete!(block, inst)
    end

    # Phase 4: Promote any remaining scalar Number types to 0D tiles.
    # This catches intrinsics that natively return scalars (get_tile_block_id,
    # item, etc.) and any scalar arithmetic on them (addi of two block IDs).
    # Tuple types (from IfOp/LoopOp results) have their elements promoted.
    for inst in instructions(block)
        current_type = value_type(inst)
        current_type === nothing && continue
        is_token_type(current_type) && continue
        new_type = promote_scalar_type(CC.widenconst(current_type))
        new_type === nothing && continue
        update_type!(block, inst, new_type)
    end

    # Phase 5: Promote block argument types (loop IVs, carries).
    # BlockArgument is immutable, so we create a new one and replace all uses.
    for (i, arg) in enumerate(block.args)
        is_token_type(arg.type) && continue
        T = CC.widenconst(arg.type)
        T <: Tile && continue
        T <: Number || continue
        new_arg = BlockArgument(arg.id, Tile{T, Tuple{}})
        replace_uses!(block, arg, new_arg)
        block.args[i] = new_arg
    end
end

"""Promote scalar Number types to 0D Tile types. Returns the promoted type,
or `nothing` if no promotion needed. Also promotes Tuple element types."""
function promote_scalar_type(@nospecialize(T))
    T <: Number && return Tile{T, Tuple{}}
    if T <: Tuple
        params = T.parameters
        any_promoted = false
        new_params = map(params) do P
            P = CC.widenconst(P)
            if P <: Number && !(P <: Tile)
                any_promoted = true
                Tile{P, Tuple{}}
            else
                P
            end
        end
        any_promoted || return nothing
        return Tuple{new_params...}
    end
    return nothing
end
