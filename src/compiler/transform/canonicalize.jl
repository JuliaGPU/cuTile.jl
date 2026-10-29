# Canonicalization Passes
#
# Imperative passes that canonicalize the SCI before optimization.

function canonicalize!(sci::StructuredIRCode)
    scalar_elim_pass!(sci)
    lower_intr_pass!(sci)
    if_convert!(sci)
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
    @rewrite Core.Intrinsics.ule_int(~x, ~y) =>
            Intrinsics.cmpi(~x, ~y, $(ComparisonPredicate.LessThanOrEqual), $(Signedness.Unsigned))

    # Bitwise
    @rewrite Core.Intrinsics.and_int(~x, ~y) => Intrinsics.andi(~x, ~y)
    @rewrite Core.Intrinsics.or_int(~x, ~y)  => Intrinsics.ori(~x, ~y)
    @rewrite Core.Intrinsics.xor_int(~x, ~y) => Intrinsics.xori(~x, ~y)

    # not_int: xori with all-ones constant (type-dependent). Unlike the binary
    # bitwise ops above, the constant makes these rules type-directed, so each
    # one has to admit the scalar form too (`!` on a Bool, `~` on an integer).
    @rewrite Core.Intrinsics.not_int(~x::Union{Bool, Tile{Bool}})     => Intrinsics.xori(~x, $(true))
    @rewrite Core.Intrinsics.not_int(~x::Union{Int32, Tile{Int32}})   => Intrinsics.xori(~x, $(Int32(-1)))
    @rewrite Core.Intrinsics.not_int(~x::Union{Int64, Tile{Int64}})   => Intrinsics.xori(~x, $(Int64(-1)))
    @rewrite Core.Intrinsics.not_int(~x::Union{UInt32, Tile{UInt32}}) => Intrinsics.xori(~x, $(~UInt32(0)))
    @rewrite Core.Intrinsics.not_int(~x::Union{UInt64, Tile{UInt64}}) => Intrinsics.xori(~x, $(~UInt64(0)))

    # Float arithmetic
    @rewrite Core.Intrinsics.add_float(~x, ~y) => Intrinsics.addf(~x, ~y)
    @rewrite Core.Intrinsics.sub_float(~x, ~y) => Intrinsics.subf(~x, ~y)
    @rewrite Core.Intrinsics.mul_float(~x, ~y) => Intrinsics.mulf(~x, ~y)
    @rewrite Core.Intrinsics.div_float(~x, ~y) => Intrinsics.divf(~x, ~y)
    @rewrite Core.Intrinsics.neg_float(~x)     => Intrinsics.negf(~x)
    # `muladd` may be computed fused (rounded once) or as a multiply and an add;
    # Tile IR has a native `fma`, so always fuse.
    @rewrite Core.Intrinsics.muladd_float(~x, ~y, ~z) => Intrinsics.fma(~x, ~y, ~z)
    # Base's `asin` and `acos` call `sqrt_llvm` directly.
    @rewrite Core.Intrinsics.sqrt_llvm(~x) => Intrinsics.sqrt(~x)

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
    @rewrite (===)(~x, ~y) => Intrinsics.egal(~x, ~y)
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
#
# A scalar function lifted over a tile (`f.(tile)`, `map(f, tile)`) becomes
# scalar code operating on `to_scalar` results. Its control flow ends up in
# nested regions, so shapes are propagated over the whole tree in program
# order. A branch on a shaped condition is later flattened by `if_convert!`.

function scalar_elim_pass!(sci::StructuredIRCode)
    # Phase 1: Eliminate to_scalar (forward tile-typed operand). Uses may be
    # in nested regions, so this must happen before any shape propagation.
    eliminate_calls!(sci, Intrinsics.to_scalar)

    # Phase 2: Propagate tile types from operands to results.
    propagate_tile_types!(sci.entry)

    # Phase 3: Eliminate from_scalar (operand now has correct tile type)
    eliminate_calls!(sci, Intrinsics.from_scalar)

    for block in eachblock(sci)
        promote_scalar_types!(block)
    end
end

# Replace every call to `func` with its first operand.
function eliminate_calls!(sci::StructuredIRCode, func)
    for block in eachblock(sci)
        to_delete = Instruction[]
        for inst in instructions(block)
            call = resolve_call(block, inst[:stmt])
            call === nothing && continue
            call[1] === func || continue
            replace_uses!(block, ssavalue(inst), call[2][1])
            push!(to_delete, inst)
        end
        for inst in to_delete
            delete!(block, inst)
        end
    end
end

# Shape of a tile type as a tuple of integers (column-major, as in Julia).
tile_type_shape(@nospecialize(T)) = Tuple(T.parameters[2].parameters)

# Broadcast shape of the shaped tiles among `types`, or `nothing` when no type
# is a tile. Incompatible shapes keep the first one; codegen reports those.
function broadcast_tile_shape(types)
    shape = nothing
    for T in types
        T isa DataType && T <: Tile || continue
        s = tile_type_shape(T)
        shape = shape === nothing ? s :
                try Base.Broadcast.broadcast_shape(shape, s) catch; shape end
    end
    return shape
end

# Type of an operand, or `nothing` for tokens and unknown types.
function tile_operand_type(block::Block, @nospecialize(op))
    T = value_type(block, op)
    (T === nothing || is_token_type(T)) ? nothing : CC.widenconst(T)
end

# Instructions that consumed to_scalar results still have scalar type
# annotations but now receive tile-typed operands; give them the broadcast
# shape of those operands. Regions are visited in program order, so values
# defined outside a region are typed before the region uses them.
function propagate_tile_types!(block::Block)
    for inst in instructions(block)
        stmt = inst[:stmt]
        if stmt isa ControlFlowOp
            foreach(propagate_tile_types!, IRStructurizer.blocks(stmt))
            stmt isa IfOp && propagate_if_type!(block, inst, stmt)
            continue
        end

        call = resolve_call(block, stmt)
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

        # A field of a control-flow result, which carries its own shape.
        if func === getfield && length(ops) >= 2 && ops[2] isa Int
            PT = tile_operand_type(block, ops[1])
            if PT isa DataType && PT <: Tuple && 1 <= ops[2] <= length(PT.parameters)
                FT = CC.widenconst(PT.parameters[ops[2]])
                FT <: Tile{T} && (inst[:type] = FT)
                continue
            end
        end

        shape = broadcast_tile_shape(tile_operand_type(block, op) for op in ops)
        shape === nothing && continue
        inst[:type] = Tile{T, Tuple{shape...}}
    end
end

# Give each scalar result of an `IfOp` the broadcast shape of the values its
# arms yield. A shaped condition will be flattened into selects, whose result
# also has the condition's shape.
function propagate_if_type!(block::Block, inst::Instruction, op::IfOp)
    RT = CC.widenconst(inst[:type])
    RT <: Tuple || return
    cond_type = tile_operand_type(block, op.condition)
    params = map(enumerate(RT.parameters)) do (i, P)
        P = CC.widenconst(P)
        P <: Number && !(P <: Tile) || return P
        types = Any[cond_type]
        for region in (op.then_region, op.else_region)
            term = terminator(region)
            term isa YieldOp && i <= length(term.values) &&
                push!(types, tile_operand_type(region, term.values[i]))
        end
        shape = broadcast_tile_shape(types)
        shape === nothing ? P : Tile{P, Tuple{shape...}}
    end
    inst[:type] = Tuple{params...}
end

function promote_scalar_types!(block::Block)
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
        inst[:type] = new_type
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
    # a Union or UnionAll of tuples passes `<: Tuple` but has no `parameters`
    if T isa DataType && T <: Tuple
        params = T.parameters
        any(Base.isvarargtype, params) && return nothing
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
