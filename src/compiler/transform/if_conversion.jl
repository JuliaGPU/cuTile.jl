# If-conversion of branches on shaped conditions
#
# A scalar function lifted over a tile (`f.(tile)`, `map(f, tile)`) may branch
# on a value derived from its arguments: `x > 0 ? a : b`, or the special cases
# in Base's `asin` and `hypot`. After scalar elimination such a branch has a
# shaped `Tile{Bool}` condition, which Tile IR's `if` cannot take. Both arms
# are executed instead and their results combined lane-wise with `select`.
# Arms executed this way may only contain operations without memory effects,
# and nested branches must have been flattened already.
#
# A throw becomes an assertion that no lane takes its path. Julia ends a
# throwing arm in `unreachable`, so the arm yields nothing. Once the assertion
# holds, every lane takes the other arm, which therefore runs unchanged even if
# it has side effects. When nothing follows the branch, Julia places the rest
# of the enclosing block in that arm.

function is_shaped_condition(block::Block, @nospecialize(cond))
    T = tile_operand_type(block, cond)
    T isa DataType && T <: Tile && !isempty(tile_type_shape(T))
end

"""
    if_convert!(sci)

Flatten every `IfOp` whose condition is a shaped tile into its parent block,
innermost first. Branches that cannot be flattened are left in place.
"""
function if_convert!(sci::StructuredIRCode)
    if_convert_block!(sci, sci.entry)
    return sci
end

function if_convert_block!(sci::StructuredIRCode, block::Block)
    for inst in collect(instructions(block))
        # Flattening a branch deletes the extractions of its results.
        haskey(block, inst.ssa_idx) || continue
        stmt = inst[:stmt]
        stmt isa ControlFlowOp || continue
        for region in IRStructurizer.blocks(stmt)
            if_convert_block!(sci, region)
        end
        stmt isa IfOp && is_shaped_condition(block, stmt.condition) || continue
        convert_if!(sci, block, inst, stmt)
    end
end

# `:pure` for an arm without side effects that yields, `:throw` for one that
# ends in a throw, `:other` for any other arm.
function arm_kind(region::Block)
    throws = false
    pure = true
    for inst in instructions(region)
        stmt = inst[:stmt]
        if stmt isa ThrowNode
            throws = true
        elseif stmt isa ControlFlowOp
            pure = false
        else
            call = resolve_call(region, stmt)
            call !== nothing && classify_memory_op(call[1]) !== MEM_NONE && (pure = false)
        end
    end
    # Nothing after a throw executes, whatever the terminator (`unreachable`, or
    # a yield once `hoist_returns!` has rewritten it).
    throws && return pure ? :throw : :other
    term = terminator(region)
    return pure && (term === nothing || term isa YieldOp) ? :pure : :other
end

function convert_if!(sci::StructuredIRCode, parent::Block, inst::Instruction, op::IfOp)
    kinds = (arm_kind(op.then_region), arm_kind(op.else_region))
    # Either both arms are selected from, or one throws and the other continues.
    kinds === (:pure, :pure) || count(==(:throw), kinds) == 1 || return false
    selected = kinds === (:pure, :pure)

    if_ssa = SSAValue(inst)
    RT = CC.widenconst(inst[:type])
    RT === Nothing || RT <: Tuple || return false
    result_types = RT === Nothing ? Any[] : collect(Any, RT.parameters)
    extracts = Tuple{Block, SSAValue, Int}[]
    if !isempty(result_types)
        extracts = tuple_result_extracts(sci, Rewriter(sci), if_ssa, length(result_types))
        extracts === nothing && return false
    end

    # An arm that continues past the branch replaces the parent's terminator.
    continuing = selected ? nothing :
                 kinds[1] === :throw ? op.else_region : op.then_region
    new_terminator = nothing
    if continuing !== nothing
        term = terminator(continuing)
        if !(term === nothing || term isa YieldOp)
            isempty(result_types) || return false
            last(collect(instructions(parent))).ssa_idx == inst.ssa_idx || return false
            new_terminator = term
        end
    end

    then_vals = kinds[1] === :throw ? nothing : yield_values(op.then_region)
    else_vals = kinds[2] === :throw ? nothing : yield_values(op.else_region)
    if new_terminator === nothing
        for vals in (then_vals, else_vals)
            vals === nothing || length(vals) == length(result_types) || return false
        end
    end
    # Only tiles can be selected; any other result must be the same in both arms.
    if selected
        for (i, T) in enumerate(result_types)
            CC.widenconst(T) <: Tile || then_vals[i] === else_vals[i] || return false
        end
    end

    cond = op.condition
    cond_type = tile_operand_type(parent, cond)
    not_cond = SSAValue(insert_before!(parent, inst, Expr(:call, Intrinsics.xori, cond, true),
                                       cond_type; flag=inferred_flags(Intrinsics.xori)))

    # Move both arms into the parent. An assertion inside an arm (from a throw,
    # possibly in an already flattened nested branch) must only fail for the
    # lanes that take the arm.
    for (region, off_arm) in ((op.then_region, not_cond), (op.else_region, cond))
        for rinst in collect(instructions(region))
            stmt = rinst[:stmt]
            if stmt isa ThrowNode
                rinst[:stmt] = Expr(:call, Intrinsics.assert, off_arm, stmt.message)
                rinst[:type] = Nothing
                rinst[:flag] = inferred_flags(Intrinsics.assert)
            else
                call = resolve_call(region, stmt)
                if call !== nothing && call[1] === Intrinsics.assert
                    ok = call[2][1]
                    ok_type = Tile{Bool, Tuple{broadcast_tile_shape(
                        (tile_operand_type(region, ok), cond_type))...}}
                    weakened = insert_before!(region, rinst,
                                              Expr(:call, Intrinsics.ori, ok, off_arm), ok_type;
                                              flag=inferred_flags(Intrinsics.ori))
                    rinst[:stmt] = Expr(:call, Intrinsics.assert, SSAValue(weakened), call[2][2])
                end
            end
        end
        for rinst in collect(instructions(region))
            move_before!(rinst, inst)
        end
    end

    results = map(enumerate(result_types)) do (i, T)
        T = CC.widenconst(T)
        if selected
            then_vals[i] === else_vals[i] && return then_vals[i]
            return SSAValue(insert_before!(parent, inst,
                Expr(:call, Intrinsics.select, cond, then_vals[i], else_vals[i]), T;
                flag=inferred_flags(Intrinsics.select)))
        end
        val = something(then_vals, else_vals)[i]
        # The surviving value may be narrower than the branch result, which
        # has the condition's shape.
        T <: Tile && tile_operand_type(parent, val) !== T || return val
        SSAValue(insert_before!(parent, inst,
            Expr(:call, Intrinsics.broadcast, val, tile_type_shape(T)), T;
            flag=inferred_flags(Intrinsics.broadcast)))
    end

    for (block, extract_ssa, field) in extracts
        replace_uses!(sci.entry, extract_ssa, results[field])
        delete!(block, extract_ssa.id)
    end
    delete!(parent, if_ssa.id)
    new_terminator === nothing || terminator!(parent, new_terminator)
    return true
end
