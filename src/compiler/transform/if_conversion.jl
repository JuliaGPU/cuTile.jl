# If-conversion of branches on shaped conditions
#
# A scalar function lifted over a tile (`f.(tile)`, `map(f, tile)`) may branch
# on a value derived from its arguments: `x > 0 ? a : b`, or the special cases
# in Base's `asin` and `hypot`. After scalar elimination such a branch has a
# shaped `Tile{Bool}` condition, which Tile IR's `if` cannot take. Both arms
# are executed instead and their results combined lane-wise with `select`.
# Arms executed this way may only contain operations without side effects, and
# nested branches must have been flattened already. Integer divisions are
# made safe for the inputs they are undefined for, which lanes that do not
# take the arm may well have.
#
# A throw becomes an assertion that no lane takes its path. Julia ends a
# throwing arm in `unreachable`, so the arm yields nothing. Once the assertion
# holds, every lane takes the other arm, which therefore runs unchanged even if
# it has side effects. When nothing follows the branch, Julia places the rest
# of the enclosing block in that arm.

function is_shaped_condition(block::Block, @nospecialize(cond))
    T = tile_operand_type(block, cond)
    T isa DataType && T <: Tile && !isempty(size(T))
end

"""
    if_convert!(sci)

Flatten every `IfOp` whose condition is a shaped tile into its parent block,
innermost first. Branches that cannot be flattened are left in place.
"""
function if_convert!(sci::StructuredIRCode)
    if_convert_block!(Rewriter(sci), sci.entry)
    return sci
end

function if_convert_block!(r::Rewriter, block::Block)
    for inst in collect(instructions(block))
        # Flattening a branch deletes the extractions of its results.
        haskey(block, inst.ssa_idx) || continue
        stmt = inst[:stmt]
        stmt isa ControlFlowOp || continue
        for region in IRStructurizer.blocks(stmt)
            if_convert_block!(r, region)
        end
        stmt isa IfOp && is_shaped_condition(block, stmt.condition) || continue
        convert_if!(r, block, inst, stmt)
    end
end

# Whether an instruction may also run for lanes that do not take its arm.
# Assertions are weakened for those lanes, and `@fpmode` markers only scope
# codegen. Loads are removable but not speculatable: the lanes that do not take
# the arm may not be allowed to access that memory.
function speculatable(block::Block, inst::Instruction)
    call = resolve_call(block, inst[:stmt])
    if call !== nothing
        func = call[1]
        func in (Intrinsics.assert, Intrinsics.fpmode_begin, Intrinsics.fpmode_end) &&
            return true
        classify_memory_op(func) === MEM_NONE || return false
    end
    return !must_keep(block, inst)
end

# `:pure` for an arm without side effects that yields, `:throw` for one that
# ends in a throw, `:other` for any other arm.
function arm_kind(region::Block)
    throws = false
    for inst in instructions(region)
        stmt = inst[:stmt]
        if stmt isa ThrowNode
            throws = true
        elseif stmt isa ControlFlowOp || !speculatable(region, inst)
            return :other
        end
    end
    # Nothing after a throw executes, whatever the terminator (`unreachable`, or
    # a yield once `hoist_returns!` has rewritten it).
    throws && return :throw
    return plain_yield(region) ? :pure : :other
end

# Integer division is undefined for a zero divisor, and signed division also
# for MIN / -1, even if the quotient is then discarded: tileiras assumes the
# divisor is valid and drops the select that guarded it. Divide by 1 instead
# for those inputs, which only changes results that were undefined anyway.
const INTEGER_DIVISIONS = (Intrinsics.divi, Intrinsics.fldi, Intrinsics.cldi, Intrinsics.remi)

function make_division_safe!(r::Rewriter, block::Block, inst::Instruction, func, ops)
    x, y, signedness = ops
    y isa Integer && !iszero(y) && y != -1 && return
    T = eltype(something(tile_operand_type(block, y), typeof(y)))
    shaped(S, vals...) = Tile{S, Tuple{something(
        broadcast_tile_shape(Any[tile_operand_type(block, v) for v in vals]), ())...}}
    insert(stmt, type, f) = ssavalue(insert_before!(r, block, inst, stmt, type;
                                                    flag=inferred_flags(f)))
    cmp(a, b) = insert(Expr(:call, Intrinsics.cmpi, a, b, ComparisonPredicate.Equal,
                            signedness), shaped(Bool, a, b), Intrinsics.cmpi)
    invalid = cmp(y, zero(T))
    if signedness === Signedness.Signed && func !== Intrinsics.remi
        overflow = cmp(x, typemin(T))
        minus_one = cmp(y, -one(T))
        overflow = insert(Expr(:call, Intrinsics.andi, overflow, minus_one),
                          shaped(Bool, x, y), Intrinsics.andi)
        invalid = insert(Expr(:call, Intrinsics.ori, invalid, overflow),
                         shaped(Bool, x, y), Intrinsics.ori)
    end
    divisor = insert(Expr(:call, Intrinsics.select, invalid, one(T), y),
                     shaped(T, invalid, y), Intrinsics.select)
    replace_stmt!(r, block, ssavalue(inst), Expr(:call, func, x, divisor, signedness))
end

function convert_if!(r::Rewriter, parent::Block, inst::Instruction, op::IfOp)
    sci = r.sci
    kinds = (arm_kind(op.then_region), arm_kind(op.else_region))
    # Either both arms are selected from, or one throws and the other continues.
    kinds === (:pure, :pure) || count(==(:throw), kinds) == 1 || return false
    selected = kinds === (:pure, :pure)

    if_ssa = ssavalue(inst)
    RT = CC.widenconst(inst[:type])
    RT === Nothing || RT <: Tuple || return false
    result_types = RT === Nothing ? Any[] : collect(Any, RT.parameters)
    extracts = Tuple{Block, SSAValue, Int}[]
    if !isempty(result_types)
        extracts = tuple_result_extracts(sci, r, if_ssa, length(result_types))
        extracts === nothing && return false
    end

    # An arm that continues past the branch replaces the parent's terminator.
    continuing = selected ? nothing :
                 kinds[1] === :throw ? op.else_region : op.then_region
    new_terminator = nothing
    if continuing !== nothing && !plain_yield(continuing)
        isempty(result_types) || return false
        last(collect(instructions(parent))).ssa_idx == inst.ssa_idx || return false
        new_terminator = terminator(continuing)
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
    not_cond = ssavalue(insert_before!(r, parent, inst, Expr(:call, Intrinsics.xori, cond, true),
                                       cond_type; flag=inferred_flags(Intrinsics.xori)))

    # Move both arms into the parent. An assertion inside an arm (from a throw,
    # possibly in an already flattened nested branch) must only fail for the
    # lanes that take the arm.
    for (region, off_arm) in ((op.then_region, not_cond), (op.else_region, cond))
        for rinst in collect(instructions(region))
            stmt = rinst[:stmt]
            if stmt isa ThrowNode
                replace_stmt!(r, region, ssavalue(rinst),
                              Expr(:call, Intrinsics.assert, off_arm, stmt.message);
                              flag=inferred_flags(Intrinsics.assert))
                rinst[:type] = Nothing
                continue
            end
            call = resolve_call(region, stmt)
            call === nothing && continue
            func, ops = call
            if func === Intrinsics.assert
                ok = ops[1]
                ok_type = Tile{Bool, Tuple{broadcast_tile_shape(
                    (tile_operand_type(region, ok), cond_type))...}}
                weakened = insert_before!(r, region, rinst,
                                          Expr(:call, Intrinsics.ori, ok, off_arm), ok_type;
                                          flag=inferred_flags(Intrinsics.ori))
                replace_stmt!(r, region, ssavalue(rinst),
                              Expr(:call, Intrinsics.assert, ssavalue(weakened), ops[2]))
            elseif selected && func in INTEGER_DIVISIONS
                make_division_safe!(r, region, rinst, func, ops)
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
            return ssavalue(insert_before!(r, parent, inst,
                Expr(:call, Intrinsics.select, cond, then_vals[i], else_vals[i]), T;
                flag=inferred_flags(Intrinsics.select)))
        end
        val = something(then_vals, else_vals)[i]
        # The surviving value may be narrower than the branch result, which
        # has the condition's shape.
        T <: Tile && tile_operand_type(parent, val) !== T || return val
        ssavalue(insert_before!(r, parent, inst,
            Expr(:call, Intrinsics.broadcast, val, size(T)), T;
            flag=inferred_flags(Intrinsics.broadcast)))
    end

    for (block, extract_ssa, field) in extracts
        replace_uses!(r, extract_ssa, results[field])
        erase!(r, block, extract_ssa)
    end
    erase!(r, parent, if_ssa)
    new_terminator === nothing || terminator!(parent, new_terminator)
    return true
end
