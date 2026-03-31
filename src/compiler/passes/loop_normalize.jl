# Loop Normalization for Structured IR
#
# Canonicalizes 1-based for-loops to 0-based, eliminating redundant
# subi(iv, lower_bound) operations inside loop bodies.
#
# Julia's while-loop pattern `j=1; while j<=N; ...; j+=1; end` compiles
# to `for iv in (1 to N+1, step 1)` with `subi(iv, 1)` at every use site
# inside the body. This pass transforms to `for iv in (0 to N, step 1)`
# and replaces all `subi(iv, 1)` results with the IV directly, matching
# Python cuTile's 0-based loop output.
#
# Inspired by MLIR's `normalizeLoopBounds` and LLVM's IndVarSimplify.

using Core: SSAValue

"""
    loop_normalize_pass!(sci::StructuredIRCode)

Normalize for-loop induction variables to start at 0. For each ForOp where:
1. The lower bound is a constant `c > 0`
2. Every use of the IV inside the body is `subi(iv, c)` with the same constant

Transforms: `for iv in (c to upper, step s)` → `for iv in (0 to upper-c, step s)`
and replaces `subi(iv, c)` with `iv`.
"""
function loop_normalize_pass!(sci::StructuredIRCode)
    changed = true
    while changed
        changed = false
        changed |= _normalize_loops_in_block!(sci, sci.entry)
    end
end

function _get_constant_int(block::Block, @nospecialize(val))
    if val isa Int32 || val isa Int64 || val isa Int
        return val
    end
    if val isa SSAValue
        haskey(block.body, val.id) || return nothing
        entry = block.body[val.id]
        s = entry.stmt
        if s isa Int32 || s isa Int64 || s isa Int
            return s
        end
        if s isa QuoteNode && (s.value isa Int32 || s.value isa Int64 || s.value isa Int)
            return s.value
        end
    end
    return nothing
end

function _normalize_loops_in_block!(sci::StructuredIRCode, block::Block)
    changed = false
    for inst in instructions(block)
        s = stmt(inst)
        if s isa ForOp
            changed |= _try_normalize_for!(sci, block, inst, s)
            changed |= _normalize_loops_in_block!(sci, s.body)
        elseif s isa IfOp
            changed |= _normalize_loops_in_block!(sci, s.then_region)
            changed |= _normalize_loops_in_block!(sci, s.else_region)
        elseif s isa WhileOp
            changed |= _normalize_loops_in_block!(sci, s.before)
            changed |= _normalize_loops_in_block!(sci, s.after)
        elseif s isa LoopOp
            changed |= _normalize_loops_in_block!(sci, s.body)
        end
    end
    return changed
end

function _try_normalize_for!(sci::StructuredIRCode, parent_block::Block,
                              for_inst::Instruction, for_op::ForOp)
    # 1. Check that lower bound is a constant > 0
    lower_const = _get_constant_int(parent_block, for_op.lower)
    lower_const === nothing && return false
    lower_const <= 0 && return false

    iv_arg = for_op.iv_arg
    body = for_op.body

    # 2. Find all uses of the IV in the loop body
    iv_users = users(body, iv_arg)

    # 3. Check that EVERY user is `Intrinsics.subi(iv, c)` where c is
    #    a constant equal to lower_const
    subi_insts = Instruction[]
    for user_inst in iv_users
        s = stmt(user_inst)
        call = resolve_call(body, s)
        call === nothing && return false
        func, ops = call
        func === Intrinsics.subi || return false
        length(ops) == 2 || return false
        ops[1] === iv_arg || return false
        c_val = _get_constant_int(body, ops[2])
        c_val === nothing && return false
        c_val == lower_const || return false
        push!(subi_insts, user_inst)
    end

    # Also check that the IV is not used in the terminator directly
    term = terminator(body)
    if term !== nothing
        for v in operands(term)
            v === iv_arg && return false
        end
    end

    # All checks pass — apply the transformation

    # 4. Compute new upper: subi(upper, lower). The algebra pass's
    #    subi(addi(x, c), c) → x rule will simplify this when possible.
    new_upper_stmt = Expr(:call, Intrinsics.subi, for_op.upper, for_op.lower)
    new_upper_inst = insert_before!(parent_block, for_inst, new_upper_stmt, Int32)
    for_op.lower = Int32(0)
    for_op.upper = SSAValue(new_upper_inst)

    # 5. Replace all subi(iv, c) results with iv directly
    for subi_inst in subi_insts
        replace_uses!(body, SSAValue(subi_inst), iv_arg)
        delete!(body, subi_inst)
    end

    return true
end
