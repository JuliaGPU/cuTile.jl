# Loop-Invariant Code Motion (LICM)
#
# Hoists loop-invariant operations out of loops. Runs AFTER token_order_pass!
# so that token dependencies correctly prevent unsafe hoisting of aliasing loads.
#
# Hoist gate: the per-stmt `IR_FLAG_EFFECT_FREE` bit (Julia inference,
# incorporating every `efunc(...) = effect_free=ALWAYS_FALSE` override). This is
# the analogue of LLVM LICM's `!I.mayHaveSideEffects()` and MLIR LICM's
# `isMemoryEffectFree(op)` — pin anything that can be observed from outside its
# operands, not just memory writes. Stores, atomics, asserts, fpmode_*,
# print_tko, RNG state ops are all rejected without a per-pass skip list.
#
# Loads carry `IR_FLAG_EFFECT_FREE` (the load itself doesn't write), but
# token-threading installs a token operand pointing at the latest store, so
# `is_defined_outside` on that operand naturally rejects loads whose memory
# state can change inside the loop. The two checks compose: the flag pins
# side-effecting ops, the operand check pins memory-dependent ones.
#
# Uses IRStructurizer's `is_defined_outside`, `move_before!`, and `operands`
# primitives. Processes innermost loops first and repeats until fixpoint.
#
# This mirrors cuTile Python's code_motion.py:hoist_loop_invariants.

"""
    licm_pass!(sci::StructuredIRCode)

Hoist loop-invariant operations out of loops. Must run after token_order_pass!.
"""
function licm_pass!(sci::StructuredIRCode)
    for (loop_inst, loop_op) in collect_loops(sci.entry)
        hoist_from_loop!(loop_inst, loop_op)
    end
end

# Collect (instruction, loop_op) pairs in post-order (innermost first).
function collect_loops(root::Block)
    result = Tuple{Instruction, Union{ForOp, LoopOp, WhileOp}}[]
    collect_loops!(result, root)
    return result
end

function collect_loops!(result, block::Block)
    for inst in instructions(block)
        s = inst[:stmt]
        if s isa ForOp || s isa LoopOp
            collect_loops!(result, s.body)
            push!(result, (inst, s))
        elseif s isa WhileOp
            collect_loops!(result, s.before)
            collect_loops!(result, s.after)
            push!(result, (inst, s))
        elseif s isa ControlFlowOp
            for b in blocks(s)
                collect_loops!(result, b)
            end
        end
    end
end

function hoist_from_loop!(loop_inst::Instruction, loop_op)
    changed = true
    while changed
        changed = false
        for body in blocks(loop_op)
            for inst in collect(instructions(body))
                inst[:stmt] isa ControlFlowOp && continue
                CC.has_flag(inst[:flag], CC.IR_FLAG_EFFECT_FREE) || continue
                all(v -> is_defined_outside(v, loop_op), operands(body, inst)) || continue
                move_before!(inst, loop_inst)
                changed = true
            end
        end
    end
end
