# Loop-Invariant Code Motion (LICM)
#
# Hoists loop-invariant operations out of loops. Runs AFTER token_order_pass!
# so that token dependencies correctly prevent unsafe hoisting of aliasing loads.
#
# Operations classified as stores (store_partition_view, store_ptr_tko, atomics,
# print_tko) and control flow exits (return) are never hoisted. All other
# operations — including loads, arithmetic, partition views, token nodes — are
# hoisted when all their data dependencies are defined outside the loop.
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
        s = stmt(inst)
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
                stmt(inst) isa ControlFlowOp && continue
                is_store(body, stmt(inst)) && continue
                all(v -> is_defined_outside(v, loop_op), operands(body, inst)) || continue
                move_before!(inst, loop_inst)
                changed = true
            end
        end
    end
end

# Check if a statement is a store/atomic (side-effecting memory write).
function is_store(block::Block, @nospecialize(s))
    call = resolve_call(block, s)
    call === nothing && return false
    resolved_func, _ = call
    return classify_memory_op(resolved_func) == MEM_STORE
end
