# Memory-effect classification
#
# Generic, pass-independent classifier for the side effects of a resolved
# intrinsic. Used by `token_order_pass!` (to decide which ops need token
# threading), `licm_pass!` (to keep stores pinned), and `cse_pass!` (to skip
# memory ops as CSE candidates).
#
# Mirrors LLVM's `Instruction::mayReadOrWriteMemory` / MLIR's
# `MemoryEffectOpInterface`: a single source of truth for "what does this op
# do to memory", consumed by every pass that cares.

@enum MemoryEffect MEM_NONE MEM_LOAD MEM_STORE

"""
    classify_memory_op(resolved_func) -> MemoryEffect

Return the memory effect of a resolved intrinsic call. `MEM_NONE` for pure
ops, `MEM_LOAD` for reads, `MEM_STORE` for writes (including atomics, which
are conservatively classified as stores, and `print_tko` which has an
externally observable side effect).
"""
function classify_memory_op(resolved_func)
    if resolved_func === Intrinsics.load_partition_view ||
       resolved_func === Intrinsics.load_ptr_tko
        return MEM_LOAD
    elseif resolved_func === Intrinsics.store_partition_view ||
           resolved_func === Intrinsics.store_ptr_tko
        return MEM_STORE
    elseif resolved_func === Intrinsics.print_tko
        return MEM_STORE
    elseif is_atomic_intrinsic(resolved_func)
        return MEM_STORE
    else
        return MEM_NONE
    end
end

function is_atomic_intrinsic(func)
    isdefined(Intrinsics, :atomic_cas) && func === Intrinsics.atomic_cas && return true
    for op in (:atomic_xchg, :atomic_add, :atomic_max, :atomic_min,
               :atomic_or, :atomic_and, :atomic_xor)
        isdefined(Intrinsics, op) && func === getfield(Intrinsics, op) && return true
    end
    return false
end
