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

"""
    intrinsic_effects(func) -> Union{CC.Effects, Nothing}

Declared effects of a cuTile intrinsic, or `nothing` for non-intrinsic callees.
Single source of truth for transform passes that need per-intrinsic effect
information (rewriter flag recomputation, DCE root classification).

Starts from `EFFECTS_TOTAL` — intrinsic methods are `not_callable()` bodies with
no observable effect — and applies any `efunc` override. Returns `nothing` for
non-intrinsic callees: purity of arbitrary Julia functions isn't ours to claim,
and callers should treat `nothing` as "unknown, be conservative".
"""
function intrinsic_effects(@nospecialize(func))
    func isa Function || return nothing
    parentmodule(func) === Intrinsics || return nothing
    effects = CC.EFFECTS_TOTAL
    override = efunc(func, effects)
    override !== nothing && (effects = override)
    return effects
end

"""
    inferred_flags(func) -> UInt32

IR flags corresponding to `func`'s declared effects, mirroring inference's
`flags_for_effects`. Used by the rewriter to set fresh flags on inserted or
opcode-changed instructions, so downstream gates (CSE, LICM) see the same
information they would have gotten from a fresh inference.

Returns `IR_FLAG_NULL` for non-intrinsic callees — see `intrinsic_effects`.
"""
function inferred_flags(@nospecialize(func))
    effects = intrinsic_effects(func)
    effects === nothing && return CC.IR_FLAG_NULL
    return CC.flags_for_effects(effects)
end
