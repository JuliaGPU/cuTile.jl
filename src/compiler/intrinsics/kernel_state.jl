# KernelState intrinsic
#
# `kernel_state()` returns the host-supplied `KernelState` struct that
# `emit_kernel!` registers at the trailing codegen arg_idx
# (`length(sci.argtypes) + 1`) — its primitive fields are destructured into
# the trailing kernel parameters. The intrinsic resolves to a lazy arg-ref
# pointing at that virtual destructured arg; `getfield(:field)` then flows
# through the standard destructured-arg path, so no per-field emit plumbing
# is needed.
#
# This works because every cuTile-emitted Tile IR function is an entry kernel
# and subprograms compile inline as nested regions (outer SSA is in scope);
# `emit_subprogram!` re-keys the parent's state flat values into the sub-ctx's
# own trailing slot. If we ever add non-entry callable subroutines, they would
# need GPUCompiler-style state threading via a leading parameter on every
# reachable function instead.

@intrinsic kernel_state()
tfunc(𝕃, ::typeof(Intrinsics.kernel_state)) = KernelState
efunc(::typeof(Intrinsics.kernel_state), effects::CC.Effects) =
    CC.Effects(effects; effect_free=CC.ALWAYS_FALSE)
emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.kernel_state), args) =
    arg_ref_value(length(ctx.sci.argtypes) + 1, Int[], KernelState)
