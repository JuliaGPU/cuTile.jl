# KernelState — per-launch ambient state implicitly threaded into every kernel
#
# A small struct that the host appends to every `cuTile.launch`. Its primitive
# fields are destructured into trailing kernel parameters; inside the kernel,
# the IR retrieves the value via the `kernel_state()` intrinsic, which resolves
# to a lazy arg-ref into the destructured arg — `state.field` accesses flow
# through the standard `getfield` path with no extra emit plumbing.
#
# `KernelState` is currently empty (a ghost type), so it adds zero kernel
# parameters and zero per-launch overhead. The plumbing is wired so that
# adding fields later (e.g. for a host-supplied RNG seed) requires no
# additional codegen, host, or subprogram changes — just a field declaration.

# Internal — not in `public`.
struct KernelState
end
