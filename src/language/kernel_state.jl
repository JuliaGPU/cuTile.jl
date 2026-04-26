# KernelState — per-launch ambient state implicitly threaded into every kernel
#
# A small struct that the host appends to every `cuTile.launch`. Its primitive
# fields are destructured into trailing kernel parameters; inside the kernel,
# the IR retrieves the value via the `kernel_state()` intrinsic, which resolves
# to a lazy arg-ref into the destructured arg — `state.field` accesses flow
# through the standard `getfield` path with no extra emit plumbing.
#
# `seed` is the only field today: a fresh `RandomDevice`-derived `UInt32` per
# launch, consumed by `lower_rng_state!` to seed the default RNG stream so
# bare `rand()` produces uncorrelated output across launches. Kernels that
# don't reference `kernel_state()` pay only the cost of one unused `UInt32`
# parameter slot.

using Random

# Internal — not in `public`. The no-arg constructor draws fresh entropy so
# consecutive launches see distinct seeds.
struct KernelState
    seed::UInt32
end

KernelState() = KernelState(Base.rand(Random.RandomDevice(), UInt32))
