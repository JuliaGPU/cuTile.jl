# Host-level Operations

cuTile.jl provides a limited set of host-level APIs to use cuTile without writing custom
kernels.


## Fused broadcast

For element-wise operations on `CuArray`s, cuTile can automatically generate and launch a
fused kernel using Julia's broadcast machinery:

```julia
using CUDA
import cuTile as ct

A = CUDA.rand(Float32, 1024)
B = CUDA.rand(Float32, 1024)
C = CUDA.zeros(Float32, 1024)

# Wrap arrays in Tiled() to route through cuTile
ct.Tiled(C) .= ct.Tiled(A) .+ ct.Tiled(B)

# Or use the @. macro for convenience
ct.@. C = A + sin(B)

# Allocating form (returns a new CuArray)
D = ct.@. A + B
```

The entire broadcast expression is fused into a single cuTile kernel. Tile sizes are
automatically chosen based on array dimensions (power-of-2, budget-based). Works with 1D
through N-dimensional arrays.

Standard broadcast shape semantics apply: size-1 dimensions are expanded to the destination's
size (`Tiled(C) .= Tiled(row) .+ Tiled(B)` with a `(1, N)` row), scalars fill the destination
(`Tiled(C) .= 0`), and incompatible shapes throw a `DimensionMismatch`. For in-place
assignment, `ct.@.` returns the original destination array.


## Random number generation

`cuTile.RNG` fills `CuArray`s on the device using the same Philox2x32-7 generator as the
in-kernel `rand` / `randn` / `randexp`:

```julia
using CUDA
import cuTile as ct

A = CUDA.zeros(Float32, 1024)
rng = ct.RNG(42)
rand!(rng, A)                  # `Random.rand!`, uniform in-place
randn!(rng, A)                 # `Random.randn!`, standard normal
randexp!(rng, A)               # `Random.randexp!`, standard exponential
B = rand(rng, Float64, 16)     # out-of-place
N = randn(rng, Float32, 1024)  # out-of-place normal

# Or via the global helpers (match CUDA.rand! / CUDA.seed!)
ct.rand!(A)
ct.randn!(A)
ct.randexp!(A)
ct.seed!(0xdeadbeef)
```

Supports the same output types as the [in-kernel API](random.md). The counter is
auto-advanced after each fill, so consecutive calls on the same `RNG` produce disjoint
streams.
