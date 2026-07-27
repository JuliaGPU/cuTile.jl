# Host-level Operations

```@meta
DocTestSetup = quote
    using CUDA, Random
    import cuTile as ct
end
```

cuTile.jl provides a limited set of host-level APIs to use cuTile without
writing custom kernels. CUDA.jl remains the general-purpose home for `CuArray`
operations, allocation, streams and device management; this page only covers
the small host surface implemented directly by cuTile.

!!! note

    These implementations are temporary, and will disappear in the future once
    they get folded into CUDA.jl itself.


## Fused broadcast

For element-wise operations on `CuArray`s, cuTile can automatically generate and
launch a fused kernel using Julia's broadcast machinery:

```jldoctest host_broadcast
julia> A = CUDA.rand(Float32, 1024);

julia> B = CUDA.rand(Float32, 1024);

julia> C = CUDA.zeros(Float32, 1024);

julia> ct.Tiled(C) .= ct.Tiled(A) .+ ct.Tiled(B);

julia> @assert C == A .+ B

julia> ct.@. C = A + sin(B);

julia> @assert C == A .+ sin.(B)

julia> D = ct.@. A + B;

julia> @assert D == A .+ B
```

The entire broadcast expression is fused into a single cuTile kernel. Tile sizes
are automatically chosen based on array dimensions (power-of-2, budget-based).
Works with 1D through N-dimensional arrays.

Standard broadcast shape semantics apply: size-1 dimensions are expanded to the
destination's size (`Tiled(C) .= Tiled(row) .+ Tiled(B)` with a `(1, N)` row),
scalars fill the destination (`Tiled(C) .= 0`), and incompatible shapes throw a
`DimensionMismatch`. For in-place assignment, `ct.@.` returns the original
destination array.


## Random number generation

`cuTile.RNG` fills `CuArray`s on the device using the same Philox2x32-7
generator as the in-kernel `rand` / `randn` / `randexp`:

```jldoctest host_random
julia> A = CUDA.zeros(Float32, 1024);

julia> rng = ct.RNG(42);

julia> rand!(rng, A);

julia> randn!(rng, A);

julia> randexp!(rng, A);

julia> B = rand(rng, Float64, 16);

julia> N = randn(rng, Float32, 1024);

julia> ct.rand!(A);

julia> ct.randn!(A);

julia> ct.randexp!(A);

julia> ct.seed!(0xdeadbeef);
```

Supports the same output types as the [in-kernel API](random.md). The counter is
auto-advanced after each fill, so consecutive calls on the same `RNG` produce
disjoint streams.
