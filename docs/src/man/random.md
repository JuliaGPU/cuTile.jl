# Random Numbers

```@meta
DocTestSetup = quote
    using CUDA, cuTile, Random
    import cuTile as ct
end
```

cuTile provides random number generation inside kernels, using a tile-vectorised
Philox2x32-7 generator.

| Operation | Description |
|-----------|-------------|
| `rand()` | Uniform `Float32` scalar in `(0, 1]` (uses `Random.default_rng()`) |
| `rand(T)` / `rand(T, dims)` / `rand(dims)` | Uniform scalar or tile |
| `randn()` / `randn(T)` / `randn(T, dims)` | Standard normal scalar or tile (Box-Muller) |
| `randexp()` / `randexp(T)` / `randexp(T, dims)` | Standard exponential scalar or tile (`-log(U)`) |
| `ct.DeviceRNG()` | Open an independent RNG stream |
| `Random.rand(rng, ...)` / `Random.randn(rng, ...)` / `Random.randexp(rng, ...)` | Explicit-stream variants |
| `Random.seed!(rng, seed)` | Re-seed a stream |

`rand` supports all of `Int{8,16,32,64}`, `UInt{8,16,32,64}`, `Float16`,
`BFloat16`, `Float32` and `Float64`; `randn` and `randexp` cover the four
floating-point types. Different `DeviceRNG()` call sites yield independent
streams, all keyed on a per-launch host seed for cross-launch divergence.

The floating-point generator deliberately excludes zero so that `randn` and
`randexp` can safely take a logarithm. The `Float32` conversion can round its
largest values to exactly `1.0f0`; the narrower and `Float64` conversions
produce values strictly between zero and one. This differs from Base's usual
`[0, 1)` interval.

```jldoctest device_random
julia> function noise(out)
           pid = ct.bid(1)
           t = randn(Float32, (256,))         # default RNG, standard normal
           ct.store(out; index=pid, tile=t)
           return
       end;

julia> out = CUDA.zeros(Float32, 1024);

julia> @cuda backend=cuTile blocks=4 noise(out);

julia> @assert all(isfinite, Array(out))
```

The same generator is available on the host, for filling `CuArray`s without
writing a kernel; see [Host-level Operations](host.md#Random-number-generation).
