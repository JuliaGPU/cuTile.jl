# Performance

## Kernel configuration

`ct.@compiler_options` sets optimization hints inside a kernel function body:

```julia
function matmul(A, B, C, ...)
    ct.@compiler_options num_ctas=ct.ByTarget(v"10.0" => 2) occupancy=8
    ...
end
```

| Option | Description | Valid values |
|--------|-------------|--------------|
| `num_ctas` | Number of CTAs in a CGA | Powers of 2 |
| `occupancy` | Target concurrent CTAs per SM | 1–32 |
| `opt_level` | Optimization level | 0–3 |
| `num_worker_warps` | Worker warps per CTA in a warp-specialized kernel | 4 or 8 |

Values can be plain scalars or `ct.ByTarget(...)` for per-architecture dispatch. `ByTarget`
maps compute capabilities to values, with an optional default:

```julia
ct.@compiler_options num_ctas=ct.ByTarget(v"10.0" => 4, v"12.0" => 2; default=1)
```

Hints can also be passed as keyword arguments to `@cuda backend=cuTile` or `ct.code_tiled`,
which take precedence over `@compiler_options`.

These correspond to the hints described in the Tile IR
[optimization guide](https://docs.nvidia.com/cuda/tile-ir/latest/sections/optimization_guide.html),
which documents what each one does to code generation.


## Load/store hints

`ct.load` and `ct.store` accept optional keyword arguments that influence memory traffic
scheduling:

| Hint | Description |
|------|-------------|
| `latency` | DRAM traffic weight hint, integer 1 (low) to 10 (high). Default: compiler-inferred. |
| `allow_tma` | Whether to allow Tensor Memory Accelerator lowering. Default: allowed. |


## Array specialization

Kernels are specialized on the properties encoded in [`ct.ArraySpec`](types.md#TileArray) —
pointer alignment, contiguity, and stride/shape divisibility — which are derived from each
`CuArray`'s runtime layout at launch. A contiguous, 128-byte-aligned array whose dimensions
divide evenly by the tile shape compiles to code without tile-boundary handling and with
wider vectorized accesses. Views and unusual strides weaken those guarantees, which shows up
as a different (and slower) specialization rather than as an error.


## Benchmarks

Run the benchmark suite with:

```bash
julia --project=examples examples/benchmarks.jl  # Julia
uv run python examples/benchmarks.py             # Python (for comparison)
```

Results comparing cuTile.jl against cuTile Python on an RTX 5080 (`tileiras` 13.2.51, 20
runs, 5 warmup, min time reported):

| Kernel | Size | Julia | Python | Status |
|--------|------|-------|--------|--------|
| Vector Addition | 2^27 f32 | 844 GB/s | 845 GB/s | OK (=) |
| Matrix Transpose | 8192² f32 | 812 GB/s | 814 GB/s | OK (=) |
| Layer Norm fwd | 4096² f32 | 986 GB/s | 716 GB/s | +38% |
| Layer Norm bwd | 4096² f32 | 246 GB/s | 251 GB/s | OK (-2%) |
| Matrix Multiplication | 4096³ f32 | 47.4 TFLOPS | 43.5 TFLOPS | +9% |
| Batch Matrix Multiply | 1024×512×2048 ×8 f32 | 34.2 TFLOPS | 30.9 TFLOPS | +11% |
| FFT (3-stage Cooley-Tukey) | 4096-pt ×256 c64 | 209 μs | 204 μs | OK (-2%) |
| Mixture of Experts | 256tok 1024h 32e 2048i f16 | 27.7 TFLOPS | 20.3 TFLOPS | +36% |
| Attention (FMHA) | 8×16×1024² ×64 f16 causal | 102.7 TFLOPS | 63.3 TFLOPS | +62% |
| Softmax (TMA) | 4096² f32 | 838 GB/s | 843 GB/s | OK (-1%) |
| Softmax (Chunked) | 4096² f32 | 1672 GB/s | 1636 GB/s | OK (+2%) |

!!! note
    These numbers are a snapshot of one machine and one toolchain version, taken with the
    GPU clocks locked. Both the hardware and the `tileiras` version materially affect the
    result, and the single-digit-percent rows are within the noise of an unlocked-clock run.
    Re-measure on your own hardware before drawing conclusions.
