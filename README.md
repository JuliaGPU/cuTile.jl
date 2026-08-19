# cuTile.jl

*Tile-based GPU programming in Julia.*

[![][docs-stable-img]][docs-stable-url] [![][docs-dev-img]][docs-dev-url]

[docs-stable-img]: https://img.shields.io/badge/docs-stable-blue.svg
[docs-stable-url]: https://juliagpu.github.io/cuTile.jl/stable/

[docs-dev-img]: https://img.shields.io/badge/docs-dev-blue.svg
[docs-dev-url]: https://juliagpu.github.io/cuTile.jl/dev/

The cuTile.jl package compiles Julia functions to [Tile
IR](https://docs.nvidia.com/cuda/tile-ir/), NVIDIA's portable tile-based instruction set.
Rather than writing kernels that describe what a single GPU thread does, you write kernels
that operate on *tiles* — multi-dimensional array fragments — and leave it to the Tile IR
compiler to map those onto the hardware, including tensor cores and the tensor memory
accelerator.


## Requirements

cuTile.jl requires Julia 1.11 or newer, an NVIDIA driver supporting CUDA 13 (version 580 or
later), and a GPU with [compute capability](https://developer.nvidia.com/cuda/gpus) 8.0 or
newer. You do not need to install the CUDA toolkit yourself: CUDA.jl downloads the
appropriate artifacts automatically.


## Installation

cuTile.jl can be installed with the Julia package manager. From the Julia REPL, type `]` to
enter the Pkg REPL mode and run:

```
pkg> add cuTile
```

Or, equivalently, via the `Pkg` API:

```julia
julia> import Pkg; Pkg.add("cuTile")
```

Launching kernels additionally requires CUDA.jl to be imported.
[CUDA.jl's documentation](https://cuda.juliagpu.org/stable/) covers driver and
toolkit setup, `CuArray`s, streams and device management; cuTile's
documentation focuses on the tile programming model.


## Quick start

```julia
using CUDA, cuTile
import cuTile as ct

function vadd(a, b, c, tile_size::Int)
    pid = ct.bid(1)
    tile_a = ct.load(a; index=pid, shape=(tile_size,))
    tile_b = ct.load(b; index=pid, shape=(tile_size,))
    ct.store(c; index=pid, tile=tile_a + tile_b)
    return
end

vector_size = 2^20
tile_size = 16
grid = (cld(vector_size, tile_size), 1, 1)

a, b = CUDA.rand(Float32, vector_size), CUDA.rand(Float32, vector_size)
c = CUDA.zeros(Float32, vector_size)

@cuda backend=cuTile blocks=grid vadd(a, b, c, ct.Constant(tile_size))

@assert c == a .+ b
```

For more usage instructions and other information, please refer to [the
documentation][docs-stable-url]. If you are porting from cuTile Python, start with the
[comparison page](https://juliagpu.github.io/cuTile.jl/stable/man/comparison/). Further
examples are available under `examples/`.


## Performance

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

These numbers are a snapshot of one machine and one toolchain version, taken with the GPU
clocks locked. Both the hardware and the `tileiras` version materially affect the result, and
the single-digit-percent rows are within the noise of an unlocked-clock run. For tuning
knobs, see the [performance
chapter](https://juliagpu.github.io/cuTile.jl/stable/man/performance/) of the documentation.


## Acknowledgments

cuTile.jl is inspired by [cuTile-Python](https://github.com/NVIDIA/cutile-python/),
licensed under Apache 2.0 by NVIDIA Corporation & Affiliates.
