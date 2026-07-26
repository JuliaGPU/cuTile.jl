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

**This package is in beta.** Most Tile IR features are implemented and the package has been
verified on the benchmarks and tests included in the repository. Interfaces and APIs may
still change without notice.


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


## Acknowledgments

cuTile.jl is inspired by [cuTile-Python](https://github.com/NVIDIA/cutile-python/),
licensed under Apache 2.0 by NVIDIA Corporation & Affiliates.

The IRStructurizer component is based on [SPIRV.jl](https://github.com/serenity4/SPIRV.jl)
by [Cédric Belmant](https://github.com/serenity4).
