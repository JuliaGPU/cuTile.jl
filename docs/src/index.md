# cuTile.jl

*Tile-based GPU programming in Julia.*

cuTile.jl compiles Julia functions to [Tile IR](https://docs.nvidia.com/cuda/tile-ir/),
NVIDIA's portable tile-based instruction set. Rather than writing kernels that describe what
a single GPU thread does, you write kernels that operate on *tiles* — multi-dimensional
array fragments — and leave it to the Tile IR compiler to map those onto the hardware,
including tensor cores and the tensor memory accelerator.

!!! warning "This package is in beta"
    Most Tile IR features are implemented, and the package has been verified on the
    benchmarks and tests included in the repository. Interfaces and APIs may still change
    without notice. See [Compatibility](man/compatibility.md) for what is and isn't
    guaranteed.


## Quick start

A vector addition kernel:

```julia
using CUDA, cuTile
import cuTile as ct

# Define kernel
function vadd(a, b, c, tile_size::Int)
    pid = ct.bid(1)
    tile_a = ct.load(a; index=pid, shape=(tile_size,))
    tile_b = ct.load(b; index=pid, shape=(tile_size,))
    ct.store(c; index=pid, tile=tile_a + tile_b)
    return
end

# Launch
vector_size = 2^20
tile_size = 16

blocks = cld(vector_size, tile_size)
grid = (blocks, 1, 1)

a, b = CUDA.rand(Float32, vector_size), CUDA.rand(Float32, vector_size)
c = CUDA.zeros(Float32, vector_size)

@cuda backend=cuTile blocks=grid vadd(a, b, c, ct.Constant(tile_size))

@assert c == a .+ b
```

Kernels are ordinary Julia functions: no decorator or macro is needed, though they must
return `nothing`. They take array arguments, use [`ct.load`](man/memory.md) and
[`ct.store`](man/memory.md) to move data between global memory and tiles, and operate on
those tiles with standard Julia syntax — `+`, `sum`, `reshape`, broadcasting, and so on.


## Where to go next

- [Installation](installation.md) — driver and hardware requirements.
- [Vector Addition](tutorials/vector_addition.md) — the above kernel, explained line by line.
- [Matrix Multiplication](tutorials/matmul.md) — a GEMM built up in four steps.
- [Programming Model](man/programming_model.md) — grids, blocks, arrays and tiles.
- [Writing Kernels](man/kernels.md) — kernel definition, launching, control flow.
- [Comparison with cuTile Python](man/comparison.md) — if you are porting from
  `cuda.tile`, start here.
- [Debugging](man/debugging.md) — inspecting generated Tile IR.

The [Tile IR specification](https://docs.nvidia.com/cuda/tile-ir/) is the authoritative
reference for the underlying model; these docs cross-reference it rather than restate it.


## Acknowledgments

cuTile.jl is inspired by [cuTile-Python](https://github.com/NVIDIA/cutile-python/),
licensed under Apache 2.0 by NVIDIA Corporation & Affiliates.

The IRStructurizer component is based on [SPIRV.jl](https://github.com/serenity4/SPIRV.jl)
by [Cédric Belmant](https://github.com/serenity4).
