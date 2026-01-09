# cuTile.jl

A Julia package for writing GPU kernels using NVIDIA's tile-based programming model.

**This package is under active development.** Not all Tile IR features are implemented, and
support for the Julia language is limited and only verified on the examples provided here.
Interfaces and APIs may change without notice.


## Installation

cuTile.jl has not been registered, and depends on several unregistered packages, so you
have to clone the repository and activate the environment within:

```
$ git clone https://github.com/JuliaGPU/cuTile.jl
$ julia --project=cuTile.jl
julia> using Pkg
julia> Pkg.instantiate()
julia> using cuTile
```

Execution of cuTile kernels requires CUDA.jl to be installed and imported. Furthermore,
only Blackwell GPUs (compute capability 10+) are supported at this time, and the CUDA driver
needs to be version 13 or higher.


## Quick Start

A simple vector addition kernel using cuTile looks like this:

```julia
using CUDA
import cuTile as ct

# Define kernel
function vadd(a, b, c, tile_size::ct.Constant{Int})
    pid = ct.bid(1)
    tile_a = ct.load(a, pid, (tile_size[],))
    tile_b = ct.load(b, pid, (tile_size[],))
    ct.store(c, pid, tile_a + tile_b)
    return
end

# Launch
vector_size = 2^20
tile_size = 16
a, b = CUDA.rand(Float32, vector_size), CUDA.rand(Float32, vector_size)
c = CUDA.zeros(Float32, vector_size)

ct.launch(vadd, (cld(vector_size, tile_size), 1, 1), a, b, c, ct.Constant(tile_size))

@assert c == a .+ b
```

### Inspecting Generated Tile IR

The generated Tile IR can be inspected using the `code_tiled` function:

```julia
ct.code_tiled(vadd, Tuple{ct.TileArray{Float32, 1, ct.ArraySpec{1}(128, true, (0,), (32,))},
                          ct.TileArray{Float32, 1, ct.ArraySpec{1}(128, true, (0,), (32,))},
                          ct.TileArray{Float32, 1, ct.ArraySpec{1}(128, true, (0,), (32,))},
                          ct.Constant{Int64, 16}})
```

Since these types can be verbose, and are derived from the runtime properties
of arrays, it's often easier to use the `@code_tiled` macro instead:

```julia-repl
julia> ct.@code_tiled ct.launch(vadd, (cld(vector_size, tile_size), 1, 1), a, b, c, ct.Constant(tile_size))
// vadd(cuTile.TileArray{Float32, 1, cuTile.ArraySpec{1}(128, true, (0,), (32,))}, cuTile.TileArray{Float32, 1, cuTile.ArraySpec{1}(128, true, (0,), (32,))}, cuTile.TileArray{Float32, 1, cuTile.ArraySpec{1}(128, true, (0,), (32,))}, cuTile.Constant{Int64, 16})

cuda_tile.module @kernels {
  entry @vadd(...) {
    ...
    return
  }
}
```

The former form can be useful on systems without a GPU, since it does not require CUDA.jl,
while the latter needs valid `CuArray`s to be passed to the kernel.


## Performance

Run benchmarks with:

```bash
julia --project examples/benchmarks.jl  # Julia
uv run python examples/benchmarks.py    # Python (for comparison)
```

Benchmarks comparing cuTile.jl against cuTile Python on an RTX 5080:

| Kernel | Julia | Python | Status |
|--------|-------|--------|--------|
| Vector Addition | 813 GB/s | 834 GB/s | OK (-3%) |
| Matrix Transpose | 769 GB/s | 795 GB/s | OK (-3%) |
| Matrix Multiplication | 48.3 TFLOPS | 48.6 TFLOPS | OK (=) |
| Layer Normalization | 254 GB/s | 683 GB/s | https://github.com/JuliaGPU/cuTile.jl/issues/1 (-63%) |
| Batch Matrix Multiply | 31.7 TFLOPS | 31.6 TFLOPS | OK (=) |

Compute-intensive kernels (matmul, batch matmul) perform identically to Python. Memory-bound
kernels (vadd, transpose) are within ~3% of Python. The layernorm kernel is slower due to
conservative token threading in the compiler (see https://github.com/JuliaGPU/cuTile.jl/issues/1).


## Supported Operations

### Memory
| Operation | Description |
|-----------|-------------|
| `load(arr, index, shape)` | Load a tile from array |
| `store(arr, index, tile)` | Store a tile to array |
| `gather(arr, indices)` | Gather elements by index tile |
| `scatter(arr, indices, tile)` | Scatter elements by index tile |

### Grid
| Operation | Description |
|-----------|-------------|
| `bid(axis)` | Block ID (1=x, 2=y, 3=z) |
| `num_blocks(axis)` | Grid size along axis |
| `num_tiles(arr, axis, shape)` | Number of tiles along axis |

### Arithmetic
| Operation | Description |
|-----------|-------------|
| `+`, `-` | Element-wise (same shape only) |
| `tile * scalar`, `tile / scalar` | Scalar multiply/divide |
| `.+`, `.-`, `.*`, `./` | Broadcasting element-wise |
| `.^` | Power (float only, broadcast) |

### Construction
| Operation | Description |
|-----------|-------------|
| `zeros(shape, T)` | Zero-filled tile |
| `full(shape, value, T)` | Constant-filled tile |
| `arange(shape, T)` | Sequence `[1, 2, 3, ...]` |

### Shape
| Operation | Description |
|-----------|-------------|
| `broadcast_to(tile, shape)` | Broadcast to target shape |
| `transpose(tile)` | Transpose 2D tile |
| `reshape(tile, shape)` | Reshape (same element count) |
| `permute(tile, perm)` | Permute dimensions |
| `extract(tile, index, shape)` | Extract sub-tile |
| `cat((a, b), axis)` | Concatenate tiles |

### Matrix
| Operation | Description |
|-----------|-------------|
| `a * b` | Matrix multiplication: `a @ b` |
| `muladd(a, b, acc)` | Matrix multiply-accumulate: `a * b + acc` |

### Reductions
| Operation | Description |
|-----------|-------------|
| `reduce_sum(tile, axis)` | Sum along axis |
| `reduce_max(tile, axis)` | Maximum along axis |

### Math
| Operation | Description |
|-----------|-------------|
| `sqrt.(tile)`, `sqrt(x)` | Square root (tile broadcast or scalar) |
| `rsqrt.(tile)`, `rsqrt(x)` | Reciprocal square root |
| `exp.(tile)`, `exp(x)` | Natural exponential |
| `exp2.(tile)`, `exp2(x)` | Base-2 exponential |
| `log.(tile)`, `log(x)` | Natural logarithm |
| `log2.(tile)`, `log2(x)` | Base-2 logarithm |
| `sin.(tile)`, `cos.(tile)`, etc. | Trigonometric functions |
| `fma.(a, b, c)`, `fma(x, y, z)` | Fused multiply-add (tile broadcast or scalar) |
| `abs.(tile)`, `abs(x)` | Absolute value |
| `max(a, b)`, `min(a, b)` | Maximum/minimum (scalars) |

### Comparison
| Operation | Description |
|-----------|-------------|
| `.<`, `.>`, `.<=`, `.>=` | Element-wise comparisons (return `Tile{Bool}`) |
| `.==`, `.!=` | Element-wise equality |
| `where(cond, x, y)` | Conditional selection |

### Type Conversion
| Operation | Description |
|-----------|-------------|
| `astype(tile, T)` | Convert element type |
| `convert(Tile{T}, tile)` | Julia-style conversion |

### Integer Arithmetic
| Operation | Description |
|-----------|-------------|
| `cld(a, b)` | Ceiling division |
| `fld(a, b)` | Floor division |
| `div(a, b)` | Truncating division |
| `mul_hi.(tile_a, tile_b)`, `mul_hi(x, y)` | High bits of integer multiply (use `Base.mul_hi` on Julia 1.13+) |

### Atomics
| Operation | Description |
|-----------|-------------|
| `atomic_cas(arr, idx, expected, desired)` | Compare-and-swap |
| `atomic_xchg(arr, idx, val)` | Exchange |
| `atomic_add(arr, idx, val)` | Atomic add |


## Differences from cuTile Python

cuTile.jl follows Julia conventions, which differ from the Python API in several ways:


### Kernel definition syntax

Kernels don't need a decorator, but do have to return `nothing`:

```python
# Python
@ct.kernel
def vadd(a, b, c):
    pid = ct.bid(0)

    a_tile = ct.load(a, index=(pid,), shape=(16,))
    b_tile = ct.load(b, index=(pid,), shape=(16,))
    result = a_tile + b_tile
    ct.store(c, index=(pid, ), tile=result)
```

```julia
# Julia
function vadd(a, b, c)
    pid = ct.bid(1)

    a_tile = ct.load(a, pid, (16,))
    b_tile = ct.load(b, pid, (16,))
    result = a_tile + b_tile
    ct.store(c, pid, result)

    return
end
```

### Launch Syntax

cuTile.jl implicitly uses the current task-bound stream from CUDA.jl:

```python
# Python
import cupy as cp
ct.launch(cp.cuda.get_current_stream(), grid, vadd, (a, b, c))
```

```julia
# Julia
ct.launch(vadd, grid, a, b, c)
```

### 1-Based Indexing

All index-based operations use Julia's 1-based convention:

```python
# Python
bid_x = ct.bid(0)
bid_y = ct.bid(1)
ct.permute(tile, (2, 0, 1))
```

```julia
# Julia
bid_x = ct.bid(1)
bid_y = ct.bid(2)
ct.permute(tile, (3, 1, 2))
```

This applies to `bid`, `num_blocks`, `permute`, `reshape`, dimension arguments, etc.

### `Val`-like constants

CuTile.jl uses `ct.Constant{T}` to encode compile-time constant values in the type domain, similar to how `Val` works. An explicit `[]` is needed to extract the value at runtime:

```python
# Python
@ct.kernel
def kernel(a, b, tile_size):
    tile = ct.load(a, index=(0,), shape=(tile_size,))

ct.launch(stream, grid, kernel, (a, b, 16))
```

```julia
# Julia
function kernel(a, b, tile_size::ct.Constant{Int})
    tile = ct.load(a, 1, (tile_size[],))
end

ct.launch(kernel, grid, a, b, ct.Constant(16))
```

### Broadcasting and Math Functions

Python's operators and math functions work directly on tiles with automatic broadcasting.
Julia cuTile follows standard Julia conventions: Operators and math functions can generally only be applied to scalars, while elementwise application requires broadcast syntax (`.+`, `exp.(...)`, etc).

Some exceptions:

- Scaling operations (`*` and `/`) can be applied directly to tiles and scalars.
- Addition and subtraction can be applied directly to tiles with matching shapes.

```python
# Python
a + b              # Automatically broadcasts (16,) + (1, 16) → (1, 16)
a * b              # Element-wise multiply
result = ct.exp(tile)
```

```julia
# Julia
a + b              # Same shape only
a .+ b             # Broadcasts different shapes
a .* b             # Element-wise multiply (broadcast)
a * b              # Matrix multiplication
tile * 2.0f0       # Scalar multiply
result = exp.(tile)
```


## Limitations

### `for` loops

The compiler recognizes simple while-loop patterns but not Julia's iterator-based `for` loops. Write such loops as:

```julia
# Do this:
i = 1
while i <= n
    # ...
    i += 1
end

# Not this:
for i in 1:n
    # ...
end
```

Also make sure `i`, `n`, and the increment all have the same type.

### Keyword arguments

`load` and `store` use positional arguments instead of keyword arguments:

```python
# Python
ct.load(arr, index=(i, j), shape=(m, n))
ct.store(arr, index=(i, j), tile=t)
```

```julia
# Julia
ct.load(arr, (i, j), (m, n))
ct.store(arr, (i, j), t)
```


## Acknowledgments

cuTile.jl is inspired by [cuTile-Python](https://github.com/NVIDIA/cutile-python/),
licensed under Apache 2.0 by NVIDIA Corporation & Affiliates.

The IRStructurizer component is based on [SPIRV.jl](https://github.com/serenity4/SPIRV.jl)
by [Cédric Belmant](https://github.com/serenity4).
