# cuTile.jl

A Julia package for writing GPU kernels using NVIDIA's tile-based programming model.

**This package is under active development.** Not all Tile IR features are implemented, and
support for the Julia language is limited and only verified on the examples provided here.


## Installation

cuTile.jl has not been registered, so you have to install it directly from the GitHub repository:

```julia
using Pkg
Pkg.add(url="https://github.com/JuliaGPU/cuTile.jl")
```

Requires CUDA.jl for execution and access to `tileiras` for assembling Tile IR to cubins.


## Quick Start

```julia
using CUDA
import cuTile as ct

# Define kernel
function vadd(a, b, c, tile_size::ct.Constant{Int})
    pid = ct.bid(0)
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
```


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
| `bid(axis)` | Block ID (0=x, 1=y, 2=z) |
| `num_blocks(axis)` | Grid size along axis |
| `num_tiles(arr, axis, shape)` | Number of tiles along axis |

### Arithmetic
| Operation | Description |
|-----------|-------------|
| `+`, `-`, `*`, `/` | Element-wise (same shape) |
| `.+`, `.-`, `.*`, `./` | Broadcasting element-wise |
| `^`, `.^` | Power (float only) |

### Construction
| Operation | Description |
|-----------|-------------|
| `zeros(shape, T)` | Zero-filled tile |
| `full(shape, value, T)` | Constant-filled tile |
| `arange(shape, T)` | Sequence `[0, 1, 2, ...]` |

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
| `mma(a, b, acc)` | Matrix multiply-accumulate: `a @ b + acc` |
| `matmul(a, b)` | Matrix multiplication: `a @ b` |

### Reductions
| Operation | Description |
|-----------|-------------|
| `reduce_sum(tile, axis)` | Sum along axis |
| `reduce_max(tile, axis)` | Maximum along axis |

### Math
| Operation | Description |
|-----------|-------------|
| `sqrt(tile)` | Element-wise square root |
| `rsqrt(tile)` | Element-wise reciprocal square root |

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
| `cdiv(a, b)` | Ceiling division |
| `floordiv(a, b)` | Floor division |

### Atomics
| Operation | Description |
|-----------|-------------|
| `atomic_cas(arr, idx, expected, desired)` | Compare-and-swap |
| `atomic_xchg(arr, idx, val)` | Exchange |
| `atomic_add(arr, idx, val)` | Atomic add |


## Gotchas

**Use `while` loops instead of `for`:**
The compiler recognizes simple while-loop patterns but not Julia's iterator-based `for` loops. Write loops as:

```julia
# Do this:
i = 0
while i < n
    # ...
    i += 1
end

# Not this:
for i in 0:n-1
    # ...
end
```

**Tile shapes must be compile-time constants:**
The shape argument to `load`, `zeros`, etc. must be known at compile time. Use `ct.Constant{Int}` for values that need to be passed as arguments but remain constant:

```julia
function kernel(arr, tile_size::ct.Constant{Int})
    tile = ct.load(arr, 0, (tile_size[],))  # tile_size[] extracts the value
end
```

**No scalar returns:**
Kernels must return `nothing`. Use `store` to write results to arrays.

**Broadcasting requires explicit shapes:**
Same-shape operations use `+`, `-`, `*`, `/`. Different shapes require broadcast syntax `.+`, `.-`, `.*`, `./`.
