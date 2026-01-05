# cuTile.jl

A Julia package for writing GPU kernels using NVIDIA's tile-based programming model.

**This package is under active development.** Not all Tile IR features are implemented, and
support for the Julia language is limited and only verified on the examples provided here. Interfaces and APIs may change without notice.


## Installation

cuTile.jl has not been registered, so you have to install it directly from the GitHub repository:

```julia
using Pkg
Pkg.add(url="https://github.com/JuliaGPU/cuTile.jl")
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

Since these types can be verbosed, and are derived from the runtime properties
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

### Broadcasting

Python's binary operators automatically broadcast different shapes. Julia uses standard broadcast syntax:

```python
# Python
a + b  # Automatically broadcasts (16,) + (1, 16) → (1, 16)
```

```julia
# Julia
a + b   # Same shape only
a .+ b  # Broadcasts different shapes
```

This matches how regular Julia arrays behave.


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
