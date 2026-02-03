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
| FFT (3-stage Cooley-Tukey) | 508 μs | 230 μs | (-55%) |

Compute-intensive kernels (matmul, batch matmul) perform identically to Python. Memory-bound
kernels (vadd, transpose) are within ~3% of Python. The layernorm kernel is slower due to
conservative token threading in the compiler (see https://github.com/JuliaGPU/cuTile.jl/issues/1).


## Supported Operations

cuTile.jl aims to expose as much functionality as possible through Julia-native constructs
(`+`, `sum`, `reshape`, `broadcast`, etc.) rather than cuTile-specific functions. Operations
prefixed with `ct.` are cuTile intrinsics with no direct Julia equivalent; everything else
uses standard Julia syntax and is overlaid on `Base`.

### Memory
| Operation | Description |
|-----------|-------------|
| `ct.load(arr, index, shape)` | Load a tile from array |
| `ct.store(arr, index, tile)` | Store a tile to array |
| `ct.gather(arr, indices)` | Gather elements by index tile |
| `ct.scatter(arr, indices, tile)` | Scatter elements by index tile |

### Grid
| Operation | Description |
|-----------|-------------|
| `ct.bid(axis)` | Block ID (1=x, 2=y, 3=z) |
| `ct.num_blocks(axis)` | Grid size along axis |
| `ct.num_tiles(arr, axis, shape)` | Number of tiles along axis |

### Arithmetic
| Operation | Description |
|-----------|-------------|
| `+`, `-` | Element-wise (same shape only) |
| `tile * scalar`, `tile / scalar` | Scalar multiply/divide |
| `.+`, `.-`, `.*`, `./`, `.^` | Broadcasting element-wise |

### Construction
| Operation | Description |
|-----------|-------------|
| `ct.zeros(shape, T)` | Zero-filled tile |
| `ct.full(shape, value, T)` | Constant-filled tile |
| `ct.arange(shape, T)` | Sequence `[1, 2, 3, ...]` |

### Shape
| Operation | Description |
|-----------|-------------|
| `ct.broadcast_to(tile, shape)` | Broadcast to target shape |
| `transpose(tile)` | Transpose 2D tile |
| `reshape(tile, shape)` | Reshape (same element count) |
| `ct.permute(tile, perm)` | Permute dimensions |
| `ct.extract(tile, index, shape)` | Extract sub-tile |
| `ct.cat((a, b), axis)` | Concatenate tiles |
| `dropdims(tile; dims)` | Remove singleton dimensions |

### Matrix
| Operation | Description |
|-----------|-------------|
| `a * b` | Matrix multiplication: `a @ b` |
| `muladd(a, b, acc)` | Matrix multiply-accumulate: `a * b + acc` |

### Higher-Order Functions
| Operation | Description |
|-----------|-------------|
| `map(f, tiles...)` | Apply function element-wise (same shape) |
| `f.(tiles...)`, `broadcast(f, tiles...)` | Apply function with shape broadcasting |
| `reduce(f, tile; dims, init)` | Reduction with arbitrary function |
| `accumulate(f, tile; dims, init, rev)` | Scan/prefix-sum with arbitrary function |

### Reductions
| Operation | Description |
|-----------|-------------|
| `sum(tile; dims)` | Sum along axis |
| `prod(tile; dims)` | Product along axis |
| `maximum(tile; dims)` | Maximum along axis |
| `minimum(tile; dims)` | Minimum along axis |
| `any(tile; dims)` | Logical OR along axis |
| `all(tile; dims)` | Logical AND along axis |
| `count(tile; dims)` | Count `true` elements along axis |
| `argmax(tile; dims)` | 1-based index of maximum along axis |
| `argmin(tile; dims)` | 1-based index of minimum along axis |
| `cumsum(tile; dims, rev)` | Cumulative sum |
| `cumprod(tile; dims, rev)` | Cumulative product |

### Math
| Operation | Description |
|-----------|-------------|
| `sqrt.(tile)` | Square root |
| `rsqrt.(tile)` | Reciprocal square root |
| `exp.(tile)` | Natural exponential |
| `exp2.(tile)` | Base-2 exponential |
| `log.(tile)` | Natural logarithm |
| `log2.(tile)` | Base-2 logarithm |
| `sin.(tile)`, `cos.(tile)`, etc. | Trigonometric functions |
| `fma.(a, b, c)` | Fused multiply-add |
| `abs.(tile)` | Absolute value |
| `max(a, b)`, `min(a, b)` | Maximum/minimum (scalars) |

### Comparison
| Operation | Description |
|-----------|-------------|
| `.<`, `.>`, `.<=`, `.>=` | Element-wise comparisons (return `Tile{Bool}`) |
| `.==`, `.!=` | Element-wise equality |
| `ifelse.(cond, x, y)` | Conditional selection |

### Type Conversion
| Operation | Description |
|-----------|-------------|
| `ct.astype(tile, T)` | Convert element type |
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
| `ct.atomic_cas(arr, idx, expected, desired)` | Compare-and-swap |
| `ct.atomic_xchg(arr, idx, val)` | Exchange |
| `ct.atomic_add(arr, idx, val)` | Atomic add |


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
Julia cuTile follows standard Julia conventions: operators and math functions apply to
scalars, while element-wise application requires broadcast syntax (`.+`, `exp.(...)`, etc).

`map(f, tiles...)` applies an arbitrary function element-wise to tiles of the same shape.
Broadcast syntax (`.+`, `f.(x, y)`, etc.) combines `map` with automatic shape broadcasting,
so any function that works on scalars "just works" when broadcast over tiles.

Some non-broadcast shortcuts:

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
map(x -> x * x, tile)  # map with arbitrary lambda
```

### Reductions

Python reductions (`ct.sum`, `ct.max`, etc.) drop the reduced dimension by default (`keepdims=False`). Julia reductions (`sum`, `maximum`, etc.) always keep it as size 1 (matching `Base` semantics). Use `dropdims` to remove singleton dims afterward.

```python
# Python
result = ct.sum(tile, axis=1)           # (M, N) → (M,)
result = ct.sum(tile, axis=1, keepdims=True)  # (M, N) → (M, 1)
```

```julia
# Julia
result = sum(tile; dims=2)              # (M, N) → (M, 1)
result = dropdims(sum(tile; dims=2); dims=2)  # (M, N) → (M,)
```

### Store reshaping

`ct.store` automatically reshapes the tile to match the target array's rank by dropping singleton dimensions (e.g., storing a `(1, N)` tile into a 1D array reshapes it to `(N,)`). Scalar `()` tiles are reshaped to `(1,)`.

### Broadcasting shape alignment

cuTile.jl uses Julia's standard left-aligned broadcast shape rules: dimensions are matched
starting from the first (leftmost) dimension. cuTile Python uses NumPy-style right-aligned
rules, where dimensions are matched from the last (rightmost) dimension.

This means a 1D `(N,)` tile cannot broadcast with a 2D `(M, N)` tile in Julia, because
dimension 1 has size `N` vs `M`. In NumPy/Python, `(N,)` would be right-aligned to `(1, N)`
and broadcast to `(M, N)`.

Use `reshape` to get the desired alignment, just as with regular Julia arrays:

```julia
# Julia: explicitly reshape to align dimensions
a = ct.load(...)              # (N,)
b = ct.load(...)              # (M, N)
result = reshape(a, (1, N)) .+ b  # (1, N) .+ (M, N) → (M, N)
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
