# Comparison with cuTile Python

cuTile.jl follows Julia conventions, which differ from the [cuTile
Python](https://github.com/NVIDIA/cutile-python/) API in several ways. This page
maps the two APIs onto each other, then explains the semantic differences that a
mechanical translation would miss.

Both implementations target the same Tile IR and expose nearly the same
capabilities, so most differences are surface ones: naming, argument
conventions, and where Julia's own semantics take precedence.


## Kernel definition syntax

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

    a_tile = ct.load(a; index=pid, shape=(16,))
    b_tile = ct.load(b; index=pid, shape=(16,))
    result = a_tile + b_tile
    ct.store(c; index=pid, tile=result)

    return
end
```


## Optimization hints

Python passes optimization hints as `@ct.kernel` decorator arguments. Julia uses
`ct.@compiler_options` inside the function body (like `@inline`). See
[Performance](performance.md) for full details.

```python
# Python
@ct.kernel(num_ctas=ct.ByTarget(sm_100=2), occupancy=8)
def matmul(A, B, C, ...):
    ...
```

```julia
# Julia
function matmul(A, B, C, ...)
    ct.@compiler_options num_ctas=ct.ByTarget(v"10.0" => 2) occupancy=8
    ...
end
```


## Launch syntax

cuTile.jl implicitly uses the current task-bound stream from CUDA.jl:

```python
# Python
import cupy as cp
ct.launch(cp.cuda.get_current_stream(), grid, vadd, (a, b, c))
```

```julia
# Julia
@cuda backend=cuTile blocks=grid vadd(a, b, c)
```


## 1-based indexing

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
permutedims(tile, (3, 1, 2))
```

This applies to `bid`, `num_blocks`, `permutedims`, reduction dimensions and
other axis or index arguments.


## Compile-time constants

Python annotates constant parameters in the kernel signature and passes plain
values at launch. Julia is the reverse: kernel signatures use plain types, and
constants are wrapped at launch:

```python
# Python
@ct.kernel
def kernel(a, b, tile_size: ct.Constant[int]):
    tile = ct.load(a, index=(0,), shape=(tile_size,))

ct.launch(stream, grid, kernel, (a, b, 16))
```

```julia
# Julia
function kernel(a, b, tile_size::Int)
    tile = ct.load(a; index=1, shape=(tile_size,))
end

@cuda backend=cuTile blocks=grid kernel(a, b, ct.Constant(16))
```

`ct.Constant` arguments generate no kernel parameter; the value is embedded
directly in the compiled code. Different constant values produce different
kernel specializations.


## Broadcasting and math functions

Python's operators and math functions work directly on tiles with automatic
broadcasting. Julia cuTile follows standard Julia conventions: operators and
math functions apply to scalars, while element-wise application requires
broadcast syntax (`.+`, `exp.(...)`, etc).

`map(f, tiles...)` applies an arbitrary function element-wise to tiles of the
same shape. Broadcast syntax (`.+`, `f.(x, y)`, etc.) combines `map` with
automatic shape broadcasting, so any function that works on scalars "just works"
when broadcast over tiles.

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


## Broadcasting shape alignment

cuTile.jl uses Julia's standard left-aligned broadcast shape rules: dimensions are matched
starting from the first (leftmost) dimension. cuTile Python uses NumPy-style right-aligned
rules, where dimensions are matched from the last (rightmost) dimension.

This means a 1D `(N,)` tile cannot broadcast with a 2D `(M, N)` tile in Julia, because
dimension 1 has size `N` vs `M`. In NumPy/Python, `(N,)` would be right-aligned to `(1, N)`
and broadcast to `(M, N)`. Use `reshape` to get the desired alignment, as shown in
[Supported Operations](../lib/operations.md#Broadcasting-shape-alignment).


## Reductions

Python reductions (`ct.sum`, `ct.max`, etc.) drop the reduced dimension by default
(`keepdims=False`). Julia reductions (`sum`, `maximum`, etc.) always keep it as size 1
(matching `Base` semantics). Use `dropdims` to remove singleton dims afterward.

```python
# Python
result = ct.sum(tile, axis=1)                 # (M, N) → (M,)
result = ct.sum(tile, axis=1, keepdims=True)  # (M, N) → (M, 1)
result = ct.sum(tile, axis=(0, 1))            # (M, N) → ()
```

```julia
# Julia
result = sum(tile; dims=2)                    # (M, N) → (M, 1)
result = dropdims(sum(tile; dims=2); dims=2)  # (M, N) → (M,)
result = sum(tile; dims=(1, 2))               # (M, N) → (1, 1)
result = sum(tile; dims=:)                    # (M, N) → scalar
```

Julia `dims` is a region: its order and repeated entries do not affect the result.


## Scalar access and 0-D tiles

cuTile Python represents single-element loads as 0-D tiles (`shape=()`), which can be used
directly as indices. cuTile.jl uses Julia's standard indexing syntax instead — `getindex`
returns a scalar `T` and `setindex!` stores a scalar:

```python
# Python
expert_id = ct.load(ids, index=bid_m, shape=())
b = ct.load(B, (expert_id, k, bid_n), shape=(1, TILE_K, TILE_N))
```

```julia
# Julia
expert_id = ids[bid_m]
b = ct.load(B; index=(expert_id, k, bid_n), shape=(1, TILE_K, TILE_N))
```


## Automatic rank matching

Both implementations match the tile rank to the target's rank automatically; the
rules are documented under [Memory](memory.md#Automatic-rank-matching).


## Operations

Both packages are conventionally imported as `ct` (`import cuda.tile as ct`,
`import cuTile as ct`), so most operations are spelled identically on both
sides: `ct.bid`, `ct.num_blocks`, `ct.num_tiles`, `ct.load`, `ct.store`,
`ct.gather`, `ct.scatter`, `ct.arange`, `ct.broadcast_to`,
`ct.extract`, `ct.insert`, `ct.divmod`, `ct.assume_divisible_by` and the
`ct.atomic_*` family all keep their names, and so do their keyword arguments
(`index`, `shape`, `order`, `padding_mode`, `padding_value`, `check_bounds`,
`latency`, `memory_order`, `memory_scope`, `mask`).

Two rules cover most of the rest:

- **What `Base` already provides is used unprefixed.** cuTile.jl overlays the
  `Base` function instead of adding a `ct.` one, so `ct.sum(x, axis=0)` becomes
  `sum(x; dims=1)`. The same holds for `prod`, `argmax`, `argmin`, `reduce`,
  `cumsum`, `cumprod`, `reshape`, `transpose`, `abs`, `isnan`, `mod`, `exp`,
  `exp2`, `log`, `log2`, `sqrt`, `sin`, `cos`, `tan`, `sinh`, `cosh`, `tanh`,
  `floor` and `ceil`. `ct.rsqrt` is the exception, as `Base` has no `rsqrt`.
- **Element-wise application needs a dot.** Python's operators and math
  functions map over tiles implicitly; in Julia that is broadcast syntax, so
  `ct.exp(t)` is `exp.(t)`, and `ct.mod(a, b)` is `mod.(a, b)`. See
  [Broadcasting and math functions](#Broadcasting-and-math-functions).

Axis and dimension arguments are 1-based, as described
[above](#1-based-indexing). That also shifts `ct.arange`'s `start` keyword,
which defaults to `1` in Julia and to `0` in Python.

What follows are the operations that don't fall out of those rules, grouped as
in cuTile Python's operation reference.

### Load/store

| cuTile Python | cuTile.jl |
|---------------|-----------|
| `array.tiled_view(tile_shape, traversal_steps=...)` | `eachtile(array, tile_shape; step=...)` |
| `load_advanced_indexing` | `ct.load(@view a[idx, …], shape)` |
| `store_advanced_indexing` | `ct.store(@view a[idx, …], tile)` |

`eachtile` returns the same kind of device-side tiled view. Its indices and
dimension arguments are 1-based, and `step` is cuTile.jl's name for Python's
`traversal_steps`.

Both APIs keep tile loads and stores checked by default: partial loads use their
requested padding and partial stores clip to the array. In both, an explicit
`check_bounds=false` is the unsafe promise that the whole tile is in bounds.

### Factory

| cuTile Python | cuTile.jl |
|---------------|-----------|
| `astile(value, dtype=T)` | `ct.Tile(x)` (scalars only; there is no tuple-literal form) |
| `full(shape, value, dtype)` | `fill(value, dims)` |
| `ones(shape, dtype)` / `zeros(shape, dtype)` | `ones(T, dims)` / `zeros(T, dims)` |

### Shape and dtype

| cuTile Python | cuTile.jl |
|---------------|-----------|
| `expand_dims` | `reshape` |
| `permute` | `permutedims` |
| `astype` | `convert(ct.Tile{T}, x)` |
| `bitcast`, `pack_to_bytes`, `unpack_from_bytes` | `reinterpret` |

Julia's `reinterpret` covers all three of Python's reinterpretation functions,
dispatching on whether the source and target element widths match.
Unlike Python's `reshape`, Julia's tile `reshape` does not accept `-1` as an
inferred dimension; compute and pass every output dimension explicitly.

### Reduction and scan

| cuTile Python | cuTile.jl |
|---------------|-----------|
| `max` / `min` | `maximum` / `minimum` |
| `reduce` | `reduce`, `mapreduce` |
| `scan` | `accumulate` |

!!! warning "`max` means different things"

    Python's `ct.max`/`ct.min` are *reductions*, and its `ct.maximum`/`ct.minimum` are
    *element-wise*. Julia is the other way around, following `Base`: `maximum`/`minimum`
    reduce, `max`/`min` are element-wise. Translating `ct.max(tile, axis=0)` to `max` rather
    than `maximum` compiles and computes the wrong thing.

Julia's reductions also keep the reduced dimension, as covered under
[Reductions](#Reductions).

### Matmul

| cuTile Python | cuTile.jl |
|---------------|-----------|
| `mma` | `muladd` |
| `mma_scaled` | `ct.muladd_scaled` |
| `matmul`, `x @ y` | `*` |

### Selection

| cuTile Python | cuTile.jl |
|---------------|-----------|
| `where` | `ifelse.` |

### Math, bitwise and comparison

Python offers both operators and named functions here; Julia has only the
operators, broadcast:

| cuTile Python | cuTile.jl |
|---------------|-----------|
| `add`, `sub`, `mul`, `truediv`, `pow` | `.+`, `.-`, `.*`, `./`, `.^` |
| `negative` | `-` (no dot needed) |
| `floordiv`, `cdiv` | `fld.`, `cld.` |
| `minimum`, `maximum` (element-wise) | `min.`, `max.` |
| `atan2(y, x)` | `atan.(y, x)` |
| `bitwise_and`, `bitwise_or`, `bitwise_xor`, `bitwise_not` | `.&`, `.\|`, `xor.`, `.~` |
| `bitwise_lshift`, `bitwise_rshift` | `.<<`, `.>>` (`.>>>` shifts in zeros) |
| `greater`, `greater_equal`, `less`, `less_equal` | `.>`, `.>=`, `.<`, `.<=` |
| `equal`, `not_equal` | `.==`, `.!=` |

`*` is the one operator that means something different: element-wise multiply in
Python (`@` is matrix multiply), matrix multiply in Julia.

### Utility and metaprogramming

| cuTile Python | cuTile.jl |
|---------------|-----------|
| `printf`, `print` | `print`, `println` |
| `assert_` | `ct.@assert` |
| `static_eval` | compile-time propagation from `ct.Constant` arguments |
| `static_assert` | `Base.@assert` in generated/host code, or `ct.@assert` in a kernel |
| `static_iter` | generated Julia code when unrolling is required; otherwise a normal loop |

Python needs explicit metaprogramming helpers because its kernels are traced.
Julia inference propagates `ct.Constant` values through ordinary expressions,
so many `static_eval` uses need no wrapper. A normal Julia `for` loop still
compiles to a Tile IR loop; it is not implicitly unrolled merely because its
bounds are literals. Use a generated function or another Julia
code-generation construct when compile-time unrolling is actually required.

### Types and enums

| cuTile Python | cuTile.jl |
|---------------|-----------|
| `Array` | `ct.TileArray` |
| `TiledView` | result of `eachtile` |
| `Slice` | result of `@view` / `view` |
| `RoundingMode` | `ct.Rounding` |
| `MemoryScope` | `ct.MemScope` |

Enum members follow Julia's capitalization: `PaddingMode.NEG_ZERO` is
`ct.PaddingMode.NegZero`, `MemoryOrder.ACQ_REL` is `ct.MemoryOrder.AcqRel`, and
`RoundingMode.RN`/`RZ`/`RM`/`RP` are
`ct.Rounding.NearestEven`/`Zero`/`NegInf`/`PosInf`. Julia exposes no equivalent
of `RoundingMode.FULL` or `.RZI`, nor of `MemoryScope.NONE`, while weak ordering
is expressed through `memory_order` instead.
