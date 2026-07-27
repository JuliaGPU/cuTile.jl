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

This applies to `bid`, `num_blocks`, `permutedims`, `reshape`, dimension arguments, etc.


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
```

```julia
# Julia
result = sum(tile; dims=2)                    # (M, N) → (M, 1)
result = dropdims(sum(tile; dims=2); dims=2)  # (M, N) → (M,)
```


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

Grouped as in cuTile Python's operation reference. `ct.` on the Julia side means
the cuTile module; unprefixed names are `Base` functions that cuTile overlays.

### Load/store

| cuTile Python | cuTile.jl |
|---------------|-----------|
| `bid(i)` | `ct.bid(i+1)` |
| `num_blocks(i)` | `ct.num_blocks(i+1)` |
| `num_tiles` | `ct.num_tiles` |
| `load` | `ct.load` |
| `store` | `ct.store` |
| `load_advanced_indexing` | `ct.load(@view a[idx, …], shape)` |
| `store_advanced_indexing` | `ct.store(@view a[idx, …], tile)` |
| `gather` | `ct.gather` |
| `scatter` | `ct.scatter` |

### Factory

| cuTile Python | cuTile.jl |
|---------------|-----------|
| `arange` | `ct.arange` |
| `astile` | `ct.Tile(x)` |
| `full` | `fill` |
| `ones` | `ones` |
| `zeros` | `zeros` |

### Shape and dtype

| cuTile Python | cuTile.jl |
|---------------|-----------|
| `cat` | `ct.cat` |
| `broadcast_to` | `ct.broadcast_to` |
| `expand_dims` | `reshape` |
| `reshape` | `reshape` |
| `permute` | `permutedims` |
| `transpose` | `transpose` |
| `astype` | `convert(ct.Tile{T}, x)` |
| `bitcast` | `reinterpret` |
| `pack_to_bytes` | `reinterpret` |
| `unpack_from_bytes` | `reinterpret` |

Julia's `reinterpret` covers all three of Python's reinterpretation functions,
dispatching on whether the source and target element widths match.

### Reduction and scan

| cuTile Python | cuTile.jl |
|---------------|-----------|
| `sum` | `sum` |
| `max` | `maximum` |
| `min` | `minimum` |
| `prod` | `prod` |
| `argmax` / `argmin` | `argmax` / `argmin` |
| `reduce` | `reduce`, `mapreduce` |
| `cumsum` / `cumprod` | `cumsum` / `cumprod` |
| `scan` | `accumulate` |

!!! warning "`max` means different things"

    Python's `ct.max`/`ct.min` are *reductions*, and its `ct.maximum`/`ct.minimum` are
    *element-wise*. Julia is the other way around, following `Base`: `maximum`/`minimum`
    reduce, `max`/`min` are element-wise. Translating `ct.max(tile, axis=0)` to `max` rather
    than `maximum` compiles and computes the wrong thing.

### Matmul

| cuTile Python | cuTile.jl |
|---------------|-----------|
| `mma` | `muladd` |
| `mma_scaled` | `ct.muladd_scaled` |
| `matmul` | `*` |

### Selection

| cuTile Python | cuTile.jl |
|---------------|-----------|
| `where` | `ifelse` |
| `extract` | `ct.extract` |
| `insert` | `ct.insert` |

### Math

| cuTile Python | cuTile.jl |
|---------------|-----------|
| `add`, `sub`, `mul` | `.+`, `.-`, `.*` |
| `truediv` | `./` |
| `floordiv` | `fld` |
| `cdiv` | `cld` |
| `pow` | `.^` |
| `atan2(y, x)` | `atan(y, x)` |
| `mod` | `mod.` |
| `divmod` | `ct.divmod` |
| `minimum`, `maximum` | `min.`, `max.` |
| `negative` | `-` |
| `abs` | `abs` |
| `isnan` | `isnan` |
| `exp`, `exp2`, `log`, `log2` | same |
| `sqrt`, `rsqrt` | same |
| `sin`, `cos`, `tan` | same |
| `sinh`, `cosh`, `tanh` | same |
| `floor`, `ceil` | same |
| — | `fma` |
| — | `mul_hi` |

### Bitwise and comparison

| cuTile Python | cuTile.jl |
|---------------|-----------|
| `bitwise_and`, `bitwise_or`, `bitwise_xor` | `&`, `\|`, `xor` |
| `bitwise_lshift`, `bitwise_rshift` | `<<`, `>>` (`>>>` unsigned) |
| `bitwise_not` | `~` |
| `greater`, `greater_equal` | `>`, `>=` |
| `less`, `less_equal` | `<`, `<=` |
| `equal`, `not_equal` | `==`, `!=` |

### Atomics

| cuTile Python | cuTile.jl |
|---------------|-----------|
| `atomic_cas` | `ct.atomic_cas` |
| `atomic_xchg`, `atomic_add`, `atomic_max`, `atomic_min`, `atomic_and`, `atomic_or`, `atomic_xor` | same, `ct.`-prefixed |
| — | `ct.atomic_store_*`, `ct.@atomic` |

### Utility and metaprogramming

| cuTile Python | cuTile.jl |
|---------------|-----------|
| `printf`, `print` | `print`, `println` |
| `assert_` | `ct.@assert` |
| `assume_divisible_by` | `ct.assume_divisible_by` |
| `static_assert`, `static_eval`, `static_iter` | ordinary Julia code |

Python needs explicit metaprogramming helpers because its kernels are traced. In
Julia, compile-time evaluation is what the compiler does by default: a `for`
loop over a literal range unrolls, `@assert` on a constant folds away, and
`ct.Constant` values participate in inference.

### Types and enums

| cuTile Python | cuTile.jl |
|---------------|-----------|
| `Array` | `ct.TileArray` |
| `TiledView` | result of `eachtile` |
| `Slice` | result of `@view` / `view` |
| `RoundingMode` | `ct.Rounding` |
| `PaddingMode` | `ct.PaddingMode` |

### Not available in cuTile.jl

`tune.exhaustive_search`, `tune.TuningResult`, `tune.Measurement`, `kernel.replace_hints`
and `compiler_timeout` have no equivalent: there is no autotuning interface. Neither is
there a JAX foreign-function interface (`jax.cutile_call`, `jax.OutputPlaceholder`,
`jax.InputOutput`).

### Not available in cuTile Python

| cuTile.jl | Description |
|-----------|-------------|
| `eachtile` | Indexable device-side collection of tile windows, with controllable step |
| `rand`, `randn`, `randexp`, `ct.DeviceRNG` | In-kernel random numbers ([Random Numbers](random.md)) |
| `ct.RNG`, `ct.rand!` | Host-side array filling with the same generator |
| `ct.Tiled`, `ct.@.` | Fused broadcast over `CuArray`s without writing a kernel ([Host-level Operations](host.md)) |
| `map`, `reduce`, `mapreduce`, `accumulate` with closures | Arbitrary Julia functions, not a fixed operator set |
| `ct.@atomic` | Julia-style atomic reduction syntax |
| `repeat`, `dropdims`, `count`, `any`, `all` | Additional `Base` operations |

### Coverage

Neither implementation is a subset of the other, and both cover essentially the
whole underlying instruction set: of the 100 operations in the Tile IR 13.3
specification, cuTile.jl emits all but four, none of which has a Julia-level
surface (stack allocation, module-level globals, and pointer-to-pointer casts).
