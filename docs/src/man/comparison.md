# Comparison with cuTile Python

cuTile.jl follows Julia conventions, which differ from the
[cuTile Python](https://github.com/NVIDIA/cutile-python/) API in several ways. This page maps
the two APIs onto each other, then explains the semantic differences that a mechanical
translation would miss.

Both implementations target the same Tile IR, so the table below also gives the Tile IR
operation each construct lowers to. That column is useful when reading `code_tiled` output
(see [Debugging](debugging.md)) or the
[Tile IR operation reference](https://docs.nvidia.com/cuda/tile-ir/latest/sections/operations.html).

## Symbol map

Grouped as in cuTile Python's operation reference. `ct.` on the Julia side means the cuTile
module; unprefixed names are `Base` functions that cuTile overlays.

### Load/store

| cuTile Python | cuTile.jl | Tile IR |
|---------------|-----------|---------|
| `bid(i)` | `ct.bid(i+1)` | `get_tile_block_id` |
| `num_blocks(i)` | `ct.num_blocks(i+1)` | `get_num_tile_blocks` |
| `num_tiles` | `ct.num_tiles` | `get_index_space_shape` |
| `load` | `ct.load` | `load_view_tko` |
| `store` | `ct.store` | `store_view_tko` |
| `load_advanced_indexing` | `ct.load(@view a[idx, …], shape)` | `make_gather_scatter_view` + `load_view_tko` |
| `store_advanced_indexing` | `ct.store(@view a[idx, …], tile)` | `make_gather_scatter_view` + `store_view_tko` |
| `gather` | `ct.gather` | `offset` + `load_ptr_tko` |
| `scatter` | `ct.scatter` | `offset` + `store_ptr_tko` |

### Factory

| cuTile Python | cuTile.jl | Tile IR |
|---------------|-----------|---------|
| `arange` | `ct.arange` | `iota` |
| `astile` | `ct.Tile(x)` | `constant` |
| `full` | `fill` | `constant` |
| `ones` | `ones` | `constant` |
| `zeros` | `zeros` | `constant` |

### Shape and dtype

| cuTile Python | cuTile.jl | Tile IR |
|---------------|-----------|---------|
| `cat` | `ct.cat` | `cat` |
| `broadcast_to` | `ct.broadcast_to` | `broadcast` |
| `expand_dims` | `reshape` | `reshape` |
| `reshape` | `reshape` | `reshape` |
| `permute` | `permutedims` | `permute` |
| `transpose` | `transpose` | `permute` |
| `astype` | `convert(ct.Tile{T}, x)` | `ftof`, `ftoi`, `itof`, `exti`, `trunci` |
| `bitcast` | `reinterpret` | `bitcast` |
| `pack_to_bytes` | `reinterpret` | `pack` |
| `unpack_from_bytes` | `reinterpret` | `unpack` |

Julia's `reinterpret` covers all three of Python's reinterpretation functions: it emits a
`bitcast` when the widths match and `pack`/`unpack` when they don't.

### Reduction and scan

| cuTile Python | cuTile.jl | Tile IR |
|---------------|-----------|---------|
| `sum` | `sum` | `reduce` |
| `max` | `maximum` | `reduce` |
| `min` | `minimum` | `reduce` |
| `prod` | `prod` | `reduce` |
| `argmax` / `argmin` | `argmax` / `argmin` | `reduce` |
| `reduce` | `reduce`, `mapreduce` | `reduce` |
| `cumsum` / `cumprod` | `cumsum` / `cumprod` | `scan` |
| `scan` | `accumulate` | `scan` |

!!! warning "`max` means different things"
    Python's `ct.max`/`ct.min` are *reductions*, and its `ct.maximum`/`ct.minimum` are
    *element-wise*. Julia is the other way around, following `Base`: `maximum`/`minimum`
    reduce, `max`/`min` are element-wise. Translating `ct.max(tile, axis=0)` to `max` rather
    than `maximum` compiles and computes the wrong thing.

### Matmul

| cuTile Python | cuTile.jl | Tile IR |
|---------------|-----------|---------|
| `mma` | `muladd` | `mmaf`, `mmai` |
| `mma_scaled` | `ct.muladd_scaled` | `mmaf_scaled` |
| `matmul` | `*` | `mmaf`, `mmai` |

### Selection

| cuTile Python | cuTile.jl | Tile IR |
|---------------|-----------|---------|
| `where` | `ifelse` | `select` |
| `extract` | `ct.extract` | `extract` |
| `insert` | `ct.insert` | `insert` |

### Math

| cuTile Python | cuTile.jl | Tile IR |
|---------------|-----------|---------|
| `add`, `sub`, `mul` | `.+`, `.-`, `.*` | `addf`/`addi`, `subf`/`subi`, `mulf`/`muli` |
| `truediv` | `./` | `divf` |
| `floordiv` | `fld` | `divi` |
| `cdiv` | `cld` | `divi` |
| `pow` | `.^` | `pow` |
| `atan2(y, x)` | `atan(y, x)` | `atan2` |
| `mod` | `mod.` | `remf`, `remi` |
| `divmod` | `ct.divmod` | `divi` + `remi` |
| `minimum`, `maximum` | `min.`, `max.` | `minf`/`mini`, `maxf`/`maxi` |
| `negative` | `-` | `negf`, `negi` |
| `abs` | `abs` | `absf`, `absi` |
| `isnan` | `isnan` | `cmpf` |
| `exp`, `exp2`, `log`, `log2` | same | `exp`, `exp2`, `log`, `log2` |
| `sqrt`, `rsqrt` | same | `sqrt`, `rsqrt` |
| `sin`, `cos`, `tan` | same | `sin`, `cos`, `tan` |
| `sinh`, `cosh`, `tanh` | same | `sinh`, `cosh`, `tanh` |
| `floor`, `ceil` | same | `floor`, `ceil` |
| — | `fma` | `fma` |
| — | `mul_hi` | `mulhii` |

### Bitwise and comparison

| cuTile Python | cuTile.jl | Tile IR |
|---------------|-----------|---------|
| `bitwise_and`, `bitwise_or`, `bitwise_xor` | `&`, `\|`, `xor` | `andi`, `ori`, `xori` |
| `bitwise_lshift`, `bitwise_rshift` | `<<`, `>>` (`>>>` unsigned) | `shli`, `shri` |
| `bitwise_not` | `~` | `xori` |
| `greater`, `greater_equal` | `>`, `>=` | `cmpf`, `cmpi` |
| `less`, `less_equal` | `<`, `<=` | `cmpf`, `cmpi` |
| `equal`, `not_equal` | `==`, `!=` | `cmpf`, `cmpi` |

### Atomics

| cuTile Python | cuTile.jl | Tile IR |
|---------------|-----------|---------|
| `atomic_cas` | `ct.atomic_cas` | `atomic_cas_tko` |
| `atomic_xchg`, `atomic_add`, `atomic_max`, `atomic_min`, `atomic_and`, `atomic_or`, `atomic_xor` | same, `ct.`-prefixed | `atomic_rmw_tko` |
| — | `ct.atomic_store_*`, `ct.@atomic` | `atomic_red_view_tko` |

### Utility and metaprogramming

| cuTile Python | cuTile.jl | Tile IR |
|---------------|-----------|---------|
| `printf`, `print` | `print`, `println` | `print_tko` |
| `assert_` | `ct.@assert` | `assert` |
| `assume_divisible_by` | `ct.assume_divisible_by` | `assume` |
| `static_assert`, `static_eval`, `static_iter` | ordinary Julia code | — |

Python needs explicit metaprogramming helpers because its kernels are traced. In Julia,
compile-time evaluation is what the compiler does by default: a `for` loop over a literal
range unrolls, `@assert` on a constant folds away, and `ct.Constant` values participate in
inference.

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

### Tile IR coverage

Of the 100 operations in the Tile IR 13.3 specification, cuTile.jl emits all but
`cuda_tile.alloca`, `cuda_tile.global`, `cuda_tile.get_global` and `cuda_tile.ptr_to_ptr`,
and additionally emits `cuda_tile.insert` from v13.4.

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

Python annotates constant parameters in the kernel signature and passes plain values at
launch. Julia is the reverse: kernel signatures use plain types, and constants are wrapped at
launch:

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

`ct.Constant` arguments generate no kernel parameter; the value is embedded directly in the
compiled code. Different constant values produce different kernel specializations.

## Broadcasting and math functions

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

## Broadcasting shape alignment

cuTile.jl uses Julia's standard left-aligned broadcast shape rules: dimensions are matched
starting from the first (leftmost) dimension. cuTile Python uses NumPy-style right-aligned
rules, where dimensions are matched from the last (rightmost) dimension.

This means a 1D `(N,)` tile cannot broadcast with a 2D `(M, N)` tile in Julia, because
dimension 1 has size `N` vs `M`. In NumPy/Python, `(N,)` would be right-aligned to `(1, N)`
and broadcast to `(M, N)`. Use `reshape` to get the desired alignment, as shown in
[Operations](operations.md#Broadcasting-shape-alignment).

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

Both implementations match the tile rank to the target's rank automatically; the rules are
documented under [Memory](memory.md#Automatic-rank-matching).
