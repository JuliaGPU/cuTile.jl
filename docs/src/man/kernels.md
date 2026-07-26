# Writing Kernels

A cuTile kernel is an ordinary Julia function. There is no decorator or macro to apply; the
only requirement is that it returns `nothing`:

```julia
import cuTile as ct

function vadd(a, b, c, tile_size::Int)
    pid = ct.bid(1)
    tile_a = ct.load(a; index=pid, shape=(tile_size,))
    tile_b = ct.load(b; index=pid, shape=(tile_size,))
    ct.store(c; index=pid, tile=tile_a + tile_b)
    return
end
```

Argument types may be left unannotated, as above, or constrained — annotating with
`ct.TileArray{T, N}` documents the expected rank and element type and gives better errors:

```julia
function vadd(a::ct.TileArray{T,1}, b::ct.TileArray{T,1}, c::ct.TileArray{T,1},
              tile_size::Int) where {T}
```


## Launching

Kernels are launched through CUDA.jl, which must be imported. cuTile.jl uses the current
task-bound stream:

```julia
using CUDA, cuTile

grid = (cld(vector_size, tile_size), 1, 1)
@cuda backend=cuTile blocks=grid vadd(a, b, c, ct.Constant(tile_size))
```

`CuArray` arguments are converted to [`ct.TileArray`](types.md) automatically. A functional
equivalent, `ct.launch(f, grid, args...)`, is also available and takes the same optimization
hints as keyword arguments; see [Performance](performance.md).


## Compile-time constants

Tile shapes must be compile-time values. To choose them on the host, wrap the value in
`ct.Constant` at the launch site — the kernel signature keeps its plain type:

```julia
function kernel(a, b, tile_size::Int)
    tile = ct.load(a; index=1, shape=(tile_size,))
    ...
end

@cuda backend=cuTile blocks=grid kernel(a, b, ct.Constant(16))
```

`ct.Constant` arguments generate no kernel parameter; the value is embedded directly in the
compiled code. Different constant values therefore produce different kernel specializations.


## Control flow

Standard Julia control flow works inside kernels and is compiled to structured Tile IR
operations:

| Construct | Description |
|-----------|-------------|
| `if`/`elseif`/`else` | Conditional branching |
| `for i in start:stop` | Counted loops (compiled to Tile IR `ForOp`) |
| `for i in start:step:stop` | Stepped loops |
| `while cond ... end` | While loops |


## Differences from Julia

### Some operations are non-throwing

cuTile kernels cannot throw Julia exceptions. Operations that would throw in standard Julia
silently produce truncated or wrapped results instead:

- **Float-to-integer conversions:** `Int32(x)`, `trunc(Int32, x)`, and
  `round(Int32, x, RoundToZero)` silently truncate toward zero rather than throwing
  `InexactError` for non-integer or out-of-range values. Use `unsafe_trunc` for the explicit
  non-throwing primitive.

Use `ct.@assert` to add runtime checks in kernels; see [Debugging](debugging.md).
