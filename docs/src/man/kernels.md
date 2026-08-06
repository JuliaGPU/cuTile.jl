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

Note that `tile_size` is annotated as a plain `Int` even though a tile shape has to be a
compile-time value. That is because the wrapping happens at the launch site, not in the
signature: see [compile-time arguments](execution.md#Compile-time-arguments). Launching,
argument conversion and specialization are all covered in
[Compiling and Launching](execution.md).


## Control flow

Standard Julia control flow works inside kernels and is compiled to structured Tile IR
operations:

| Construct | Description |
|-----------|-------------|
| `if`/`elseif`/`else` | Conditional branching |
| `for i in start:stop` | Counted loops |
| `for i in start:step:stop` | Stepped loops |
| `while cond ... end` | While loops |


## Differences from Julia

Kernels compile with Julia semantics wherever Tile IR can express them, but
exceptions, some conversions, and bounds handling need different machinery.
See [Differences from Julia](julia_differences.md).
