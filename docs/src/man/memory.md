# Memory

Kernels move data between global-memory arrays and tiles with `ct.load` and
`ct.store`.

| Operation | Description |
|-----------|-------------|
| `ct.load(arr; index, shape, ...)` | Load a tile from array |
| `ct.store(arr; index, tile, ...)` | Store a tile to array |
| `ct.gather(arr, indices; ...)` | Gather elements by index tile |
| `ct.scatter(arr, indices, tile; ...)` | Scatter elements by index tile |

Their keyword arguments differ: `load` controls padding and both `load` and
`store` accept an axis order and TMA hint; `gather` and `scatter` accept a
per-element mask. All four support bounds checking and a latency hint. See
[`load`](@ref cuTile.load), [`store`](@ref cuTile.store), [`gather`](@ref
cuTile.gather) and [`scatter`](@ref cuTile.scatter) for the exact signatures.

```julia
# Gather with user mask and custom padding for masked-out elements
tile = ct.gather(arr, indices; mask=valid_mask, padding_value=-1.0f0)

# Scatter with mask (only write where mask is true)
ct.scatter(arr, indices, tile; mask=active_mask)
```

The `latency` hint, and the `allow_tma` hint on loads and stores, influence how
memory traffic is scheduled; see [Performance](performance.md). Version
requirements mentioned on this page are collected in
[Compatibility](compatibility.md).


## Automatic rank matching

`ct.load` and `ct.store` automatically match the tile rank to that of the target:

- **Lower rank**: trailing `1`s are appended. Loading `(M, N)` from a 4D array
  internally uses `(M, N, 1, 1)`. Storing a scalar tile into a 2D array pads to
  `(1, 1)`.
- **Higher rank**: trailing `1`s are stripped. Storing `(M, 1)` into a 1D array
  reshapes to `(M,)`. Non-trailing singletons (e.g. from `sum(tile; dims=1)`)
  require explicit `dropdims`.


## Indexing

| Operation | Description |
|-----------|-------------|
| `arr[i, j, ...]` | Load scalar element from `TileArray` |
| `arr[i, j, ...] = val` | Store scalar element to `TileArray` |
| `tile[i, j, ...]` | Extract scalar from `Tile` |
| `setindex(tile, val, i, j, ...)` | Return new `Tile` with element replaced |


## Views

| Operation | Description |
|-----------|-------------|
| `@view arr[r1:r2, :, ...]` / `view(arr, ...)` | Sub-range view |
| `permutedims(arr, perm)` | Permute axes (1-indexed) |
| `transpose(arr)` | 2D transpose (`permutedims(arr, (2, 1))`) |
| `reshape(arr, dims)` | Column-major reshape, requires contiguous source |

`@view` and `view` derive a sub-range `TileArray` from an existing one. Each
index must be `:`, a `UnitRange` (e.g. `i:j`), or a positive `StepRange` (e.g.
`i:s:j`); scalar `Int` and `CartesianIndex` forms are rejected at compile time.
A StepRange changes the element stride inside the resulting TileArray. The
result can be passed to `ct.load`/`ct.store` (or sliced again). Runtime asserts
verify that ranges start at ≥ 1 and have a positive step; negative steps cannot
be represented.

```julia
function rowsum(a, b, r1::Int32, r2::Int32)
    sub = @view a[r1:r2, :]                    # sub-range TileArray
    tile = ct.load(sub, (1, 1), (4, 4))
    ct.store(b, (1, 1), sum(tile; dims=2))
    return
end
```

### Sparse views

For a 2D-or-higher array, one 1D integer `Tile` index plus integer unit ranges
(or `:`) in every other dimension creates a sparse view consumed only by
`ct.load` and `ct.store`. Public indices are one-based; the load shape is
explicit and static while range starts may be runtime values. A `:` dense
dimension starts at element 1 and takes its extent from the load shape.

```julia
rows = ct.arange(4; start=1, step=2)
selected = @view a[rows, col_start:col_start+3]
tile = ct.load(selected, (4, 4); padding_mode=ct.PaddingMode.Zero)
ct.store(selected, tile)
```

Sparse loads apply the requested padding and stores clip partially out-of-bounds
elements. Repeated sparse indices are valid for loads, but conflicting stores
are undefined. This requires Tile IR v13.3. Direct bracket access, view atomics,
and Python-style advanced-indexing function names are intentionally not
provided.


## Tile windows

`eachtile` creates a small, indexable device-side collection of fixed-shape
tiles. Its indices are 1-based and `step` (one entry per tile dimension)
controls tile origins, not the element stride inside a tile. `size(tiles, d)` is
the number of tiles along `d`: on the host it computes `cld(size(a, d),
step[d])` for launch-grid sizing, while inside a kernel it queries the Tile IR
backend for the authoritative index-space count:

```julia
adjacent = eachtile(a, (8, 8))              # default: step == (8, 8)
overlap  = eachtile(a, (8, 8); step=(4, 8)) # neighboring windows overlap
gapped   = eachtile(a, (8, 8); step=(16, 8)) # gaps between row windows

tile = overlap[2, 1]
overlap[2, 1] = tile
```

Equal shape and step work at any supported bytecode version; unequal values
require Tile IR bytecode v13.3 or newer. This is distinct from `@view a[1:2:end,
:]`, which steps individual elements rather than tile origins. See
[`eachtile`](@ref cuTile.eachtile) for the remaining keyword arguments.
