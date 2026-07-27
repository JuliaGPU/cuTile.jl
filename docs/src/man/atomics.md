# Atomics

cuTile offers atomics in three flavours, which differ in what they return and how strongly
they order:

| Family | Returns | Ordering |
|--------|---------|----------|
| [`ct.atomic_cas`](@ref cuTile.atomic_cas), [`ct.atomic_xchg`](@ref cuTile.atomic_xchg), [`ct.atomic_add`](@ref cuTile.atomic_add) and friends | the old value | configurable |
| [`ct.atomic_store_*`](@ref cuTile.atomic_store_add) | `nothing` | fixed: relaxed, device-wide |
| [`ct.@atomic`](@ref cuTile.@atomic) | depends on form | configurable |

The read-modify-write `atomic_*` functions accept `memory_order` (default:
`ct.MemoryOrder.AcqRel`) and `memory_scope` (default: `ct.MemScope.Device`)
keyword arguments; see [Memory Model](memory_model.md). Their indices may be
scalars or tiles.


## View-based reductions

The `atomic_store_*` functions use Tile IR's view-based atomic reductions and
require Tile IR bytecode v13.3 or newer. The destination may be a `TileArray` or
a `TiledView` returned by [`eachtile`](memory.md#Tile-windows); tile updates are
broadcast to the selected window.

They cover fewer element types than the read-modify-write family, and `BFloat16`
addition additionally requires Hopper (`sm_90`) or newer; see
[Compatibility](compatibility.md) and [`atomic_store_add`](@ref
cuTile.atomic_store_add).


## `ct.@atomic`

`ct.@atomic` provides statement and value forms:

```julia
ct.@atomic windows[i, j] += update
ct.@atomic counters[i] = max(counters[i], value)
old_new = ct.@atomic counters[i] + value
```

Statement forms return `nothing` and default to relaxed (`:monotonic`) ordering.
Value forms return `old => new` and default to `:acquire_release`. The supported
operators are `+`, `-`, `max`, `min`, `&`, `|`, and `⊻`. An explicit leading
order may be `:monotonic`, `:acquire`, `:release`, or `:acquire_release`;
ordered statements and value forms require a `TileArray`, while `TiledView`
reductions support only `:monotonic`.
