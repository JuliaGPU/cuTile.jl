# Atomics

| Operation | Description |
|-----------|-------------|
| `ct.atomic_cas(arr, idx, expected, desired; ...)` | Compare-and-swap |
| `ct.atomic_xchg(arr, idx, val; ...)` | Exchange, returning the old value |
| `ct.atomic_add(arr, idx, val; ...)` | Add, returning the old value |
| `ct.atomic_max(arr, idx, val; ...)` | Maximum, returning the old value |
| `ct.atomic_min(arr, idx, val; ...)` | Minimum, returning the old value |
| `ct.atomic_or(arr, idx, val; ...)` | Bitwise OR, returning the old value |
| `ct.atomic_and(arr, idx, val; ...)` | Bitwise AND, returning the old value |
| `ct.atomic_xor(arr, idx, val; ...)` | Bitwise XOR, returning the old value |
| `ct.atomic_store_add(dst, idx, update)` | Add a tile without returning old values |
| `ct.atomic_store_{max,min,or,and,xor}(dst, idx, update)` | Other tile reductions without returning old values |
| `ct.@atomic [order] expr` | Julia-style atomic reduction |

The `atomic_*` functions accept `memory_order` (default: `ct.MemoryOrder.AcqRel`) and
`memory_scope` (default: `ct.MemScope.Device`) keyword arguments; see
[Memory Model](memory_model.md). Their indices may be scalars or tiles. Unsigned
`atomic_max` and `atomic_min` use unsigned comparisons. Updates to bitwise atomics must have
exactly the array element type.


## View-based reductions

The `atomic_store_*` functions use Tile IR's view-based atomic reductions and require Tile IR
bytecode v13.3 or newer. The destination may be a `TileArray` or a `TiledView` returned by
[`eachtile`](memory.md#Tile-windows); tile updates are broadcast to the selected window.
These operations always use relaxed, device-wide ordering and return `nothing`.

Addition supports `Int32`, `Int64`, `UInt32`, `UInt64`, `Float16`, `BFloat16`, `Float32`, and
`Float64`; the other reductions support the four integer types. `BFloat16` addition also
requires Hopper (sm_90) or newer — see [Compatibility](compatibility.md).


## `ct.@atomic`

`ct.@atomic` provides statement and value forms:

```julia
ct.@atomic windows[i, j] += update
ct.@atomic counters[i] = max(counters[i], value)
old_new = ct.@atomic counters[i] + value
```

Statement forms return `nothing` and default to relaxed (`:monotonic`) ordering. Value forms
return `old => new` and default to `:acquire_release`. The supported operators are `+`, `-`,
`max`, `min`, `&`, `|`, and `⊻`. An explicit leading order may be `:monotonic`, `:acquire`,
`:release`, or `:acquire_release`; ordered statements and value forms require a `TileArray`,
while `TiledView` reductions support only `:monotonic`.
