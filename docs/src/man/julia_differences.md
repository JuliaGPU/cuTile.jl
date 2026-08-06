# Differences from Julia

cuTile kernels are ordinary Julia functions, and cuTile.jl uses Julia semantics
where Tile IR can express them. This page covers the important exceptions.


## Exceptions become traps

Tile IR has no exceptions or stack unwinding. A conditional `throw`, `error`,
or failing `Base.@assert` becomes a device-side assertion; an unconditional
one becomes a compile-time diagnostic. `try`/`catch` is not supported. Use
`ct.@assert` for explicit runtime checks.


## Some operations are non-throwing

Float-to-integer conversions such as `Int32(x)`, `trunc(Int32, x)`, and
`round(Int32, x, RoundToZero)` truncate toward zero rather than throwing
`InexactError`. They behave like Julia's explicit `unsafe_trunc` primitive.


## Tile bounds handling is not `BoundsError`

Julia bounds checks are protective: a correct access has the same result with
or without its check. Tile IR's checked memory operations instead define
partial edge tiles: loads pad missing elements and stores clip them. A tile
whose origin is valid may still extend past the array, so that padding or
clipping is part of a correct kernel's result.

Consequently, `@inbounds` does not alter `ct.load` or `ct.store` bounds
handling. `@inbounds` and `--check-bounds` still control ordinary Julia
`@boundscheck` blocks in a kernel, such as checks in Base range indexing. For
tile memory, use `check_bounds=false` only as an explicit promise that the
whole tile lies within its array. It drops padding, selects Tile IR's unchecked
encoding, and requires bytecode v13.4 or newer.
