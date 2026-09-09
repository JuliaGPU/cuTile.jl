# Element Types

Tiles and arrays are containers; this chapter is about what goes in them.

- **Integers:** `Int8`, `UInt8`, `Int16`, `UInt16`, `Int32`, `UInt32`, `Int64`,
  `UInt64`
- **Boolean:** `Bool`
- **Arithmetic floats:** `Float16`, `BFloat16`, `Float32`, `Float64`
- **Numeric floats:** `TFloat32`\*, `Float8_E4M3FN`\*\*, `Float8_E5M2`\*\*,
  `Float8_E8M0FNU`\*\*, `Float4_E2M1FN`\*\*

\* [`cuTile.TFloat32`](@ref cuTile.TFloat32) is a public 32-bit floating-point
numeric type with truncated mantissa (10 bits), made for tensor core operations.

\*\* [Microscaling
(MX)](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)
numeric types, exported by
[Microfloats.jl](https://github.com/MurrellGroup/Microfloats.jl).
`Float8_E4M3FN` and `Float8_E5M2` (FP8) are also exported by
[DLFP8Types.jl](https://github.com/chengchingwen/DLFP8Types.jl).

Which types are available depends on the Tile IR bytecode version in use; see
[Compatibility](compatibility.md#Features-by-bytecode-version).


## Arithmetic versus numeric floats

**Arithmetic** floats behave as one would expect: they support the whole
operation set. **Numeric** floats are storage and tensor-core operand formats
with intentionally restricted coverage, and do not support general arithmetic,
reductions or scans.

Operations that need those reject a numeric float up front with an error, rather
than letting it fail deeper down in `tileiras`. To compute with such values
element-wise, convert to an arithmetic float first. For example, `x .+ y` on two
`Float8_E4M3FN` tiles fails at kernel compile time with *"operations on a
restricted float element type are not supported"*; casting first is what makes
the intermediate precision explicit:

```julia
f32(t) = convert(ct.Tile{Float32}, t)
sum = f32(x) .+ f32(y)
```

Comparisons and `ifelse` selection are the exceptions, and stay available:
comparisons upcast losslessly, having no result to round, and selection leaves
the values themselves alone.

The rejection is per operation, not per function: applying a custom function
element-wise (`map`, or broadcasting a lambda) over a numeric-float tile is
rejected as well, even when every step inside it is a cast. Convert the tile
with `convert(ct.Tile{T}, tile)` or `T.(tile)` instead.

This is why a `Float32` matmul that wants tensor cores converts its *operands*
to `TFloat32` while leaving the accumulator `Float32`: the operands only ever
feed a multiply, but the accumulator is summed into. The [matrix multiplication
tutorial](../tutorials/matmul.md) works through that in context.


## Conversion

| Operation | Description |
|-----------|-------------|
| `convert(Tile{T}, tile)` | Convert element type of a whole tile |
| `T(x)`, `T.(tile)` | Scalar conversion, broadcast element-wise over a tile |
| `T(x, mode)`, `T.(tile, mode)` | Float-to-float conversion with explicit rounding |
| `reinterpret` | Reinterpret bits rather than convert values |

Converting a `Float32` tile to `TFloat32` is the usual way to opt into tensor-core
acceleration for a matmul:

```julia
a = ct.load(A; index=(bid_m, k), shape=(tm, tk))
a_tf32 = convert(ct.Tile{ct.TFloat32}, a)
```

Float conversions use round-to-nearest, ties-to-even by default. Pass a Base
rounding mode to choose a mode for one conversion:

```julia
down = Float32.(a, RoundDown)
up = map(x -> Float32(x, RoundUp), a)
```

| Mode | Meaning |
|------|---------|
| `RoundNearest` | Nearest, ties to even |
| `RoundToZero` | Toward zero |
| `RoundDown` | Toward negative infinity |
| `RoundUp` | Toward positive infinity |
| `RoundNearestTiesAway` | Nearest, ties away from zero |

Explicit rounding applies only to float-to-float conversions. The supported
source/target pairs follow Tile IR: round-to-nearest supports every float target
except `Float8_E8M0FNU`; the directed modes are limited to particular targets,
and most require bytecode v13.4. Unsupported pairs report their available modes
at compilation. `RoundNearestTiesUp` and `RoundFromZero` have no Tile IR
equivalent.

`Float8_E8M0FNU` is the exception: its ordinary conversion defaults to
`RoundToZero`. Conversions to it with `RoundToZero` or `RoundUp` work from
bytecode v13.3 for `Float32`, `TFloat32`, `Float16`, `BFloat16`,
`Float8_E8M0FNU`, and `Float4_E2M1FN`; `Float64`, `Float8_E5M2`, and
`Float8_E4M3FN` sources require v13.4.

`reinterpret` covers same-width bit reinterpretation directly. When the element
width changes, it preserves the total bit count by scaling the first tile
dimension; for example, reinterpreting a `(16,)` `UInt16` tile as `UInt8`
produces a `(32,)` tile. Differing-width reinterpretation requires Tile IR
v13.3 or newer.

!!! note

    Float-to-integer conversions do not throw in kernels; they truncate toward
    zero rather than raising `InexactError`. See [Differences from
    Julia](julia_differences.md#Some-operations-are-non-throwing).
