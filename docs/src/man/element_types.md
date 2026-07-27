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
element-wise, convert to an arithmetic float first.

This is why a `Float32` matmul that wants tensor cores converts its *operands*
to `TFloat32` while leaving the accumulator `Float32`: the operands only ever
feed a multiply, but the accumulator is summed into. The [matrix multiplication
tutorial](../tutorials/matmul.md) works through that in context.


## Conversion

| Operation | Description |
|-----------|-------------|
| `convert(Tile{T}, tile)` | Convert element type of a whole tile |
| `T(x)`, `T.(tile)` | Scalar conversion, broadcast element-wise over a tile |
| `reinterpret` | Reinterpret bits rather than convert values |

Converting a `Float32` tile to `TFloat32` is the usual way to opt into tensor-core
acceleration for a matmul:

```julia
a = ct.load(A; index=(bid_m, k), shape=(tm, tk))
a_tf32 = convert(ct.Tile{ct.TFloat32}, a)
```

`reinterpret` covers both same-width bit reinterpretation and, for differing widths, the
packing described above.

!!! note

    Float-to-integer conversions do not throw in kernels; they truncate toward
    zero rather than raising `InexactError`. See [Differences from
    Julia](kernels.md#Differences-from-Julia).
