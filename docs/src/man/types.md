# Types

## Element types

- **Integers:** `Int8`, `UInt8`, `Int16`, `UInt16`, `Int32`, `UInt32`, `Int64`, `UInt64`
- **Boolean:** `Bool`
- **Arithmetic floats:** `Float16`, `BFloat16`, `Float32`, `Float64`
- **Numeric floats:** `TFloat32`\*, `Float8_E4M3FN`\*\*, `Float8_E5M2`\*\*,
  `Float8_E8M0FNU`\*\*, `Float4_E2M1FN`\*\*

\* `cuTile.TFloat32` is a public 32-bit floating-point numeric type with truncated mantissa
(10 bits), made for tensor core operations.

\*\* [Microscaling (MX)](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)
numeric types, exported by [Microfloats.jl](https://github.com/MurrellGroup/Microfloats.jl).
`Float8_E4M3FN` and `Float8_E5M2` (FP8) are also exported by
[DLFP8Types.jl](https://github.com/chengchingwen/DLFP8Types.jl).

The distinction between *arithmetic* and *numeric* float types matters. Numeric floats are
storage and tensor-core operand formats with intentionally restricted operation coverage:
they do not support general arithmetic, reductions or scans. Operations that need those
reject the type up front with an error rather than failing downstream in `tileiras`. Convert
to an arithmetic float to compute with the values element-wise.


## `TileArray`

`ct.TileArray{T, N, Spec}` is the type a kernel's array arguments have. It carries a base
pointer, sizes and strides, all of which are flattened into individual kernel parameters at
compile time. `CuArray` arguments are converted to `TileArray` automatically at launch.

The third type parameter, `ct.ArraySpec{N}`, encodes properties the compiler can specialize
on, derived from the array's runtime layout:

- base pointer alignment in bytes,
- whether the array is contiguous (`stride[1] == 1`),
- per-dimension stride and shape divisibility,
- whether two distinct in-bounds indices may alias.

These drive kernel specialization without runtime overhead — a 128-byte-aligned, contiguous
array compiles to different (and faster) code than one with unknown alignment. Because the
spec is part of the type, arrays with different properties produce different kernel
specializations. See [Performance](performance.md).


## `Tile`

`ct.Tile{T, Shape}` is a tile value: element type `T` and a static, power-of-two `Shape`. It
is the result type of [`ct.load`](memory.md) and of most [operations](operations.md).

`Tile` is a mutable struct in Julia terms — deliberately, so that the compiler preserves SSA
references to it through inlining rather than folding tiles into constants.


## `Constant`

`ct.Constant{T, V}` encodes a value in its type parameters, embedding it directly into the
compiled code rather than passing it as a kernel parameter. This is how tile shapes and other
compile-time quantities reach a kernel; see [Writing Kernels](kernels.md#Compile-time-constants).


## Type conversion

| Operation | Description |
|-----------|-------------|
| `convert(Tile{T}, tile)` | Convert element type |
| `T(x)` | Scalar type conversion (e.g. `Float16.(tile)` via broadcast) |

Converting a `Float32` tile to `TFloat32` is the usual way to opt into tensor-core
acceleration for a matmul:

```julia
a = ct.load(A; index=(bid_m, k), shape=(tm, tk))
a_tf32 = convert(ct.Tile{ct.TFloat32}, a)
```

!!! note
    Float-to-integer conversions do not throw in kernels; they truncate. See
    [Differences from Julia](kernels.md#Differences-from-Julia).
