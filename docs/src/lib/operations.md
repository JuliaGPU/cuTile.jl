# Operations

```@meta
CurrentModule = cuTile
```

Most scalar operations listed here work on both scalars and tiles. Use Julia's broadcast
syntax (the `.` operator) to apply any scalar function element-wise over tiles:
`sqrt.(tile)`, `max.(a, b)`, `cld.(tile, 4)`, and so on.

Operations prefixed with `ct.` are cuTile intrinsics without a direct Julia equivalent;
everything else is standard Julia syntax overlaid on `Base`. Operations that need a
particular Tile IR version or GPU architecture are noted inline and collected in
[Compatibility](../man/compatibility.md).


## Arithmetic

| Operation | Description |
|-----------|-------------|
| `+`, `-` | Element-wise (same shape only) |
| `tile * scalar`, `tile / scalar` | Scalar multiply/divide |
| `.+`, `.-`, `.*`, `./`, `.^` | Broadcasting element-wise |
| `mod.(x, y)` | Floored remainder with Julia sign semantics |
| `ct.divmod(x, y)` | Floored quotient and remainder |

Note that `+` and `-` apply directly to tiles of matching shape, and `*` and `/` to a tile
and a scalar, but combining tiles of *different* shapes requires broadcast syntax. `a * b`
between two tiles is matrix multiplication, not an element-wise product.


## Construction

| Operation | Description |
|-----------|-------------|
| `zeros(T, dims...)` | Zero-filled tile |
| `ones(T, dims...)` | One-filled tile |
| `fill(value, dims...)` | Constant-filled tile |
| `ct.arange(n; dtype, start, step)` | Configurable arithmetic sequence (defaults to `1:n`) |


## Shape

| Operation | Description |
|-----------|-------------|
| `reshape(tile, shape)` | Reshape (same element count) |
| `transpose(tile)` | Transpose 2D tile |
| `permutedims(tile, perm)` | Permute dimensions |
| `repeat(tile, counts...)` `repeat(tile; inner, outer)` | Repeat values along dimensions |
| `ct.extract(tile, index, shape)` | Extract sub-tile |
| `ct.insert(tile, index, value)` | Replace a non-overlapping sub-tile (Tile IR v13.4+) |
| `ct.cat((a, b), axis)` | Concatenate tiles |
| `ct.broadcast_to(tile, shape)` | Broadcast to target shape |
| `dropdims(tile; dims)` | Remove singleton dimensions |


## Matrix multiplication

| Operation | Description |
|-----------|-------------|
| `a * b` | Matrix multiplication: `a @ b` |
| `muladd(a, b, acc; fast_acc=false)` | Matrix multiply-accumulate: `a * b + acc` |
| `ct.muladd_scaled(a, a_scale, b, b_scale, acc)` | Block-scaled multiply-accumulate |

Each operation follows `Base.:*` / `Base.muladd`'s shape rules, with the addition of allowing
trailing batch dimensions.

`fast_acc=true` enables fast accumulation for FP8 inputs, and has an effect only on Hopper
(sm_90; silently ignored on other architectures), and requires Tile IR v13.3+.

`ct.muladd_scaled` multiplies each operand by a low-precision block scale before the matmul:
each scale element covers a contiguous block of `B = K ÷ K_s` elements along the K dimension.
Requires Blackwell. The supported operand/scale/accumulator dtypes and block sizes are:

| Input (`a`/`b`) | Scale | Acc/Output | B |
|-----------------|-------|------------|--------|
| `Float8_E4M3FN`, `Float8_E5M2` | `Float8_E8M0FNU` | `Float32` | 32 |
| `Float4_E2M1FN` | `Float8_E8M0FNU` | `Float32` | 16, 32 |
| `Float4_E2M1FN` | `Float8_E4M3FN` | `Float32` | 16 |


## Reductions and scans

| Operation | Description |
|-----------|-------------|
| `sum(tile; dims)` | Sum along axis |
| `prod(tile; dims)` | Product along axis |
| `maximum(tile; dims, propagate_nan)` | Maximum along axis |
| `minimum(tile; dims, propagate_nan)` | Minimum along axis |
| `any(tile; dims)` | Logical OR along axis |
| `all(tile; dims)` | Logical AND along axis |
| `count(tile; dims)` | Count `true` elements along axis |
| `argmax(tile; dims, propagate_nan)` | 1-based index of maximum along axis |
| `argmin(tile; dims, propagate_nan)` | 1-based index of minimum along axis |
| `cumsum(tile; dims, rev)` | Cumulative sum |
| `cumprod(tile; dims, rev)` | Cumulative product |

Following `Base` semantics, reductions keep the reduced dimension as size 1. Use `dropdims`
to remove it.


## Higher-order functions

| Operation | Description |
|-----------|-------------|
| `map(f, tiles...)` | Apply function element-wise (same shape) |
| `f.(tiles...)`, `broadcast(f, tiles...)` | Apply function with shape broadcasting |
| `reduce(f, tile; dims, init)` | Reduction with arbitrary function |
| `mapreduce(f, op, tile; dims, init)` | Map then reduce |
| `accumulate(f, tile; dims, init, rev)` | Scan/prefix-sum with arbitrary function |

Any function that works on scalars "just works" when broadcast over tiles, so these cover
element-wise operations that have no dedicated entry in the tables above.

### Broadcasting shape alignment

cuTile.jl uses Julia's standard left-aligned broadcast shape rules: dimensions are matched
starting from the first (leftmost) dimension. A 1D `(N,)` tile therefore cannot broadcast
with a 2D `(M, N)` tile, because dimension 1 has size `N` vs `M`. Use `reshape` to align
dimensions, just as with regular Julia arrays:

```julia
a = ct.load(...)              # (N,)
b = ct.load(...)              # (M, N)
result = reshape(a, (1, N)) .+ b  # (1, N) .+ (M, N) → (M, N)
```

This differs from NumPy's (and cuTile Python's) right-aligned rules; see
[Comparison with cuTile Python](../man/comparison.md).


## Math

| Operation | Description |
|-----------|-------------|
| `sqrt(x)` | Square root |
| `rsqrt(x)` | Reciprocal square root |
| `exp(x)`, `exp2(x)` | Exponential |
| `log(x)`, `log2(x)` | Logarithm |
| `sin(x)`, `cos(x)`, `tan(x)` | Trigonometric functions |
| `sinh(x)`, `cosh(x)`, `tanh(x)` | Hyperbolic functions |
| `fma(a, b, c)` | Fused multiply-add |
| `abs(x)` | Absolute value |
| `isnan(x)` | NaN test |
| `max(a, b)`, `min(a, b)` | Maximum/minimum |
| `ceil(x)`, `floor(x)` | Rounding |
| `ct.@fpmode rounding_mode=ct.Rounding.Approx flush_to_zero=true begin ... end` | Scoped FP rounding mode and flush-to-zero |


## Comparison

| Operation | Description |
|-----------|-------------|
| `<`, `>`, `<=`, `>=` | Comparison (returns `Bool` tile when broadcast) |
| `==`, `!=` | Equality |
| `ifelse(cond, x, y)` | Conditional selection |


## Integer and bitwise

| Operation | Description |
|-----------|-------------|
| `cld(a, b)` | Ceiling division |
| `fld(a, b)` | Floor division |
| `div(a, b)` | Truncating division |
| `mul_hi(a, b)` | High bits of integer multiply (`Base.mul_hi` on Julia 1.13+) |
| `~x` | Bitwise NOT |
| `&`, `\|`, `xor` | Bitwise AND, OR, XOR |


## Type conversion

Element-type conversion and reinterpretation are covered in
[Element Types](../man/element_types.md).


## Reference

The entries below are the operations with cuTile-specific behaviour or no `Base` counterpart.
Everything else in this chapter is a `Base` function that cuTile overlays, documented by
`Base` itself.

```@docs
arange
cat
broadcast_to
extract
insert
Base.reinterpret(::Type, ::Tile)
Base.reinterpret(::typeof(reshape), ::Type, ::Tile)
divmod
rsqrt
Base.muladd(::Tile, ::Tile, ::Tile)
muladd_scaled
```
