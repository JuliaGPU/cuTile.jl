# Programming Model

A cuTile kernel is a Julia function that executes in parallel on a logical
*grid* of *blocks*. Each block computes on **tiles**: fixed-shape,
multi-dimensional fragments of the arrays passed to the kernel. This is the same
programming model as the one described in the [Tile IR
specification](https://docs.nvidia.com/cuda/tile-ir/latest/sections/prog_model.html),
which cuTile.jl compiles to.

The tile abstraction is what distinguishes this model from ordinary CUDA
kernels: you do not write code describing what one thread does, and there is no
`threadIdx`. A block's work is expressed as whole-tile operations, and the Tile
IR compiler decides how to map those onto threads, tensor cores and the tensor
memory accelerator.


## Arrays and tiles

Two data structures show up in every kernel, and the difference between them
matters:

**Arrays** ([`ct.TileArray{T,N,I,Spec}`](@ref cuTile.TileArray)) live in global
memory. They are mutable, have a strided memory layout, and their shapes are
*runtime* values. Inside a kernel they support only a limited set of operations,
mostly [loading and storing](memory.md) tiles and deriving views. A `CuArray`
passed to a kernel arrives as a `TileArray`, carrying with it what the compiler
may assume about its layout; see [Compiling and
Launching](execution.md#Argument-conversion).

**Tiles** ([`ct.Tile{T,Shape}`](@ref cuTile.Tile)) are values without defined
storage that exist only inside a kernel. Their shapes are *compile-time*
constants. Tiles support the bulk of the [operation set](../lib/operations.md):
element-wise arithmetic, matrix multiplication, reductions, shape manipulation,
and so on.

```julia
function kernel(a)              # a::TileArray — global memory, dynamic shape
    pid = ct.bid(1)
    tile = ct.load(a; index=pid, shape=(16,))   # tile::Tile — a value, static shape
    ...
end
```

Static versus dynamic shape is the distinction that drives everything else. It
is why tile shapes have to arrive as [compile-time
arguments](execution.md#Compile-time-arguments), and why a kernel is specialized
on the properties of the arrays you hand it.


## The grid

A kernel is launched over a grid of up to three dimensions. Blocks find their
place in it with:

| Operation | Description |
|-----------|-------------|
| `ct.bid(axis)` | Block ID (1=x, 2=y, 3=z) |
| `ct.num_blocks(axis)` | Grid size along axis |
| `ct.num_tiles(arr, axis, shape)` | Number of tiles along axis |

All of these are 1-based, following Julia convention: `ct.bid(1)` is the x axis
and returns `1` for the first block. `ct.num_tiles(arr, axis, shape)` is
equivalent to `cld(size(arr, axis), shape[axis])`, and is the usual way to size
a loop over the tiles an array decomposes into.


## Tile shapes

Tile shapes are part of the type, which imposes two constraints:

- **Every dimension must be a power of two.**
- **The shape must be type-inferrable.** A shape that Julia's compiler can only
  infer as a union type is rejected; the shape has to be statically known at
  each use.

Passing tile sizes as [`ct.Constant`](execution.md#Compile-time-arguments)
arguments is how you keep them compile-time values while still choosing them on
the host.


## A Julia-native surface

cuTile.jl aims to expose as much functionality as possible through Julia-native
constructs (`+`, `sum`, `reshape`, `broadcast`, etc.) rather than
cuTile-specific functions. Operations prefixed with `ct.` are cuTile intrinsics
with no direct Julia equivalent; everything else uses standard Julia syntax and
is overlaid on `Base`.

This is the main structural difference from cuTile Python, where every operation
is a `cuda.tile` function. See [Comparison with cuTile Python](comparison.md)
for a mapping between the two.
