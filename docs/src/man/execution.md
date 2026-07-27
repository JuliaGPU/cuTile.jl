# Compiling and Launching

A cuTile kernel is an ordinary Julia function until you launch it. This chapter
covers what happens at that point: how arguments cross to the device, which of
them are compile-time values, what counts as a distinct kernel, and how the
results are cached.

Launching requires CUDA.jl to be imported. It supplies the `CuArray` type, the
compiler artifacts, and the stream the kernel runs on.


## Launch paths

Two equivalent entry points:

```julia
using CUDA, cuTile
import cuTile as ct

grid = cld(n, tile_size)

@cuda backend=cuTile blocks=grid vadd(a, b, c, ct.Constant(tile_size))   # macro form
ct.launch(vadd, grid, a, b, c, ct.Constant(tile_size))                   # functional form
```

`@cuda backend=cuTile` routes CUDA.jl's launch macro through cuTile's compiler;
[`ct.launch`](@ref cuTile.launch) is the same operation without the macro, which
is easier to call from generated code or when the argument list is built
programmatically. Both accept the optimization hints described in
[Performance](performance.md), and both use the current task-bound CUDA stream;
cuTile has no stream argument of its own. To run a kernel on a different stream,
set the task's stream as you would for any CUDA.jl operation.

`blocks` sizes the grid, in up to three dimensions. It counts *blocks*, so it is
almost always a ceiling division of the problem size by the tile shape. Inside
the kernel, a block finds its place in the grid with `ct.bid`; see [Programming
Model](programming_model.md#The-grid).


## Argument conversion

Arguments do not reach the kernel as the types you passed. Each is converted
first:

| Host argument | Kernel parameter |
|---------------|------------------|
| `CuArray{T,N}` | [`ct.TileArray{T,N,Spec}`](programming_model.md#Arrays-and-tiles) |
| `ct.Constant(x)` | `ct.Constant{typeof(x), x}`, and no kernel parameter at all |
| other `isbits` values | themselves, as a by-value parameter |

A `TileArray` is then flattened further: its base pointer, sizes and strides
each become a separate kernel parameter. This is why a kernel taking three
vectors and a tile size has ten parameters in the generated code rather than
four.


## Compile-time arguments

A tile's shape is part of its type, so any value used as a tile shape has to be
known to the compiler rather than passed at runtime. [`ct.Constant`](@ref
cuTile.Constant) is how a host value becomes a compile-time one: it encodes the
value in its own type parameters, so the compiler sees the value itself and not
merely its type.

Wrap it at the *launch site*; the kernel signature keeps its plain annotation:

```julia
function kernel(a, b, tile_size::Int)
    tile = ct.load(a; index=1, shape=(tile_size,))
    ...
end

@cuda backend=cuTile blocks=grid kernel(a, b, ct.Constant(16))
```

That asymmetry is deliberate, and is the opposite of cuTile Python, which
annotates the parameter and passes a plain value; see
[Comparison](comparison.md#Compile-time-constants).

A `Constant` argument generates no kernel parameter: the value is embedded in
the code, so it costs nothing at runtime. It does mean that each distinct value
produces a distinct kernel.


## What makes a distinct kernel

cuTile compiles one kernel per combination of *converted* argument types,
exactly as Julia compiles one method instance per combination of argument types.
Three things therefore trigger a recompile:

**The element type and rank of each array**, as you would expect.

**Each `Constant` value**, since it lands in the type. `ct.Constant(64)` and
`ct.Constant(128)` have types `Constant{Int64,64}` and `Constant{Int64,128}`, so
a tile-size sweep compiles a kernel per size, which is the point, but worth
knowing before sweeping a hundred of them.

**Each array's memory layout**, via the [`ct.ArraySpec`](@ref cuTile.ArraySpec)
carried in the `TileArray` type. The spec records what the compiler may assume
about the array:

- base pointer alignment in bytes,
- whether the array is contiguous (`stride[1] == 1`),
- per-dimension stride and shape divisibility,
- whether two distinct in-bounds indices may alias.

These are read off each `CuArray`'s runtime layout at launch, so you never write
one down. But they do mean that arrays which look interchangeable are not:

```julia-repl
julia> typeof(ct.TileArray(CUDA.rand(Float32, 1024)))
cuTile.TileArray{Float32, 1, cuTile.ArraySpec{1, 128, true, (0,), (16,), false}()}

julia> typeof(ct.TileArray(view(CUDA.rand(Float32, 2048), 1:2:2048)))
cuTile.TileArray{Float32, 1, cuTile.ArraySpec{1, 128, false, (0,), (16,), false}()}
```

The strided view is not contiguous, so it compiles to a second, more
conservative specialization. That is a correctness feature, but it is also a
common reason a kernel is unexpectedly slow; see
[Performance](performance.md#Array-specialization).


## Caching

Compilation results are cached at two levels, and neither needs configuring for
normal use.

Within a session, a compiled kernel is attached to the underlying Julia
`CodeInstance`, so invalidation rides on Julia's ordinary method-invalidation
machinery: redefining the kernel function recompiles it, and nothing else does.

Across sessions, the Tile IR → CUBIN step is cached on disk, so the second run
of a program skips the `tileiras` invocation entirely. The cache is
content-addressed on the bytecode plus the toolkit version, architecture and
optimization level, so a toolchain upgrade simply produces new entries rather
than stale hits. Two environment variables control it:

| Variable | Effect |
|----------|--------|
| `JULIA_CUTILE_CACHE_DIR` | Override the cache directory. `0`, `off`, `none` or empty disables the disk cache. |
| `JULIA_CUTILE_CACHE_SIZE` | Override the maximum cache size (default 1 GiB). |

These knobs are considered internal, and may disappear when cuTile.jl integrates
more deeply with Julia's integrated code cache.
