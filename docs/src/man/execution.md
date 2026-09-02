# Compiling and Launching

A cuTile kernel is an ordinary Julia function until you launch it. This chapter
covers what happens at that point: how arguments cross to the device, which of
them are compile-time values, what counts as a distinct kernel, and how the
results are cached.

Persistent settings are package preferences. For example:

```toml
[cuTile]
compiler_timeout_seconds = 60
disk_cache = true
```

The timeout covers each `tileiras` invocation. On expiry, cuTile terminates the
compiler and reports a [`TileCompilerTimeoutError`](@ref cuTile.TileCompilerTimeoutError).

```@docs
cuTile.TileCompilerTimeoutError
```

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
pass `stream=other_stream` to either form to choose a different one. See
CUDA.jl's [task and stream
documentation](https://cuda.juliagpu.org/stable/usage/multitasking/) for the
stream model shared by both packages.

`blocks` sizes the grid, in up to three dimensions. It counts *blocks*, so it is
almost always a ceiling division of the problem size by the tile shape. Inside
the kernel, a block finds its place in the grid with `ct.bid`; see [Programming
Model](programming_model.md#The-grid).


## Programmatic dependent launch

[Programmatic dependent launch](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/programmatic-dependent-launch.html)
can overlap the tail of a producer kernel with an independent preamble in the
next kernel on the same stream. The producer signals when the consumer may
start, and the consumer waits before reading the producer's results:

```julia
function producer(a, producer_out)
    ct.grid_dependency_control_launch_dependents()

    # This work may overlap with the consumer's preamble.
    tile = ct.load(a, 1, (32,))
    ct.store(producer_out, 1, tile)
    return
end

function consumer(b, producer_out, out)
    tile = ct.load(b, 1, (32,)) # Independent work can run before the wait.

    ct.grid_dependency_control_wait()
    ct.store(out, 1, tile + ct.load(producer_out, 1, (32,)))
    return
end

stream = CUDA.stream()
@cuda backend=cuTile blocks=1 stream producer(a, producer_out)
@cuda backend=cuTile blocks=1 dependent=true stream consumer(b, producer_out, out)
```

The `dependent=true` attribute belongs on the consumer launch. Every producer
block should call `ct.grid_dependency_control_launch_dependents`; a block that
exits without calling it triggers completion implicitly. The consumer must call
`ct.grid_dependency_control_wait` before accessing any producer results because
the producer's signal does not make its writes visible.

Overlap is opportunistic. Correctness must not require the kernels to run
concurrently, as doing so can deadlock. Without `dependent=true`, normal stream
serialization still applies. Programmatic dependent launch requires Tile IR
13.4 and compute capability 9.0 or newer.


## Argument conversion

Arguments do not reach the kernel as the types you passed. Each is converted
first:

| Host argument | Kernel parameter |
|---------------|------------------|
| `CuArray{T,N}` | [`ct.TileArray{T,N,I,Spec}`](programming_model.md#Arrays-and-tiles) |
| `ct.Constant(x)` | `x` with its ordinary type, and no runtime parameter |
| other `isbits` values | themselves, as a by-value parameter |

A `TileArray` is then flattened further: its base pointer, sizes and strides
each become a separate kernel parameter. This is why a kernel taking three
vectors and a constant tile size has nine runtime parameters rather than four.

Array sizes and strides use `Int32` when they fit, and switch to `Int64`
automatically when either exceeds the 32-bit range. The index width is part of
the converted argument type, so crossing that boundary compiles a new kernel.
Use `ct.TileArray(array; index=Int64)` to select the wide path explicitly.


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

**Each array's index width.** Sizes and strides normally use `Int32`, but an
array with a size or stride outside that range uses `Int64` and therefore has a
different converted type.

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
cuTile.TileArray{Float32, 1, Int32, cuTile.ArraySpec{1, 128, true, (0,), (16,), false, (false,)}()}

julia> typeof(ct.TileArray(view(CUDA.rand(Float32, 2048), 1:2:2048)))
cuTile.TileArray{Float32, 1, Int32, cuTile.ArraySpec{1, 128, false, (0,), (16,), false, (false,)}()}
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
machinery: redefining the kernel or a method it depends on invalidates the
corresponding compiled result.

Across sessions, the Tile IR → CUBIN step is cached in Julia's object cache, so
the second run of a program skips the `tileiras` invocation entirely. Entries
are keyed on the bytecode plus the compiler identity, architecture and
optimization level, so a toolchain upgrade simply produces new entries rather
than stale hits. The `disk_cache` preference disables this tier when set to
`false`. The store itself is Julia's: on Julia 1.14 it is the runtime's object
cache, shared with the JIT and living in the depot; on older versions
CompilerCaching.jl provides an equivalent store in its scratch space. Either way
it is configured through the same environment variables: `JULIA_OBJCACHE=0`
disables it, `JULIA_OBJCACHE_PATH` relocates it, and `JULIA_OBJCACHE_CAPACITY`
sets its size in bytes (default 512 MiB), with least-recently-used eviction
beyond that. The former `cache_dir` and `cache_size_bytes` preferences no longer
apply; cuTile warns at load time if they are still set.
