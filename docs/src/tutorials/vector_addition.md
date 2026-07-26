# Vector Addition

Almost every cuTile kernel has the same three-part shape:

1. load one or more tiles from global memory,
2. compute on those tiles, producing new tiles,
3. store the result tiles back to global memory.

Vector addition is the smallest problem that shows all three, so it is where we start.


## The kernel

```julia
using CUDA, cuTile
import cuTile as ct

function vadd(a, b, c, tile_size::Int)
    i = ct.bid(1)
    tile_a = ct.load(a; index=i, shape=(tile_size,))
    tile_b = ct.load(b; index=i, shape=(tile_size,))
    ct.store(c; index=i, tile=tile_a + tile_b)
    return
end
```

That is the whole kernel. A few things are worth pointing out, because they are where the
tile-based model differs from what you may expect.

**There is no decorator and no `threadIdx`.** A cuTile kernel is an ordinary Julia function.
It also never talks about individual threads: `vadd` describes what one *block* does to one
*tile* of 128-or-so elements, and the Tile IR compiler decides how many threads run it and
how they divide the work.

**`ct.bid(1)` is a tile index, not an element index.** It returns which block we are, along
grid dimension 1, counting from `1` as Julia does. Block 2 handles the second tile of the
vector, not its second element.

**`index` is in units of tiles too.** `ct.load(a; index=i, shape=(tile_size,))` reads the
`i`-th `tile_size`-element window of `a` — elements `(i-1)*tile_size+1 : i*tile_size`. You
never compute a flat offset yourself. Compare this with a SIMT kernel, where you would write
`i = (blockIdx().x - 1) * blockDim().x + threadIdx().x` and index one element.

**`tile_a + tile_b` adds two whole tiles.** Tiles of matching shape support `+` and `-`
directly, as single whole-tile operations rather than loops.

**Kernels must return `nothing`.** A bare `return` is the idiomatic way to say so.


## Launching it

```julia
n = 1_000_000
tile_size = 128

a = CUDA.rand(Float32, n)
b = CUDA.rand(Float32, n)
c = CUDA.zeros(Float32, n)

@cuda backend=cuTile blocks=cld(n, tile_size) vadd(a, b, c, ct.Constant(tile_size))

@assert Array(c) == Array(a) .+ Array(b)
```

`blocks=cld(n, tile_size)` sizes the grid so that there is exactly one block per tile,
rounding up. `CuArray`s are converted to [`ct.TileArray`](../man/types.md) automatically.

The one piece of ceremony is [`ct.Constant`](../man/kernels.md#Compile-time-constants). A
tile's shape is part of its type, so `tile_size` has to be known to the compiler, not passed
as a runtime parameter. Wrapping it at the launch site embeds the value in the compiled code
— which is why the kernel signature can keep the plain `tile_size::Int` annotation, and why
launching with a different tile size compiles a different kernel.


## What happens at the end of the vector?

`1_000_000` is not a multiple of `128`, so the last block's tile hangs over the end of the
array. The kernel above already handles this: loads and stores are bounds-checked by default,
so the final store writes only the elements that exist.

The loaded values in that overhang *are* undefined, because the default padding mode leaves
out-of-bounds reads unspecified. Here that is harmless — those lanes are exactly the ones the
store discards, so whatever they contained never reaches memory. That reasoning does not hold
for every kernel, and the [matrix multiplication tutorial](matmul.md#Sizes-that-are-not-a-multiple-of-the-tile-shape)
shows a case where the garbage does contaminate a real result, and what to do about it.


## Choosing a tile size

Every tile dimension must be a power of two. Beyond that, the tile size trades the number of
blocks against the work each one does; a few hundred to a few thousand elements per tile is a
reasonable starting point for a memory-bound kernel like this one. `ct.Constant` means you can
sweep it from the host without touching the kernel. See
[Performance](../man/performance.md) for the knobs that matter once you care about the
numbers.


## One kernel per argument type

`vadd` has no type annotations, so it compiles afresh for each combination of argument types
it is called with — as any Julia function does. What counts as a distinct type here includes
more than the element type: a `CuArray`'s alignment, contiguity and divisibility are encoded
in its [`ct.ArraySpec`](../man/types.md#TileArray), so a contiguous 128-byte-aligned vector
whose length divides evenly by the tile size compiles to different, faster code than a
strided view does. You get that specialization without asking for it, which is also why
passing a `@view` can quietly cost performance.

If you want to see what came out, [Debugging](../man/debugging.md) covers the inspection
entry points.


## When you don't need a kernel at all

Element-wise work on whole `CuArray`s does not require writing a kernel. cuTile can generate
and launch one from a broadcast expression:

```julia
ct.@. c = a + b
```

The entire expression is fused into a single cuTile kernel, with tile sizes chosen for you.
See [Host-level Operations](../man/host.md). Writing the kernel by hand is what you do when
the computation is not a plain element-wise map — which brings us to matrix multiplication.
