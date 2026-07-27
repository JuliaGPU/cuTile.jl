# Matrix Multiplication

Matrix multiplication is where the tile-based model earns its keep: a tile is
exactly the operand shape a tensor core wants, so `a * b` on two tiles is a
single Tile IR instruction rather than a hand-written blocking scheme.

This tutorial builds `C = A * B` — with `A` of size `(M, K)`, `B` of size `(K,
N)` and `C` of size `(M, N)` — in four steps, each one lifting a restriction
from the last. It follows the same progression as chapter 2 of the [Tile IR
specification](https://docs.nvidia.com/cuda/tile-ir/latest/sections/prog_model.html).

The goal here is understanding, not speed. The final kernel is a real, correct
GEMM, but it leaves out the scheduling work that a tuned implementation needs;
the last section says what that is.


## Step 1: one output tile

Start with the smallest possible problem: matrices exactly one tile in each
dimension, so a single block does all the work.

```julia
using CUDA, cuTile
import cuTile as ct

function matmul_one_tile(A, B, C, tm::Int, tn::Int, tk::Int)
    a = ct.load(A; index=(1, 1), shape=(tm, tk))
    b = ct.load(B; index=(1, 1), shape=(tk, tn))
    ct.store(C; index=(1, 1), tile=a * b)
    return
end
```

```julia
tm = tn = tk = 64
A, B = CUDA.rand(Float32, tm, tk), CUDA.rand(Float32, tk, tn)
C = CUDA.zeros(Float32, tm, tn)

@cuda(backend=cuTile, blocks=1,
      matmul_one_tile(A, B, C,
                      ct.Constant(tm), ct.Constant(tn), ct.Constant(tk)))

@assert isapprox(Array(C), Array(A) * Array(B); rtol=1e-4)
```

The shape is the same load–compute–store as [vector
addition](vector_addition.md), with two differences.

`index` and `shape` are now tuples, one entry per array dimension: `index=(1,
1)` is the top-left tile, and `shape=(tm, tk)` its extent. As before, `index`
counts tiles, not elements.

More importantly, **`a * b` between two tiles is matrix multiplication**,
following the `*` function from Base: `(tm, tk) * (tk, tn)` gives a `(tm, tn)`
tile. This is different from broadcasting: `a .* b` would be an element-wise
product. It is also one of the areas cuTile.jl deviates from cuTile Python,
where operators are element-wise, and it does so to match Julia rather than to
match its own conventions.


## Step 2: a grid of output tiles

Nothing about `matmul_one_tile` scales: it computes tile `(1, 1)` and ignores
the rest. Since each output tile depends on no other output tile, they can all
be computed in parallel, one block each, on a two-dimensional grid.

```julia
function matmul_grid(A, B, C, tm::Int, tn::Int, tk::Int)
    i, j = ct.bid(1), ct.bid(2)
    a = ct.load(A; index=(i, 1), shape=(tm, tk))
    b = ct.load(B; index=(1, j), shape=(tk, tn))
    ct.store(C; index=(i, j), tile=a * b)
    return
end
```

```julia
M, N, K = 256, 128, 64      # K is still a single tile
tm, tn, tk = 64, 64, 64
A, B = CUDA.rand(Float32, M, K), CUDA.rand(Float32, K, N)
C = CUDA.zeros(Float32, M, N)

@cuda(backend=cuTile, blocks=(cld(M, tm), cld(N, tn)),
      matmul_grid(A, B, C,
                  ct.Constant(tm), ct.Constant(tn), ct.Constant(tk)))
```

Block `(i, j)` now computes output tile `(i, j)`, reading tile `(i, 1)` of `A`
(its band of rows) and tile `(1, j)` of `B` (its band of columns). The grid is
sized so that every output tile gets a block.

`K` is still restricted to a single tile, which is the last thing standing
between this and a general GEMM.


## Step 3: looping over K

For larger `K`, an output tile is a sum of products over all the `K`-tiles:
`C[i,j] = Σₖ A[i,k] * B[k,j]`. That sum is a loop with an accumulator.

```julia
function matmul_kloop(A, B, C, tm::Int, tn::Int, tk::Int)
    i, j = ct.bid(1), ct.bid(2)

    acc = zeros(Float32, tm, tn)
    for k in 1:ct.num_tiles(A, 2, (tm, tk))
        a = ct.load(A; index=(i, k), shape=(tm, tk))
        b = ct.load(B; index=(k, j), shape=(tk, tn))
        acc = muladd(a, b, acc)
    end

    ct.store(C; index=(i, j), tile=acc)
    return
end
```

`K` is now unconstrained, and the launch is the one from step 2 with a larger `K`:

```julia
M, N, K = 256, 128, 512
tm, tn, tk = 64, 64, 64
A, B = CUDA.rand(Float32, M, K), CUDA.rand(Float32, K, N)
C = CUDA.zeros(Float32, M, N)

@cuda(backend=cuTile, blocks=(cld(M, tm), cld(N, tn)),
      matmul_kloop(A, B, C,
                   ct.Constant(tm), ct.Constant(tn), ct.Constant(tk)))

@assert isapprox(Array(C), Array(A) * Array(B); rtol=1e-4)
```

Four things to note.

**`zeros(Float32, tm, tn)` creates a tile** inside a kernel, not an array, just
like `ones` and `fill`. Its shape is static, so `tm` and `tn` must be
`ct.Constant`s at the launch site, as they already are.

**`ct.num_tiles(A, 2, (tm, tk))` is how many tiles fit along a dimension**, here
it is `cld(size(A, 2), tk)`, the same rounding-up we did on the host to size the grid, done on the device where the array's real size is known.

**`muladd(a, b, acc)` is a multiply-accumulate**, the fused operation a tensor
core performs directly. You could write `a * b + acc` instead and get the same
code; `muladd` just names it.

**The accumulator is `Float32` regardless of the input type.** Summing many
products in the input precision loses accuracy quickly, so accumulating wider is
the norm. Here the inputs are already `Float32`; step 4 makes the distinction
matter.

The `for` loop is not unrolled or traced away; it compiles to a real counted
loop, with `acc` carried between iterations. Ordinary Julia control flow works
in kernels; see [Writing Kernels](../man/kernels.md#Control-flow).


## Step 4: using the tensor cores

`Float32` operands do not reach the tensor cores directly. The usual way to opt
in is `TFloat32`, a 32-bit format with a truncated 10-bit mantissa, which trades
some precision for a large throughput gain:

```julia
function matmul(A, B, C, tm::Int, tn::Int, tk::Int)
    i, j = ct.bid(1), ct.bid(2)

    acc = zeros(Float32, tm, tn)
    for k in 1:ct.num_tiles(A, 2, (tm, tk))
        a = ct.load(A; index=(i, k), shape=(tm, tk), padding_mode=ct.PaddingMode.Zero)
        b = ct.load(B; index=(k, j), shape=(tk, tn), padding_mode=ct.PaddingMode.Zero)
        acc = muladd(convert(ct.Tile{ct.TFloat32}, a),
                     convert(ct.Tile{ct.TFloat32}, b), acc)
    end

    ct.store(C; index=(i, j), tile=acc)
    return
end
```

`convert(ct.Tile{ct.TFloat32}, a)` narrows the operands while the accumulator
stays `Float32`, which is exactly why the accumulator's type was worth being
explicit about. Because `TFloat32` is a lossy format, verify against a looser
tolerance:

```julia
@assert isapprox(Array(C), Array(A) * Array(B); rtol=1e-2)
```

`Float16` and `BFloat16` inputs are already tensor-core operand formats and need
no conversion. Keep the `Float32` accumulator and convert on the way out
instead:

```julia
function matmul_f16(A, B, C, tm::Int, tn::Int, tk::Int)
    i, j = ct.bid(1), ct.bid(2)

    acc = zeros(Float32, tm, tn)
    for k in 1:ct.num_tiles(A, 2, (tm, tk))
        a = ct.load(A; index=(i, k), shape=(tm, tk), padding_mode=ct.PaddingMode.Zero)
        b = ct.load(B; index=(k, j), shape=(tk, tn), padding_mode=ct.PaddingMode.Zero)
        acc = muladd(a, b, acc)          # Float16 × Float16 → Float32
    end

    ct.store(C; index=(i, j), tile=convert(ct.Tile{Float16}, acc))
    return
end
```

The other new argument is `padding_mode`, which the next section explains.


## Sizes that are not a multiple of the tile shape

Loads and stores are bounds-checked by default, so a ragged `M` or `N` needs no
special handling: the overhanging rows and columns of the output tile are simply
not stored. Their values are undefined, but nothing reads them.

A ragged `K` is different, and this is the subtle part. The out-of-bounds
columns of `A` and rows of `B` in the final `k` iteration are multiplied and
*summed into `acc`*, the same accumulator as the in-bounds data. They land in
output elements that do get stored, so their values matter, and the default
padding mode leaves them unspecified.

`padding_mode=ct.PaddingMode.Zero` makes out-of-bounds reads return zero. Zeros
contribute nothing to a sum of products, so the ragged iteration adds exactly
the partial products that should be there. With it in place, the step 4 kernel
handles sizes that divide by nothing:

```julia
M, N, K = 200, 100, 300     # none of them a multiple of 64
tm, tn, tk = 64, 64, 64
A, B = CUDA.rand(Float32, M, K), CUDA.rand(Float32, K, N)
C = CUDA.zeros(Float32, M, N)

@cuda(backend=cuTile, blocks=(cld(M, tm), cld(N, tn)),
      matmul(A, B, C,
             ct.Constant(tm), ct.Constant(tn), ct.Constant(tk)))

@assert isapprox(Array(C), Array(A) * Array(B); rtol=1e-2)
```

The rule generalizes past GEMM: when out-of-bounds lanes feed a reduction whose
result is stored, request a padding value that is neutral for that reduction.
When they only feed output positions that get clipped, you can leave the
default.


## Where to go from here

The kernel above is a correct GEMM, and it is not a fast one. Two things
separate it from a tuned implementation, neither of which changes what it
computes:

- **Block scheduling.** Neighbouring blocks in a naive 2D grid re-read `A` and
  `B` tiles that a better traversal order would have found in L2.
  `examples/matmul.jl` swizzles a 1D grid into `(i, j)` pairs in grouped
  row-major order for this reason.
- **Compiler hints.** `ct.@compiler_options` controls CTA clustering and
  occupancy, which matter a lot for GEMM. See
  [Performance](../man/performance.md).

`examples/matmul.jl` in the repository is this kernel plus both of those, and
`examples/batchmatmul.jl` extends it to batched operands using trailing batch
dimensions on `muladd`. For block-scaled FP8 and FP4 matmuls, see
[`ct.muladd_scaled`](../lib/operations.md#Matrix-multiplication).
