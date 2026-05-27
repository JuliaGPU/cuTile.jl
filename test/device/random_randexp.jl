using CUDA
using Random

# Distribution config shared by the device + host randexp testsets. `edges`
# are 3 quartile-or-equivalent CDF cuts; `probs` are the matching per-bin
# probabilities; `mean` is the population mean; `in_range` is the per-element
# validity predicate (exponential support is `[0, ∞)`).
const RANDEXP_DIST = (label = "randexp",
                      f     = Random.randexp,
                      f!    = Random.randexp!,
                      edges = (-log(0.75), -log(0.5), -log(0.25)),
                      probs = (0.25, 0.25, 0.25, 0.25),
                      mean  = 1.0,
                      in_range = >=(0.0))

function bin_counts(v, edges)
    counts = zeros(Int, length(edges) + 1)
    for x in v
        idx = something(findfirst(>=(Float64(x)), edges), length(edges) + 1)
        counts[idx] += 1
    end
    counts
end

# Per-bin tolerance: 10% of N. Smallest bin under either CDF (p≈0.16 for
# normal, p=0.25 for exp) clears 5σ at N≥1024, so the check is flake-free
# across the per-launch-randomized default-stream seed.
function check_dist(v, dist; N = length(v), tol_factor = 0.10)
    counts = bin_counts(v, dist.edges)
    for (i, p) in enumerate(dist.probs)
        @test abs(counts[i] - N * p) < N * tol_factor
    end
end

# Mean tolerance: SE/√N at N=1024 is ≤ 0.031; 5σ ≈ 0.16. Narrow floats lose
# precision in the BM/log path, so widen for those.
mean_tol(::Type{T}) where {T} =
    T <: Union{Float16, ct.BFloat16} ? 0.20 : 0.15

@testset "device randexp" begin
    dist = RANDEXP_DIST
    f, in_range = dist.f, dist.in_range

    @testset "typed surfaces (T=$T, dims=$dims)" for (T, dims) in
            ((Float32, (16,)), (Float32, (32,)), (Float32, (64,)),
             (Float64, (16,)), (Float16, (16,)), (ct.BFloat16, (16,)))
        # `o1`: scalar form. `o2`: tile form via the default stream.
        # `o3`: tile form via an explicit `DeviceRNG` (different stream).
        function k(o1, o2, o3, ::Type{T_}, dims_::NTuple{N, Int}) where {T_, N}
            pid = ct.bid(1)
            rng = ct.DeviceRNG(); Random.seed!(rng, 1)
            o1[pid] = f(T_)
            ct.store(o2, pid, f(T_, dims_))
            ct.store(o3, pid, f(rng, T_, prod(dims_)))
            return
        end

        n_blocks = 64
        m = prod(dims)
        o1 = CUDA.zeros(T, n_blocks)
        o2 = CUDA.zeros(T, n_blocks * m)
        o3 = CUDA.zeros(T, n_blocks * m)
        @cuda backend=cuTile blocks=n_blocks k(o1, o2, o3, T, ct.Constant(dims))

        for v in (Array(o1), Array(o2), Array(o3))
            @test eltype(v) === T
            @test all(isfinite, v)
            @test all(in_range, v)
        end

        # Distribution shape + mean on the larger draws (o1's N=64 is too
        # small for shape testing).
        for v in (Array(o2), Array(o3))
            check_dist(v, dist)
            @test abs(sum(Float64, v) / length(v) - dist.mean) < mean_tol(T)
        end
    end

    @testset "untyped surface defaults to Float32" begin
        function k(o::ct.TileArray{Float32, 1})
            pid = ct.bid(1)
            ct.store(o, pid, ct.reshape(f(4, 4), (16,)))
            return
        end
        n_blocks = 64
        o = CUDA.zeros(Float32, n_blocks * 16)
        @cuda backend=cuTile blocks=n_blocks k(o)
        v = Array(o)
        @test eltype(v) === Float32
        @test all(in_range, v)
        check_dist(v, dist)
    end

    @testset "in-kernel `Random.seed!` matches host RNG output" begin
        # Single-block draw with an in-kernel `Random.seed!(default_rng(),s)`
        # plumbs the same `(seed, counter)` as `cuTile.RNG(s, 0)`, so the
        # outputs must be byte-identical.
        function k(out::ct.TileArray{Float32, 1})
            Random.seed!(Random.default_rng(), UInt32(42))
            pid = ct.bid(1)
            ct.store(out, pid, f(Float32, (512,)))
            return
        end
        out = CUDA.zeros(Float32, 512); @cuda backend=cuTile k(out)
        @test Array(out) == Array(f(ct.RNG(UInt32(42), UInt32(0)), Float32, 512))
    end
end


@testset "host randexp" begin
    dist = RANDEXP_DIST
    f, f!, in_range = dist.f, dist.f!, dist.in_range
    N = 4096

    @testset "randexp! basics + counter advance" begin
        rng = ct.RNG(42)
        A = CUDA.zeros(Float32, N)
        f!(rng, A)
        v = Array(A)
        @test all(isfinite, v)
        @test all(in_range, v)
        @test rng.counter == UInt32(N)
        @test abs(sum(v) / N - dist.mean) < 0.1
    end

    @testset "determinism + Philox re-keying" begin
        # Same seed → byte-identical output.
        @test Array(f(ct.RNG(123), Float32, N)) == Array(f(ct.RNG(123), Float32, N))

        # Different seeds → uncorrelated streams. Set-disjoint isn't safe
        # for either distribution at this N due to Float32 birthday flake
        # (especially randexp, whose output concentrates near 0); use
        # element-wise equality instead — uncorrelated streams collide in
        # ≤ N²/2^24 ≈ 1 position.
        c = Array(f(ct.RNG(42),  Float32, N))
        d = Array(f(ct.RNG(100), Float32, N))
        @test sum(c .== d) < N ÷ 100
    end

    @testset "consecutive disjoint; seed! resets counter" begin
        # Two back-to-back draws on the same RNG use disjoint counter
        # ranges, so the underlying Philox outputs (and post-transform
        # samples) are disjoint up to Float32 birthday flake.
        rng = ct.RNG(7)
        a = Array(f(rng, Float32, N))
        b = Array(f(rng, Float32, N))
        @test sum(a .== b) < N ÷ 100

        Random.seed!(rng, 7)
        @test rng.counter == UInt32(0)
    end

    @testset "T-coverage" for T in (Float16, ct.BFloat16, Float32, Float64)
        v = Array(f(ct.RNG(13), T, N))
        @test eltype(v) === T
        @test all(in_range, v)
        @test all(isfinite, v)
        @test abs(sum(Float64, v) / N - dist.mean) < mean_tol(T)
        check_dist(v, dist)
    end

    @testset "ct.randexp / ct.randexp! aliases default to Float32" begin
        @test eltype(ct.randexp(N)) === Float32
        @test eltype(ct.randexp(Float32, N)) === Float32
        B = CUDA.zeros(Float32, N); ct.randexp!(B)
        @test all(in_range, Array(B))
    end

    @testset "arbitrary length (partial last tile)" begin
        # `store_partition_view` clips OOB writes, so `length(A)` can be
        # any value. The host advances by `n_blocks * RAND_FILL_TILE`,
        # not `length(A)`, so consecutive partial-length calls remain
        # disjoint up to Float32 birthday flake.
        rng = ct.RNG(0)
        A = CUDA.zeros(Float32, 513)
        f!(rng, A)
        v = Array(A)
        @test all(isfinite, v)
        @test rng.counter == UInt32(2 * cuTile.RAND_FILL_TILE)

        rng2 = ct.RNG(0)
        a = Array(f(rng2, Float32, 100))
        b = Array(f(rng2, Float32, 100))
        @test sum(a .== b) < 100 ÷ 10
    end
end
