using CUDA
using Random

# Device-side `rand` is backed by Philox2x32-7 with a per-block-derived key.
# `KernelState.seed` is internal — a fresh `RandomDevice` seed per launch — so
# bare `rand()` produces uncorrelated output across launches. Kernels that
# want determinism take a seed as an argument and call `Random.seed!` at
# entry. Different blocks under the same seed produce disjoint streams.

@testset "scalar rand(Float32) diverges per block" begin
    function k(out::ct.TileArray{Float32, 1})
        pid = ct.bid(1)
        out[pid] = rand(Float32)
        return
    end

    n = 128
    out = CUDA.zeros(Float32, n)
    ct.launch(k, n, out)
    v = Array(out)

    @test all(0f0 .< v .< 1f0)
    # All per-block values should be distinct (birthday paradox is trivial at n=128).
    @test length(unique(v)) == n
end

@testset "tile rand(Float32, (N,)) fills a tile" begin
    function k(out::ct.TileArray{Float32, 1})
        pid = ct.bid(1)
        t = rand(Float32, (16,))
        ct.store(out, pid, t)
        return
    end

    n_blocks = 64
    out = CUDA.zeros(Float32, n_blocks * 16)
    ct.launch(k, n_blocks, out)
    v = Array(out)

    @test all(0f0 .< v .< 1f0)
    # Within a block, values must be distinct. Across blocks too (rare collisions
    # are astronomically unlikely under Philox with disjoint counters).
    @test length(unique(v)) == n_blocks * 16
end

@testset "variadic-Integer dim forms" begin
    function k_typed(out::ct.TileArray{Float32, 1})
        pid = ct.bid(1); ct.store(out, pid, rand(Float32, 16)); return
    end
    function k_untyped(out::ct.TileArray{Float32, 1})
        pid = ct.bid(1); ct.store(out, pid, rand(16)); return
    end
    function k_rng_typed(out::ct.TileArray{Float32, 1})
        pid = ct.bid(1); a = ct.DeviceRNG()
        ct.store(out, pid, rand(a, Float32, 16)); return
    end
    function k_rng_untyped(out::ct.TileArray{Float32, 1})
        pid = ct.bid(1); a = ct.DeviceRNG()
        ct.store(out, pid, rand(a, 16)); return
    end
    function k_multidim(out::ct.TileArray{Float32, 1})
        pid = ct.bid(1); ct.store(out, pid, ct.reshape(rand(Float32, 4, 4), (16,))); return
    end

    for k in (k_typed, k_untyped, k_rng_typed, k_rng_untyped, k_multidim)
        out = CUDA.zeros(Float32, 16)
        ct.launch(k, 1, out)
        v = Array(out)
        @test all(0f0 .< v .< 1f0)
        @test length(unique(v)) == 16
    end
end

@testset "two rand() calls in the same kernel are disjoint" begin
    function k(a::ct.TileArray{Float32, 1}, b::ct.TileArray{Float32, 1})
        pid = ct.bid(1)
        ta = rand(Float32, (16,))
        tb = rand(Float32, (16,))
        ct.store(a, pid, ta)
        ct.store(b, pid, tb)
        return
    end

    n_blocks = 32
    a = CUDA.zeros(Float32, n_blocks * 16)
    b = CUDA.zeros(Float32, n_blocks * 16)
    ct.launch(k, n_blocks, a, b)

    va, vb = Array(a), Array(b)
    @test va != vb
    @test isempty(intersect(Set(va), Set(vb)))
end

@testset "in-kernel seed: same seed → same bytes across launches" begin
    function k(out::ct.TileArray{Float32, 1}, seed::UInt32)
        Random.seed!(Random.default_rng(), seed)
        pid = ct.bid(1)
        t = rand(Float32, (16,))
        ct.store(out, pid, t)
        return
    end

    n_blocks = 16
    out1 = CUDA.zeros(Float32, n_blocks * 16)
    out2 = CUDA.zeros(Float32, n_blocks * 16)
    ct.launch(k, n_blocks, out1, UInt32(42))
    ct.launch(k, n_blocks, out2, UInt32(42))

    @test Array(out1) == Array(out2)
end

@testset "default seed: consecutive launches diverge" begin
    function k(out::ct.TileArray{Float32, 1})
        pid = ct.bid(1)
        t = rand(Float32, (16,))
        ct.store(out, pid, t)
        return
    end

    n_blocks = 16
    out1 = CUDA.zeros(Float32, n_blocks * 16)
    out2 = CUDA.zeros(Float32, n_blocks * 16)
    ct.launch(k, n_blocks, out1)
    ct.launch(k, n_blocks, out2)

    @test Array(out1) != Array(out2)
end

@testset "default seed reaches DeviceRNG: launches diverge" begin
    # A kernel that only uses `DeviceRNG()` (no default stream, no explicit
    # seeding) must still see a fresh per-launch seed propagated through
    # `KernelState`. Without this, both launches' DeviceRNG streams start at
    # seed=0 and produce byte-identical output.
    function k(out::ct.TileArray{Float32, 1})
        a = ct.DeviceRNG()
        pid = ct.bid(1)
        t = rand(a, Float32, (16,))
        ct.store(out, pid, t)
        return
    end

    n_blocks = 16
    out1 = CUDA.zeros(Float32, n_blocks * 16)
    out2 = CUDA.zeros(Float32, n_blocks * 16)
    ct.launch(k, n_blocks, out1)
    ct.launch(k, n_blocks, out2)

    @test Array(out1) != Array(out2)
end

@testset "rand(UInt32, ...) raw output" begin
    function k(out::ct.TileArray{UInt32, 1})
        pid = ct.bid(1)
        t = rand(UInt32, (16,))
        ct.store(out, pid, t)
        return
    end

    n_blocks = 256
    out = CUDA.zeros(UInt32, n_blocks * 16)
    ct.launch(k, n_blocks, out)
    v = Array(out)

    # Mean should be roughly typemax(UInt32)/2 for N≈4096. The relative SE of
    # the sample mean is ≈0.9% (uniform U[0,2^32) has σ ≈ 2^32/√12, sample mean
    # σ ≈ that/√4096), so a 2% tolerance is only ~2σ and flakes ~5% of runs.
    # Use 5% (~5σ) to keep CI reliable while still catching gross bias.
    target = Float64(typemax(UInt32)) / 2
    observed = sum(Float64.(v)) / length(v)
    rel = abs(observed - target) / target
    @test rel < 0.05
end

@testset "loop iteration divergence" begin
    # `for` loop with rand() inside. rng_state_pass! threads the counter as
    # a loop carry, so each iteration sees a bumped counter → distinct values.
    function k(out::ct.TileArray{Float32, 1})
        pid = ct.bid(1)
        for i in Int32(1):Int32(4)
            out[(pid - Int32(1)) * Int32(4) + i] = rand(Float32)
        end
        return
    end

    n_blocks = 16
    out = CUDA.zeros(Float32, n_blocks * 4)
    ct.launch(k, n_blocks, out)
    v = Array(out)

    @test all(0f0 .< v .< 1f0)
    @test length(unique(v)) == length(v)
    # Each block's 4 values must be distinct (per-iteration divergence).
    for b in 1:n_blocks
        @test length(unique(v[(b - 1) * 4 + 1 : b * 4])) == 4
    end
end

@testset "rand() untyped defaults to Float32" begin
    function k(out::ct.TileArray{Float32, 1})
        pid = ct.bid(1)
        out[pid] = rand()
        return
    end

    n = 64
    out = CUDA.zeros(Float32, n)
    ct.launch(k, n, out)
    v = Array(out)

    @test all(0f0 .< v .< 1f0)
    @test length(unique(v)) == n
end

@testset "host RNG wrapper" begin
    # Fill tile size; length(A) must be a multiple.
    N = 2048

    @testset "Random.rand! with explicit RNG" begin
        rng = ct.RNG(42)
        A = CUDA.zeros(Float32, N)
        Random.rand!(rng, A)
        v = Array(A)
        @test all(0f0 .< v .< 1f0)
        @test length(unique(v)) == N
        @test rng.counter == UInt32(N)      # auto-advanced
    end

    @testset "same seed → byte-equal output" begin
        r1 = ct.RNG(123); r2 = ct.RNG(123)
        A1 = Random.rand(r1, Float32, N)
        A2 = Random.rand(r2, Float32, N)
        @test Array(A1) == Array(A2)
    end

    @testset "different seeds → different output" begin
        r1 = ct.RNG(123); r2 = ct.RNG(999)
        A1 = Random.rand(r1, Float32, N)
        A2 = Random.rand(r2, Float32, N)
        @test Array(A1) != Array(A2)
    end

    @testset "seed re-keys Philox (no shifted equality)" begin
        # Different seeds go through `Random.seed!(default_rng(), s)` at kernel
        # entry, which XORs into the Philox key. The resulting streams are
        # mathematically independent — no shift produces equality (unlike a
        # counter-offset seeding scheme).
        r1 = ct.RNG(42);  A1 = Array(Random.rand(r1, Float32, N))
        r2 = ct.RNG(100); A2 = Array(Random.rand(r2, Float32, N))
        @test A1[59:N] != A2[1:N-58]    # direct "off by seed difference" check
        @test !any(k -> A1[k+1:N] == A2[1:N-k], 1:64)   # no shift anywhere
        @test isempty(intersect(Set(A1), Set(A2)))      # no pointwise overlap
    end

    @testset "multiple DeviceRNG handles are independent" begin
        # Two `ct.DeviceRNG()` handles in one kernel, independently seeded,
        # produce uncorrelated output. This is the core multi-handle test.
        function kmulti(outA::ct.TileArray{Float32, 1}, outB::ct.TileArray{Float32, 1})
            pid = ct.bid(1)
            a = ct.DeviceRNG(); b = ct.DeviceRNG()
            Random.seed!(a, 1); Random.seed!(b, 2)
            ct.store(outA, pid, rand(a, Float32, (32,)))
            ct.store(outB, pid, rand(b, Float32, (32,)))
            return
        end
        A = CUDA.zeros(Float32, 32); B = CUDA.zeros(Float32, 32)
        ct.launch(kmulti, 1, A, B)
        va, vb = Array(A), Array(B)
        @test va != vb
        @test isempty(intersect(Set(va), Set(vb)))
    end

    @testset "unseeded DeviceRNG handles are independent" begin
        # Two `ct.DeviceRNG()` handles without explicit seeding. Without the
        # stream-ID mix in `rng_key`, both would share `seed = 0` and produce
        # byte-identical Philox output. With the mix, each stream ID folds
        # into the key, so the two streams are uncorrelated.
        function kunseed(outA::ct.TileArray{Float32, 1}, outB::ct.TileArray{Float32, 1})
            pid = ct.bid(1)
            a = ct.DeviceRNG(); b = ct.DeviceRNG()
            ct.store(outA, pid, rand(a, Float32, (32,)))
            ct.store(outB, pid, rand(b, Float32, (32,)))
            return
        end
        A = CUDA.zeros(Float32, 32); B = CUDA.zeros(Float32, 32)
        ct.launch(kunseed, 1, A, B)
        va, vb = Array(A), Array(B)
        @test va != vb
        @test isempty(intersect(Set(va), Set(vb)))
    end

    @testset "seeding one handle does not affect another" begin
        # Kernel seeds only `a`, leaves `b` at default (0). Output A should
        # reflect the seed; output B should match the bid-only stream.
        function kone(outA::ct.TileArray{Float32, 1}, outB::ct.TileArray{Float32, 1})
            pid = ct.bid(1)
            a = ct.DeviceRNG(); b = ct.DeviceRNG()
            Random.seed!(a, 0xDEADBEEF)
            ct.store(outA, pid, rand(a, Float32, (16,)))
            ct.store(outB, pid, rand(b, Float32, (16,)))   # b unseeded
            return
        end
        A = CUDA.zeros(Float32, 16); B = CUDA.zeros(Float32, 16)
        ct.launch(kone, 1, A, B)
        @test Array(A) != Array(B)
    end

    @testset "multiple default_rng calls share state" begin
        # Two `rand(Float32)` calls in the same kernel use `default_rng()`
        # which resolves to the singleton `DeviceRNG`. They share a single
        # counter slot — the second call sees an advanced counter, so the
        # two outputs differ.
        function kdef(out::ct.TileArray{Float32, 1})
            pid = ct.bid(1)
            x = rand(Float32)
            y = rand(Float32)
            out[pid] = x - y
            return
        end
        D = CUDA.zeros(Float32, 32); ct.launch(kdef, 32, D)
        @test all(!iszero, Array(D))
    end

    @testset "in-kernel Random.seed! matches host-wrapper output" begin
        # A kernel that explicitly calls `Random.seed!(default_rng(), 42)` at
        # entry should produce byte-equal output to `cuTile.RNG(42, 0)` on the
        # same-sized draw — they both route through the same threaded state.
        function kseeded(out::ct.TileArray{Float32, 1})
            pid = ct.bid(1)
            Random.seed!(Random.default_rng(), UInt32(42))
            t = rand(Float32, (512,))
            ct.store(out, pid, t)
            return
        end
        B = CUDA.zeros(Float32, 512); ct.launch(kseeded, 1, B)
        host = ct.RNG(UInt32(42), UInt32(0))
        A = Array(Random.rand(host, Float32, 512))
        @test Array(B) == A
    end

    @testset "consecutive rand! calls produce disjoint streams" begin
        rng = ct.RNG(7)
        A1 = Array(Random.rand(rng, Float32, N))
        A2 = Array(Random.rand(rng, Float32, N))
        @test A1 != A2
        @test isempty(intersect(Set(A1), Set(A2)))
    end

    @testset "Random.seed! resets counter" begin
        rng = ct.RNG(1)
        Random.rand(rng, Float32, N)        # advances counter
        @test rng.counter == UInt32(N)
        Random.seed!(rng, 1)
        @test rng.counter == UInt32(0)
    end

    @testset "UInt32 raw output" begin
        rng = ct.RNG(99)
        A = Random.rand(rng, UInt32, N)
        v = Array(A)
        @test eltype(v) === UInt32
        # sample mean near typemax/2
        target = Float64(typemax(UInt32)) / 2
        observed = sum(Float64.(v)) / length(v)
        @test abs(observed - target) / target < 0.02
    end

    @testset "ct.rand / ct.rand! top-level aliases" begin
        A = ct.rand(Float32, N)
        @test all(0f0 .< Array(A) .< 1f0)

        B = CUDA.zeros(Float32, N)
        ct.rand!(B)
        @test all(0f0 .< Array(B) .< 1f0)

        # Untyped defaults to Float32
        C = ct.rand(N)
        @test eltype(C) === Float32
    end

    @testset "arbitrary length (non-multiple of fill tile)" begin
        # `store_partition_view` clips OOB writes, so the kernel handles
        # partial last tiles automatically. Each block still consumes a
        # full RAND_FILL_TILE counters — the host advance rounds up to
        # match, keeping consecutive `rand!` calls disjoint.
        rng = ct.RNG(0)
        A = CUDA.zeros(Float32, 513)   # 512 + 1, partial second tile
        Random.rand!(rng, A)
        v = Array(A)
        @test all(0f0 .< v .< 1f0)
        @test length(unique(v)) == 513
        @test rng.counter == UInt32(2 * cuTile.RAND_FILL_TILE)   # 2 blocks × tile

        # Two consecutive partial-length calls must produce disjoint output
        # — would overlap if the counter advanced by `n` instead of
        # `n_blocks * tile` (n=100 < tile=512 means call 2's counter range
        # would still overlap call 1's).
        rng2 = ct.RNG(0)
        A1 = Array(Random.rand(rng2, Float32, 100))
        A2 = Array(Random.rand(rng2, Float32, 100))
        @test isempty(intersect(Set(A1), Set(A2)))
    end

    @testset "advance_counter! bumps seed on UInt32 wrap" begin
        rng = ct.RNG(UInt32(7), typemax(UInt32) - UInt32(3))
        cuTile.advance_counter!(rng, UInt32(10))
        @test rng.counter == UInt32(6)        # wrapped: 0xFFFFFFFC + 10 = 6
        @test rng.seed == UInt32(8)            # seed bumped by 1
    end
end
