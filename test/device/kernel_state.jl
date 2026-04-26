using CUDA

# `Intrinsics.kernel_state()` returns the implicit per-launch `KernelState`
# struct. The host appends a fresh `RandomDevice`-derived seed to every
# `cuTile.launch`, so two consecutive launches see distinct seeds, while
# every block within a single launch sees the same seed.

@testset "kernel_state seed: distinct per launch, shared per block" begin
    function k(out::ct.TileArray{UInt32, 1})
        pid = ct.bid(1)
        state = ct.Intrinsics.kernel_state()
        out[pid] = state.seed
        return
    end

    n = 64
    out1 = CUDA.zeros(UInt32, n)
    out2 = CUDA.zeros(UInt32, n)
    ct.launch(k, n, out1)
    ct.launch(k, n, out2)
    v1 = Array(out1)
    v2 = Array(out2)

    # Within one launch, every block sees the same seed.
    @test all(==(v1[1]), v1)
    @test all(==(v2[1]), v2)

    # Across launches, the seed is fresh entropy — collisions are 1/2^32.
    @test v1[1] != v2[1]
end
