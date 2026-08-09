# Gather/Scatter operations

using CUDA

#=========================================================================
 Basic gather/scatter (regression: mask was never applied due to
 get_constant() bug in emit_intrinsic! for load_ptr_tko/store_ptr_tko)
=========================================================================#

@testset "gather basic" begin
    function gather_kernel(a::ct.TileArray{Float32,1},
                           b::ct.TileArray{Float32,1})
        pid = ct.bid(1)
        base = (pid - 1) * 16
        indices = base .+ ct.arange(16)
        tile = ct.gather(a, indices)
        ct.store(b, pid, tile)
        return nothing
    end

    n = 1024
    a = CUDA.rand(Float32, n)
    b = CUDA.zeros(Float32, n)

    @cuda backend=cuTile blocks=64 gather_kernel(a, b)

    @test Array(b) ≈ Array(a)
end

@testset "scatter basic" begin
    function scatter_kernel(a::ct.TileArray{Float32,1},
                            b::ct.TileArray{Float32,1})
        pid = ct.bid(1)
        tile = ct.load(a, pid, (16,))
        base = (pid - 1) * 16
        indices = base .+ ct.arange(16)
        ct.scatter(b, indices, tile)
        return nothing
    end

    n = 1024
    a = CUDA.rand(Float32, n)
    b = CUDA.zeros(Float32, n)

    @cuda backend=cuTile blocks=64 scatter_kernel(a, b)

    @test Array(b) ≈ Array(a)
end

@testset "implicit store and scatter conversion" begin
    function convert_store_kernel(a::ct.TileArray{Float32,1},
                                  b::ct.TileArray{Float16,1},
                                  c::ct.TileArray{Float16,1})
        tile = ct.load(a, 1, (16,))
        ct.store(b, 1, tile)
        ct.scatter(c, ct.arange(16), tile)
        return
    end

    a = CUDA.rand(Float32, 16)
    b = CUDA.zeros(Float16, 16)
    c = CUDA.zeros(Float16, 16)
    @cuda backend=cuTile convert_store_kernel(a, b, c)
    expected = Float16.(Array(a))
    @test Array(b) == expected
    @test Array(c) == expected
end

#=========================================================================
 Gather/scatter with OOB bounds checking (regression: mask emission bug
 meant bounds checks were silently skipped — load_ptr_tko always loaded
 without mask, so OOB accesses were unchecked)
=========================================================================#

@testset "gather OOB returns zero" begin
    function gather_oob_kernel(out::ct.TileArray{Float32,1},
                                arr::ct.TileArray{Float32,1})
        # arr has 16 elements, we access indices 1-32 → last 16 are OOB
        idx = ct.arange(32)
        tile = ct.gather(arr, idx)
        ct.store(out, (ct.bid(1),), tile)
        return nothing
    end

    arr = CUDA.ones(Float32, 16)
    out = CUDA.zeros(Float32, 32)

    @cuda backend=cuTile gather_oob_kernel(out, arr)

    result = Array(out)
    @test all(result[1:16] .== 1.0f0)
    @test all(result[17:32] .== 0.0f0)
end

@testset "scatter OOB ignored" begin
    function scatter_oob_kernel(arr::ct.TileArray{Float32,1})
        # arr has 16 elements, scatter to indices 1-32 → last 16 are OOB and ignored
        idx = ct.arange(32)
        tile = ct.broadcast_to(ct.Tile(42.0f0), (32,))
        ct.scatter(arr, idx, tile)
        return nothing
    end

    arr = CUDA.zeros(Float32, 16)

    @cuda backend=cuTile scatter_oob_kernel(arr)

    result = Array(arr)
    @test all(result .== 42.0f0)
end

#=========================================================================
 Gather with user mask kwarg
=========================================================================#

@testset "gather with mask" begin
    function masked_gather_kernel(out::ct.TileArray{Float32,1},
                                   arr::ct.TileArray{Float32,1})
        idx = ct.arange(32)
        # Only gather first 16 elements
        user_mask = idx .<= ct.Tile(Int32(16))
        tile = ct.gather(arr, idx; mask=user_mask)
        ct.store(out, (ct.bid(1),), tile)
        return nothing
    end

    arr = CuArray(Float32.(1:64))
    out = CUDA.zeros(Float32, 32)

    @cuda backend=cuTile masked_gather_kernel(out, arr)

    result = Array(out)
    @test result[1:16] == Float32.(1:16)
    @test all(result[17:32] .== 0.0f0)
end

#=========================================================================
 Gather with padding_value kwarg
=========================================================================#

@testset "gather with padding_value" begin
    function padding_gather_kernel(out::ct.TileArray{Float32,1},
                                    arr::ct.TileArray{Float32,1})
        # arr has 16 elements, access 1-32 → last 16 OOB get padding
        idx = ct.arange(32)
        tile = ct.gather(arr, idx; padding_value=-1.0f0)
        ct.store(out, (ct.bid(1),), tile)
        return nothing
    end

    arr = CUDA.ones(Float32, 16)
    out = CUDA.zeros(Float32, 32)

    @cuda backend=cuTile padding_gather_kernel(out, arr)

    result = Array(out)
    @test all(result[1:16] .== 1.0f0)
    @test all(result[17:32] .== -1.0f0)
end

#=========================================================================
 Gather with mask + padding_value combined
=========================================================================#

@testset "gather with mask and padding_value" begin
    function mask_pad_gather_kernel(out::ct.TileArray{Float32,1},
                                     arr::ct.TileArray{Float32,1})
        idx = ct.arange(32)
        # User mask: only first 8 elements
        user_mask = idx .<= ct.Tile(Int32(8))
        tile = ct.gather(arr, idx; mask=user_mask, padding_value=-999.0f0)
        ct.store(out, (ct.bid(1),), tile)
        return nothing
    end

    arr = CuArray(Float32.(1:64))
    out = CUDA.zeros(Float32, 32)

    @cuda backend=cuTile mask_pad_gather_kernel(out, arr)

    result = Array(out)
    @test result[1:8] == Float32.(1:8)
    @test all(result[9:32] .== -999.0f0)
end

#=========================================================================
 Scatter with user mask kwarg
=========================================================================#

@testset "scatter with mask" begin
    function masked_scatter_kernel(arr::ct.TileArray{Float32,1})
        idx = ct.arange(32)
        tile = ct.broadcast_to(ct.Tile(42.0f0), (32,))
        # Only scatter first 16 elements
        user_mask = idx .<= ct.Tile(Int32(16))
        ct.scatter(arr, idx, tile; mask=user_mask)
        return nothing
    end

    arr = CUDA.zeros(Float32, 64)

    @cuda backend=cuTile masked_scatter_kernel(arr)

    result = Array(arr)
    @test all(result[1:16] .== 42.0f0)
    @test all(result[17:64] .== 0.0f0)
end

#=========================================================================
 check_bounds=false (fast path: no bounds mask, no padding, maskless load)
=========================================================================#

@testset "gather with check_bounds=false" begin
    function unchecked_gather_kernel(out::ct.TileArray{Float32,1},
                                      arr::ct.TileArray{Float32,1})
        idx = ct.arange(32)
        tile = ct.gather(arr, idx; check_bounds=false)
        ct.store(out, (ct.bid(1),), tile)
        return nothing
    end

    arr = CuArray(Float32.(1:32))
    out = CUDA.zeros(Float32, 32)

    @cuda backend=cuTile unchecked_gather_kernel(out, arr)

    @test Array(out) == Float32.(1:32)
end

@testset "scatter with check_bounds=false" begin
    function unchecked_scatter_kernel(a::ct.TileArray{Float32,1},
                                      b::ct.TileArray{Float32,1})
        pid = ct.bid(1)
        tile = ct.load(a, pid, (16,))
        base = (pid - 1) * 16
        indices = base .+ ct.arange(16)
        ct.scatter(b, indices, tile; check_bounds=false)
        return nothing
    end

    n = 1024
    a = CUDA.rand(Float32, n)
    b = CUDA.zeros(Float32, n)

    @cuda backend=cuTile blocks=64 unchecked_scatter_kernel(a, b)

    @test Array(b) ≈ Array(a)
end

#=========================================================================
 kwargs in @generated function (regression: Core.kwcall was documented
 as failing in bug.jl, verified to work through launch())
=========================================================================#

@testset "kwargs in @generated kernel" begin
    @generated function gen_kwargs_kernel(dest::ct.TileArray{Float32,1})
        quote
            ct.atomic_add(dest, 1, 1.0f0; memory_order=ct.MemoryOrder.Relaxed)
            return
        end
    end

    out = CUDA.zeros(Float32, 1)
    n_blocks = 100
    @cuda backend=cuTile blocks=n_blocks gen_kwargs_kernel(out)
    @test Array(out)[1] == Float32(n_blocks)
end
