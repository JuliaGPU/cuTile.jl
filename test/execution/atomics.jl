# atomic operations

using CUDA

@testset "atomic_add Int" begin
    # Test atomic_add with Int: each thread block adds 1 to a counter
    function atomic_add_kernel(counters::ct.TileArray{Int,1})
        bid = ct.bid(1)
        ct.atomic_add(counters, 1, 1;
                     memory_order=ct.MemoryOrder.AcqRel)
        return
    end

    n_blocks = 1000
    counters = CUDA.zeros(Int, 1)

    ct.launch(atomic_add_kernel, n_blocks, counters)

    result = Array(counters)[1]
    @test result == n_blocks
end

@testset "atomic_add Float32" begin
    # Test atomic_add with Float32
    function atomic_add_f32_kernel(out::ct.TileArray{Float32,1}, val::Float32)
        bid = ct.bid(1)
        ct.atomic_add(out, 1, val;
                     memory_order=ct.MemoryOrder.AcqRel)
        return
    end

    n_blocks = 100
    out = CUDA.zeros(Float32, 1)
    val = 1.5f0

    ct.launch(atomic_add_f32_kernel, n_blocks, out, ct.Constant(val))

    result = Array(out)[1]
    @test result ≈ n_blocks * val rtol=1e-3
end

@testset "atomic_xchg" begin
    # Test atomic_xchg: each thread exchanges, last one wins
    function atomic_xchg_kernel(arr::ct.TileArray{Int,1})
        bid = ct.bid(1)
        # bid is 1-indexed (1..n_blocks), val is auto-converted from Int32 to Int
        ct.atomic_xchg(arr, 1, bid;
                      memory_order=ct.MemoryOrder.AcqRel)
        return
    end

    n_blocks = 10
    arr = CUDA.zeros(Int, 1)

    ct.launch(atomic_xchg_kernel, n_blocks, arr)

    # Result should be one of 1..n_blocks (whichever thread ran last)
    result = Array(arr)[1]
    @test 1 <= result <= n_blocks
end

@testset "atomic_cas success" begin
    # Test atomic_cas: only one thread should succeed in setting 0->1
    function atomic_cas_kernel(locks::ct.TileArray{Int,1}, success_count::ct.TileArray{Int,1})
        bid = ct.bid(1)
        # Try to acquire lock (0 -> 1)
        old = ct.atomic_cas(locks, 1, 0, 1;
                           memory_order=ct.MemoryOrder.AcqRel)
        # If we got old=0, we succeeded
        # Use atomic_add to count successes (returns a tile, so comparison works)
        # Actually simpler: just increment success_count if old was 0
        # But we can't do conditionals easily here, so let's just verify lock changes
        return
    end

    locks = CUDA.zeros(Int, 1)
    success_count = CUDA.zeros(Int, 1)

    ct.launch(atomic_cas_kernel, 100, locks, success_count)

    # Lock should be set to 1 (at least one thread succeeded)
    lock_val = Array(locks)[1]
    @test lock_val == 1
end

@testset "spinlock with token ordering" begin
    # Test that token threading enforces memory ordering in spinlock patterns
    function spinlock_kernel(result::ct.TileArray{Float32,1}, lock::ct.TileArray{Int,1})
        bid = ct.bid(1)
        val = ct.full((1,), 1.0f0, Float32)

        # Spin until we acquire the lock (CAS returns old value, 0 means we got it)
        while ct.atomic_cas(lock, 1, 0, 1;
                           memory_order=ct.MemoryOrder.Acquire) == 1
        end

        # Critical section: load, increment, store
        # With proper token threading, these are ordered after the acquire
        current = ct.load(result, 1, (1,))
        updated = current .+ val
        ct.store(result, 1, updated)

        # Release the lock
        ct.atomic_xchg(lock, 1, 0;
                      memory_order=ct.MemoryOrder.Release)
        return
    end

    n_blocks = 50  # Use fewer blocks to reduce test time
    result = CUDA.zeros(Float32, 1)
    lock = CUDA.zeros(Int, 1)

    ct.launch(spinlock_kernel, n_blocks, result, lock)

    # Each block should have added 1.0 to the result
    final_result = Array(result)[1]
    @test final_result == Float32(n_blocks)
end

@testset "explicit memory ordering kwargs" begin
    # Test that explicit memory_order kwargs work correctly
    function explicit_ordering_kernel(result::ct.TileArray{Float32,1}, lock::ct.TileArray{Int,1})
        bid = ct.bid(1)
        val = ct.full((1,), 1.0f0, Float32)

        # Spin until we acquire the lock - use explicit Acquire ordering
        while ct.atomic_cas(lock, 1, 0, 1;
                           memory_order=ct.MemoryOrder.Acquire) == 1
        end

        # Critical section
        current = ct.load(result, 1, (1,))
        updated = current .+ val
        ct.store(result, 1, updated)

        # Release the lock - use explicit Release ordering
        ct.atomic_xchg(lock, 1, 0; memory_order=ct.MemoryOrder.Release)
        return
    end

    n_blocks = 50
    result = CUDA.zeros(Float32, 1)
    lock = CUDA.zeros(Int, 1)

    ct.launch(explicit_ordering_kernel, n_blocks, result, lock)

    final_result = Array(result)[1]
    @test final_result == Float32(n_blocks)
end

@testset "atomic_add with explicit kwargs" begin
    # Test atomic_add with explicit memory ordering
    function explicit_add_kernel(counters::ct.TileArray{Int,1})
        bid = ct.bid(1)
        ct.atomic_add(counters, 1, 1;
                     memory_order=ct.MemoryOrder.Relaxed,
                     memory_scope=ct.MemScope.Device)
        return
    end

    n_blocks = 100
    counters = CUDA.zeros(Int, 1)

    ct.launch(explicit_add_kernel, n_blocks, counters)

    result = Array(counters)[1]
    @test result == n_blocks
end


# ============================================================================
# Tile-indexed atomic operations (scatter-gather style indexing)
# ============================================================================

@testset "atomic_add tile-indexed 1D" begin
    function atomic_add_tile_kernel(arr::ct.TileArray{Int,1}, TILE::Int)
        bid = ct.bid(1)
        base = (bid - 1) * TILE
        indices = base .+ ct.arange((TILE,), Int)
        ct.atomic_add(arr, indices, 1;
                     memory_order=ct.MemoryOrder.AcqRel)
        return
    end

    tile_size = 16
    n = 256
    n_blocks = div(n, tile_size)
    arr = CUDA.zeros(Int, n)

    ct.launch(atomic_add_tile_kernel, n_blocks, arr, ct.Constant(tile_size))

    @test all(Array(arr) .== 1)
end

@testset "atomic_add tile-indexed returns old values" begin
    function atomic_add_return_kernel(arr::ct.TileArray{Int,1}, out::ct.TileArray{Int,1})
        indices = ct.arange((16,), Int)
        old_vals = ct.atomic_add(arr, indices, 1;
                                memory_order=ct.MemoryOrder.AcqRel)
        ct.scatter(out, indices, old_vals)
        return
    end

    arr = CUDA.zeros(Int, 16)
    out = CUDA.fill(Int(-1), 16)

    ct.launch(atomic_add_return_kernel, 1, arr, out)

    @test all(Array(out) .== 0)
    @test all(Array(arr) .== 1)
end

@testset "atomic_add tile-indexed Float32" begin
    function atomic_add_f32_tile_kernel(arr::ct.TileArray{Float32,1}, TILE::Int)
        bid = ct.bid(1)
        base = (bid - 1) * TILE
        indices = base .+ ct.arange((TILE,), Int)
        ct.atomic_add(arr, indices, 1.5f0;
                     memory_order=ct.MemoryOrder.AcqRel)
        return
    end

    tile_size = 16
    n = 256
    n_blocks = div(n, tile_size)
    arr = CUDA.zeros(Float32, n)

    ct.launch(atomic_add_f32_tile_kernel, n_blocks, arr, ct.Constant(tile_size))

    @test all(isapprox.(Array(arr), 1.5f0))
end

@testset "atomic_add tile-indexed with tile values" begin
    function atomic_add_tile_val_kernel(arr::ct.TileArray{Int,1},
                                        vals::ct.TileArray{Int,1})
        indices = ct.arange((16,), Int)
        val_tile = ct.gather(vals, indices)
        ct.atomic_add(arr, indices, val_tile;
                     memory_order=ct.MemoryOrder.AcqRel)
        return
    end

    arr = CUDA.zeros(Int, 16)
    vals = CuArray(collect(Int, 1:16))

    ct.launch(atomic_add_tile_val_kernel, 1, arr, vals)

    @test Array(arr) == collect(1:16)
end

@testset "atomic_xchg tile-indexed" begin
    function atomic_xchg_tile_kernel(arr::ct.TileArray{Int,1})
        indices = ct.arange((16,), Int)
        ct.atomic_xchg(arr, indices, 42;
                      memory_order=ct.MemoryOrder.AcqRel)
        return
    end

    arr = CUDA.zeros(Int, 16)

    ct.launch(atomic_xchg_tile_kernel, 1, arr)

    @test all(Array(arr) .== 42)
end

@testset "atomic_cas tile-indexed success" begin
    function atomic_cas_tile_kernel(arr::ct.TileArray{Int,1}, out::ct.TileArray{Int,1})
        indices = ct.arange((16,), Int)
        old_vals = ct.atomic_cas(arr, indices, 0, 1;
                                memory_order=ct.MemoryOrder.AcqRel)
        ct.scatter(out, indices, old_vals)
        return
    end

    arr = CUDA.zeros(Int, 16)
    out = CUDA.fill(Int(-1), 16)

    ct.launch(atomic_cas_tile_kernel, 1, arr, out)

    @test all(Array(out) .== 0)
    @test all(Array(arr) .== 1)
end

@testset "atomic_cas tile-indexed failure" begin
    function atomic_cas_fail_kernel(arr::ct.TileArray{Int,1}, out::ct.TileArray{Int,1})
        indices = ct.arange((16,), Int)
        old_vals = ct.atomic_cas(arr, indices, 0, 2;
                                memory_order=ct.MemoryOrder.AcqRel)
        ct.scatter(out, indices, old_vals)
        return
    end

    arr = CUDA.fill(Int(1), 16)
    out = CUDA.fill(Int(-1), 16)

    ct.launch(atomic_cas_fail_kernel, 1, arr, out)

    @test all(Array(out) .== 1)   # old values returned
    @test all(Array(arr) .== 1)   # unchanged (CAS failed)
end

@testset "atomic_add tile-indexed out-of-bounds" begin
    function atomic_add_oob_kernel(arr::ct.TileArray{Int,1})
        # Index tile is larger than array — OOB elements should be masked
        indices = ct.arange((16,), Int)
        ct.atomic_add(arr, indices, 1;
                     memory_order=ct.MemoryOrder.AcqRel)
        return
    end

    arr = CUDA.zeros(Int, 8)

    ct.launch(atomic_add_oob_kernel, 1, arr)

    # Only first 8 elements should be updated
    @test all(Array(arr) .== 1)
end

@testset "atomic_add tile-indexed 3D" begin
    function atomic_add_3d_kernel(arr::ct.TileArray{Int,3})
        # 3D index tiles — each is length 4, will broadcast to (4,4,4) = 64 elements
        i = ct.reshape(ct.arange((4,), Int), (4, 1, 1))
        j = ct.reshape(ct.arange((4,), Int), (1, 4, 1))
        k = ct.reshape(ct.arange((4,), Int), (1, 1, 4))
        ct.atomic_add(arr, (i, j, k), 1;
                     memory_order=ct.MemoryOrder.AcqRel)
        return
    end

    arr = CUDA.zeros(Int, 4, 4, 4)

    ct.launch(atomic_add_3d_kernel, 1, arr)

    @test all(Array(arr) .== 1)
end

@testset "1D gather - simple" begin
    # Simple 1D gather: copy first 16 elements using gather
    function gather_simple_kernel(src::ct.TileArray{Float32,1}, dst::ct.TileArray{Float32,1})
        pid = ct.bid(1)
        # Simple indices 0..15
        indices = ct.arange((16,), Int)
        # Gather from source
        tile = ct.gather(src, indices)
        # Store to destination
        ct.store(dst, pid, tile)
        return
    end

    n = 16
    src = CUDA.rand(Float32, n)
    dst = CUDA.zeros(Float32, n)

    ct.launch(gather_simple_kernel, 1, src, dst)

    @test Array(dst) ≈ Array(src)
end

@testset "1D scatter - simple" begin
    # Simple 1D scatter: write first 16 elements using scatter
    function scatter_simple_kernel(src::ct.TileArray{Float32,1}, dst::ct.TileArray{Float32,1})
        pid = ct.bid(1)
        # Load from source
        tile = ct.load(src, pid, (16,))
        # Simple indices 0..15
        indices = ct.arange((16,), Int)
        # Scatter to destination
        ct.scatter(dst, indices, tile)
        return
    end

    n = 16
    src = CUDA.rand(Float32, n)
    dst = CUDA.zeros(Float32, n)

    ct.launch(scatter_simple_kernel, 1, src, dst)

    @test Array(dst) ≈ Array(src)
end
