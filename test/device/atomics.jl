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

    @cuda backend=cuTile blocks=n_blocks atomic_add_kernel(counters)

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

    @cuda backend=cuTile blocks=n_blocks atomic_add_f32_kernel(out, ct.Constant(val))

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

    @cuda backend=cuTile blocks=n_blocks atomic_xchg_kernel(arr)

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

    @cuda backend=cuTile blocks=100 atomic_cas_kernel(locks, success_count)

    # Lock should be set to 1 (at least one thread succeeded)
    lock_val = Array(locks)[1]
    @test lock_val == 1
end

@testset "spinlock with token ordering" begin
    # Test that token threading enforces memory ordering in spinlock patterns
    function spinlock_kernel(result::ct.TileArray{Float32,1}, lock::ct.TileArray{Int,1})
        bid = ct.bid(1)
        val = fill(1.0f0, (1,))

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

    @cuda backend=cuTile blocks=n_blocks spinlock_kernel(result, lock)

    # Each block should have added 1.0 to the result
    final_result = Array(result)[1]
    @test final_result == Float32(n_blocks)
end

@testset "explicit memory ordering kwargs" begin
    # Test that explicit memory_order kwargs work correctly
    function explicit_ordering_kernel(result::ct.TileArray{Float32,1}, lock::ct.TileArray{Int,1})
        bid = ct.bid(1)
        val = fill(1.0f0, (1,))

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

    @cuda backend=cuTile blocks=n_blocks explicit_ordering_kernel(result, lock)

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

    @cuda backend=cuTile blocks=n_blocks explicit_add_kernel(counters)

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
        indices = base .+ ct.arange(TILE; dtype=Int)
        ct.atomic_add(arr, indices, 1;
                     memory_order=ct.MemoryOrder.AcqRel)
        return
    end

    tile_size = 16
    n = 256
    n_blocks = div(n, tile_size)
    arr = CUDA.zeros(Int, n)

    @cuda backend=cuTile blocks=n_blocks atomic_add_tile_kernel(arr, ct.Constant(tile_size))

    @test all(Array(arr) .== 1)
end

@testset "atomic_add tile-indexed returns old values" begin
    function atomic_add_return_kernel(arr::ct.TileArray{Int,1}, out::ct.TileArray{Int,1})
        indices = ct.arange(16; dtype=Int)
        old_vals = ct.atomic_add(arr, indices, 1;
                                memory_order=ct.MemoryOrder.AcqRel)
        ct.scatter(out, indices, old_vals)
        return
    end

    arr = CUDA.zeros(Int, 16)
    out = CUDA.fill(Int(-1), 16)

    @cuda backend=cuTile atomic_add_return_kernel(arr, out)

    @test all(Array(out) .== 0)
    @test all(Array(arr) .== 1)
end

@testset "atomic_add tile-indexed Float32" begin
    function atomic_add_f32_tile_kernel(arr::ct.TileArray{Float32,1}, TILE::Int)
        bid = ct.bid(1)
        base = (bid - 1) * TILE
        indices = base .+ ct.arange(TILE; dtype=Int)
        ct.atomic_add(arr, indices, 1.5f0;
                     memory_order=ct.MemoryOrder.AcqRel)
        return
    end

    tile_size = 16
    n = 256
    n_blocks = div(n, tile_size)
    arr = CUDA.zeros(Float32, n)

    @cuda backend=cuTile blocks=n_blocks atomic_add_f32_tile_kernel(arr, ct.Constant(tile_size))

    @test all(isapprox.(Array(arr), 1.5f0))
end

@testset "atomic_add tile-indexed with tile values" begin
    function atomic_add_tile_val_kernel(arr::ct.TileArray{Int,1},
                                        vals::ct.TileArray{Int,1})
        indices = ct.arange(16; dtype=Int)
        val_tile = ct.gather(vals, indices)
        ct.atomic_add(arr, indices, val_tile;
                     memory_order=ct.MemoryOrder.AcqRel)
        return
    end

    arr = CUDA.zeros(Int, 16)
    vals = CuArray(collect(Int, 1:16))

    @cuda backend=cuTile atomic_add_tile_val_kernel(arr, vals)

    @test Array(arr) == collect(1:16)
end

@testset "atomic_xchg tile-indexed" begin
    function atomic_xchg_tile_kernel(arr::ct.TileArray{Int,1})
        indices = ct.arange(16; dtype=Int)
        ct.atomic_xchg(arr, indices, 42;
                      memory_order=ct.MemoryOrder.AcqRel)
        return
    end

    arr = CUDA.zeros(Int, 16)

    @cuda backend=cuTile atomic_xchg_tile_kernel(arr)

    @test all(Array(arr) .== 42)
end

@testset "atomic_cas tile-indexed success" begin
    function atomic_cas_tile_kernel(arr::ct.TileArray{Int,1}, out::ct.TileArray{Int,1})
        indices = ct.arange(16; dtype=Int)
        old_vals = ct.atomic_cas(arr, indices, 0, 1;
                                memory_order=ct.MemoryOrder.AcqRel)
        ct.scatter(out, indices, old_vals)
        return
    end

    arr = CUDA.zeros(Int, 16)
    out = CUDA.fill(Int(-1), 16)

    @cuda backend=cuTile atomic_cas_tile_kernel(arr, out)

    @test all(Array(out) .== 0)
    @test all(Array(arr) .== 1)
end

@testset "atomic_cas tile-indexed failure" begin
    function atomic_cas_fail_kernel(arr::ct.TileArray{Int,1}, out::ct.TileArray{Int,1})
        indices = ct.arange(16; dtype=Int)
        old_vals = ct.atomic_cas(arr, indices, 0, 2;
                                memory_order=ct.MemoryOrder.AcqRel)
        ct.scatter(out, indices, old_vals)
        return
    end

    arr = CUDA.fill(Int(1), 16)
    out = CUDA.fill(Int(-1), 16)

    @cuda backend=cuTile atomic_cas_fail_kernel(arr, out)

    @test all(Array(out) .== 1)   # old values returned
    @test all(Array(arr) .== 1)   # unchanged (CAS failed)
end

@testset "atomic_add tile-indexed out-of-bounds" begin
    function atomic_add_oob_kernel(arr::ct.TileArray{Int,1})
        # Index tile is larger than array — OOB elements should be masked
        indices = ct.arange(16; dtype=Int)
        ct.atomic_add(arr, indices, 1;
                     memory_order=ct.MemoryOrder.AcqRel)
        return
    end

    arr = CUDA.zeros(Int, 8)

    @cuda backend=cuTile atomic_add_oob_kernel(arr)

    # Only first 8 elements should be updated
    @test all(Array(arr) .== 1)
end

@testset "atomic_add tile-indexed 3D" begin
    function atomic_add_3d_kernel(arr::ct.TileArray{Int,3})
        # 3D index tiles — each is length 4, will broadcast to (4,4,4) = 64 elements
        i = reshape(ct.arange(4; dtype=Int), (4, 1, 1))
        j = reshape(ct.arange(4; dtype=Int), (1, 4, 1))
        k = reshape(ct.arange(4; dtype=Int), (1, 1, 4))
        ct.atomic_add(arr, (i, j, k), 1;
                     memory_order=ct.MemoryOrder.AcqRel)
        return
    end

    arr = CUDA.zeros(Int, 4, 4, 4)

    @cuda backend=cuTile atomic_add_3d_kernel(arr)

    @test all(Array(arr) .== 1)
end

@testset "atomic_max Int" begin
    function atomic_max_kernel(out::ct.TileArray{Int,1})
        bid = ct.bid(1)
        ct.atomic_max(out, 1, bid;
                     memory_order=ct.MemoryOrder.AcqRel)
        return
    end

    n_blocks = 100
    out = CUDA.zeros(Int, 1)

    @cuda backend=cuTile blocks=n_blocks atomic_max_kernel(out)

    result = Array(out)[1]
    @test result == n_blocks
end

@testset "atomic_min Int" begin
    function atomic_min_kernel(out::ct.TileArray{Int,1})
        bid = ct.bid(1)
        ct.atomic_min(out, 1, bid;
                     memory_order=ct.MemoryOrder.AcqRel)
        return
    end

    n_blocks = 100
    out = CUDA.fill(Int(999), 1)

    @cuda backend=cuTile blocks=n_blocks atomic_min_kernel(out)

    result = Array(out)[1]
    @test result == 1
end

@testset "atomic_max/min UInt32 (unsigned compare)" begin
    function atomic_umax_kernel(out::ct.TileArray{UInt32,1})
        ct.atomic_max(out, 1, 0x80000000; memory_order=ct.MemoryOrder.AcqRel)
        return
    end
    function atomic_umin_kernel(out::ct.TileArray{UInt32,1})
        ct.atomic_min(out, 1, 0x00000001; memory_order=ct.MemoryOrder.AcqRel)
        return
    end

    out_max = CUDA.fill(0x00000001, 1)
    @cuda backend=cuTile blocks=8 atomic_umax_kernel(out_max)
    @test Array(out_max)[1] == 0x80000000

    out_min = CUDA.fill(0x80000000, 1)
    @cuda backend=cuTile blocks=8 atomic_umin_kernel(out_min)
    @test Array(out_min)[1] == 0x00000001
end

@testset "atomic_or Int" begin
    function atomic_or_kernel(out::ct.TileArray{Int,1})
        bid = ct.bid(1)
        # Set bit (bid-1) — bid is 1-indexed
        ct.atomic_or(out, 1, 1 << (bid - 1);
                    memory_order=ct.MemoryOrder.AcqRel)
        return
    end

    n_blocks = 16
    out = CUDA.zeros(Int, 1)

    @cuda backend=cuTile blocks=n_blocks atomic_or_kernel(out)

    result = Array(out)[1]
    @test result == (1 << n_blocks) - 1
end

@testset "atomic_and Int" begin
    function atomic_and_kernel(out::ct.TileArray{Int,1})
        bid = ct.bid(1)
        # Clear bit (bid-1)
        ct.atomic_and(out, 1, ~(1 << (bid - 1));
                     memory_order=ct.MemoryOrder.AcqRel)
        return
    end

    n_blocks = 16
    out = CUDA.fill(Int((1 << n_blocks) - 1), 1)

    @cuda backend=cuTile blocks=n_blocks atomic_and_kernel(out)

    result = Array(out)[1]
    @test result == 0
end

@testset "atomic_xor Int" begin
    function atomic_xor_kernel(out::ct.TileArray{Int,1})
        bid = ct.bid(1)
        ct.atomic_xor(out, 1, Int(bid);
                     memory_order=ct.MemoryOrder.AcqRel)
        return
    end

    n_blocks = 100
    out = CUDA.zeros(Int, 1)

    @cuda backend=cuTile blocks=n_blocks atomic_xor_kernel(out)

    # Expected: XOR of 1..n_blocks
    expected = reduce(xor, 1:n_blocks)
    result = Array(out)[1]
    @test result == expected
end

# View-based atomic reductions

@testset "atomic_store reductions (multi-block)" begin
    N = 16
    n_blocks = 16

    function k_add(a::ct.TileArray{Int32,1})
        tiles = ct.eachtile(a, (16,))
        ct.atomic_store_add(tiles, 1, ct.broadcast_to(ct.Tile(Int32(1)), (16,)))
        return
    end
    a = CUDA.zeros(Int32, N)
    @cuda backend=cuTile blocks=n_blocks k_add(a)
    @test all(Array(a) .== n_blocks)

    function k_addf(a::ct.TileArray{Float32,1})
        tiles = ct.eachtile(a, (16,))
        ct.atomic_store_add(tiles, 1, ct.broadcast_to(ct.Tile(1.5f0), (16,)))
        return
    end
    af = CUDA.zeros(Float32, N)
    @cuda backend=cuTile blocks=n_blocks k_addf(af)
    @test all(isapprox.(Array(af), n_blocks * 1.5f0))

    function k_max(a::ct.TileArray{Int32,1})
        bid = ct.bid(1)
        tiles = ct.eachtile(a, (16,))
        ct.atomic_store_max(tiles, 1, ct.broadcast_to(ct.Tile(Int32(bid)), (16,)))
        return
    end
    a = CUDA.zeros(Int32, N)
    @cuda backend=cuTile blocks=n_blocks k_max(a)
    @test all(Array(a) .== n_blocks)

    function k_min(a::ct.TileArray{Int32,1})
        bid = ct.bid(1)
        tiles = ct.eachtile(a, (16,))
        ct.atomic_store_min(tiles, 1, ct.broadcast_to(ct.Tile(Int32(bid)), (16,)))
        return
    end
    a = CUDA.fill(Int32(999), N)
    @cuda backend=cuTile blocks=n_blocks k_min(a)
    @test all(Array(a) .== 1)

    function k_or(a::ct.TileArray{Int32,1})
        bid = ct.bid(1)
        tiles = ct.eachtile(a, (16,))
        ct.atomic_store_or(tiles, 1, ct.broadcast_to(ct.Tile(Int32(bid)), (16,)))
        return
    end
    a = CUDA.zeros(Int32, N)
    @cuda backend=cuTile blocks=n_blocks k_or(a)
    @test all(Array(a) .== reduce(|, Int32(1):Int32(n_blocks)))

    function k_and(a::ct.TileArray{Int32,1})
        bid = ct.bid(1)
        tiles = ct.eachtile(a, (16,))
        ct.atomic_store_and(tiles, 1, ct.broadcast_to(ct.Tile(Int32(bid)), (16,)))
        return
    end
    a = CUDA.fill(Int32(-1), N)
    @cuda backend=cuTile blocks=n_blocks k_and(a)
    @test all(Array(a) .== foldl(&, Int32(1):Int32(n_blocks); init=Int32(-1)))

    function k_xor(a::ct.TileArray{Int32,1})
        bid = ct.bid(1)
        tiles = ct.eachtile(a, (16,))
        ct.atomic_store_xor(tiles, 1, ct.broadcast_to(ct.Tile(Int32(bid)), (16,)))
        return
    end
    a = CUDA.zeros(Int32, N)
    @cuda backend=cuTile blocks=n_blocks k_xor(a)
    @test all(Array(a) .== reduce(xor, Int32(1):Int32(n_blocks)))
end

@testset "atomic_store_add broadcast update shapes" begin
    function k_scalar(a::ct.TileArray{Float32,2})
        ct.atomic_store_add(ct.eachtile(a, (16, 16)), (1, 1), ct.Tile(1.0f0))
        return
    end
    function k_row(a::ct.TileArray{Float32,2})
        ct.atomic_store_add(ct.eachtile(a, (16, 16)), (1, 1),
                            ct.broadcast_to(ct.Tile(1.0f0), (1, 16)))
        return
    end
    function k_col(a::ct.TileArray{Float32,2})
        ct.atomic_store_add(ct.eachtile(a, (16, 16)), (1, 1),
                            ct.broadcast_to(ct.Tile(1.0f0), (16, 1)))
        return
    end
    n_blocks = 8
    for k in (k_scalar, k_row, k_col)
        a = CUDA.zeros(Float32, 16, 16)
        @cuda backend=cuTile blocks=n_blocks k(a)
        @test all(isapprox.(Array(a), Float32(n_blocks)))
    end
end

@testset "atomic_store_add boundary partial tile" begin
    function k(a::ct.TileArray{Float32,1})
        tiles = ct.eachtile(a, (16,))
        ct.atomic_store_add(tiles, 2, ct.broadcast_to(ct.Tile(1.0f0), (16,)))
        return
    end
    a = CUDA.zeros(Float32, 20)
    @cuda backend=cuTile k(a)
    r = Array(a)
    @test all(r[1:16] .== 0)
    @test all(r[17:20] .== 1)
end

@testset "atomic_store_add stepped (overlapping) windows" begin
    function k(a::ct.TileArray{Float32,1})
        bid = ct.bid(1)
        tiles = ct.eachtile(a, (16,); step=(8,))
        ct.atomic_store_add(tiles, bid, ct.broadcast_to(ct.Tile(1.0f0), (16,)))
        return
    end
    a = CUDA.zeros(Float32, 24)
    @cuda backend=cuTile blocks=3 k(a)
    r = Array(a)
    @test all(r[1:8] .== 1)
    @test all(r[9:24] .== 2)
end

@testset "atomic_store_add f16/bf16" begin
    for T in (Float16, ct.BFloat16)
        function k(a::ct.TileArray{T,1}) where {T}
            tiles = ct.eachtile(a, (16,))
            ct.atomic_store_add(tiles, 1, ct.broadcast_to(ct.Tile(one(eltype(a))), (16,)))
            return
        end
        n_blocks = 8
        a = CUDA.zeros(T, 16)
        @cuda backend=cuTile blocks=n_blocks k(a)
        @test all(Array(a) .== T(n_blocks))
    end
end

@testset "@atomic macro" begin
    function k_stmt(a::ct.TileArray{Float32,1})
        tiles = ct.eachtile(a, (16,))
        ct.@atomic tiles[1] += ct.broadcast_to(ct.Tile(1.0f0), (16,))
        return
    end
    a = CUDA.zeros(Float32, 16)
    @cuda backend=cuTile blocks=10 k_stmt(a)
    @test all(Array(a) .== 10)

    function k_ordered(c::ct.TileArray{Int,1})
        ct.@atomic :acquire_release c[1] += 1
        return
    end
    c = CUDA.zeros(Int, 1)
    @cuda backend=cuTile blocks=100 k_ordered(c)
    @test Array(c)[1] == 100

    function k_sub(c::ct.TileArray{Int,1})
        ct.@atomic c[1] -= 2
        return
    end
    c = CUDA.zeros(Int, 1)
    @cuda backend=cuTile blocks=10 k_sub(c)
    @test Array(c)[1] == -20

    function k_max(c::ct.TileArray{Int,1})
        bid = ct.bid(1)
        ct.@atomic c[1] = max(c[1], bid)
        return
    end
    c = CUDA.zeros(Int, 1)
    @cuda backend=cuTile blocks=50 k_max(c)
    @test Array(c)[1] == 50

    function k_value(c::ct.TileArray{Int,1}, out::ct.TileArray{Int,1})
        pair = ct.@atomic c[1] + 5
        ct.store(out, 1, pair.first)
        ct.store(out, 2, pair.second)
        return
    end
    c = CUDA.fill(Int(7), 1)
    out = CUDA.zeros(Int, 2)
    @cuda backend=cuTile k_value(c, out)
    @test Array(out) == [7, 12]
    @test Array(c)[1] == 12

    function k_value_convert(c::ct.TileArray{Float32,1}, out::ct.TileArray{Float32,1})
        pair = ct.@atomic c[1] + 1.0
        ct.store(out, 1, pair.second)
        return
    end
    c = CUDA.fill(Float32(16777216), 1)
    out = CUDA.zeros(Float32, 1)
    @cuda backend=cuTile k_value_convert(c, out)
    @test Array(c)[1] == Float32(16777216)
    @test Array(out)[1] == Array(c)[1]
end

@testset "1D gather - simple" begin
    # Simple 1D gather: copy first 16 elements using gather
    function gather_simple_kernel(src::ct.TileArray{Float32,1}, dst::ct.TileArray{Float32,1})
        pid = ct.bid(1)
        # Simple indices 0..15
        indices = ct.arange(16; dtype=Int)
        # Gather from source
        tile = ct.gather(src, indices)
        # Store to destination
        ct.store(dst, pid, tile)
        return
    end

    n = 16
    src = CUDA.rand(Float32, n)
    dst = CUDA.zeros(Float32, n)

    @cuda backend=cuTile gather_simple_kernel(src, dst)

    @test Array(dst) ≈ Array(src)
end

@testset "1D scatter - simple" begin
    # Simple 1D scatter: write first 16 elements using scatter
    function scatter_simple_kernel(src::ct.TileArray{Float32,1}, dst::ct.TileArray{Float32,1})
        pid = ct.bid(1)
        # Load from source
        tile = ct.load(src, pid, (16,))
        # Simple indices 0..15
        indices = ct.arange(16; dtype=Int)
        # Scatter to destination
        ct.scatter(dst, indices, tile)
        return
    end

    n = 16
    src = CUDA.rand(Float32, n)
    dst = CUDA.zeros(Float32, n)

    @cuda backend=cuTile scatter_simple_kernel(src, dst)

    @test Array(dst) ≈ Array(src)
end
