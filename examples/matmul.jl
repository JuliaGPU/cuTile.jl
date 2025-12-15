# Matrix multiplication example - Julia port of cuTile Python's matmul sample

using CUDA
import cuTile as ct

# 2D swizzle for better L2 cache locality
# Groups blocks to access nearby memory regions together
@inline function swizzle_2d(M, N, tm, tn, GROUP_SIZE_M, bid)
    num_bid_m = ct.cdiv(M, Int32(tm))
    num_bid_n = ct.cdiv(N, Int32(tn))
    num_bid_in_group = Int32(GROUP_SIZE_M) * num_bid_n
    group_id = ct.floordiv(bid, num_bid_in_group)
    first_bid_m = group_id * Int32(GROUP_SIZE_M)
    group_size_m = min(num_bid_m - first_bid_m, Int32(GROUP_SIZE_M))
    bid_m = first_bid_m + rem(bid, group_size_m)
    bid_n = ct.floordiv(rem(bid, num_bid_in_group), group_size_m)
    return bid_m, bid_n
end

# Matrix multiplication kernel with K reduction loop and 2D swizzle
# C = A @ B where A is (M, K), B is (K, N), C is (M, N)
function matmul_kernel(A::ct.TileArray{T,2}, B::ct.TileArray{T,2}, C::ct.TileArray{T,2},
                       tm::ct.Constant{Int}, tn::ct.Constant{Int}, tk::ct.Constant{Int}) where {T}
    # Use 1D grid with swizzle for better cache locality
    bid = ct.bid(0)
    M = A.sizes[1]
    N = B.sizes[2]
    bid_m, bid_n = swizzle_2d(M, N, tm[], tn[], 8, bid)

    # Number of K tiles to iterate over
    num_k = ct.num_tiles(A, 1, (tm[], tk[]))

    # Initialize accumulator with Float32 for precision
    acc = ct.full((tm[], tn[]), zero(Float32), Float32)

    # Convert fp32 to tf32 to use tensor cores (like Python cuTile)
    dtype = T === Float32 ? ct.TFloat32 : T

    # K reduction loop - accumulate partial products
    k = Int32(0)
    while k < num_k
        a = ct.astype(ct.load(A, (bid_m, k), (tm[], tk[])), dtype)
        b = ct.astype(ct.load(B, (k, bid_n), (tk[], tn[])), dtype)
        acc = ct.mma(a, b, acc)
        k += Int32(1)
    end

    # Convert accumulator to output type and store
    result = ct.astype(acc, T)
    ct.store(C, (bid_m, bid_n), result)

    return nothing
end

function test_matmul(::Type{T}, M, N, K, tm, tn, tk; name=nothing) where T
    name = something(name, "matmul ($M x $K) @ ($K x $N), $T, tiles=$tm x $tn x $tk")
    println("--- $name ---")

    A = CUDA.rand(T, M, K)
    B = CUDA.rand(T, K, N)
    C = CUDA.zeros(T, M, N)

    # Use 1D grid - swizzle_2d converts to 2D indices
    grid_m = cld(M, tm)
    grid_n = cld(N, tn)
    grid = grid_m * grid_n

    # Launch kernel
    ct.launch(matmul_kernel, grid, A, B, C,
              ct.Constant(tm), ct.Constant(tn), ct.Constant(tk))

    # Verify result
    expected = Array(A) * Array(B)
    result = Array(C)

    if isapprox(result, expected, rtol=1e-2, atol=1e-2)
        println("  passed")
    else
        max_diff = maximum(abs.(result - expected))
        println("  FAILED (max diff: $max_diff)")
    end
end

function main()
    println("--- cuTile Matrix Multiplication Examples ---\n")

    # Small matrices with Float32
    test_matmul(Float32, 256, 256, 256, 32, 32, 32)
    test_matmul(Float32, 512, 512, 512, 64, 64, 64)

    # Non-square matrices
    test_matmul(Float32, 256, 512, 128, 32, 32, 32)
    test_matmul(Float32, 512, 256, 384, 64, 64, 64)

    # Larger matrices
    test_matmul(Float32, 1024, 1024, 1024, 32, 32, 32)

    println("\n--- All matmul examples completed ---")
end

isinteractive() || main()
