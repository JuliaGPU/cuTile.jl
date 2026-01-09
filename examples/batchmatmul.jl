# Batch matrix multiplication example - Julia port of cuTile Python's BatchMatMul.py sample
#
# SPDX-License-Identifier: Apache-2.0
#
# Uses Julia-idiomatic batch-last ordering: A(M, K, Batch), B(K, N, Batch), C(M, N, Batch)
# This provides optimal memory access with Julia's column-major layout.

using CUDA
import cuTile as ct

# Batch matrix multiplication kernel
# A: (M, K, Batch), B: (K, N, Batch), C: (M, N, Batch)
# Grid: (M_tiles, N_tiles, Batch)
function batch_matmul_kernel(A::ct.TileArray{T,3}, B::ct.TileArray{T,3}, C::ct.TileArray{T,3},
                             tm::ct.Constant{Int}, tn::ct.Constant{Int},
                             tk::ct.Constant{Int}) where {T}
    # Grid dimensions (1-indexed)
    bid_m = ct.bid(1)      # M tile index
    bid_n = ct.bid(2)      # N tile index
    pid_batch = ct.bid(3)  # Batch index

    # Number of K tiles to iterate over
    K = A.sizes[2]
    num_k = cld(K, Int32(tk[]))

    # Initialize accumulator with Float32 for precision
    acc = ct.full((tm[], tn[]), zero(Float32), Float32)

    # K reduction loop
    k = Int32(1)
    while k <= num_k
        # Load 3D tiles: (tm, tk, 1) and (tk, tn, 1)
        a = ct.load(A, (bid_m, k, pid_batch), (tm[], tk[], 1);
                    padding_mode=ct.PaddingMode.Zero)
        b = ct.load(B, (k, bid_n, pid_batch), (tk[], tn[], 1);
                    padding_mode=ct.PaddingMode.Zero)

        # Reshape 3D tiles to 2D for mma
        a_2d = ct.reshape(a, (tm[], tk[]))
        b_2d = ct.reshape(b, (tk[], tn[]))

        # Convert to TF32 for tensor cores (Float32 inputs only)
        if T === Float32
            a_2d = convert(ct.Tile{ct.TFloat32}, a_2d)
            b_2d = convert(ct.Tile{ct.TFloat32}, b_2d)
        end

        acc = muladd(a_2d, b_2d, acc)
        k += Int32(1)
    end

    # Convert to output type, reshape to 3D, and store
    result = convert(ct.Tile{T}, acc)
    result_3d = ct.reshape(result, (tm[], tn[], 1))
    ct.store(C, (bid_m, bid_n, pid_batch), result_3d)

    return nothing
end

function test_batch_matmul(::Type{T}, M, K, N, Batch, tm, tn, tk; name=nothing) where T
    name = something(name, "batch_matmul ($M x $K x $Batch) @ ($K x $N x $Batch), $T, tiles=$tm x $tn x $tk")
    println("--- $name ---")

    # Batch-last ordering for optimal column-major access
    A = CUDA.rand(T, M, K, Batch)
    B = CUDA.rand(T, K, N, Batch)
    C = CUDA.zeros(T, M, N, Batch)

    # 3D grid: (M_tiles, N_tiles, Batch)
    grid = (cld(M, tm), cld(N, tn), Batch)

    # Launch kernel
    ct.launch(batch_matmul_kernel, grid, A, B, C,
              ct.Constant(tm), ct.Constant(tn), ct.Constant(tk))

    # Verify result - compute batched matmul on CPU
    A_cpu = Array(A)
    B_cpu = Array(B)
    expected = similar(A_cpu, M, N, Batch)
    for b in 1:Batch
        expected[:, :, b] = A_cpu[:, :, b] * B_cpu[:, :, b]
    end
    result = Array(C)

    if isapprox(result, expected, rtol=1e-2, atol=1e-2)
        println("  passed")
    else
        max_diff = maximum(abs.(result - expected))
        println("  FAILED (max diff: $max_diff)")
    end
end

function main()
    println("--- cuTile Batch Matrix Multiplication Examples ---\n")

    # Float32 tests with smaller tile sizes
    test_batch_matmul(Float32, 256, 128, 256, 4, 32, 32, 32)
    test_batch_matmul(Float32, 512, 256, 512, 4, 64, 64, 64)

    # Float16 tests - can use larger tiles for tensor cores
    test_batch_matmul(Float16, 512, 256, 1024, 4, 128, 256, 64)

    # Non-square matrices
    test_batch_matmul(Float32, 256, 512, 128, 2, 32, 32, 32)

    println("\n--- All batch matmul examples completed ---")
end

isinteractive() || main()
