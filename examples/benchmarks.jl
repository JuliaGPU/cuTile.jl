# EXCLUDE FROM TESTING
#
# Comprehensive benchmarks for cuTile.jl
# Compares: GPUArrays (generic), SIMT (CUDA.jl), cuTile
# Kernels: vadd, transpose, matmul

using CUDA
using LinearAlgebra
using CUDA: GPUArrays
using FFTW
import cuTile as ct

#=============================================================================
 Configuration
=============================================================================#

const NRUNS = 10
const WARMUP = 3

# Data sizes - large enough to saturate GPU and minimize launch overhead
const VADD_SIZE = 2^27           # 512 MB (128M elements)
const TRANSPOSE_DIM = 8192       # 8192x8192 = 268 MB
const MATMUL_DIM = 4096          # 4096x4096x4096

# Tile sizes
const VADD_TILE = 1024
const TRANSPOSE_TILE_M = 64
const TRANSPOSE_TILE_N = 64
const MATMUL_TM = 64
const MATMUL_TN = 64
const MATMUL_TK = 64

#=============================================================================
 Benchmark Utilities
=============================================================================#

struct BenchmarkResult
    name::String
    min_ms::Float64
    mean_ms::Float64
end

function benchmark_kernel(f, nruns::Int=NRUNS, warmup::Int=WARMUP)
    # Warmup
    for _ in 1:warmup
        f()
    end
    CUDA.synchronize()

    # Benchmark
    times = Float64[]
    for _ in 1:nruns
        t = CUDA.@elapsed f()
        push!(times, t * 1000)  # Convert to ms
    end

    return minimum(times), sum(times) / length(times)
end

function print_table(title::String, results::Vector{BenchmarkResult}; extra_col=nothing)
    println()
    println("=" ^ 60)
    println("  ", title)
    println("=" ^ 60)

    if extra_col !== nothing
        println(rpad("Implementation", 20), rpad("Min (ms)", 12), rpad("Mean (ms)", 12), extra_col[1])
        println("-" ^ 60)
        for (i, r) in enumerate(results)
            extra = extra_col[2][i]
            println(rpad(r.name, 20), rpad(round(r.min_ms, digits=3), 12),
                    rpad(round(r.mean_ms, digits=3), 12), extra)
        end
    else
        println(rpad("Implementation", 20), rpad("Min (ms)", 12), "Mean (ms)")
        println("-" ^ 60)
        for r in results
            println(rpad(r.name, 20), rpad(round(r.min_ms, digits=3), 12),
                    round(r.mean_ms, digits=3))
        end
    end
    println("-" ^ 60)
end

#=============================================================================
 Vector Addition
=============================================================================#

# SIMT kernel
function vadd_simt_kernel!(a, b, c)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= length(c)
        @inbounds c[i] = a[i] + b[i]
    end
    return
end

# cuTile kernel
function vadd_cutile_kernel(a, b, c, tile_size::ct.Constant{Int})
    pid = ct.bid(1)
    tile_a = ct.load(a, pid, (tile_size[],))
    tile_b = ct.load(b, pid, (tile_size[],))
    result = tile_a + tile_b
    ct.store(c, pid, result)
    return
end

function benchmark_vadd()
    println("\nBenchmarking Vector Addition...")
    println("  Size: $VADD_SIZE elements ($(VADD_SIZE * 4 / 1e6) MB)")

    a = CUDA.rand(Float32, VADD_SIZE)
    b = CUDA.rand(Float32, VADD_SIZE)
    c = similar(a)
    expected = Array(a) .+ Array(b)

    results = BenchmarkResult[]

    # GPUArrays (broadcast)
    gpuarrays_f = () -> begin
        c .= a .+ b
    end
    gpuarrays_f()
    CUDA.synchronize()
    @assert Array(c) ≈ expected "GPUArrays incorrect!"
    min_t, mean_t = benchmark_kernel(gpuarrays_f)
    push!(results, BenchmarkResult("GPUArrays", min_t, mean_t))

    # SIMT
    threads = 1024
    blocks = cld(VADD_SIZE, threads)
    simt_f = () -> @cuda threads=threads blocks=blocks vadd_simt_kernel!(a, b, c)
    simt_f()
    CUDA.synchronize()
    @assert Array(c) ≈ expected "SIMT incorrect!"
    min_t, mean_t = benchmark_kernel(simt_f)
    push!(results, BenchmarkResult("SIMT (CUDA.jl)", min_t, mean_t))

    # cuTile
    grid = (cld(VADD_SIZE, VADD_TILE), 1, 1)
    cutile_f = () -> ct.launch(vadd_cutile_kernel, grid, a, b, c, ct.Constant(VADD_TILE))
    cutile_f()
    CUDA.synchronize()
    @assert Array(c) ≈ expected "cuTile incorrect!"
    min_t, mean_t = benchmark_kernel(cutile_f)
    push!(results, BenchmarkResult("cuTile.jl", min_t, mean_t))

    # Calculate bandwidth
    bytes = 3 * VADD_SIZE * sizeof(Float32)  # 2 reads + 1 write
    bandwidths = [string(round(bytes / (r.min_ms / 1000) / 1e9, digits=1), " GB/s") for r in results]

    print_table("Vector Addition (Float32)", results; extra_col=("Bandwidth", bandwidths))
    return results
end

#=============================================================================
 Matrix Transpose
=============================================================================#

# SIMT naive kernel
function transpose_simt_naive_kernel!(input, output, M, N)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    if i <= M && j <= N
        @inbounds output[j, i] = input[i, j]
    end
    return
end

# SIMT shared memory kernel
function transpose_simt_shared_kernel!(input, output, M, N)
    TILE = 32
    tile = CuStaticSharedArray(Float32, (TILE+1, TILE))

    x = (blockIdx().x - 1) * TILE + threadIdx().x
    y = (blockIdx().y - 1) * TILE + threadIdx().y

    if x <= M && y <= N
        @inbounds tile[threadIdx().x, threadIdx().y] = input[x, y]
    end
    sync_threads()

    x = (blockIdx().y - 1) * TILE + threadIdx().x
    y = (blockIdx().x - 1) * TILE + threadIdx().y

    if x <= N && y <= M
        @inbounds output[x, y] = tile[threadIdx().y, threadIdx().x]
    end
    return
end

# cuTile kernel
function transpose_cutile_kernel(input, output, tile_m::ct.Constant{Int}, tile_n::ct.Constant{Int})
    pid_m = ct.bid(1)
    pid_n = ct.bid(2)
    tile = ct.load(input, (pid_m, pid_n), (tile_m[], tile_n[]))
    tile_t = ct.transpose(tile)
    ct.store(output, (pid_n, pid_m), tile_t)
    return
end

function benchmark_transpose()
    println("\nBenchmarking Matrix Transpose...")
    M, N = TRANSPOSE_DIM, TRANSPOSE_DIM
    println("  Size: $(M)x$(N) ($(M * N * 4 / 1e6) MB)")

    input = CUDA.rand(Float32, M, N)
    output = CUDA.zeros(Float32, N, M)
    expected = Array(permutedims(input, (2, 1)))

    results = BenchmarkResult[]

    # GPUArrays (permutedims)
    gpuarrays_f = () -> permutedims!(output, input, (2, 1))
    gpuarrays_f()
    CUDA.synchronize()
    @assert Array(output) ≈ expected "GPUArrays incorrect!"
    min_t, mean_t = benchmark_kernel(gpuarrays_f)
    push!(results, BenchmarkResult("GPUArrays", min_t, mean_t))

    # SIMT naive
    fill!(output, 0)
    threads_naive = (16, 16)
    blocks_naive = (cld(M, 16), cld(N, 16))
    simt_naive_f = () -> @cuda threads=threads_naive blocks=blocks_naive transpose_simt_naive_kernel!(input, output, M, N)
    simt_naive_f()
    CUDA.synchronize()
    @assert Array(output) ≈ expected "SIMT naive incorrect!"
    min_t, mean_t = benchmark_kernel(simt_naive_f)
    push!(results, BenchmarkResult("SIMT naive", min_t, mean_t))

    # SIMT shared
    fill!(output, 0)
    threads_shared = (32, 32)
    blocks_shared = (cld(M, 32), cld(N, 32))
    simt_shared_f = () -> @cuda threads=threads_shared blocks=blocks_shared transpose_simt_shared_kernel!(input, output, M, N)
    simt_shared_f()
    CUDA.synchronize()
    @assert Array(output) ≈ expected "SIMT shared incorrect!"
    min_t, mean_t = benchmark_kernel(simt_shared_f)
    push!(results, BenchmarkResult("SIMT shared", min_t, mean_t))

    # cuTile
    fill!(output, 0)
    grid = (cld(M, TRANSPOSE_TILE_M), cld(N, TRANSPOSE_TILE_N), 1)
    cutile_f = () -> ct.launch(transpose_cutile_kernel, grid, input, output,
                               ct.Constant(TRANSPOSE_TILE_M), ct.Constant(TRANSPOSE_TILE_N))
    cutile_f()
    CUDA.synchronize()
    @assert Array(output) ≈ expected "cuTile incorrect!"
    min_t, mean_t = benchmark_kernel(cutile_f)
    push!(results, BenchmarkResult("cuTile.jl", min_t, mean_t))

    # Calculate bandwidth
    bytes = 2 * M * N * sizeof(Float32)  # read + write
    bandwidths = [string(round(bytes / (r.min_ms / 1000) / 1e9, digits=1), " GB/s") for r in results]

    print_table("Matrix Transpose (Float32)", results; extra_col=("Bandwidth", bandwidths))
    return results
end

#=============================================================================
 Matrix Multiplication
=============================================================================#

# 2D swizzle for better L2 cache locality (using 0-indexed block IDs)
@inline function swizzle_2d(M, N, tm, tn, GROUP_SIZE_M, bid)
    num_bid_m = cld(M, Int32(tm))
    num_bid_n = cld(N, Int32(tn))
    num_bid_in_group = Int32(GROUP_SIZE_M) * num_bid_n
    group_id = fld(bid, num_bid_in_group)
    first_bid_m = group_id * Int32(GROUP_SIZE_M)
    group_size_m = min(num_bid_m - first_bid_m, Int32(GROUP_SIZE_M))
    bid_m = first_bid_m + rem(bid, group_size_m)
    bid_n = fld(rem(bid, num_bid_in_group), group_size_m)
    return bid_m, bid_n
end

# cuTile matmul kernel with TF32 tensor cores
function matmul_cutile_kernel(A::ct.TileArray{T,2}, B::ct.TileArray{T,2}, C::ct.TileArray{T,2},
                              tm::ct.Constant{Int}, tn::ct.Constant{Int}, tk::ct.Constant{Int}) where {T}
    bid = ct.bid(1)
    M = A.sizes[1]
    N = B.sizes[2]
    bid_m_0, bid_n_0 = swizzle_2d(M, N, tm[], tn[], 8, bid - Int32(1))
    bid_m = bid_m_0 + Int32(1)
    bid_n = bid_n_0 + Int32(1)

    num_k = ct.num_tiles(A, 2, (tm[], tk[]))
    acc = ct.full((tm[], tn[]), zero(Float32), Float32)

    # Use TF32 for tensor cores
    dtype = T === Float32 ? ct.TFloat32 : T

    k = Int32(1)
    while k <= num_k
        a = ct.astype(ct.load(A, (bid_m, k), (tm[], tk[])), dtype)
        b = ct.astype(ct.load(B, (k, bid_n), (tk[], tn[])), dtype)
        acc = muladd(a, b, acc)
        k += Int32(1)
    end

    result = ct.astype(acc, T)
    ct.store(C, (bid_m, bid_n), result)
    return nothing
end

function benchmark_matmul()
    println("\nBenchmarking Matrix Multiplication...")
    M, N, K = MATMUL_DIM, MATMUL_DIM, MATMUL_DIM
    println("  Size: $(M)x$(K) * $(K)x$(N)")

    A = CUDA.rand(Float32, M, K)
    B = CUDA.rand(Float32, K, N)
    C = CUDA.zeros(Float32, M, N)

    # Reference result (cuBLAS)
    C_ref = similar(C)
    mul!(C_ref, A, B)
    CUDA.synchronize()

    results = BenchmarkResult[]
    flops = 2.0 * M * N * K

    # GPUArrays (generic matmul)
    gpuarrays_f = () -> GPUArrays.generic_matmatmul!(C, A, B, one(Float32), zero(Float32))
    gpuarrays_f()
    CUDA.synchronize()
    @assert isapprox(Array(C), Array(C_ref), rtol=1e-2, atol=1e-2) "GPUArrays incorrect!"
    min_t, mean_t = benchmark_kernel(gpuarrays_f)
    push!(results, BenchmarkResult("GPUArrays", min_t, mean_t))

    # cuBLAS
    fill!(C, 0)
    cublas_f = () -> mul!(C, A, B)
    cublas_f()
    CUDA.synchronize()
    min_t, mean_t = benchmark_kernel(cublas_f)
    push!(results, BenchmarkResult("cuBLAS", min_t, mean_t))

    # cuTile
    fill!(C, 0)
    grid_m = cld(M, MATMUL_TM)
    grid_n = cld(N, MATMUL_TN)
    grid = (grid_m * grid_n, 1, 1)
    cutile_f = () -> ct.launch(matmul_cutile_kernel, grid, A, B, C,
                               ct.Constant(MATMUL_TM), ct.Constant(MATMUL_TN), ct.Constant(MATMUL_TK))
    cutile_f()
    CUDA.synchronize()
    @assert isapprox(Array(C), Array(C_ref), rtol=1e-2, atol=1e-2) "cuTile incorrect!"
    min_t, mean_t = benchmark_kernel(cutile_f)
    push!(results, BenchmarkResult("cuTile.jl", min_t, mean_t))

    # Calculate TFLOPS
    tflops_vals = [string(round(flops / (r.min_ms * 1e-3) / 1e12, digits=2), " TFLOPS") for r in results]

    print_table("Matrix Multiplication (Float32, TF32 cores)", results; extra_col=("Performance", tflops_vals))
    return results
end

#=============================================================================
 Layer Normalization
=============================================================================#

const LAYERNORM_M = 4096
const LAYERNORM_N = 4096
const LAYERNORM_TILE_N = 1024
const LAYERNORM_EPS = 1f-5

# Batch matmul sizes
const BATCHMATMUL_BATCH = 8
const BATCHMATMUL_M = 1024
const BATCHMATMUL_K = 512
const BATCHMATMUL_N = 2048
const BATCHMATMUL_TM = 128
const BATCHMATMUL_TN = 256
const BATCHMATMUL_TK = 64

# FFT sizes
# Tile size is (D, BS, N2D), limited by tileiras compiler.
# Current kernel loads all batches per block, limiting scalability.
const FFT_BATCH = 64
const FFT_SIZE = 512
const FFT_FACTORS = (8, 8, 8)
const FFT_ATOM_PACKING_DIM = 2

# SIMT naive kernel (2-pass: compute mean/var, then normalize)
function layernorm_simt_kernel!(X, W, B, Y, Mean, Rstd, N, eps)
    m = blockIdx().x

    # First pass: compute mean
    mean_acc = 0.0f0
    for i in 1:N
        @inbounds mean_acc += X[m, i]
    end
    mean = mean_acc / N
    @inbounds Mean[m] = mean

    # Second pass: compute variance
    var_acc = 0.0f0
    for i in 1:N
        @inbounds diff = X[m, i] - mean
        var_acc += diff * diff
    end
    var = var_acc / N
    rstd = 1.0f0 / sqrt(var + eps)
    @inbounds Rstd[m] = rstd

    # Third pass: normalize and apply affine
    for i in 1:N
        @inbounds Y[m, i] = (X[m, i] - mean) * rstd * W[i] + B[i]
    end

    return
end

# cuTile kernel (from layernorm.jl)
function layernorm_cutile_kernel(X::ct.TileArray{Float32, 2}, W::ct.TileArray{Float32, 1},
                                  B::ct.TileArray{Float32, 1}, Y::ct.TileArray{Float32, 2},
                                  Mean::ct.TileArray{Float32, 1}, Rstd::ct.TileArray{Float32, 1},
                                  eps::ct.Constant{Float32}, TILE_N::ct.Constant{Int})
    bid_m = ct.bid(1)
    num_tiles = ct.num_tiles(X, 2, (1, TILE_N[]))
    N = X.sizes[2]

    # Compute mean
    mean = ct.full((1, TILE_N[]), 0.0f0, Float32)
    j = Int32(1)
    while j <= num_tiles
        tx = ct.load(X, (bid_m, j), (1, TILE_N[]); padding_mode=ct.PaddingMode.Zero)
        mean = mean .+ tx
        j += Int32(1)
    end
    mean = ct.reduce_sum(mean, 2) / N
    ct.store(Mean, bid_m, mean)

    # Compute variance
    var = ct.full((1, TILE_N[]), 0.0f0, Float32)
    j = Int32(1)
    while j <= num_tiles
        tx = ct.load(X, (bid_m, j), (1, TILE_N[]); padding_mode=ct.PaddingMode.Zero)
        # Mask for valid elements
        mask = ct.broadcast_to(((j - Int32(1)) * Int32(TILE_N[]) .+ ct.arange((TILE_N[],), Int32)) .<= N, (1, TILE_N[]))
        centered_tx = ct.where(mask, tx .- mean, ct.full((1, TILE_N[]), 0.0f0, Float32))
        var = var .+ (centered_tx .^ 2.0f0)
        j += Int32(1)
    end
    var = ct.reduce_sum(var, 2) / N
    rstd = 1.0f0 ./ sqrt.(var .+ eps[])
    ct.store(Rstd, bid_m, rstd)

    # Normalize and apply affine transformation
    j = Int32(1)
    while j <= num_tiles
        tx = ct.load(X, (bid_m, j), (1, TILE_N[]); padding_mode=ct.PaddingMode.Zero)
        tw = ct.load(W, j, (TILE_N[],); padding_mode=ct.PaddingMode.Zero)
        tb = ct.load(B, j, (TILE_N[],); padding_mode=ct.PaddingMode.Zero)
        ty = (tx .- mean) .* rstd
        ty = ty .* tw .+ tb
        ct.store(Y, (bid_m, j), ty)
        j += Int32(1)
    end

    return
end

function benchmark_layernorm()
    println("\nBenchmarking Layer Normalization...")
    M, N = LAYERNORM_M, LAYERNORM_N
    println("  Size: $(M)x$(N) ($(M * N * 4 / 1e6) MB)")

    X = -2.3f0 .+ 0.5f0 .* CUDA.rand(Float32, M, N)
    W = CUDA.randn(Float32, N)
    B = CUDA.randn(Float32, N)
    Y = CUDA.zeros(Float32, M, N)
    Mean = CUDA.zeros(Float32, M)
    Rstd = CUDA.zeros(Float32, M)

    # Reference result
    X_cpu = Array(X)
    W_cpu = Array(W)
    B_cpu = Array(B)
    expected_mean = vec(sum(X_cpu, dims=2) ./ N)
    expected_var = vec(sum((X_cpu .- expected_mean) .^ 2, dims=2) ./ N)
    expected_rstd = 1.0f0 ./ sqrt.(expected_var .+ LAYERNORM_EPS)
    normalized = (X_cpu .- expected_mean) .* expected_rstd
    expected_Y = normalized .* W_cpu' .+ B_cpu'

    results = BenchmarkResult[]

    # SIMT naive (single thread per row)
    fill!(Y, 0); fill!(Mean, 0); fill!(Rstd, 0)
    simt_f = () -> @cuda threads=1 blocks=M layernorm_simt_kernel!(X, W, B, Y, Mean, Rstd, N, LAYERNORM_EPS)
    simt_f()
    CUDA.synchronize()
    @assert isapprox(Array(Y), expected_Y, rtol=1e-2, atol=1e-2) "SIMT incorrect!"
    min_t, mean_t = benchmark_kernel(simt_f)
    push!(results, BenchmarkResult("SIMT naive", min_t, mean_t))

    # cuTile
    fill!(Y, 0); fill!(Mean, 0); fill!(Rstd, 0)
    cutile_f = () -> ct.launch(layernorm_cutile_kernel, M, X, W, B, Y, Mean, Rstd,
                               ct.Constant(LAYERNORM_EPS), ct.Constant(LAYERNORM_TILE_N))
    cutile_f()
    CUDA.synchronize()
    @assert isapprox(Array(Y), expected_Y, rtol=1e-2, atol=1e-2) "cuTile incorrect!"
    min_t, mean_t = benchmark_kernel(cutile_f)
    push!(results, BenchmarkResult("cuTile.jl", min_t, mean_t))

    # Calculate bandwidth (rough estimate: 3 reads of X + W + B, 1 write of Y)
    bytes = (3 * M * N + N + N + M * N) * sizeof(Float32)
    bandwidths = [string(round(bytes / (r.min_ms / 1000) / 1e9, digits=1), " GB/s") for r in results]

    print_table("Layer Normalization (Float32)", results; extra_col=("Bandwidth", bandwidths))
    return results
end

#=============================================================================
 Batch Matrix Multiplication
=============================================================================#

# Batch matmul kernel (3D arrays with batch-last ordering)
# A: (M, K, Batch), B: (K, N, Batch), C: (M, N, Batch)
function batchmatmul_cutile_kernel(A::ct.TileArray{T,3}, B::ct.TileArray{T,3}, C::ct.TileArray{T,3},
                                   tm::ct.Constant{Int}, tn::ct.Constant{Int},
                                   tk::ct.Constant{Int}) where {T}
    bid_m = ct.bid(1)
    bid_n = ct.bid(2)
    pid_batch = ct.bid(3)

    K = A.sizes[2]
    num_k = cld(K, Int32(tk[]))

    acc = ct.full((tm[], tn[]), zero(Float32), Float32)

    k = Int32(1)
    while k <= num_k
        a = ct.load(A, (bid_m, k, pid_batch), (tm[], tk[], 1);
                    padding_mode=ct.PaddingMode.Zero)
        b = ct.load(B, (k, bid_n, pid_batch), (tk[], tn[], 1);
                    padding_mode=ct.PaddingMode.Zero)

        a_2d = ct.reshape(a, (tm[], tk[]))
        b_2d = ct.reshape(b, (tk[], tn[]))

        if T === Float32
            a_2d = convert(ct.Tile{ct.TFloat32}, a_2d)
            b_2d = convert(ct.Tile{ct.TFloat32}, b_2d)
        end

        acc = muladd(a_2d, b_2d, acc)
        k += Int32(1)
    end

    result = convert(ct.Tile{T}, acc)
    result_3d = ct.reshape(result, (tm[], tn[], 1))
    ct.store(C, (bid_m, bid_n, pid_batch), result_3d)

    return nothing
end

function benchmark_batchmatmul()
    println("\nBenchmarking Batch Matrix Multiplication...")
    Batch, M, K, N = BATCHMATMUL_BATCH, BATCHMATMUL_M, BATCHMATMUL_K, BATCHMATMUL_N
    println("  Size: ($M x $K x $Batch) @ ($K x $N x $Batch), Float16")

    # Batch-last ordering for optimal column-major access
    A = CUDA.rand(Float16, M, K, Batch)
    B = CUDA.rand(Float16, K, N, Batch)
    C = CUDA.zeros(Float16, M, N, Batch)

    # Reference result (batched matmul on CPU)
    A_cpu = Float32.(Array(A))
    B_cpu = Float32.(Array(B))
    C_ref = zeros(Float32, M, N, Batch)
    for b in 1:Batch
        C_ref[:, :, b] = A_cpu[:, :, b] * B_cpu[:, :, b]
    end

    results = BenchmarkResult[]
    flops = 2.0 * Batch * M * N * K

    # cuBLAS batched gemm (via loop)
    fill!(C, 0)
    cublas_f = () -> begin
        for b in 1:Batch
            mul!(view(C, :, :, b), view(A, :, :, b), view(B, :, :, b))
        end
    end
    cublas_f()
    CUDA.synchronize()
    @assert isapprox(Float32.(Array(C)), C_ref, rtol=1e-1, atol=1e-1) "cuBLAS incorrect!"
    min_t, mean_t = benchmark_kernel(cublas_f)
    push!(results, BenchmarkResult("cuBLAS (loop)", min_t, mean_t))

    # cuTile
    fill!(C, 0)
    grid = (cld(M, BATCHMATMUL_TM), cld(N, BATCHMATMUL_TN), Batch)
    cutile_f = () -> ct.launch(batchmatmul_cutile_kernel, grid, A, B, C,
                               ct.Constant(BATCHMATMUL_TM), ct.Constant(BATCHMATMUL_TN),
                               ct.Constant(BATCHMATMUL_TK))
    cutile_f()
    CUDA.synchronize()
    @assert isapprox(Float32.(Array(C)), C_ref, rtol=1e-1, atol=1e-1) "cuTile incorrect!"
    min_t, mean_t = benchmark_kernel(cutile_f)
    push!(results, BenchmarkResult("cuTile.jl", min_t, mean_t))

    # Calculate TFLOPS
    tflops_vals = [string(round(flops / (r.min_ms * 1e-3) / 1e12, digits=2), " TFLOPS") for r in results]

    print_table("Batch Matrix Multiplication (Float16)", results; extra_col=("Performance", tflops_vals))
    return results
end

#=============================================================================
 FFT (3-stage Cooley-Tukey) - Column-Major Version
=============================================================================#

# FFT kernel - 3-stage Cooley-Tukey decomposition (column-major)
# Uses swapped dimensions and right-multiply for column-major compatibility.
# Input/output layout: (D, BS, N2D) where D=2 for real/imag interleaving.
function fft_kernel(
    x_packed_in::ct.TileArray{Float32, 3},
    y_packed_out::ct.TileArray{Float32, 3},
    W0::ct.TileArray{Float32, 3},
    W1::ct.TileArray{Float32, 3},
    W2::ct.TileArray{Float32, 3},
    T0::ct.TileArray{Float32, 3},
    T1::ct.TileArray{Float32, 3},
    n_const::ct.Constant{Int},
    f0_const::ct.Constant{Int},
    f1_const::ct.Constant{Int},
    f2_const::ct.Constant{Int},
    f0f1_const::ct.Constant{Int},
    f1f2_const::ct.Constant{Int},
    f0f2_const::ct.Constant{Int},
    bs_const::ct.Constant{Int},
    d_const::ct.Constant{Int},
    n2d_const::ct.Constant{Int}
)
    N = n_const[]
    F0 = f0_const[]
    F1 = f1_const[]
    F2 = f2_const[]
    F0F1 = f0f1_const[]
    F1F2 = f1f2_const[]
    F0F2 = f0f2_const[]
    BS = bs_const[]
    D = d_const[]
    N2D = n2d_const[]

    bid = ct.bid(1)

    # Load input (D, BS, N2D) and reshape to (2, BS, N)
    X_ri = ct.reshape(ct.load(x_packed_in, (1, bid, 1), (D, BS, N2D)), (2, BS, N))
    X_r = ct.reshape(ct.extract(X_ri, (1, 1, 1), (1, BS, N)), (BS, F1F2, F0))
    X_i = ct.reshape(ct.extract(X_ri, (2, 1, 1), (1, BS, N)), (BS, F1F2, F0))

    # Load DFT matrices
    W0_ri = ct.reshape(ct.load(W0, (1, 1, 1), (F0, F0, 2)), (F0, F0, 2))
    W0_r = ct.broadcast_to(ct.reshape(ct.extract(W0_ri, (1, 1, 1), (F0, F0, 1)), (1, F0, F0)), (BS, F0, F0))
    W0_i = ct.broadcast_to(ct.reshape(ct.extract(W0_ri, (1, 1, 2), (F0, F0, 1)), (1, F0, F0)), (BS, F0, F0))

    W1_ri = ct.reshape(ct.load(W1, (1, 1, 1), (F1, F1, 2)), (F1, F1, 2))
    W1_r = ct.broadcast_to(ct.reshape(ct.extract(W1_ri, (1, 1, 1), (F1, F1, 1)), (1, F1, F1)), (BS, F1, F1))
    W1_i = ct.broadcast_to(ct.reshape(ct.extract(W1_ri, (1, 1, 2), (F1, F1, 1)), (1, F1, F1)), (BS, F1, F1))

    W2_ri = ct.reshape(ct.load(W2, (1, 1, 1), (F2, F2, 2)), (F2, F2, 2))
    W2_r = ct.broadcast_to(ct.reshape(ct.extract(W2_ri, (1, 1, 1), (F2, F2, 1)), (1, F2, F2)), (BS, F2, F2))
    W2_i = ct.broadcast_to(ct.reshape(ct.extract(W2_ri, (1, 1, 2), (F2, F2, 1)), (1, F2, F2)), (BS, F2, F2))

    # Load twiddle factors (column-major layout)
    T0_ri = ct.reshape(ct.load(T0, (1, 1, 1), (F1F2, F0, 2)), (F1F2, F0, 2))
    T0_r = ct.reshape(ct.extract(T0_ri, (1, 1, 1), (F1F2, F0, 1)), (1, N))
    T0_i = ct.reshape(ct.extract(T0_ri, (1, 1, 2), (F1F2, F0, 1)), (1, N))

    T1_ri = ct.reshape(ct.load(T1, (1, 1, 1), (F0F2, F1, 2)), (F0F2, F1, 2))
    T1_r = ct.reshape(ct.extract(T1_ri, (1, 1, 1), (F0F2, F1, 1)), (1, F0F2 * F1))
    T1_i = ct.reshape(ct.extract(T1_ri, (1, 1, 2), (F0F2, F1, 1)), (1, F0F2 * F1))

    # Stage 0: F0-point DFT via right-multiply
    X_r_ = X_r * W0_r - X_i * W0_i
    X_i_ = X_r * W0_i + X_i * W0_r

    # Twiddle & Permute 0
    X_r_flat = ct.reshape(X_r_, (BS, N))
    X_i_flat = ct.reshape(X_i_, (BS, N))
    X_r2 = T0_r .* X_r_flat .- T0_i .* X_i_flat
    X_i2 = T0_i .* X_r_flat .+ T0_r .* X_i_flat

    X_r3 = ct.reshape(X_r2, (BS, F2, F1, F0))
    X_i3 = ct.reshape(X_i2, (BS, F2, F1, F0))
    X_r4 = ct.permute(X_r3, (1, 2, 4, 3))
    X_i4 = ct.permute(X_i3, (1, 2, 4, 3))
    X_r5 = ct.reshape(X_r4, (BS, F0F2, F1))
    X_i5 = ct.reshape(X_i4, (BS, F0F2, F1))

    # Stage 1: F1-point DFT
    X_r6 = X_r5 * W1_r - X_i5 * W1_i
    X_i6 = X_r5 * W1_i + X_i5 * W1_r

    # Twiddle & Permute 1
    X_r_flat2 = ct.reshape(X_r6, (BS, N))
    X_i_flat2 = ct.reshape(X_i6, (BS, N))
    X_r7 = T1_r .* X_r_flat2 .- T1_i .* X_i_flat2
    X_i7 = T1_i .* X_r_flat2 .+ T1_r .* X_i_flat2

    X_r8 = ct.reshape(X_r7, (BS, F2, F0, F1))
    X_i8 = ct.reshape(X_i7, (BS, F2, F0, F1))
    X_r9 = ct.permute(X_r8, (1, 3, 4, 2))
    X_i9 = ct.permute(X_i8, (1, 3, 4, 2))
    X_r10 = ct.reshape(X_r9, (BS, F0F1, F2))
    X_i10 = ct.reshape(X_i9, (BS, F0F1, F2))

    # Stage 2: F2-point DFT
    X_r11 = X_r10 * W2_r - X_i10 * W2_i
    X_i11 = X_r10 * W2_i + X_i10 * W2_r

    # Final output
    X_r_final = ct.reshape(X_r11, (1, BS, N))
    X_i_final = ct.reshape(X_i11, (1, BS, N))

    # Concatenate and Store
    Y_ri = ct.reshape(ct.cat((X_r_final, X_i_final), 1), (D, BS, N2D))
    ct.store(y_packed_out, (1, bid, 1), Y_ri)

    return
end

# Helper: Generate DFT matrix
function fft_dft_matrix(size::Int)
    W = zeros(ComplexF32, size, size)
    for i in 0:size-1, j in 0:size-1
        W[i+1, j+1] = exp(-2π * im * i * j / size)
    end
    result = zeros(Float32, size, size, 2)
    result[:, :, 1] = Float32.(real.(W))
    result[:, :, 2] = Float32.(imag.(W))
    return result
end

# Twiddle factors T0 for column-major layout (F1F2, F0)
function fft_make_twiddles_T0(F0::Int, F1F2::Int, N::Int)
    T0 = zeros(Float32, F1F2, F0, 2)
    for j in 0:F1F2-1, i in 0:F0-1
        val = exp(-2π * im * i * j / N)
        T0[j+1, i+1, 1] = Float32(real(val))
        T0[j+1, i+1, 2] = Float32(imag(val))
    end
    return T0
end

# Twiddle factors T1 for column-major layout (F0F2, F1)
function fft_make_twiddles_T1(F0::Int, F1::Int, F2::Int)
    F0F2 = F0 * F2
    F1F2 = F1 * F2
    T1 = zeros(Float32, F0F2, F1, 2)
    for k in 0:F0F2-1, j in 0:F1-1
        f2 = k % F2
        val = exp(-2π * im * j * f2 / F1F2)
        T1[k+1, j+1, 1] = Float32(real(val))
        T1[k+1, j+1, 2] = Float32(imag(val))
    end
    return T1
end

function fft_make_twiddles(factors::NTuple{3, Int})
    F0, F1, F2 = factors
    N = F0 * F1 * F2
    F1F2 = F1 * F2
    W0 = fft_dft_matrix(F0)
    W1 = fft_dft_matrix(F1)
    W2 = fft_dft_matrix(F2)
    T0 = fft_make_twiddles_T0(F0, F1F2, N)
    T1 = fft_make_twiddles_T1(F0, F1, F2)
    return (W0, W1, W2, T0, T1)
end

function benchmark_fft()
    println("\nBenchmarking FFT...")
    BS, N = FFT_BATCH, FFT_SIZE
    F0, F1, F2 = FFT_FACTORS
    D = FFT_ATOM_PACKING_DIM
    println("  Size: $BS batches × $N FFT ($(BS * N * 8 / 1e6) MB)")

    # Create complex input
    CUDA.seed!(42)
    input = CUDA.randn(ComplexF32, BS, N)

    # Reference result (FFTW)
    reference = FFTW.fft(Array(input), 2)

    results = BenchmarkResult[]

    # Pre-compute twiddles (one-time CPU cost)
    W0, W1, W2, T0, T1 = fft_make_twiddles(FFT_FACTORS)
    W0_gpu, W1_gpu, W2_gpu = CuArray(W0), CuArray(W1), CuArray(W2)
    T0_gpu, T1_gpu = CuArray(T0), CuArray(T1)

    # Pre-pack input (zero-copy)
    N2D = N * 2 ÷ D
    x_packed = reinterpret(reshape, Float32, input)
    y_packed = CUDA.zeros(Float32, D, BS, N2D)

    # Kernel launch parameters
    F0F1, F1F2, F0F2 = F0 * F1, F1 * F2, F0 * F2
    grid = (BS, 1, 1)

    # Kernel-only timing function
    cutile_kernel_f = () -> ct.launch(fft_kernel, grid,
        x_packed, y_packed,
        W0_gpu, W1_gpu, W2_gpu, T0_gpu, T1_gpu,
        ct.Constant(N), ct.Constant(F0), ct.Constant(F1), ct.Constant(F2),
        ct.Constant(F0F1), ct.Constant(F1F2), ct.Constant(F0F2),
        ct.Constant(BS), ct.Constant(D), ct.Constant(N2D))

    # Verify correctness
    cutile_kernel_f()
    CUDA.synchronize()
    y_complex = reinterpret(reshape, ComplexF32, y_packed)
    output = copy(y_complex)
    @assert isapprox(Array(output), reference, rtol=1e-3) "cuTile FFT incorrect!"

    # Benchmark kernel only
    min_t, mean_t = benchmark_kernel(cutile_kernel_f)
    push!(results, BenchmarkResult("cuTile.jl", min_t, mean_t))

    # Performance metric: GFLOPS (5 * N * log2(N) per complex FFT)
    flops_per_fft = 5.0 * N * log2(N)
    total_flops = BS * flops_per_fft
    gflops = [string(round(total_flops / (r.min_ms * 1e-3) / 1e9, digits=1), " GFLOPS") for r in results]

    print_table("FFT (ComplexF32)", results; extra_col=("Performance", gflops))
    return results
end

#=============================================================================
 Main
=============================================================================#

function main()
    println("=" ^ 60)
    println("  cuTile.jl Comprehensive Benchmarks")
    println("=" ^ 60)
    println()
    println("Configuration:")
    println("  Runs: $NRUNS (+ $WARMUP warmup)")
    println("  GPU: ", CUDA.name(CUDA.device()))
    println()

    vadd_results = benchmark_vadd()
    transpose_results = benchmark_transpose()
    matmul_results = benchmark_matmul()
    layernorm_results = benchmark_layernorm()
    batchmatmul_results = benchmark_batchmatmul()
    fft_results = benchmark_fft()

    println()
    println("=" ^ 60)
    println("  Benchmark Complete")
    println("=" ^ 60)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
