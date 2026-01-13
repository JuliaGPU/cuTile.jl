# EXCLUDE FROM TESTING
#
# Comprehensive benchmarks for cuTile.jl
# Compares: GPUArrays (generic), SIMT (CUDA.jl), cuTile
# Kernels: vadd, transpose, matmul

# Include example files to reuse their kernels
include("vadd.jl")
include("transpose.jl")
include("matmul.jl")
include("batchmatmul.jl")
include("layernorm.jl")
include("fft.jl")

using LinearAlgebra
using CUDA: GPUArrays

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

# SIMT kernel (benchmark-specific)
function vadd_simt_kernel!(a, b, c)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= length(c)
        @inbounds c[i] = a[i] + b[i]
    end
    return
end

# cuTile kernel: use vec_add_kernel_1d from vadd.jl

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

    # cuTile (uses vec_add_kernel_1d from vadd.jl)
    grid = (cld(VADD_SIZE, VADD_TILE), 1, 1)
    cutile_f = () -> ct.launch(vec_add_kernel_1d, grid, a, b, c, ct.Constant(VADD_TILE))
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

# SIMT shared memory kernel (benchmark-specific)
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

# cuTile kernel: use transpose_kernel from transpose.jl

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

    # cuTile (uses transpose_kernel from transpose.jl)
    fill!(output, 0)
    grid = (cld(M, TRANSPOSE_TILE_M), cld(N, TRANSPOSE_TILE_N), 1)
    cutile_f = () -> ct.launch(transpose_kernel, grid, input, output,
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

# cuTile kernel: use matmul_kernel and swizzle_2d from matmul.jl

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

    # cuTile (uses matmul_kernel from matmul.jl)
    fill!(C, 0)
    grid_m = cld(M, MATMUL_TM)
    grid_n = cld(N, MATMUL_TN)
    grid = (grid_m * grid_n, 1, 1)
    cutile_f = () -> ct.launch(matmul_kernel, grid, A, B, C,
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

# SIMT naive kernel (benchmark-specific, 2-pass: compute mean/var, then normalize)
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

# cuTile kernel: use layer_norm_fwd from layernorm.jl

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

    # cuTile (uses layer_norm_fwd from layernorm.jl)
    fill!(Y, 0); fill!(Mean, 0); fill!(Rstd, 0)
    cutile_f = () -> ct.launch(layer_norm_fwd, M, X, W, B, Y, Mean, Rstd,
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

# cuTile kernel: use batch_matmul_kernel from batchmatmul.jl

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

    # cuTile (uses batch_matmul_kernel from batchmatmul.jl)
    fill!(C, 0)
    grid = (cld(M, BATCHMATMUL_TM), cld(N, BATCHMATMUL_TN), Batch)
    cutile_f = () -> ct.launch(batch_matmul_kernel, grid, A, B, C,
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

# cuTile kernel: use fft_kernel and make_twiddles from fft.jl

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

    # Pre-compute twiddles (one-time CPU cost, uses make_twiddles from fft.jl)
    W0, W1, W2, T0, T1 = make_twiddles(FFT_FACTORS)
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
