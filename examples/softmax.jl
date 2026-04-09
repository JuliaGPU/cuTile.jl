# Softmax example - Julia port of cuTile Python TileGym softmax
#
# Column-major layout: input is (N, M) where N (softmax dimension) is contiguous.
# Two strategies:
# 1. TMA: loads entire column in one tile (small-to-medium N, TILE_SIZE >= N)
# 2. Chunked: 3-pass with gather/scatter (large N, arbitrary TILE_SIZE)
#
# SPDX-License-Identifier: Apache-2.0

using CUDA
import cuTile as ct

#=============================================================================
 TMA Softmax Kernel (single-tile per column, persistent scheduling)
=============================================================================#

function softmax_tma_kernel(output::ct.TileArray{T,2}, input::ct.TileArray{T,2},
                            TILE_SIZE::Int) where {T}
    ct.@compiler_options occupancy=2

    pid = ct.bid(1)
    num_programs = ct.num_blocks(1)
    M = size(input, 2)

    col_idx = pid
    while col_idx <= M
        col = ct.load(input; index=(Int32(1), col_idx), shape=(TILE_SIZE, 1),
                      padding_mode=ct.PaddingMode.NegInf)
        col = convert(ct.Tile{Float32}, col)

        col_max = maximum(col; dims=1)
        numerator = exp.(col .- col_max)
        denominator = sum(numerator; dims=1)
        softmax_output = numerator ./ denominator

        ct.store(output; index=(Int32(1), col_idx),
                 tile=convert(ct.Tile{T}, softmax_output))
        col_idx += num_programs
    end
    return
end

#=============================================================================
 Chunked Softmax Kernel (3-pass with gather/scatter, persistent scheduling)
=============================================================================#

function softmax_chunked_kernel(output::ct.TileArray{T,2}, input::ct.TileArray{T,2},
                                n_elems::Int, TILE_SIZE::Int) where {T}
    ct.@compiler_options occupancy=4

    pid = ct.bid(1)
    num_programs = ct.num_blocks(1)
    M = size(input, 2)
    num_chunks = (n_elems + TILE_SIZE - Int32(1)) ÷ Int32(TILE_SIZE)
    row_offsets_base = ct.arange(TILE_SIZE)

    col_idx = pid
    while col_idx <= M
        col_tile = ct.Tile(col_idx)
        row_max = fill(-Inf32, (1,))
        denominator = zeros(Float32, TILE_SIZE)

        # Pass 1: Find maximum across all chunks
        for chunk_idx in Int32(0):num_chunks - Int32(1)
            row_indices = ct.broadcast_to(ct.Tile(chunk_idx * Int32(TILE_SIZE)), (TILE_SIZE,)) .+ row_offsets_base
            chunk = ct.gather(input, (row_indices, col_tile);
                             check_bounds=true, padding_value=T(-Inf))
            chunk = convert(ct.Tile{Float32}, chunk)
            chunk_max = maximum(chunk)
            row_max = max.(row_max, ct.Tile(chunk_max))
        end

        # Pass 2: Compute denominator (sum of all exp values)
        for chunk_idx in Int32(0):num_chunks - Int32(1)
            row_indices = ct.broadcast_to(ct.Tile(chunk_idx * Int32(TILE_SIZE)), (TILE_SIZE,)) .+ row_offsets_base
            chunk = ct.gather(input, (row_indices, col_tile);
                             check_bounds=true, padding_value=T(-Inf))
            chunk = convert(ct.Tile{Float32}, chunk)
            denominator = denominator .+ exp.(chunk .- row_max)
        end
        denom_sum = ct.Tile(sum(denominator))

        # Pass 3: Compute final softmax and scatter
        for chunk_idx in Int32(0):num_chunks - Int32(1)
            row_indices = ct.broadcast_to(ct.Tile(chunk_idx * Int32(TILE_SIZE)), (TILE_SIZE,)) .+ row_offsets_base
            chunk = ct.gather(input, (row_indices, col_tile);
                             check_bounds=true, padding_value=T(-Inf))
            chunk = convert(ct.Tile{Float32}, chunk)
            softmax_output = exp.(chunk .- row_max) ./ denom_sum
            ct.scatter(output, (row_indices, col_tile), convert(ct.Tile{T}, softmax_output);
                      check_bounds=true)
        end

        col_idx += num_programs
    end
    return
end


#=============================================================================
 Example harness
=============================================================================#

function next_power_of_2(n::Int)
    n <= 0 && return 1
    p = 1
    while p < n
        p <<= 1
    end
    return p
end

function prepare(; benchmark::Bool=false,
                  M::Int=benchmark ? 4096 : 256,
                  N::Int=benchmark ? 4096 : 256,
                  T::DataType=Float32)
    # (N, M) layout: softmax dimension N is contiguous in column-major
    input = CUDA.randn(T, N, M)
    return (;
        input,
        output_tma = similar(input),
        output_chunked = similar(input),
        M, N
    )
end

function run(data; tile_tma::Int=next_power_of_2(data.N),
                   tile_chunked::Int=1024,
                   nruns::Int=1, warmup::Int=0)
    (; input, output_tma, output_chunked, M, N) = data

    function run_tma()
        ct.launch(softmax_tma_kernel, M, output_tma, input, ct.Constant(tile_tma))
    end

    function run_chunked()
        ct.launch(softmax_chunked_kernel, M, output_chunked, input,
                  ct.Constant(N), ct.Constant(tile_chunked))
    end

    # Warmup
    CUDA.@sync for _ in 1:warmup
        run_tma()
        run_chunked()
    end

    # Timed TMA runs
    times_tma = Float64[]
    for _ in 1:nruns
        t = CUDA.@elapsed run_tma()
        push!(times_tma, t * 1000)
    end

    # Timed chunked runs
    times_chunked = Float64[]
    for _ in 1:nruns
        t = CUDA.@elapsed run_chunked()
        push!(times_chunked, t * 1000)
    end

    return (; output_tma, output_chunked,
             times=Dict("cuTile TMA" => times_tma, "cuTile Chunked" => times_chunked))
end

function verify(data, result)
    M, N = data.M, data.N
    x = Array(data.input)  # (N, M)
    for label in (:output_tma, :output_chunked)
        out = Array(getproperty(result, label))
        for j in 1:M
            col = x[:, j]
            col_max = maximum(col)
            exps = exp.(col .- col_max)
            expected = exps ./ sum(exps)
            @assert isapprox(out[:, j], expected; atol=1e-5, rtol=1e-4) "$label column $j mismatch"
        end
    end
end

function metric(data)
    MN = data.M * data.N * sizeof(Float32)
    return Dict(
        # TMA: 1 read + 1 write
        "cuTile TMA" => (2 * MN, "GB/s"),
        # Chunked: 3 reads (gather per pass) + 1 write (scatter)
        "cuTile Chunked" => (4 * MN, "GB/s"),
    )
end


#=============================================================================
 Reference implementations for benchmarking
=============================================================================#

function run_others(data; nruns::Int=1, warmup::Int=0)
    (; input) = data
    results = Dict{String, Vector{Float64}}()

    # GPUArrays softmax via broadcasting
    out = similar(input)
    function gpu_softmax!()
        col_max = maximum(input; dims=1)
        exps = exp.(input .- col_max)
        out .= exps ./ sum(exps; dims=1)
    end

    CUDA.@sync for _ in 1:warmup
        gpu_softmax!()
    end
    times = Float64[]
    for _ in 1:nruns
        t = CUDA.@elapsed gpu_softmax!()
        push!(times, t * 1000)
    end
    results["GPUArrays"] = times

    return results
end


#=============================================================================
 Main
=============================================================================#

function test_softmax(M, N; tile_tma::Int=next_power_of_2(N), tile_chunked::Int=1024, name=nothing)
    name = something(name, "softmax ($M x $N), tma_tile=$tile_tma, chunked_tile=$tile_chunked")
    println("--- $name ---")
    data = prepare(; M, N)
    result = run(data; tile_tma, tile_chunked)
    verify(data, result)
    println("  tma passed, chunked passed")
end

function main()
    println("--- cuTile Softmax Examples ---\n")

    test_softmax(256, 256)
    test_softmax(1024, 1024)
    test_softmax(4096, 4096)

    println("\n--- All softmax examples completed ---")
end

isinteractive() || main()
