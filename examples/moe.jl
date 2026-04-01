# Mixture of Experts example - Julia port of cuTile Python's MoE.py sample
#
# Expert weights use (K, N, num_experts) layout with K contiguous for efficient loads.
# Intermediate caches use (topk, num_tokens, dim) so that Python-style flattening
# (slot varies fastest) matches Julia's column-major vec().
#
# SPDX-License-Identifier: Apache-2.0

using CUDA
using Random: randperm
import cuTile as ct

#=============================================================================
 Helper: 2D swizzle (same pattern as matmul.jl)
=============================================================================#

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

#=============================================================================
 Kernels
=============================================================================#

# Fused MoE kernel: multiplies routed tokens by their assigned expert weights.
#
# A: (num_tokens, K) - input tokens/activations
# B: (K, N, num_experts) - expert weight matrices (K contiguous in Julia)
# C: (total_tokens, N) - output (flattened token-topk dimension)
# topk_weights: (total_tokens,) - flattened routing weights
# sorted_token_ids: (M_padded,) - 1-indexed token replica IDs, sorted by expert
# sorted_expert_ids: (num_blocks,) - 1-indexed expert ID per TILE_M block
function fused_moe_kernel(A::ct.TileArray{T, 2}, B::ct.TileArray{T, 3},
                          C::ct.TileArray{T, 2},
                          topk_weights::ct.TileArray{T, 1},
                          sorted_token_ids::ct.TileArray{Int32, 1},
                          sorted_expert_ids::ct.TileArray{Int32, 1},
                          num_token_replicas::Int, mul_routed_weight::Bool,
                          TILE_M::Int, TILE_N::Int, TILE_K::Int) where {T}
    ct.@compiler_options opt_level=0
    M = size(sorted_token_ids, 1)
    K = size(B, 1)
    N = size(B, 2)

    bid = ct.bid(1) - Int32(1)  # 0-indexed for swizzle
    bid_m, bid_n = swizzle_2d(M, N, TILE_M, TILE_N, Int32(8), bid)

    # Gather 1-indexed token IDs for this block
    # arange returns [1, ..., TILE_M], bid_m is 0-indexed → indices are 1-indexed
    token_id_indices = bid_m * Int32(TILE_M) .+ ct.arange(TILE_M)
    token_ids = ct.gather(sorted_token_ids, token_id_indices)

    # Map 1-indexed flat token_id to 1-indexed row in A
    # token_id k → original token = (k-1) ÷ num_token_replicas + 1
    a_row_indices = (token_ids .- Int32(1)) .÷ Int32(num_token_replicas) .+ Int32(1)

    # Expert for this block (scalar, 1-indexed tile index for load)
    expert_id = sorted_expert_ids[bid_m + Int32(1)]

    acc = zeros(Float32, TILE_M, TILE_N)
    num_k = cld(K, Int32(TILE_K))

    k = Int32(1)
    while k <= num_k
        # 1-indexed column indices into A's K dimension
        a_col_indices = (k - Int32(1)) * Int32(TILE_K) .+ ct.arange(TILE_K)

        a = ct.gather(A, (reshape(a_row_indices, (TILE_M, 1)),
                          reshape(a_col_indices, (1, TILE_K))))  # (TILE_M, TILE_K)

        # B is (K, N, num_experts): load (TILE_K, TILE_N) slice for this expert
        b = ct.load(B; index=(k, bid_n + Int32(1), expert_id),
                    shape=(TILE_K, TILE_N, 1),
                    padding_mode=ct.PaddingMode.Zero)
        b = reshape(b, (TILE_K, TILE_N))

        acc = muladd(a, b, acc)
        k += Int32(1)
    end

    if mul_routed_weight
        moe_weight = convert(ct.Tile{Float32}, ct.gather(topk_weights, token_ids))
        acc = acc .* reshape(moe_weight, (TILE_M, 1))
    end

    # Scatter result to C at token_id positions
    c_col_indices = bid_n * Int32(TILE_N) .+ ct.arange(TILE_N)  # 1-indexed
    output = convert(ct.Tile{T}, acc)
    ct.scatter(C, (reshape(token_ids, (TILE_M, 1)),
                   reshape(c_col_indices, (1, TILE_N))), output)

    return nothing
end


# Element-wise SiLU activation: computes SiLU(A) * B
# Each block processes one row.
function silu_and_mul_kernel(A::ct.TileArray{T, 2}, B::ct.TileArray{T, 2},
                             C::ct.TileArray{T, 2}, TILE_N::Int) where {T}
    bid_m = ct.bid(1)
    ta = convert(ct.Tile{Float32}, ct.load(A; index=(bid_m, Int32(1)), shape=(1, TILE_N)))
    tb = convert(ct.Tile{Float32}, ct.load(B; index=(bid_m, Int32(1)), shape=(1, TILE_N)))

    # SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
    denom = 1.0f0 .+ exp.(.-ta)
    sigmoid_ta = 1.0f0 ./ denom
    silu_ta = ta .* sigmoid_ta
    tc = silu_ta .* tb

    ct.store(C; index=(bid_m, Int32(1)), tile=convert(ct.Tile{T}, tc))
    return nothing
end


#=============================================================================
 Host-side helpers
=============================================================================#

"""
Sort, replicate, and pad token indices by expert so every expert processes
a TILE_M-aligned number of tokens.

`topk_ids` is (num_tokens, topk) with 1-indexed expert IDs.
Returns GPU arrays with 1-indexed values.
"""
function moe_align_tile_size(topk_ids::Matrix{Int}, tile_m::Int, num_experts::Int)
    num_tokens, topk = size(topk_ids)
    total_tokens = num_tokens * topk

    # Flatten in Python-style order (slot varies fastest within each token).
    # permutedims gives (topk, num_tokens); vec in column-major = slot-first ordering.
    # This matches the intermediate cache layout: (topk, num_tokens, dim).
    flat_expert_ids = vec(permutedims(topk_ids))

    sorted_perm = sortperm(flat_expert_ids; stable=true)  # 1-indexed

    expert_token_counts = zeros(Int, num_experts)
    for eid in flat_expert_ids
        expert_token_counts[eid] += 1
    end
    expert_block_counts = cld.(expert_token_counts, tile_m)
    total_blocks = sum(expert_block_counts)

    # Sentinel value: total_tokens + 1 (out-of-bounds for 1-indexed arrays)
    sorted_token_ids = fill(Int32(total_tokens + 1), total_blocks * tile_m)
    sorted_expert_ids = zeros(Int32, total_blocks)

    current_block = 0
    current_token = 0
    for eid in 1:num_experts
        tc = expert_token_counts[eid]
        bc = expert_block_counts[eid]

        for i in 1:bc
            sorted_expert_ids[current_block + i] = Int32(eid)
        end

        start = current_block * tile_m + 1
        for i in 1:tc
            sorted_token_ids[start + i - 1] = Int32(sorted_perm[current_token + i])
        end

        current_token += tc
        current_block += bc
    end

    return CuArray(sorted_token_ids), CuArray(sorted_expert_ids)
end


function invoke_fused_moe_kernel(A, B, C, topk_weights, sorted_token_ids, sorted_expert_ids;
                                 mul_routed_weight, num_token_replicas, tile_m, tile_n, tile_k)
    m = length(sorted_token_ids)
    n = size(B, 2)
    grid = cld(m, tile_m) * cld(n, tile_n)

    # Flatten in Python-style order (slot varies fastest) to match token_id ordering
    topk_weights_flat = vec(permutedims(topk_weights))
    C_flat = reshape(C, size(C, 1) * size(C, 2), size(C, 3))

    ct.launch(fused_moe_kernel, grid,
              A, B, C_flat, topk_weights_flat,
              sorted_token_ids, sorted_expert_ids,
              ct.Constant(num_token_replicas), ct.Constant(mul_routed_weight),
              ct.Constant(tile_m), ct.Constant(tile_n), ct.Constant(tile_k))
end


function invoke_silu_and_mul_kernel(AB, C)
    inter = size(C, 2)
    A_half = AB[:, 1:inter]
    B_half = AB[:, inter+1:2*inter]
    tile_n = nextpow(2, inter)
    ct.launch(silu_and_mul_kernel, size(AB, 1),
              A_half, B_half, C, ct.Constant(tile_n))
end


function cutile_moe(hidden_states::CuArray{T}, w1, w2, topk_weights, topk_ids,
                    tile_m, tile_n, tile_k) where {T}
    num_tokens, hidden_size = size(hidden_states)
    intermediate_size = size(w2, 1)
    num_experts = size(w1, 3)
    _, topk = size(topk_ids)
    total_tokens = num_tokens * topk

    # Intermediate caches: (topk, num_tokens, dim) so column-major vec() matches
    # the Python-style flat token_id ordering used by the kernels.
    cache1 = CUDA.zeros(T, topk, num_tokens, intermediate_size * 2)
    cache2 = CUDA.zeros(T, total_tokens, intermediate_size)
    cache3 = CUDA.zeros(T, topk, num_tokens, hidden_size)

    sorted_token_ids, sorted_expert_ids = moe_align_tile_size(
        Array(topk_ids), tile_m, num_experts)

    # First matmul: hidden_states @ w1^T → gate+up projection
    invoke_fused_moe_kernel(
        hidden_states, w1, cache1,
        topk_weights, sorted_token_ids, sorted_expert_ids;
        mul_routed_weight=false, num_token_replicas=topk,
        tile_m, tile_n, tile_k)

    # SiLU activation: SiLU(gate) * up
    cache1_flat = reshape(cache1, total_tokens, intermediate_size * 2)
    invoke_silu_and_mul_kernel(cache1_flat, cache2)

    # Second matmul: activated @ w2^T → down projection, with routing weights
    invoke_fused_moe_kernel(
        cache2, w2, cache3,
        topk_weights, sorted_token_ids, sorted_expert_ids;
        mul_routed_weight=true, num_token_replicas=1,
        tile_m, tile_n, tile_k)

    # Sum over topk slots (first dimension) for each token
    return dropdims(sum(cache3; dims=1); dims=1)
end


#=============================================================================
 Reference implementation
=============================================================================#

function ref_moe(hidden_states, w1, w2, topk_weights, topk_ids)
    hs = Float32.(Array(hidden_states))     # (num_tokens, hidden_size)
    w1_cpu = Float32.(Array(w1))            # (hidden_size, intermediate_size*2, num_experts)
    w2_cpu = Float32.(Array(w2))            # (intermediate_size, hidden_size, num_experts)
    tw_cpu = Float32.(Array(topk_weights))  # (num_tokens, topk)
    ti_cpu = Array(topk_ids)                # (num_tokens, topk)

    num_tokens, hidden_size = size(hs)
    intermediate_size = size(w2_cpu, 1)
    num_experts = size(w1_cpu, 3)

    result = zeros(Float32, num_tokens, hidden_size)

    for eid in 1:num_experts
        positions = findall(==(eid), ti_cpu)
        isempty(positions) && continue

        token_indices = [p[1] for p in positions]
        slot_indices = [p[2] for p in positions]

        tokens = hs[token_indices, :]  # (count, hidden_size)

        w1e = w1_cpu[:, :, eid]  # (hidden_size, intermediate_size*2)
        gate_proj = w1e[:, 1:intermediate_size]
        up_proj = w1e[:, intermediate_size+1:end]

        gate_out = tokens * gate_proj   # (count, intermediate_size)
        up_out = tokens * up_proj
        silu_out = gate_out ./ (1.0f0 .+ exp.(.-gate_out)) .* up_out

        down_proj = w2_cpu[:, :, eid]   # (intermediate_size, hidden_size)
        expert_out = silu_out * down_proj  # (count, hidden_size)

        for (i, (tok, slot)) in enumerate(zip(token_indices, slot_indices))
            result[tok, :] .+= expert_out[i, :] .* tw_cpu[tok, slot]
        end
    end

    return result
end


#=============================================================================
 Example harness
=============================================================================#

function prepare(; benchmark::Bool=false,
                  num_tokens::Int = benchmark ? 256 : 48,
                  hidden_size::Int = benchmark ? 1024 : 512,
                  num_experts::Int = benchmark ? 32 : 64,
                  intermediate_size::Int = benchmark ? 2048 : 1024,
                  topk::Int = 8,
                  T::DataType = Float16)
    # Generate on GPU (Float32) then convert to avoid OOM from large Float64 intermediates
    hidden_states = T.(CUDA.rand(num_tokens, hidden_size) .- 0.5f0)
    w1 = T.((CUDA.rand(hidden_size, intermediate_size * 2, num_experts) .- 0.5f0) .* 0.2f0)
    w2 = T.((CUDA.rand(intermediate_size, hidden_size, num_experts) .- 0.5f0) .* 0.2f0)

    # Unique expert IDs per token (1-indexed)
    topk_ids = Matrix{Int}(undef, num_tokens, topk)
    for i in 1:num_tokens
        topk_ids[i, :] = randperm(num_experts)[1:topk]
    end

    # Softmax routing weights
    raw = rand(Float32, num_tokens, topk)
    raw_exp = exp.(raw)
    topk_weights = T.(raw_exp ./ sum(raw_exp; dims=2))

    return (;
        hidden_states,
        w1, w2,
        topk_weights = CuArray(topk_weights),
        topk_ids = CuArray(topk_ids),
        num_tokens, hidden_size, num_experts, intermediate_size, topk,
        tile_m = 128, tile_n = 128, tile_k = 64
    )
end


function run(data; nruns::Int=1, warmup::Int=0)
    (; hidden_states, w1, w2, topk_weights, topk_ids, tile_m, tile_n, tile_k) = data

    CUDA.@sync for _ in 1:warmup
        cutile_moe(hidden_states, w1, w2, topk_weights, topk_ids, tile_m, tile_n, tile_k)
    end

    times = Float64[]
    out = nothing
    for _ in 1:nruns
        t = CUDA.@elapsed begin
            out = cutile_moe(hidden_states, w1, w2, topk_weights, topk_ids,
                             tile_m, tile_n, tile_k)
        end
        push!(times, t * 1000)  # ms
    end

    return (; out, times)
end


function verify(data, result)
    expected = ref_moe(data.hidden_states, data.w1, data.w2, data.topk_weights, data.topk_ids)
    actual = Float32.(Array(result.out))
    @assert isapprox(actual, expected; rtol=1e-2, atol=1e-2) "MoE incorrect! max diff: $(maximum(abs.(actual .- expected)))"
end


function metric(data)
    # Two matmuls per token: gate+up (2*intermediate) and down (hidden)
    # matmul1: tokens*topk * hidden * intermediate*2 * 2 (multiply-add)
    # matmul2: tokens*topk * intermediate * hidden * 2
    # Total: 2 * tokens * topk * hidden * intermediate * 3
    nt = data.num_tokens
    topk = data.topk
    hs = data.hidden_size
    inter = data.intermediate_size
    return 2 * nt * topk * hs * inter * 3, "TFLOPS"
end


#=============================================================================
 Main
=============================================================================#

function test_moe(num_tokens, hidden_size, num_experts, intermediate_size, topk;
                  tile_m=128, tile_n=128, tile_k=64, T=Float16, name=nothing)
    name = something(name, "moe tokens=$num_tokens, hidden=$hidden_size, experts=$num_experts, " *
                           "inter=$intermediate_size, topk=$topk, $T")
    println("--- $name ---")
    data = prepare(; num_tokens, hidden_size, num_experts, intermediate_size, topk, T)
    result = run(data)
    verify(data, result)
    println("  passed")
end


function main()
    println("--- cuTile Mixture of Experts Examples ---\n")

    test_moe(48, 512, 64, 1024, 8)
    test_moe(128, 512, 32, 1024, 4)
    test_moe(64, 1024, 64, 2048, 8)

    println("\n--- All MoE examples completed ---")
end

isinteractive() || main()
