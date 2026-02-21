# Fused Multi-Head Attention example - Julia port of cuTile Python's AttentionFMHA.py sample
#
# SPDX-License-Identifier: Apache-2.0

using CUDA
import cuTile as ct

import NNlib
import CUDA.GPUArrays: AllocCache, @cached # more fair NNlib comparison

const INV_LOG_2 = 1 / log(2f0)

# cuTile kernel for Fused Multi-Head Attention (FMHA)
#
# Computes attention output for a psecific batch item and head,
# using tiling and online softmax.
#
# Layout: (D, SeqLen, Heads, Batch)
function fmha_kernel(
    Q::ct.TileArray{T,4},
    K::ct.TileArray{T,4},
    V::ct.TileArray{T,4},
    Out::ct.TileArray{Tout,4},
    qk_scale::Float32,
    input_pos::Int, # 32?
    D_K::Int,   # Head dimension of Q and K
    D_V::Int,   # Head dimension of V
    H::Int,
    TILE_M::Int,
    TILE_N::Int,
    QUERY_GROUP_SIZE::Int,
    CAUSAL::Bool,
    EVEN_K::Bool
) where {T,Tout}
    # Map block IDs to batch and head indices
    bid_x = ct.bid(1)
    bid_y = ct.bid(2)
    batch_idx, head_idx = fldmod1(bid_y, H) # floored division and modulus for 1-based indexing
    off_kv_h = cld(head_idx, QUERY_GROUP_SIZE)

    # Adjust qk_scale for exp2
    qk_scale = qk_scale * INV_LOG_2

    # Initialize offsets for current query tile (M-dimension)
    # bid_x is 1-indexed, so first tile (bid_x=1) has offsets [0, TILE_M-1]
    offs_m = (bid_x - 1) * TILE_M .+ (ct.arange((TILE_M,), Int32) .- 1)
    offs_m = offs_m .+ input_pos
    offs_m = reshape(offs_m, (1, TILE_M))

    # local offsets for key/value tile (N-dimension)
    offs_n_tile = ct.arange((TILE_N,), Int32) .- 1
    offs_n_tile = reshape(offs_n_tile, (TILE_N, 1))

    # online softmax accumulators in Float32 for stability
    m_i = ct.full((1, TILE_M), -Inf32, Float32)
    l_i = ct.zeros((1, TILE_M), Float32)
    acc = ct.zeros((D_V, TILE_M), Float32)

    # query tile for this batch, head, and M-chunk
    q = ct.load(Q, (1, bid_x, head_idx, batch_idx), (D_K, TILE_M, 1, 1))
    q = reshape(q, (D_K, TILE_M))

    # m_end: one past the last query position in this tile
    m_end = input_pos + bid_x * TILE_M
    k_seqlen = K.sizes[2]
    if CAUSAL
        # Python: mask_start = (input_pos + bid_x * TILE_M) // TILE_N
        # In Julia with 1-indexed bid_x: mask_start = (input_pos + (bid_x-1) * TILE_M) // TILE_N + 1
        mask_start = fld(input_pos + (bid_x - 1) * TILE_M, TILE_N) + 1
        # Python: mask_start = min(mask_start, k_seqlen // TILE_N)
        mask_start = min(mask_start, fld(k_seqlen, TILE_N) + 1)
        Tc = cld(min(m_end, k_seqlen), TILE_N)
    else
        Tc = cld(k_seqlen, TILE_N)
        # Python: mask_start = k_seqlen // TILE_N
        mask_start = fld(k_seqlen, TILE_N) + 1
    end

    # loop over K, V blocks (N-dimension chunks)
    j = Int32(1)
    while j <= Tc
        k = ct.load(
            K, (1, j, off_kv_h, batch_idx), (D_K, TILE_N, 1, 1),
            latency=2)
        k = reshape(k, (D_K, TILE_N))
        k = transpose(k)

        qk = ct.zeros((TILE_N, TILE_M), Float32)
        qk = ct.muladd(k, q, qk)

        # Apply masking (matches Python: if (CAUSAL or not EVEN_K) and j >= mask_start)
        if (CAUSAL || !EVEN_K) && j >= mask_start
            offs_n = (j - 1) * TILE_N .+ offs_n_tile
            # Build mask: start with all true
            mask = ct.full((TILE_N, TILE_M), true, Bool)
            # out of bound mask (Python: if not EVEN_K: mask = mask & (offs_n < k_seqlen))
            if !EVEN_K
                mask = mask .& (offs_n .< k_seqlen)
            end
            # causal mask (Python: if CAUSAL: mask = mask & (offs_m >= offs_n))
            if CAUSAL
                mask = mask .& (offs_m .>= offs_n)
            end
            # Apply mask: set invalid positions to -Inf
            qk = ifelse.(mask, qk, -Inf32)
        end

        # Online Softmax Update
        # Moving qk_scale multiplication after reduce_max is to improve performance
        m_ij = max.(m_i, maximum(qk, dims=1) * qk_scale)
        qk = qk * qk_scale .- m_ij

        # attention weights [TILE_N, TILE_M]
        p = exp2.(qk)  # XXX: flush_to_zero=True
        l_ij = sum(p, dims=1)
        alpha = exp2.(m_i .- m_ij)  # XXX: flush_to_zero=True

        l_i = l_i .* alpha .+ l_ij
        acc = acc .* alpha

        v = ct.load(
            V, (1, j, off_kv_h, batch_idx), (D_V, TILE_N, 1, 1),
            latency=4)
        v = reshape(v, (D_V, TILE_N))
        acc = ct.muladd(v, T.(p), acc)
        m_i = m_ij

        j += Int32(1)
    end

    acc = acc ./ l_i  # XXX: flush_to_zero=True, rounding_mode=APPROX
    acc = reshape(acc, (D_V, TILE_M, 1, 1))
    ct.store(Out, (1, bid_x, head_idx, batch_idx), Tout.(acc))

    return
end

function prepare(; benchmark::Bool=false,
                  D_k::Int=64,
                  SeqLen_Q::Int=benchmark ? 4096 : 256,
                  Heads::Int=4,
                  Batch::Int=4,
                  D_v::Int=D_k,
                  SeqLen_KV::Int=SeqLen_Q,
                  Heads_KV::Int=Heads,
                  causal::Bool=false,
                  T::DataType=Float32)
    return (;
        Q = CUDA.randn(T, D_k, SeqLen_Q, Heads, Batch),
        K = CUDA.randn(T, D_k, SeqLen_KV, Heads_KV, Batch),
        V = CUDA.randn(T, D_v, SeqLen_KV, Heads_KV, Batch),
        Out = CUDA.randn(T, D_v, SeqLen_Q, Heads, Batch),
        D_k, SeqLen_Q, Heads, Batch,
        D_v, SeqLen_KV, Heads_KV, causal
    )
end

function run(data; tm::Int=64, tn::Int=64, nruns::Int=1, warmup::Int=0)
    (; Q, K, V, Out, D_k, D_v, SeqLen_Q, Heads, Batch, SeqLen_KV, Heads_KV, causal) = data
    grid_x = cld(SeqLen_Q, tm)
    grid_y = Heads * Batch
    grid = (grid_x, grid_y)

    qk_scale = Float32(1 / sqrt(D_k))
    input_pos = 0

    query_group_size, remainder = divrem(Heads, Heads_KV)
    @assert remainder == 0

    even_k = (SeqLen_KV % tn) == 0

    CUDA.@sync for _ in 1:warmup
        ct.launch(fmha_kernel, grid, Q, K, V, Out,
                  qk_scale, input_pos,
                  ct.Constant(D_k), ct.Constant(D_v), ct.Constant(Heads),
                  ct.Constant(tm), ct.Constant(tn),
                  ct.Constant(query_group_size),
                  ct.Constant(causal), ct.Constant(even_k))
    end

    times = Float64[]
    for _ in 1:nruns
        t = CUDA.@elapsed ct.launch(fmha_kernel, grid, Q, K, V, Out,
                  qk_scale, input_pos,
                  ct.Constant(D_k), ct.Constant(D_v), ct.Constant(Heads),
                  ct.Constant(tm), ct.Constant(tn),
                  ct.Constant(query_group_size),
                  ct.Constant(causal), ct.Constant(even_k))
        push!(times, t * 1000)
    end

    return (; Out, times)
end

function nnlib_attention(
    Q::AbstractArray{T,4}, K::AbstractArray{T,4}, V::AbstractArray{T,4};
    causal::Bool = false,
) where T
    mask = causal ? NNlib.make_causal_mask(Q; dims=2) : nothing
    query_group_size = cld(size(Q, 3), size(K, 3))
    if query_group_size > 1
        K, V = repeat.((K, V), inner=(1, 1, query_group_size, 1))
    end
    Out, _ = NNlib.dot_product_attention(Q, K, V; mask)
    return Out
end

function verify(data, result)
    # run on GPU for proper accumulation
    expected = nnlib_attention(data.Q, data.K, data.V; data.causal)
    @assert isapprox(expected, result.Out, rtol=1e-2) "max diff: $(maximum(abs, result.Out - expected))"
end

#=============================================================================
 Reference implementations for benchmarking
=============================================================================#

function run_others(data; nruns::Int=1, warmup::Int=0)
    (; Q, K, V, causal) = data
    results = Dict{String, Vector{Float64}}()

    cache = AllocCache()

    CUDA.@sync for _ in 1:warmup
        @cached cache nnlib_attention(Q, K, V; causal)
    end
    times = Float64[]
    for _ in 1:nruns
        t = CUDA.@elapsed @cached cache nnlib_attention(Q, K, V; causal)
        push!(times, t * 1000)
    end
    results["NNlib"] = times

    return results
end

#=============================================================================
 Main
=============================================================================#

function test_attention(::Type{T},
    D_k, SeqLen_Q, Heads, Batch,
    D_v, SeqLen_KV, Heads_KV,
    causal, tm, tn;
    name=nothing
) where T
    name = something(name,
        join([
            T,
            "tile=$tm×$tn",
            "Q=$D_k×$SeqLen_Q",
            "K=$D_k×$SeqLen_KV",
            "V=$D_v×$SeqLen_KV",
            "Heads=$Heads/$Heads_KV",
            "Batch=$Batch",
            "causal=$causal"
        ], ", "))
    println("--- $name ---")
    data = prepare(; T, D_k, SeqLen_Q, Heads, Batch, D_v, SeqLen_KV, Heads_KV, causal)
    result = run(data; tm, tn)
    verify(data, result)
    println("  passed")
end

function main()
    println("--- cuTile Fused Multi-Head Attention Examples ---\n")

    for T in (Float32, Float16)
        # basic
        test_attention(T, 64, 256, 8, 2, 64, 256, 8, false, 32, 32)
        test_attention(T, 64, 256, 8, 2, 64, 128, 4, false, 32, 64)
        test_attention(T, 64, 256, 8, 2, 64, 256, 8, true, 32, 32)

        # uneven seqlen
        test_attention(T, 64, 128, 4, 1, 64, 97, 2, false, 32, 32)
        test_attention(T, 64, 127, 4, 1, 64, 127, 4, true, 32, 32)

        # D_k != D_v
        test_attention(T, 64, 256, 8, 2, 32, 256, 4, false, 32, 32)
    end

    println("\n--- All attention examples completed ---")
end

isinteractive() || main()
