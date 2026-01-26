# Fused Multi-Head Attention example - Julia port of cuTile Python's AttentionFMHA.py sample
#
# SPDX-License-Identifier: Apache-2.0

using CUDA
import cuTile as ct

import NNlib

const INV_LOG_2 = Float32(1 / log(2))
const ConstInt = ct.Constant{Int}
const ConstBool = ct.Constant{Bool}

# cuTile kernel for Fused Multi-Head Attention
# Layout: (D, SeqLen, Heads, Batch) - Julia column-major
function fmha_kernel(
    Q::ct.TileArray{T,4}, K::ct.TileArray{T,4}, V::ct.TileArray{T,4}, Out::ct.TileArray{T,4},
    qk_scale::AbstractFloat,
    input_pos::Integer,
    TILE_D::ConstInt,
    H::ConstInt,
    TILE_M::ConstInt,
    TILE_N::ConstInt,
    QUERY_GROUP_SIZE::ConstInt,
    CAUSAL::ConstBool,
    EVEN_K::ConstBool
) where T
    bid_x = ct.bid(1)
    bid_y = ct.bid(2)
    batch_idx = cld(bid_y, H[])
    head_idx = mod1(bid_y, H[])
    off_kv_h = cld(head_idx, QUERY_GROUP_SIZE[])

    qk_scale = Float32(qk_scale) * Float32(INV_LOG_2)

    # Offsets for query tile (M-dimension) - 0-indexed positions
    # bid_x is 1-indexed, so first tile (bid_x=1) has positions [0, TILE_M-1]
    # NOTE: ct.arange is 1-indexed in Julia (returns [1,2,...,N]), so subtract 1 for 0-indexed
    offs_m = ct.reshape((bid_x - Int32(1)) * TILE_M[] .+ (ct.arange((TILE_M[],), Int32) .- Int32(1)) .+ input_pos, (1, TILE_M[]))

    # local offsets for key/value tile (N-dimension)
    # NOTE: ct.arange is 1-indexed in Julia, subtract 1 for 0-indexed
    offs_n_tile = ct.reshape(ct.arange((TILE_N[],), Int32) .- Int32(1), (TILE_N[], 1))

    # online softmax accumulators in Float32 for stability
    m_i = ct.full((1, TILE_M[]), -Inf32, Float32)
    l_i = ct.zeros((1, TILE_M[]), Float32)
    acc = ct.zeros((TILE_D[], TILE_M[]), Float32)

    # query tile for this batch, head, and M-chunk
    q = ct.load(Q, (1, bid_x, head_idx, batch_idx), (TILE_D[], TILE_M[], 1, 1))
    q = ct.reshape(q, (TILE_D[], TILE_M[]))

    # m_end: one past the last query position in this tile
    m_end = input_pos + bid_x * TILE_M[]
    k_seqlen = K.sizes[2]

    if CAUSAL[]
        # Python: mask_start = (input_pos + bid_x * TILE_M) // TILE_N
        # In Julia with 1-indexed bid_x: mask_start = (input_pos + (bid_x-1) * TILE_M) // TILE_N + 1
        mask_start = div(input_pos + (bid_x - Int32(1)) * TILE_M[], TILE_N[]) + Int32(1)
        # Python: mask_start = min(mask_start, k_seqlen // TILE_N)
        mask_start = min(mask_start, div(k_seqlen, TILE_N[]) + Int32(1))
        Tc = cld(min(m_end, k_seqlen), TILE_N[])
    else
        Tc = cld(k_seqlen, TILE_N[])
        # Python: mask_start = k_seqlen // TILE_N
        mask_start = div(k_seqlen, TILE_N[]) + Int32(1)
    end

    # loop over K, V blocks (N-dimension chunks)
    j = Int32(1)
    while j <= Tc
        k = ct.load(K, (1, j, off_kv_h, batch_idx), (TILE_D[], TILE_N[], 1, 1); padding_mode=ct.PaddingMode.Zero)
        k = ct.reshape(k, (TILE_D[], TILE_N[]))
        k = ct.transpose(k)

        qk = ct.zeros((TILE_N[], TILE_M[]), Float32)
        qk = ct.muladd(k, q, qk)

        # Apply masking (matches Python: if (CAUSAL or not EVEN_K) and j >= mask_start)
        if (CAUSAL[] || !EVEN_K[]) && j >= mask_start
            offs_n = (j - Int32(1)) * TILE_N[] .+ offs_n_tile
            # Build mask: start with all true
            valid_mask = ct.full((TILE_N[], TILE_M[]), true, Bool)
            # out of bound mask (Python: if not EVEN_K: mask = mask & (offs_n < k_seqlen))
            if !EVEN_K[]
                valid_mask = valid_mask .& (offs_n .< k_seqlen)
            end
            # causal mask (Python: if CAUSAL: mask = mask & (offs_m >= offs_n))
            if CAUSAL[]
                valid_mask = valid_mask .& (offs_m .>= offs_n)
            end
            # Apply mask: set invalid positions to -Inf
            qk = ct.where(valid_mask, qk, ct.full((TILE_N[], TILE_M[]), -Inf32, Float32))
        end

        # Online Softmax Update
        # Moving qk_scale multiplication after reduce_max is to improve performance
        m_ij = max.(m_i, (ct.reduce_max(qk, 1) * qk_scale))
        qk = qk * qk_scale .- m_ij

        # attention weights [TILE_N, TILE_M]
        p = exp2.(qk)  # TODO: flush_to_zero=True
        l_ij = ct.reduce_sum(p, 1)
        alpha = exp2.(m_i .- m_ij)  # TODO: flush_to_zero=True

        l_i = l_i .* alpha .+ l_ij
        acc = acc .* alpha

        v = ct.load(V, (1, j, off_kv_h, batch_idx), (TILE_D[], TILE_N[], 1, 1); padding_mode=ct.PaddingMode.Zero)
        v = ct.reshape(v, (TILE_D[], TILE_N[]))
        p = ct.astype(p, eltype(q))
        acc = ct.muladd(v, p, acc)
        m_i = m_ij

        j += Int32(1)
    end

    acc = acc ./ l_i  # TODO: flush_to_zero=True, rounding_mode=APPROX
    acc = ct.reshape(acc, (TILE_D[], TILE_M[], 1, 1))
    ct.store(Out, (1, bid_x, head_idx, batch_idx), acc)

    return
end

function cutile_fmha(Q::AbstractArray{T,4}, K::AbstractArray{T,4}, V::AbstractArray{T,4};
    qk_scale::Union{AbstractFloat,Nothing} = nothing,
    input_pos::Integer = 0,
    tile_m::Integer = 128,
    tile_n::Integer = 128,
    query_group_size::Integer = 1,
    causal::Bool = false,
) where T
    if size(Q, 4) != size(K, 4) || size(Q, 4) != size(V, 4)
        throw(ArgumentError("Batch dimensions must match for Q, K, V."))
    end
    if size(Q, 3) % query_group_size != 0
        throw(ArgumentError("Number of query heads must be divisible by query_group_size."))
    end
    if size(K, 3) * query_group_size != size(Q, 3)
        throw(ArgumentError("K_heads * query_group_size must equal Q_heads."))
    end
    if size(Q, 1) != size(K, 1)
        throw(ArgumentError("D_k (first dim of Q and K) must match."))
    end
    if size(K, 2) != size(V, 2)
        throw(ArgumentError("SeqLen_KV (dim 2 of K and V) must match."))
    end

    D_k, SeqLen_Q, Heads, Batch = size(Q)
    D_v, SeqLen_KV, KV_heads, _ = size(V)
    even_k = (SeqLen_KV % tile_n) == 0

    isnothing(qk_scale) && (qk_scale = 1 / sqrt(D_k))

    Out = CUDA.zeros(T, D_v, SeqLen_Q, Heads, Batch)

    grid_x = cld(SeqLen_Q, tile_m)
    grid_y = Heads * Batch
    grid = (grid_x, grid_y, 1)

    ct.launch(fmha_kernel, grid,
        Q, K, V, Out,
        qk_scale, input_pos,
        ct.Constant(D_k),
        ct.Constant(Heads),
        ct.Constant(tile_m),
        ct.Constant(tile_n),
        ct.Constant(query_group_size),
        ct.Constant(causal),
        ct.Constant(even_k))

    return Out
end

function nnlib_fmha(Q::AbstractArray{T,4}, K::AbstractArray{T,4}, V::AbstractArray{T,4};
    query_group_size::Integer = 1,
    causal::Bool = false,
) where T
    mask = causal ? NNlib.make_causal_mask(Q; dims=2) : nothing
    if query_group_size > 1
        K, V = repeat.((K, V), inner=(1, 1, query_group_size, 1))
    end
    Out, _ = NNlib.dot_product_attention(Q, K, V; mask)
    return Out
end


function test_fmha(::Type{T},
    D_k, SeqLen_Q, Heads, Batch,
    D_v, SeqLen_KV, KV_heads,
    causal, tile_m, tile_n,
) where T
    query_group_size = Heads ÷ KV_heads

    Q = CUDA.randn(T, D_k, SeqLen_Q, Heads, Batch)
    K = CUDA.randn(T, D_k, SeqLen_KV, KV_heads, Batch)
    V = CUDA.randn(T, D_v, SeqLen_KV, KV_heads, Batch)

    out_cutile = cutile_fmha(Q, K, V;
        causal=causal,
        tile_m=tile_m, tile_n=tile_n,
        query_group_size=query_group_size)

    Q_cpu = Array(Q)
    K_cpu = Array(K)
    V_cpu = Array(V)
    expected = nnlib_fmha(Q_cpu, K_cpu, V_cpu; query_group_size, causal)
    result = Array(out_cutile)

    if isapprox(result, expected, rtol=1e-2, atol=1e-2)
        println("  passed")
    else
        max_diff = maximum(abs.(result - expected))
        println("  FAILED (max diff: $max_diff)")
    end
end

function main()
    println("--- cuTile Fused Multi-Head Attention Examples ---\n")

    # Float32 tests, causal=false, EVEN_K=true
    println("Non-causal, EVEN_K=true:")
    test_fmha(Float32, 64, 256, 8, 2, 64, 256, 8, false, 32, 32)
    test_fmha(Float32, 64, 256, 8, 2, 64, 128, 8, false, 32, 32)
    test_fmha(Float32, 64, 256, 8, 2, 64, 128, 4, false, 32, 32)

    # Float32 tests, causal=true, EVEN_K=true
    println("\nCausal, EVEN_K=true:")
    test_fmha(Float32, 64, 256, 8, 2, 64, 256, 8, true, 32, 32)
    test_fmha(Float32, 64, 128, 8, 2, 64, 128, 8, true, 32, 32)
    test_fmha(Float32, 64, 64, 4, 1, 64, 64, 4, true, 32, 32)

    # Float32 tests, EVEN_K=false (K_SEQLEN not divisible by tile_n)
    println("\nNon-causal, EVEN_K=false:")
    test_fmha(Float32, 64, 64, 1, 1, 64, 50, 1, false, 32, 32)  # 18 keys in partial tile
    test_fmha(Float32, 64, 64, 1, 1, 64, 33, 1, false, 32, 32)  # 1 key in partial tile

    println("\nCausal, EVEN_K=false:")
    test_fmha(Float32, 64, 50, 1, 1, 64, 50, 1, true, 32, 32)   # 18 keys in partial tile
    test_fmha(Float32, 64, 33, 1, 1, 64, 33, 1, true, 32, 32)   # 1 key in partial tile

    println("\n--- All FMHA tests completed ---")
end

isinteractive() || main()
