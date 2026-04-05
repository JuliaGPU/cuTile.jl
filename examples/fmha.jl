# Fused Multi-Head Attention (FMHA) example - Julia port of cuTile Python's
# AttentionFMHA.py sample.
#
# Implements FlashAttention-2 style online softmax with tiling.
#
# Data layout (column-major, reversed from Python's row-major):
#   Python Q(Batch, Heads, SeqLen_Q, D_k)   -> Julia Q(D_k, SeqLen_Q, Heads, Batch)
#   Python K(Batch, KVH, SeqLen_KV, D_k)    -> Julia K(D_k, SeqLen_KV, KVH, Batch)
#   Python V(Batch, KVH, SeqLen_KV, D_v)    -> Julia V(D_v, SeqLen_KV, KVH, Batch)
#   Python Out(Batch, Heads, SeqLen_Q, D_v)  -> Julia Out(D_v, SeqLen_Q, Heads, Batch)
#
# SPDX-License-Identifier: Apache-2.0

using CUDA
import cuTile as ct

const INV_LOG_2 = Float32(1.0 / log(2.0))

function fmha_kernel(Q::ct.TileArray{T, 4}, K::ct.TileArray{T, 4},
                     V::ct.TileArray{T, 4}, Out::ct.TileArray{T, 4},
                     qk_scale::Float32,
                     input_pos::Int32,
                     TILE_D::Int, H::Int,
                     TILE_M::Int, TILE_N::Int,
                     QUERY_GROUP_SIZE::Int,
                     CAUSAL::Bool, EVEN_K::Bool) where {T}
    ct.@compiler_options occupancy=2

    # Map block IDs to batch and head indices
    # Julia: bid(1) = x (seq tiles), bid(2) = y (batch * heads)
    bid_x = ct.bid(1)
    bid_y = ct.bid(2) - Int32(1)  # 0-indexed for div/mod arithmetic
    batch_idx = fld(bid_y, Int32(H)) + Int32(1)
    head_idx = rem(bid_y, Int32(H)) + Int32(1)
    off_kv_h = fld(head_idx - Int32(1), Int32(QUERY_GROUP_SIZE)) + Int32(1)

    # Adjust qk_scale for exp2
    qk_scale = qk_scale * INV_LOG_2

    # Initialize offsets for current query tile (M-dimension)
    # Julia shape: (1, TILE_M) — reversed from Python's (TILE_M, 1)
    offs_m = (bid_x - Int32(1)) * Int32(TILE_M) .+ ct.arange(TILE_M; dtype=Int32)
    offs_m = offs_m .+ input_pos
    offs_m = reshape(offs_m, (1, TILE_M))  # (1, TILE_M)

    # Initialize local offsets for key/value tile (N-dimension)
    # Julia shape: (TILE_N, 1) — reversed from Python's (1, TILE_N)
    offs_n_tile = ct.arange(TILE_N; dtype=Int32)
    offs_n_tile = reshape(offs_n_tile, (TILE_N, 1))  # (TILE_N, 1)

    # Initialize online softmax accumulators in Float32 for stability
    # Julia: m_i is (1, TILE_M), l_i is (1, TILE_M), acc is (TILE_D, TILE_M)
    m_i = fill(Float32(-Inf), (1, TILE_M))
    l_i = zeros(Float32, (1, TILE_M))
    acc = zeros(Float32, (TILE_D, TILE_M))

    # Load query tile: Q is (D_k, SeqLen_Q, Heads, Batch)
    q = reshape(ct.load(Q; index=(Int32(1), bid_x, head_idx, batch_idx),
                        shape=(TILE_D, TILE_M, 1, 1)),
                (TILE_D, TILE_M))  # (TILE_D, TILE_M)

    # Loop bounds
    m_end = input_pos + bid_x * Int32(TILE_M)
    k_seqlen = size(K, 2)  # SeqLen_KV dimension

    if CAUSAL
        mask_start = fld(input_pos + (bid_x - Int32(1)) * Int32(TILE_M), Int32(TILE_N))
        mask_start = min(mask_start, fld(k_seqlen, Int32(TILE_N)))
        Tc = cld(min(m_end, k_seqlen), Int32(TILE_N))
    else
        Tc = cld(k_seqlen, Int32(TILE_N))
        mask_start = fld(k_seqlen, Int32(TILE_N))
    end

    # Loop over K, V blocks
    for j in Int32(0):Tc-Int32(1)
        # QK product
        # K is (D_k, SeqLen_KV, KVH, Batch)
        # Load (TILE_N, TILE_D, 1, 1) with order=(2,1,3,4) to transpose D and N
        k = reshape(ct.load(K; index=(Int32(1), j + Int32(1), off_kv_h, batch_idx),
                            shape=(TILE_N, TILE_D, 1, 1), order=(2, 1, 3, 4),
                            latency=2),
                    (TILE_N, TILE_D))  # (TILE_N, TILE_D)
        qk = zeros(Float32, (TILE_N, TILE_M))
        # (TILE_N, TILE_D) @ (TILE_D, TILE_M) = (TILE_N, TILE_M)
        qk = muladd(k, q, qk)

        # Causal masking
        if (CAUSAL | !EVEN_K) & (j >= mask_start)
            offs_n = j * Int32(TILE_N) .+ offs_n_tile  # (TILE_N, 1) — 0-indexed
            mask = fill(true, (TILE_N, TILE_M))
            if !EVEN_K
                mask = mask .& (offs_n .< k_seqlen)
            end
            if CAUSAL
                # offs_m is 0-indexed, offs_n is 0-indexed
                mask = mask .& (offs_m .>= offs_n)  # broadcasts (1, TILE_M) >= (TILE_N, 1)
            end
            qk = qk .+ ifelse.(mask, 0.0f0, Float32(-Inf))
        end

        # Online softmax
        # maximum along dim 1 (N dimension): (TILE_N, TILE_M) → (1, TILE_M)
        m_ij = max.(m_i, maximum(qk; dims=1) .* qk_scale)
        qk = qk .* qk_scale .- m_ij  # broadcasts (1, TILE_M) over dim 1

        ct.@fpmode flush_to_zero=true begin
            p = exp2.(qk)                  # (TILE_N, TILE_M)
            l_ij = sum(p; dims=1)          # (1, TILE_M)
            alpha = exp2.(m_i .- m_ij)     # (1, TILE_M)
            l_i = l_i .* alpha .+ l_ij
            acc = acc .* alpha  # broadcasts (1, TILE_M) over (TILE_D, TILE_M)
        end

        # PV product
        # V is (D_v, SeqLen_KV, KVH, Batch)
        v = reshape(ct.load(V; index=(Int32(1), j + Int32(1), off_kv_h, batch_idx),
                            shape=(TILE_D, TILE_N, 1, 1),
                            latency=4),
                    (TILE_D, TILE_N))  # (TILE_D, TILE_N)
        p = convert(ct.Tile{T}, p)
        # (TILE_D, TILE_N) @ (TILE_N, TILE_M) = (TILE_D, TILE_M)
        acc = muladd(v, p, acc)
        m_i = m_ij
    end

    # Final normalization and store
    # acc is (TILE_D, TILE_M), l_i is (1, TILE_M)
    ct.@fpmode rounding_mode=ct.Rounding.Approx flush_to_zero=true begin
        acc = acc ./ l_i
    end
    acc = reshape(convert(ct.Tile{T}, acc), (TILE_D, TILE_M, 1, 1))
    ct.store(Out; index=(Int32(1), bid_x, head_idx, batch_idx), tile=acc)

    return nothing
end


#=============================================================================
 Host-side wrapper
=============================================================================#

function cutile_fmha(Q::CuArray{T}, K::CuArray{T}, V::CuArray{T};
                     qk_scale::Union{Float32,Nothing}=nothing,
                     input_pos::Int=0,
                     tile_m::Int=128, tile_n::Int=128,
                     query_group_size::Int=1,
                     causal::Bool=false) where {T}
    D_k, SeqLen_Q, Heads, Batch = size(Q)
    D_v, SeqLen_KV, KV_Heads, _ = size(V)
    even_k = (SeqLen_KV % tile_n) == 0

    if qk_scale === nothing
        qk_scale = Float32(1.0 / sqrt(D_k))
    end

    Out = CuArray{T}(undef, D_v, SeqLen_Q, Heads, Batch)

    grid_x = cld(SeqLen_Q, tile_m)
    grid_y = Batch * Heads

    ct.launch(fmha_kernel, (grid_x, grid_y),
              Q, K, V, Out,
              qk_scale, Int32(input_pos),
              ct.Constant(D_k), ct.Constant(Heads),
              ct.Constant(tile_m), ct.Constant(tile_n),
              ct.Constant(query_group_size),
              ct.Constant(causal), ct.Constant(even_k))

    return Out
end


#=============================================================================
 Reference implementation
=============================================================================#

function ref_fmha(Q, K, V; qk_scale=nothing, causal=false)
    Q_cpu = Float32.(Array(Q))  # (D_k, SeqLen_Q, Heads, Batch)
    K_cpu = Float32.(Array(K))  # (D_k, SeqLen_KV, KVH, Batch)
    V_cpu = Float32.(Array(V))  # (D_v, SeqLen_KV, KVH, Batch)

    D, M, H, B = size(Q_cpu)
    _, N, KH, _ = size(K_cpu)
    D_v = size(V_cpu, 1)

    if qk_scale === nothing
        qk_scale = Float32(1.0 / sqrt(D))
    end

    Out = zeros(Float32, D_v, M, H, B)

    for b in 1:B, h in 1:H
        kh = cld(h, H ÷ KH)  # KV head index

        # q: (D, M), k: (D, N), v: (D_v, N)
        q = Q_cpu[:, :, h, b]
        k = K_cpu[:, :, kh, b]
        v = V_cpu[:, :, kh, b]

        # scores: (M, N) = q^T @ k
        scores = (q' * k) .* qk_scale  # (M, N)

        if causal
            for i in 1:M, j in 1:N
                if j > i
                    scores[i, j] = -Inf32
                end
            end
        end

        # Softmax over N (dim 2)
        scores_max = maximum(scores; dims=2)
        scores_max = ifelse.(isinf.(scores_max), 0.0f0, scores_max)
        exp_scores = exp.(scores .- scores_max)
        attn = exp_scores ./ sum(exp_scores; dims=2)  # (M, N)

        # out: (D_v, M) = v @ attn^T
        Out[:, :, h, b] = v * attn'
    end

    return Out
end


#=============================================================================
 Example harness
=============================================================================#

function prepare(; benchmark::Bool=false,
                  batch::Int = benchmark ? 8 : 2,
                  heads::Int = benchmark ? 16 : 8,
                  seq_q::Int = benchmark ? 1024 : 128,
                  seq_kv::Int = benchmark ? 1024 : 128,
                  d_k::Int = 64, d_v::Int = 64,
                  query_group_size::Int = 1,
                  causal::Bool = true,
                  tile_m::Int = 128, tile_n::Int = 128,
                  T::DataType = Float16)
    kv_heads = heads ÷ query_group_size

    # Julia layout: (D, SeqLen, Heads, Batch) — D contiguous
    Q = T.((CUDA.rand(d_k, seq_q, heads, batch) .- 0.5f0))
    K = T.((CUDA.rand(d_k, seq_kv, kv_heads, batch) .- 0.5f0))
    V = T.((CUDA.rand(d_v, seq_kv, kv_heads, batch) .- 0.5f0))

    return (;
        Q, K, V,
        batch, heads, seq_q, seq_kv, d_k, d_v,
        query_group_size, causal,
        tile_m, tile_n
    )
end


function run(data; nruns::Int=1, warmup::Int=0)
    (; Q, K, V, causal, tile_m, tile_n, query_group_size) = data

    CUDA.@sync for _ in 1:warmup
        cutile_fmha(Q, K, V; tile_m, tile_n, query_group_size, causal)
    end

    times = Float64[]
    out = nothing
    for _ in 1:nruns
        t = CUDA.@elapsed begin
            out = cutile_fmha(Q, K, V; tile_m, tile_n, query_group_size, causal)
        end
        push!(times, t * 1000)  # ms
    end

    return (; out, times)
end


function verify(data, result)
    expected = ref_fmha(data.Q, data.K, data.V; causal=data.causal)
    actual = Float32.(Array(result.out))
    max_diff = maximum(abs.(actual .- expected))
    @assert max_diff < 1e-2 "FMHA incorrect! max diff: $max_diff"
end


function metric(data)
    B = data.batch
    H = data.heads
    M = data.seq_q
    N = data.seq_kv
    D = data.d_k
    # QK^T: 2*B*H*M*N*D, P@V: 2*B*H*M*N*D, total: 4*B*H*M*N*D
    return 4 * B * H * M * N * D, "TFLOPS"
end


#=============================================================================
 Main
=============================================================================#

function test_fmha(batch, heads, seq_q, seq_kv, d_k, d_v;
                   causal=false, tile_m=128, tile_n=128,
                   query_group_size=1, T=Float16, name=nothing)
    name = something(name, "fmha B=$batch H=$heads M=$seq_q N=$seq_kv " *
                           "D=$d_k causal=$causal $T")
    println("--- $name ---")
    data = prepare(; batch, heads, seq_q, seq_kv, d_k, d_v,
                    query_group_size, causal, tile_m, tile_n, T)
    result = run(data)
    verify(data, result)
    println("  passed")
end


function main()
    println("--- cuTile FMHA Examples ---\n")

    # Non-causal
    test_fmha(2, 8, 128, 128, 64, 64; causal=false)
    # Causal
    test_fmha(2, 8, 128, 128, 64, 64; causal=true)
    # Larger
    test_fmha(4, 16, 256, 256, 64, 64; causal=true)

    println("\n--- All FMHA examples completed ---")
end

isinteractive() || main()
