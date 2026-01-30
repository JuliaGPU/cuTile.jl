# LayerNorm example - Julia port of cuTile Python's LayerNorm.py sample
#
# SPDX-License-Identifier: Apache-2.0

using CUDA
import cuTile as ct

const ConstInt = ct.Constant{Int}

#=============================================================================
 LayerNorm Forward Kernel

 Forward pass: computes mean/var, normalizes input, and applies affine transform.

 Args:
     X: Input tensor (M, N).
     W: Weight tensor (N,).
     B: Bias tensor (N,).
     Y: Output tensor (M, N).
     Mean: Output mean tensor (M,).
     Rstd: Output reciprocal standard deviation tensor (M,).
     eps: Epsilon for numerical stability.
     TILE_N: Tile size along N dimension.
=============================================================================#
function layer_norm_fwd(X::ct.TileArray{Float32, 2}, W::ct.TileArray{Float32, 1},
                        B::ct.TileArray{Float32, 1}, Y::ct.TileArray{Float32, 2},
                        Mean::ct.TileArray{Float32, 1}, Rstd::ct.TileArray{Float32, 1},
                        eps::ct.Constant{Float32}, TILE_N::ConstInt)
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

#=============================================================================
 LayerNorm Backward Kernels

 Backward pass: computes gradients for LayerNorm.
 The full backward pass has two kernels:
 1. layer_norm_bwd_dx - Computes dX (gradient with respect to input)
 2. layer_norm_bwd_dwdb - Computes dW and dB (requires atomic accumulation)

 For now, we implement a simplified backward that just computes dX.
=============================================================================#

"""
Helper function for backward pass - loads data and computes common terms.
This gets inlined by Julia's compiler.
bid_m and j are 1-indexed (block ID and tile index).
"""
@inline function bwd_helper(X, W, DY, bid_m, j, mean, rstd, TILE_N, N)
    tx = ct.load(X, (bid_m, j), (1, TILE_N); padding_mode=ct.PaddingMode.Zero)
    tw = ct.load(W, j, (TILE_N,); padding_mode=ct.PaddingMode.Zero)
    tdy = ct.load(DY, (bid_m, j), (1, TILE_N); padding_mode=ct.PaddingMode.Zero)
    xhat = (tx .- mean) .* rstd
    wdy = tw .* tdy

    # Mask for valid elements
    indices = ct.arange((TILE_N,), Int32)
    offset = (j - Int32(1)) * Int32(TILE_N)
    global_indices = offset .+ indices
    mask = ct.broadcast_to(global_indices .<= N, (1, TILE_N))

    xhat_masked = ct.where(mask, xhat, ct.full((1, TILE_N), 0.0f0, Float32))
    wdy_masked = ct.where(mask, wdy, ct.full((1, TILE_N), 0.0f0, Float32))

    return tdy, xhat_masked, wdy_masked
end

"""
    layer_norm_bwd_dx(DX, DY, X, W, Mean, Rstd, TILE_N)

Backward pass: computes gradient with respect to input X.

Args:
    DX: Output gradient with respect to X (M, N).
    DY: Input gradient with respect to Y (M, N).
    X: Input tensor (M, N).
    W: Weight tensor (N,).
    Mean: Mean tensor (M,).
    Rstd: Reciprocal standard deviation tensor (M,).
    TILE_N: Tile size along N dimension.
"""
function layer_norm_bwd_dx(DX::ct.TileArray{Float32, 2}, DY::ct.TileArray{Float32, 2},
                           X::ct.TileArray{Float32, 2}, W::ct.TileArray{Float32, 1},
                           Mean::ct.TileArray{Float32, 1}, Rstd::ct.TileArray{Float32, 1},
                           TILE_N::ConstInt)
    bid_m = ct.bid(1)
    num_tiles = ct.num_tiles(X, 2, (1, TILE_N[]))
    N = X.sizes[2]

    # Load mean and rstd for this row
    mean = ct.load(Mean, bid_m, (1,); padding_mode=ct.PaddingMode.Zero)
    rstd = ct.load(Rstd, bid_m, (1,); padding_mode=ct.PaddingMode.Zero)

    # First pass: compute c1 and c2 reduction terms
    c1 = ct.full((1, TILE_N[]), 0.0f0, Float32)
    c2 = ct.full((1, TILE_N[]), 0.0f0, Float32)
    j = Int32(1)
    while j <= num_tiles
        _, xhat, wdy = bwd_helper(X, W, DY, bid_m, j, mean, rstd, TILE_N[], N)
        c1 = c1 .+ (xhat .* wdy)
        c2 = c2 .+ wdy
        j += Int32(1)
    end
    c1 = ct.reduce_sum(c1, 2) / N
    c2 = ct.reduce_sum(c2, 2) / N

    # Second pass: compute dX
    j = Int32(1)
    while j <= num_tiles
        _, xhat, wdy = bwd_helper(X, W, DY, bid_m, j, mean, rstd, TILE_N[], N)
        tdx = (wdy .- (xhat .* c1 .+ c2)) .* rstd
        ct.store(DX, (bid_m, j), tdx)
        j += Int32(1)
    end

    return
end

"""
    layer_norm_bwd_dx_partial_dwdb(DX, DY, DW, DB, X, W, Mean, Rstd, Locks, GROUP_SIZE_M, TILE_N)

Backward pass part 1: computes dX and partial dW/dB.
Accumulates partial gradients using atomic locks.

Args:
    DX: Output gradient with respect to X (M, N).
    DY: Input gradient with respect to Y (M, N).
    DW: Partial gradient with respect to W (GROUP_SIZE_M, N).
    DB: Partial gradient with respect to B (GROUP_SIZE_M, N).
    X: Input tensor (M, N).
    W: Weight tensor (N,).
    Mean: Mean tensor (M,).
    Rstd: Reciprocal standard deviation tensor (M,).
    Locks: Lock tensor for atomic operations (GROUP_SIZE_M,).
    GROUP_SIZE_M: Number of partial gradient groups.
    TILE_N: Tile size along N dimension.
"""
function layer_norm_bwd_dx_partial_dwdb(DX::ct.TileArray{Float32, 2}, DY::ct.TileArray{Float32, 2},
                                         DW::ct.TileArray{Float32, 2}, DB::ct.TileArray{Float32, 2},
                                         X::ct.TileArray{Float32, 2}, W::ct.TileArray{Float32, 1},
                                         Mean::ct.TileArray{Float32, 1}, Rstd::ct.TileArray{Float32, 1},
                                         Locks::ct.TileArray{Int, 1},
                                         GROUP_SIZE_M::ConstInt, TILE_N::ConstInt)
    bid_m = ct.bid(1)
    num_tiles = ct.num_tiles(X, 2, (1, TILE_N[]))
    N = X.sizes[2]
    group_bid_m = ((bid_m - Int32(1)) % Int32(GROUP_SIZE_M[])) + Int32(1)

    # Load mean and rstd for this row
    mean = ct.load(Mean, bid_m, (1,); padding_mode=ct.PaddingMode.Zero)
    rstd = ct.load(Rstd, bid_m, (1,); padding_mode=ct.PaddingMode.Zero)

    # First pass: compute c1 and c2 reduction terms
    c1 = ct.full((1, TILE_N[]), 0.0f0, Float32)
    c2 = ct.full((1, TILE_N[]), 0.0f0, Float32)
    j = Int32(1)
    while j <= num_tiles
        _, xhat, wdy = bwd_helper(X, W, DY, bid_m, j, mean, rstd, TILE_N[], N)
        c1 = c1 .+ (xhat .* wdy)
        c2 = c2 .+ wdy
        j += Int32(1)
    end
    c1 = ct.reduce_sum(c1, 2) / N
    c2 = ct.reduce_sum(c2, 2) / N

    # Second pass: compute dX and partial dW/dB
    j = Int32(1)
    while j <= num_tiles
        tdy, xhat, wdy = bwd_helper(X, W, DY, bid_m, j, mean, rstd, TILE_N[], N)
        tdx = (wdy .- (xhat .* c1 .+ c2)) .* rstd
        ct.store(DX, (bid_m, j), tdx)

        partial_dw = tdy .* xhat
        partial_db = tdy

        # Acquire spinlock
        while ct.atomic_cas(Locks, group_bid_m, 0, 1;
                           memory_order=ct.MemoryOrder.Acquire) == 1
            # spin
        end

        # Critical section: accumulate partial gradients
        partial_dw = partial_dw .+ ct.load(DW, (group_bid_m, j), (1, TILE_N[]); padding_mode=ct.PaddingMode.Zero)
        partial_db = partial_db .+ ct.load(DB, (group_bid_m, j), (1, TILE_N[]); padding_mode=ct.PaddingMode.Zero)
        ct.store(DW, (group_bid_m, j), partial_dw)
        ct.store(DB, (group_bid_m, j), partial_db)

        # Release spinlock
        ct.atomic_xchg(Locks, group_bid_m, 0;
                      memory_order=ct.MemoryOrder.Release)

        j += Int32(1)
    end

    return
end

"""
    layer_norm_bwd_dwdb(DW, DB, FINAL_DW, FINAL_DB, TILE_M, TILE_N)

Backward pass part 2: Final reduction for dW and dB.

Args:
    DW: Partial gradient with respect to W (TILE_M, N).
    DB: Partial gradient with respect to B (TILE_M, N).
    FINAL_DW: Final gradient with respect to W (N,).
    FINAL_DB: Final gradient with respect to B (N,).
    TILE_M: Number of partial gradients to reduce.
    TILE_N: Tile size along N dimension.
"""
function layer_norm_bwd_dwdb(DW::ct.TileArray{Float32, 2}, DB::ct.TileArray{Float32, 2},
                              FINAL_DW::ct.TileArray{Float32, 1}, FINAL_DB::ct.TileArray{Float32, 1},
                              TILE_M::ConstInt, TILE_N::ConstInt)
    bid_n = ct.bid(1)
    num_tiles = ct.num_tiles(DW, 1, (TILE_M[], TILE_N[]))

    dw = ct.zeros((TILE_M[], TILE_N[]), Float32)
    db = ct.zeros((TILE_M[], TILE_N[]), Float32)
    i = Int32(1)
    while i <= num_tiles
        dw = dw .+ ct.load(DW, (i, bid_n), (TILE_M[], TILE_N[]); padding_mode=ct.PaddingMode.Zero)
        db = db .+ ct.load(DB, (i, bid_n), (TILE_M[], TILE_N[]); padding_mode=ct.PaddingMode.Zero)
        i += Int32(1)
    end
    sum_dw = ct.reduce_sum(dw, 1)
    sum_db = ct.reduce_sum(db, 1)

    ct.store(FINAL_DW, bid_n, sum_dw)
    ct.store(FINAL_DB, bid_n, sum_db)

    return
end

#=============================================================================
 Example harness
=============================================================================#

function prepare(; benchmark::Bool=false,
                  M::Int=benchmark ? 4096 : 256,
                  N::Int=benchmark ? 4096 : 256,
                  eps::Float32=1f-5, GROUP_SIZE_M::Int=64)
    return (;
        # Forward inputs/outputs
        X = -2.3f0 .+ 0.5f0 .* CUDA.rand(Float32, M, N),
        W = CUDA.randn(Float32, N),
        B = CUDA.randn(Float32, N),
        Y = CuArray{Float32}(undef, M, N),
        Mean = CuArray{Float32}(undef, M),
        Rstd = CuArray{Float32}(undef, M),
        # Backward inputs/outputs
        DY = 0.1f0 .* CUDA.randn(Float32, M, N),
        DX = CuArray{Float32}(undef, M, N),
        DW_partial = CuArray{Float32}(undef, GROUP_SIZE_M, N),
        DB_partial = CuArray{Float32}(undef, GROUP_SIZE_M, N),
        Locks = CuArray{Int}(undef, GROUP_SIZE_M),
        FINAL_DW = CuArray{Float32}(undef, N),
        FINAL_DB = CuArray{Float32}(undef, N),
        # Metadata
        M, N, eps, GROUP_SIZE_M
    )
end

function run(data; TILE_N::Int=1024, TILE_M::Int=32, nruns::Int=1, warmup::Int=0)
    (; X, W, B, Y, Mean, Rstd, DY, DX, DW_partial, DB_partial, Locks, FINAL_DW, FINAL_DB,
       M, N, eps, GROUP_SIZE_M) = data

    function run_fwd()
        ct.launch(layer_norm_fwd, M, X, W, B, Y, Mean, Rstd,
                  ct.Constant(eps), ct.Constant(TILE_N))
    end

    function run_bwd()
        fill!(DW_partial, 0)
        fill!(DB_partial, 0)
        fill!(Locks, 0)
        ct.launch(layer_norm_bwd_dx_partial_dwdb, M, DX, DY, DW_partial, DB_partial, X, W,
                  Mean, Rstd, Locks, ct.Constant(GROUP_SIZE_M), ct.Constant(TILE_N))
        num_tiles_n = cld(N, TILE_N)
        ct.launch(layer_norm_bwd_dwdb, num_tiles_n, DW_partial, DB_partial, FINAL_DW, FINAL_DB,
                  ct.Constant(TILE_M), ct.Constant(TILE_N))
    end

    # Warmup
    CUDA.@sync for _ in 1:warmup
        run_fwd()
        run_bwd()
    end

    # Timed forward runs
    times_fwd = Float64[]
    for _ in 1:nruns
        t = CUDA.@elapsed run_fwd()
        push!(times_fwd, t * 1000)  # ms
    end

    # Timed backward runs
    times_bwd = Float64[]
    for _ in 1:nruns
        t = CUDA.@elapsed run_bwd()
        push!(times_bwd, t * 1000)  # ms
    end

    return (; Y, Mean, Rstd, DX, FINAL_DW, FINAL_DB, times_fwd, times_bwd)
end

function verify(data, result)
    (; X, W, B, DY, N, eps) = data

    X_cpu = Array(X)
    W_cpu = Array(W)
    B_cpu = Array(B)
    DY_cpu = Array(DY)

    # Forward verification
    expected_mean = vec(sum(X_cpu, dims=2) ./ N)
    expected_var = vec(sum((X_cpu .- expected_mean) .^ 2, dims=2) ./ N)
    expected_rstd = 1.0f0 ./ sqrt.(expected_var .+ eps)
    xhat = (X_cpu .- expected_mean) .* expected_rstd
    expected_Y = xhat .* W_cpu' .+ B_cpu'

    @assert isapprox(expected_Y, Array(result.Y); rtol=1e-2) "Y mismatch"

    # Backward verification
    wdy = W_cpu' .* DY_cpu
    c1 = sum(xhat .* wdy, dims=2) ./ N
    c2 = sum(wdy, dims=2) ./ N
    expected_DX = (wdy .- (xhat .* c1 .+ c2)) .* expected_rstd
    expected_DW = vec(sum(DY_cpu .* xhat, dims=1))
    expected_DB = vec(sum(DY_cpu, dims=1))

    @assert isapprox(expected_DX, Array(result.DX); rtol=1e-2) "dX mismatch"
    @assert isapprox(expected_DW, Array(result.FINAL_DW); rtol=1e-2) "dW mismatch"
    @assert isapprox(expected_DB, Array(result.FINAL_DB); rtol=1e-2) "dB mismatch"
end

function test_layernorm(M, N, TILE_N; TILE_M::Int=32, eps::Float32=1f-5, name=nothing)
    name = something(name, "layernorm ($M x $N), tile_n=$TILE_N, tile_m=$TILE_M")
    println("--- $name ---")
    data = prepare(; M, N, eps)
    result = run(data; TILE_N, TILE_M)
    verify(data, result)
    println("  fwd passed, bwd passed")
end

# No run_others for layernorm - no simple reference implementation to compare against

#=============================================================================
 Main
=============================================================================#

function main()
    println("=== cuTile LayerNorm Examples (fwd+bwd) ===\n")

    test_layernorm(256, 256, 256)
    test_layernorm(512, 512, 512)
    test_layernorm(1024, 2048, 1024)

    println("\n=== All layernorm examples completed ===")
end

isinteractive() || main()
