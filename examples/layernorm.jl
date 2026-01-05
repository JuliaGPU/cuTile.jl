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
    rstd = 1.0f0 / sqrt(var .+ eps[])
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
                                         Locks::ct.TileArray{Int32, 1},
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
        while ct.atomic_cas(Locks, group_bid_m, Int32(0), Int32(1);
                           memory_order=ct.MemoryOrder.Acquire) == Int32(1)
            # spin
        end

        # Critical section: accumulate partial gradients
        partial_dw = partial_dw .+ ct.load(DW, (group_bid_m, j), (1, TILE_N[]); padding_mode=ct.PaddingMode.Zero)
        partial_db = partial_db .+ ct.load(DB, (group_bid_m, j), (1, TILE_N[]); padding_mode=ct.PaddingMode.Zero)
        ct.store(DW, (group_bid_m, j), partial_dw)
        ct.store(DB, (group_bid_m, j), partial_db)

        # Release spinlock
        ct.atomic_xchg(Locks, group_bid_m, Int32(0);
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
 Test / Validation
=============================================================================#

function main()
    println("=== cuTile LayerNorm Sample ===\n")

    M, N = 1024, 2048
    TILE_N = 1024
    eps = 1f-5

    println("Input shape: ($M, $N), dtype: Float32, eps: $eps")

    # Input data
    X = -2.3f0 .+ 0.5f0 .* CUDA.rand(Float32, M, N)
    W = CUDA.randn(Float32, N)
    B = CUDA.randn(Float32, N)

    # Output buffers for forward pass
    Y = CUDA.zeros(Float32, M, N)
    Mean = CUDA.zeros(Float32, M)
    Rstd = CUDA.zeros(Float32, M)

    # =========================================================================
    # Forward Pass
    # =========================================================================
    println("\n--- Forward Pass ---")
    ct.launch(layer_norm_fwd, M, X, W, B, Y, Mean, Rstd,
              ct.Constant(eps), ct.Constant(TILE_N))

    # Compute expected values on CPU
    X_cpu = Array(X)
    W_cpu = Array(W)
    B_cpu = Array(B)

    expected_mean = vec(sum(X_cpu, dims=2) ./ N)
    expected_var = vec(sum((X_cpu .- expected_mean) .^ 2, dims=2) ./ N)
    expected_rstd = 1.0f0 ./ sqrt.(expected_var .+ eps)
    normalized = (X_cpu .- expected_mean) .* expected_rstd
    expected_Y = normalized .* W_cpu' .+ B_cpu'

    # Verify forward pass results
    Mean_cpu = Array(Mean)
    Rstd_cpu = Array(Rstd)
    Y_cpu = Array(Y)

    atol, rtol = 1f-2, 1f-2
    fwd_ok = isapprox(expected_mean, Mean_cpu; rtol, atol) &&
             isapprox(expected_rstd, Rstd_cpu; rtol, atol) &&
             isapprox(expected_Y, Y_cpu; rtol, atol)

    if fwd_ok
        println("Forward pass: PASSED")
    else
        println("Forward pass: FAILED")
        isapprox(expected_mean, Mean_cpu; rtol, atol) || println("  Mean max error: $(maximum(abs.(expected_mean .- Mean_cpu)))")
        isapprox(expected_rstd, Rstd_cpu; rtol, atol) || println("  Rstd max error: $(maximum(abs.(expected_rstd .- Rstd_cpu)))")
        isapprox(expected_Y, Y_cpu; rtol, atol) || println("  Y max error: $(maximum(abs.(expected_Y .- Y_cpu)))")
    end

    # =========================================================================
    # Backward Pass (Full: dX, dW, dB)
    # =========================================================================
    println("\n--- Backward Pass (Full: dX, dW, dB) ---")

    # Upstream gradient (random for testing)
    DY = CUDA.randn(Float32, M, N)
    DX = CUDA.zeros(Float32, M, N)

    # Parameters for partial gradient accumulation
    GROUP_SIZE_M = 64
    TILE_M = 32

    # Partial gradient buffers and locks
    DW_partial = CUDA.zeros(Float32, GROUP_SIZE_M, N)
    DB_partial = CUDA.zeros(Float32, GROUP_SIZE_M, N)
    Locks = CUDA.zeros(Int32, GROUP_SIZE_M)

    # Final gradient buffers
    FINAL_DW = CUDA.zeros(Float32, N)
    FINAL_DB = CUDA.zeros(Float32, N)

    # Launch backward kernels
    ct.launch(layer_norm_bwd_dx_partial_dwdb, M, DX, DY, DW_partial, DB_partial, X, W,
              Mean, Rstd, Locks, ct.Constant(GROUP_SIZE_M), ct.Constant(TILE_N))

    num_tiles_n = cld(N, TILE_N)
    ct.launch(layer_norm_bwd_dwdb, num_tiles_n, DW_partial, DB_partial, FINAL_DW, FINAL_DB,
              ct.Constant(TILE_M), ct.Constant(TILE_N))

    # Compute expected gradients on CPU
    # dX = rstd * (W * dY - c2 - x_hat * c1)
    # where c1 = mean(x_hat * W * dY), c2 = mean(W * dY)
    DY_cpu = Array(DY)
    wdy = W_cpu' .* DY_cpu
    xhat = normalized
    c1 = sum(xhat .* wdy, dims=2) ./ N
    c2 = sum(wdy, dims=2) ./ N
    expected_DX = (wdy .- (xhat .* c1 .+ c2)) .* expected_rstd

    # dW = sum(dY * x_hat, dim=0) and dB = sum(dY, dim=0)
    expected_DW = vec(sum(DY_cpu .* xhat, dims=1))
    expected_DB = vec(sum(DY_cpu, dims=1))

    # Verify dX
    DX_cpu = Array(DX)
    dx_ok = isapprox(expected_DX, DX_cpu; rtol, atol)
    if dx_ok
        println("  dX: PASSED")
    else
        max_err = maximum(abs.(expected_DX .- DX_cpu))
        println("  dX: FAILED (max error: $max_err)")
    end

    # Verify dW
    FINAL_DW_cpu = Array(FINAL_DW)
    dw_ok = isapprox(expected_DW, FINAL_DW_cpu; rtol, atol)
    if dw_ok
        println("  dW: PASSED")
    else
        max_err = maximum(abs.(expected_DW .- FINAL_DW_cpu))
        println("  dW: FAILED (max error: $max_err)")
    end

    # Verify dB
    FINAL_DB_cpu = Array(FINAL_DB)
    db_ok = isapprox(expected_DB, FINAL_DB_cpu; rtol, atol)
    if db_ok
        println("  dB: PASSED")
    else
        max_err = maximum(abs.(expected_DB .- FINAL_DB_cpu))
        println("  dB: FAILED (max error: $max_err)")
    end

    bwd_ok = dx_ok && dw_ok && db_ok

    # =========================================================================
    # Summary
    # =========================================================================
    println("\n=== Summary ===")
    println("Forward pass:        $(fwd_ok ? "PASSED" : "FAILED")")
    println("Backward (dX/dW/dB): $(bwd_ok ? "PASSED" : "FAILED")")

    (fwd_ok && bwd_ok) || error("LayerNorm tests failed")
end

isinteractive() || main()
