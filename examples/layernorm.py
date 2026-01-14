#!/usr/bin/env python3
"""
Layer Normalization example - cuTile Python
Forward and backward passes with unified prepare/run/verify pattern.
"""

import cupy as cp
import numpy as np
import cuda.tile as ct
from math import ceil

#=============================================================================
# Forward Kernel
#=============================================================================

@ct.kernel
def layernorm_fwd_kernel(X, W, B, Y, Mean, Rstd, eps: ct.Constant[float], TILE_N: ct.Constant[int]):
    """Forward pass: computes mean/var, normalizes input, applies affine transform."""
    bid_m = ct.bid(0)
    num_tiles = ct.num_tiles(X, axis=1, shape=(1, TILE_N))
    N = X.shape[1]

    # Compute mean
    mean = ct.full((1, TILE_N), 0, dtype=ct.float32)
    for j in range(num_tiles):
        tx = ct.load(X, index=(bid_m, j), shape=(1, TILE_N), padding_mode=ct.PaddingMode.ZERO)
        mean += tx
    mean = ct.sum(mean, axis=1) / N
    ct.store(Mean, index=(bid_m,), tile=mean)

    # Compute variance
    var = ct.full((1, TILE_N), 0, dtype=ct.float32)
    for j in range(num_tiles):
        tx = ct.load(X, index=(bid_m, j), shape=(1, TILE_N), padding_mode=ct.PaddingMode.ZERO)
        mask = (j * TILE_N + ct.arange(TILE_N, dtype=ct.int32)) < N
        centered_tx = ct.where(mask, tx - mean, 0)
        var += centered_tx ** 2
    var = ct.sum(var, axis=1) / N
    rstd = 1 / ct.sqrt(var + eps)
    ct.store(Rstd, index=(bid_m,), tile=rstd)

    # Normalize and apply affine transformation
    for j in range(num_tiles):
        tx = ct.load(X, index=(bid_m, j), shape=(1, TILE_N), padding_mode=ct.PaddingMode.ZERO)
        tw = ct.load(W, index=(j,), shape=(TILE_N,), padding_mode=ct.PaddingMode.ZERO)
        tb = ct.load(B, index=(j,), shape=(TILE_N,), padding_mode=ct.PaddingMode.ZERO)
        ty = (tx - mean) * rstd
        ty = ty * tw + tb
        ct.store(Y, index=(bid_m, j), tile=ty.astype(Y.dtype))


#=============================================================================
# Backward Kernels
#=============================================================================

def bwd_helper(X, W, DY, bid_m, j, mean, rstd, TILE_N, N):
    """Helper to load data and compute common backward terms."""
    tx = ct.load(X, index=(bid_m, j), shape=(1, TILE_N), padding_mode=ct.PaddingMode.ZERO)
    tw = ct.load(W, index=(j,), shape=(TILE_N,), padding_mode=ct.PaddingMode.ZERO)
    tdy = ct.load(DY, index=(bid_m, j), shape=(1, TILE_N), padding_mode=ct.PaddingMode.ZERO)
    xhat = (tx - mean) * rstd
    wdy = tw * tdy
    mask = j * TILE_N + ct.arange(TILE_N, dtype=ct.int32) < N
    xhat = ct.where(mask, xhat, 0)
    wdy = ct.where(mask, wdy, 0)
    return tdy, xhat, wdy


@ct.kernel
def layernorm_bwd_dx_partial_dwdb_kernel(DX, DY, DW, DB, X, W, Mean, Rstd, Locks, TILE_N: ct.Constant[int]):
    """Backward pass part 1: computes dX and partial dW/dB with atomic accumulation."""
    bid_m = ct.bid(0)
    num_tiles = ct.num_tiles(X, axis=1, shape=(1, TILE_N))
    N = X.shape[1]
    GROUP_SIZE_M = DW.shape[0]
    group_bid_m = bid_m % GROUP_SIZE_M

    mean = ct.load(Mean, index=(bid_m,), shape=(1,))
    rstd = ct.load(Rstd, index=(bid_m,), shape=(1,))

    c1 = ct.full((1, TILE_N), 0, dtype=ct.float32)
    c2 = ct.full((1, TILE_N), 0, dtype=ct.float32)
    for j in range(num_tiles):
        _, xhat, wdy = bwd_helper(X, W, DY, bid_m, j, mean, rstd, TILE_N, N)
        c1 += xhat * wdy
        c2 += wdy
    c1 = ct.sum(c1, axis=1) / N
    c2 = ct.sum(c2, axis=1) / N

    for j in range(num_tiles):
        tdy, xhat, wdy = bwd_helper(X, W, DY, bid_m, j, mean, rstd, TILE_N, N)
        tdx = (wdy - (xhat * c1 + c2)) * rstd
        ct.store(DX, index=(bid_m, j), tile=tdx.astype(DX.dtype))

        partial_dw = (tdy * xhat).astype(DW.dtype)
        partial_db = tdy.astype(DB.dtype)

        while ct.atomic_cas(Locks, group_bid_m, 0, 1, memory_order=ct.MemoryOrder.ACQUIRE) == 1:
            pass

        partial_dw += ct.load(DW, index=(group_bid_m, j), shape=(1, TILE_N), padding_mode=ct.PaddingMode.ZERO)
        partial_db += ct.load(DB, index=(group_bid_m, j), shape=(1, TILE_N), padding_mode=ct.PaddingMode.ZERO)
        ct.store(DW, index=(group_bid_m, j), tile=partial_dw)
        ct.store(DB, index=(group_bid_m, j), tile=partial_db)

        ct.atomic_xchg(Locks, group_bid_m, 0, memory_order=ct.MemoryOrder.RELEASE)


@ct.kernel
def layernorm_bwd_dwdb_kernel(DW, DB, FINAL_DW, FINAL_DB, TILE_M: ct.Constant[int], TILE_N: ct.Constant[int]):
    """Backward pass part 2: Final reduction for dW and dB."""
    bid_n = ct.bid(0)
    num_tiles = ct.num_tiles(DW, axis=0, shape=(TILE_M, TILE_N))

    dw = ct.zeros((TILE_M, TILE_N), dtype=ct.float32)
    db = ct.zeros((TILE_M, TILE_N), dtype=ct.float32)
    for i in range(num_tiles):
        dw += ct.load(DW, index=(i, bid_n), shape=(TILE_M, TILE_N), padding_mode=ct.PaddingMode.ZERO)
        db += ct.load(DB, index=(i, bid_n), shape=(TILE_M, TILE_N), padding_mode=ct.PaddingMode.ZERO)
    sum_dw = ct.sum(dw, axis=0)
    sum_db = ct.sum(db, axis=0)

    ct.store(FINAL_DW, index=(bid_n,), tile=sum_dw.astype(FINAL_DW.dtype))
    ct.store(FINAL_DB, index=(bid_n,), tile=sum_db.astype(FINAL_DB.dtype))


#=============================================================================
# Example harness
#=============================================================================

def prepare(*, benchmark: bool = False, M: int = None, N: int = None, eps: float = 1e-5, GROUP_SIZE_M: int = 64, dtype=np.float32):
    """Allocate all data for forward and backward passes."""
    if M is None:
        M = 4096 if benchmark else 256
    if N is None:
        N = 4096 if benchmark else 256
    return {
        # Forward inputs/outputs
        "X": (-2.3 + 0.5 * cp.random.randn(M, N)).astype(dtype),
        "W": cp.random.randn(N).astype(dtype),
        "B": cp.random.randn(N).astype(dtype),
        "Y": cp.empty((M, N), dtype=dtype),
        "Mean": cp.empty(M, dtype=np.float32),
        "Rstd": cp.empty(M, dtype=np.float32),
        # Backward inputs/outputs
        "DY": (0.1 * cp.random.randn(M, N)).astype(dtype),
        "DX": cp.empty((M, N), dtype=dtype),
        "DW_partial": cp.empty((GROUP_SIZE_M, N), dtype=np.float32),
        "DB_partial": cp.empty((GROUP_SIZE_M, N), dtype=np.float32),
        "Locks": cp.empty(GROUP_SIZE_M, dtype=np.int32),
        "FINAL_DW": cp.empty(N, dtype=dtype),
        "FINAL_DB": cp.empty(N, dtype=dtype),
        # Metadata
        "eps": eps,
        "M": M,
        "N": N,
        "GROUP_SIZE_M": GROUP_SIZE_M
    }


def run(data, *, tile_n: int = 1024, tile_m: int = 32, nruns: int = 1, warmup: int = 0):
    """Run both forward and backward passes with timing."""
    X, W, B, Y = data["X"], data["W"], data["B"], data["Y"]
    Mean, Rstd = data["Mean"], data["Rstd"]
    DY, DX = data["DY"], data["DX"]
    DW_partial, DB_partial = data["DW_partial"], data["DB_partial"]
    Locks = data["Locks"]
    FINAL_DW, FINAL_DB = data["FINAL_DW"], data["FINAL_DB"]
    eps, M, N = data["eps"], data["M"], data["N"]
    GROUP_SIZE_M = data["GROUP_SIZE_M"]

    stream = cp.cuda.get_current_stream()

    def run_fwd():
        ct.launch(stream, (M,), layernorm_fwd_kernel, (X, W, B, Y, Mean, Rstd, eps, tile_n))

    def run_bwd():
        DW_partial.fill(0)
        DB_partial.fill(0)
        Locks.fill(0)
        ct.launch(stream, (M,), layernorm_bwd_dx_partial_dwdb_kernel,
                  (DX, DY, DW_partial, DB_partial, X, W, Mean, Rstd, Locks, tile_n))
        num_tiles_n = ceil(N / tile_n)
        ct.launch(stream, (num_tiles_n,), layernorm_bwd_dwdb_kernel,
                  (DW_partial, DB_partial, FINAL_DW, FINAL_DB, tile_m, tile_n))

    # Warmup
    for _ in range(warmup):
        run_fwd()
        run_bwd()
    cp.cuda.runtime.deviceSynchronize()

    # Timed forward runs
    times_fwd = []
    for _ in range(nruns):
        start = cp.cuda.Event()
        end = cp.cuda.Event()
        start.record(stream)
        run_fwd()
        end.record(stream)
        end.synchronize()
        times_fwd.append(cp.cuda.get_elapsed_time(start, end))  # ms

    # Timed backward runs
    times_bwd = []
    for _ in range(nruns):
        start = cp.cuda.Event()
        end = cp.cuda.Event()
        start.record(stream)
        run_bwd()
        end.record(stream)
        end.synchronize()
        times_bwd.append(cp.cuda.get_elapsed_time(start, end))  # ms

    return {
        "Y": Y, "Mean": Mean, "Rstd": Rstd,
        "DX": DX, "DW": FINAL_DW, "DB": FINAL_DB,
        "times_fwd": times_fwd, "times_bwd": times_bwd
    }


def verify(data, result):
    """Verify both forward and backward results."""
    X_np = cp.asnumpy(data["X"])
    W_np = cp.asnumpy(data["W"])
    B_np = cp.asnumpy(data["B"])
    DY_np = cp.asnumpy(data["DY"])
    eps = data["eps"]
    N = data["N"]

    # Forward verification
    expected_mean = np.mean(X_np, axis=1, keepdims=True)
    expected_var = np.mean((X_np - expected_mean) ** 2, axis=1, keepdims=True)
    expected_rstd = 1.0 / np.sqrt(expected_var + eps)
    xhat = (X_np - expected_mean) * expected_rstd
    expected_Y = xhat * W_np + B_np

    atol, rtol = 1e-2, 1e-2
    assert np.allclose(cp.asnumpy(result["Y"]), expected_Y, rtol=rtol, atol=atol), \
        f"Y mismatch! max diff: {np.max(np.abs(cp.asnumpy(result['Y']) - expected_Y))}"

    # Backward verification
    wdy = W_np * DY_np
    c1 = np.sum(xhat * wdy, axis=1, keepdims=True) / N
    c2 = np.sum(wdy, axis=1, keepdims=True) / N
    expected_DX = (wdy - (xhat * c1 + c2)) * expected_rstd
    expected_DW = np.sum(DY_np * xhat, axis=0)
    expected_DB = np.sum(DY_np, axis=0)

    assert np.allclose(cp.asnumpy(result["DX"]), expected_DX, rtol=rtol, atol=atol), \
        f"DX mismatch! max diff: {np.max(np.abs(cp.asnumpy(result['DX']) - expected_DX))}"
    assert np.allclose(cp.asnumpy(result["DW"]), expected_DW, rtol=rtol, atol=atol), \
        f"DW mismatch! max diff: {np.max(np.abs(cp.asnumpy(result['DW']) - expected_DW))}"
    assert np.allclose(cp.asnumpy(result["DB"]), expected_DB, rtol=rtol, atol=atol), \
        f"DB mismatch! max diff: {np.max(np.abs(cp.asnumpy(result['DB']) - expected_DB))}"

# No run_others for layernorm - no simple reference implementation to compare against


#=============================================================================
# Main
#=============================================================================

def test_layernorm(M, N, tile_n, tile_m=32, eps=1e-5, dtype=np.float32, name=None):
    """Test layer normalization (fwd+bwd) with given parameters."""
    name = name or f"layernorm ({M}x{N}), tile_n={tile_n}, tile_m={tile_m}, dtype={dtype.__name__}"
    print(f"--- {name} ---")
    data = prepare(M=M, N=N, eps=eps, dtype=dtype)
    result = run(data, tile_n=tile_n, tile_m=tile_m)
    verify(data, result)
    print("  fwd passed, bwd passed")


def main():
    print("--- cuTile Layer Normalization Examples (fwd+bwd) ---\n")

    test_layernorm(256, 256, 256)
    test_layernorm(512, 512, 512)
    test_layernorm(1024, 1024, 1024)

    print("\n--- All layernorm examples completed ---")


if __name__ == "__main__":
    main()
