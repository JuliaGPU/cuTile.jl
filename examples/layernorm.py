#!/usr/bin/env python3
"""
Layer Normalization example - cuTile Python
"""

import cupy as cp
import numpy as np
import cuda.tile as ct

@ct.kernel
def layernorm_cutile_kernel(X, W, B, Y, Mean, Rstd, eps: ct.Constant[float], TILE_N: ct.Constant[int]):
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


def run_layernorm(*, M: int = 1024, N: int = 1024, tile_n: int = 1024, eps: float = 1e-5,
                  X=None, W=None, B=None, Y=None, Mean=None, Rstd=None,
                  dtype=np.float32, validate: bool = False):
    """Run layer normalization. Returns (X, W, B, Y, Mean, Rstd) arrays for benchmarking."""
    if X is None:
        X = (-2.3 + 0.5 * cp.random.randn(M, N)).astype(dtype)
    else:
        M, N = X.shape
    if W is None:
        W = cp.random.randn(N).astype(dtype)
    if B is None:
        B = cp.random.randn(N).astype(dtype)
    if Y is None:
        Y = cp.zeros((M, N), dtype=dtype)
    if Mean is None:
        Mean = cp.zeros(M, dtype=np.float32)
    if Rstd is None:
        Rstd = cp.zeros(M, dtype=np.float32)

    stream = cp.cuda.get_current_stream()
    ct.launch(stream, (M,), layernorm_cutile_kernel, (X, W, B, Y, Mean, Rstd, eps, tile_n))

    if validate:
        cp.cuda.runtime.deviceSynchronize()
        X_np = cp.asnumpy(X)
        W_np = cp.asnumpy(W)
        B_np = cp.asnumpy(B)
        expected_mean = np.mean(X_np, axis=1, keepdims=True)
        expected_var = np.mean((X_np - expected_mean) ** 2, axis=1, keepdims=True)
        expected_rstd = 1.0 / np.sqrt(expected_var + eps)
        normalized = (X_np - expected_mean) * expected_rstd
        expected_Y = normalized * W_np + B_np
        assert np.allclose(cp.asnumpy(Y), expected_Y, rtol=1e-2, atol=1e-2), \
            f"layernorm incorrect! max diff: {np.max(np.abs(cp.asnumpy(Y) - expected_Y))}"

    return X, W, B, Y, Mean, Rstd


def test_layernorm(M, N, tile_n, eps=1e-5, dtype=np.float32, name=None):
    """Test layer normalization with given parameters."""
    name = name or f"layernorm ({M}x{N}), tile={tile_n}, dtype={dtype.__name__}"
    print(f"--- {name} ---")
    run_layernorm(M=M, N=N, tile_n=tile_n, eps=eps, dtype=dtype, validate=True)
    print("  passed")


def main():
    print("--- cuTile Layer Normalization Examples ---\n")

    test_layernorm(256, 256, 256)
    test_layernorm(512, 512, 512)
    test_layernorm(1024, 1024, 1024)

    print("\n--- All layernorm examples completed ---")


if __name__ == "__main__":
    main()
