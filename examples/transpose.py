#!/usr/bin/env python3
"""
Matrix Transpose example - cuTile Python
"""

import cupy as cp
import numpy as np
import cuda.tile as ct

@ct.kernel
def transpose_cutile_kernel(input, output, tile_m: ct.Constant[int], tile_n: ct.Constant[int]):
    pid_m = ct.bid(0)
    pid_n = ct.bid(1)
    tile = ct.load(input, index=(pid_m, pid_n), shape=(tile_m, tile_n))
    tile_t = ct.transpose(tile)
    ct.store(output, index=(pid_n, pid_m), tile=tile_t)


def run_transpose(*, M: int = 1024, N: int = 1024, tile_m: int = 64, tile_n: int = 64,
                  dtype=np.float32, input=None, output=None, validate: bool = False):
    """Run matrix transpose. Returns (input, output) arrays for benchmarking."""
    if input is None:
        input = cp.random.rand(M, N).astype(dtype)
    else:
        M, N = input.shape
    if output is None:
        output = cp.zeros((N, M), dtype=dtype)

    grid = (ct.cdiv(M, tile_m), ct.cdiv(N, tile_n), 1)
    stream = cp.cuda.get_current_stream()
    ct.launch(stream, grid, transpose_cutile_kernel, (input, output, tile_m, tile_n))

    if validate:
        cp.cuda.runtime.deviceSynchronize()
        expected = cp.asnumpy(input).T
        assert np.allclose(cp.asnumpy(output), expected), "transpose incorrect!"

    return input, output


def test_transpose(M, N, tile_m, tile_n, dtype=np.float32, name=None):
    """Test transpose with given parameters."""
    name = name or f"transpose ({M}x{N}), tiles={tile_m}x{tile_n}, dtype={dtype.__name__}"
    print(f"--- {name} ---")
    run_transpose(M=M, N=N, tile_m=tile_m, tile_n=tile_n, dtype=dtype, validate=True)
    print("  passed")


def main():
    print("--- cuTile Matrix Transpose Examples ---\n")

    test_transpose(256, 256, 32, 32)
    test_transpose(512, 512, 64, 64)
    test_transpose(256, 512, 32, 64)
    test_transpose(1024, 1024, 64, 64)

    print("\n--- All transpose examples completed ---")


if __name__ == "__main__":
    main()
