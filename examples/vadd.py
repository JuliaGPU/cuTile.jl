#!/usr/bin/env python3
"""
Vector Addition example - cuTile Python
"""

import cupy as cp
import numpy as np
import cuda.tile as ct

@ct.kernel
def vadd_cutile_kernel(a, b, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    tile_a = ct.load(a, index=(pid,), shape=(tile_size,))
    tile_b = ct.load(b, index=(pid,), shape=(tile_size,))
    result = tile_a + tile_b
    ct.store(c, index=(pid,), tile=result)


def run_vadd(*, size: int = 2**20, tile: int = 1024, dtype=np.float32, validate: bool = False):
    """Run vector addition. Returns (a, b, c) arrays for benchmarking."""
    a = cp.random.rand(size).astype(dtype)
    b = cp.random.rand(size).astype(dtype)
    c = cp.zeros(size, dtype=dtype)

    grid = (ct.cdiv(size, tile), 1, 1)
    stream = cp.cuda.get_current_stream()
    ct.launch(stream, grid, vadd_cutile_kernel, (a, b, c, tile))

    if validate:
        cp.cuda.runtime.deviceSynchronize()
        expected = cp.asnumpy(a) + cp.asnumpy(b)
        assert np.allclose(cp.asnumpy(c), expected), "vadd incorrect!"

    return a, b, c


def test_vadd(size, tile, dtype=np.float32, name=None):
    """Test vector addition with given parameters."""
    name = name or f"vadd size={size}, tile={tile}, dtype={dtype.__name__}"
    print(f"--- {name} ---")
    run_vadd(size=size, tile=tile, dtype=dtype, validate=True)
    print("  passed")


def main():
    print("--- cuTile Vector Addition Examples ---\n")

    test_vadd(1_024_000, 1024)
    test_vadd(2**20, 512)
    test_vadd(2**20, 1024)

    print("\n--- All vadd examples completed ---")


if __name__ == "__main__":
    main()
