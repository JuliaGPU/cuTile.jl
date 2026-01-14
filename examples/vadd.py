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


#=============================================================================
# prepare/run/verify pattern
#=============================================================================

def vadd_prepare(*, n: int, dtype=np.float32):
    """Allocate and initialize data for vector addition."""
    return {
        "a": cp.random.rand(n).astype(dtype),
        "b": cp.random.rand(n).astype(dtype),
        "c": cp.zeros(n, dtype=dtype),
        "n": n
    }


def vadd_run(data, *, tile: int, nruns: int = 1, warmup: int = 0):
    """Run vector addition kernel with timing."""
    a, b, c = data["a"], data["b"], data["c"]
    n = data["n"]

    grid = (ct.cdiv(n, tile), 1, 1)
    stream = cp.cuda.get_current_stream()

    # Warmup
    for _ in range(warmup):
        ct.launch(stream, grid, vadd_cutile_kernel, (a, b, c, tile))
    cp.cuda.runtime.deviceSynchronize()

    # Timed runs
    times = []
    for _ in range(nruns):
        start = cp.cuda.Event()
        end = cp.cuda.Event()
        start.record(stream)
        ct.launch(stream, grid, vadd_cutile_kernel, (a, b, c, tile))
        end.record(stream)
        end.synchronize()
        times.append(cp.cuda.get_elapsed_time(start, end))  # ms

    return {"c": c, "times": times}


def vadd_verify(data, result):
    """Verify vector addition results."""
    expected = cp.asnumpy(data["a"]) + cp.asnumpy(data["b"])
    assert np.allclose(cp.asnumpy(result["c"]), expected), "vadd incorrect!"


#=============================================================================
# Test function
#=============================================================================

def test_vadd(n, tile, dtype=np.float32, name=None):
    """Test vector addition with given parameters."""
    name = name or f"vadd size={n}, tile={tile}, dtype={dtype.__name__}"
    print(f"--- {name} ---")
    data = vadd_prepare(n=n, dtype=dtype)
    result = vadd_run(data, tile=tile)
    vadd_verify(data, result)
    print("  passed")


def main():
    print("--- cuTile Vector Addition Examples ---\n")

    test_vadd(1_024_000, 1024)
    test_vadd(2**20, 512)
    test_vadd(2**20, 1024)

    print("\n--- All vadd examples completed ---")


if __name__ == "__main__":
    main()
