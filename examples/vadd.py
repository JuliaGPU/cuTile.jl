#!/usr/bin/env python3
"""
Vector Addition example - cuTile Python
"""

import cupy as cp
import numpy as np
import cuda.tile as ct

# 1D kernel
@ct.kernel
def vadd_kernel_1d(a, b, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    tile_a = ct.load(a, index=(pid,), shape=(tile_size,))
    tile_b = ct.load(b, index=(pid,), shape=(tile_size,))
    result = tile_a + tile_b
    ct.store(c, index=(pid,), tile=result)


# 2D kernel
@ct.kernel
def vadd_kernel_2d(a, b, c, tile_x: ct.Constant[int], tile_y: ct.Constant[int]):
    pid_x = ct.bid(0)
    pid_y = ct.bid(1)
    tile_a = ct.load(a, index=(pid_x, pid_y), shape=(tile_x, tile_y))
    tile_b = ct.load(b, index=(pid_x, pid_y), shape=(tile_x, tile_y))
    result = tile_a + tile_b
    ct.store(c, index=(pid_x, pid_y), tile=result)


# 1D kernel with gather/scatter
@ct.kernel
def vadd_kernel_1d_gather(a, b, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    # Create index tile for this block's elements
    offsets = ct.arange(tile_size, dtype=ct.int32)
    base = pid * tile_size
    indices = base + offsets

    # Gather, add, scatter
    tile_a = ct.gather(a, indices)
    tile_b = ct.gather(b, indices)
    result = tile_a + tile_b
    ct.scatter(c, indices, result)


#=============================================================================
# Example harness
#=============================================================================

def prepare(*, benchmark: bool = False, shape: tuple = None, use_gather: bool = False, dtype=np.float32):
    """Allocate and initialize data for vector addition."""
    if shape is None:
        shape = (2**27,) if benchmark else (1_024_000,)
    a = cp.random.rand(*shape).astype(dtype)
    return {
        "a": a,
        "b": cp.random.rand(*shape).astype(dtype),
        "c": cp.empty_like(a),
        "shape": shape,
        "use_gather": use_gather
    }


def run(data, *, tile=1024, nruns: int = 1, warmup: int = 0):
    """Run vector addition kernel with timing."""
    a, b, c = data["a"], data["b"], data["c"]
    shape = data["shape"]
    use_gather = data["use_gather"]

    stream = cp.cuda.get_current_stream()

    if len(shape) == 2:
        # 2D case
        m, n = shape
        tile_x, tile_y = tile if isinstance(tile, tuple) else (tile, tile)
        grid = (ct.cdiv(m, tile_x), ct.cdiv(n, tile_y), 1)

        def run_kernel():
            ct.launch(stream, grid, vadd_kernel_2d, (a, b, c, tile_x, tile_y))
    else:
        # 1D case
        n = shape[0]
        tile_val = tile[0] if isinstance(tile, tuple) else tile
        grid = (ct.cdiv(n, tile_val), 1, 1)

        if use_gather:
            def run_kernel():
                ct.launch(stream, grid, vadd_kernel_1d_gather, (a, b, c, tile_val))
        else:
            def run_kernel():
                ct.launch(stream, grid, vadd_kernel_1d, (a, b, c, tile_val))

    # Warmup
    for _ in range(warmup):
        run_kernel()
    cp.cuda.runtime.deviceSynchronize()

    # Timed runs
    times = []
    for _ in range(nruns):
        start = cp.cuda.Event()
        end = cp.cuda.Event()
        start.record(stream)
        run_kernel()
        end.record(stream)
        end.synchronize()
        times.append(cp.cuda.get_elapsed_time(start, end))  # ms

    return {"c": c, "times": times}


def verify(data, result):
    """Verify vector addition results."""
    expected = cp.asnumpy(data["a"]) + cp.asnumpy(data["b"])
    assert np.allclose(cp.asnumpy(result["c"]), expected), "vadd incorrect!"


#=============================================================================
# Reference implementations for benchmarking
#=============================================================================

def run_others(data, *, nruns: int = 1, warmup: int = 0):
    """Run reference implementations for comparison."""
    results = {}
    shape = data["shape"]

    if len(shape) == 1:
        a, b = data["a"], data["b"]
        c_cupy = cp.zeros_like(a)

        stream = cp.cuda.get_current_stream()

        # CuPy (broadcasting)
        for _ in range(warmup):
            cp.add(a, b, out=c_cupy)
        cp.cuda.runtime.deviceSynchronize()

        times_cupy = []
        for _ in range(nruns):
            start = cp.cuda.Event()
            end = cp.cuda.Event()
            start.record(stream)
            cp.add(a, b, out=c_cupy)
            end.record(stream)
            end.synchronize()
            times_cupy.append(cp.cuda.get_elapsed_time(start, end))
        results["CuPy"] = times_cupy

    return results


#=============================================================================
# Main
#=============================================================================

def test_vadd(shape, tile, use_gather=False, dtype=np.float32, name=None):
    """Test vector addition with given parameters."""
    if name is None:
        if len(shape) == 2:
            name = f"2D vadd ({shape[0]}x{shape[1]}), tile={tile}, dtype={dtype.__name__}"
        elif use_gather:
            name = f"1D vadd gather size={shape[0]}, tile={tile}, dtype={dtype.__name__}"
        else:
            name = f"1D vadd size={shape[0]}, tile={tile}, dtype={dtype.__name__}"
    print(f"--- {name} ---")
    data = prepare(shape=shape, use_gather=use_gather, dtype=dtype)
    result = run(data, tile=tile)
    verify(data, result)
    print("  passed")


def main():
    print("--- cuTile Vector Addition Examples ---\n")

    # 1D tests with float32
    test_vadd((1_024_000,), 1024)
    test_vadd((2**20,), 512)

    # 1D tests with float64
    test_vadd((2**18,), 512, dtype=np.float64)

    # 1D tests with float16
    test_vadd((1_024_000,), 1024, dtype=np.float16)

    # 2D tests with float32
    test_vadd((2048, 1024), (32, 32))
    test_vadd((1024, 2048), (64, 64))

    # 2D tests with float64
    test_vadd((1024, 512), (32, 32), dtype=np.float64)

    # 2D tests with float16
    test_vadd((1024, 1024), (64, 64), dtype=np.float16)

    # 1D gather/scatter tests
    test_vadd((1_024_000,), 1024, use_gather=True)
    test_vadd((2**20,), 512, use_gather=True)

    print("\n--- All vadd examples completed ---")


if __name__ == "__main__":
    main()
