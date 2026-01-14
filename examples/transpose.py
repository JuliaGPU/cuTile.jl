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


#=============================================================================
# Example harness
#=============================================================================

def transpose_prepare(*, M: int, N: int, dtype=np.float32):
    """Allocate and initialize data for transpose."""
    return {
        "input": cp.random.rand(M, N).astype(dtype),
        "output": cp.zeros((N, M), dtype=dtype),
        "M": M,
        "N": N
    }


def transpose_run(data, *, tile_m: int, tile_n: int, nruns: int = 1, warmup: int = 0):
    """Run transpose kernel with timing."""
    input_arr = data["input"]
    output_arr = data["output"]
    M, N = data["M"], data["N"]

    grid = (ct.cdiv(M, tile_m), ct.cdiv(N, tile_n), 1)
    stream = cp.cuda.get_current_stream()

    # Warmup
    for _ in range(warmup):
        ct.launch(stream, grid, transpose_cutile_kernel, (input_arr, output_arr, tile_m, tile_n))
    cp.cuda.runtime.deviceSynchronize()

    # Timed runs
    times = []
    for _ in range(nruns):
        start = cp.cuda.Event()
        end = cp.cuda.Event()
        start.record(stream)
        ct.launch(stream, grid, transpose_cutile_kernel, (input_arr, output_arr, tile_m, tile_n))
        end.record(stream)
        end.synchronize()
        times.append(cp.cuda.get_elapsed_time(start, end))  # ms

    return {"output": output_arr, "times": times}


def transpose_verify(data, result):
    """Verify transpose results."""
    expected = cp.asnumpy(data["input"]).T
    assert np.allclose(cp.asnumpy(result["output"]), expected), "transpose incorrect!"


#=============================================================================
# Main
#=============================================================================

def test_transpose(M, N, tile_m, tile_n, dtype=np.float32, name=None):
    """Test transpose with given parameters."""
    name = name or f"transpose ({M}x{N}), tiles={tile_m}x{tile_n}, dtype={dtype.__name__}"
    print(f"--- {name} ---")
    data = transpose_prepare(M=M, N=N, dtype=dtype)
    result = transpose_run(data, tile_m=tile_m, tile_n=tile_n)
    transpose_verify(data, result)
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
