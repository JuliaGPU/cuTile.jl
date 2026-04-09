#!/usr/bin/env python3
"""
Softmax example - cuTile Python
Two strategies: TMA (single-tile per row) and chunked (3-pass with gather/scatter).
"""

import math

import cupy as cp
import numpy as np
import cuda.tile as ct

#=============================================================================
# TMA Softmax Kernel (single-tile per row, persistent scheduling)
#=============================================================================

@ct.kernel(occupancy=2)
def softmax_tma_kernel(output, input, n_rows: ct.Constant[int], n_cols: ct.Constant[int],
                       TILE_SIZE: ct.Constant[int]):
    pid = ct.bid(0)
    num_programs = ct.num_blocks(0)

    for row_idx in range(pid, n_rows, num_programs):
        row = ct.load(input, index=(row_idx, 0), shape=(1, TILE_SIZE),
                      padding_mode=ct.PaddingMode.NEG_INF)
        row = ct.astype(row, ct.float32)

        row_max = ct.max(row, 1, keepdims=True)
        numerator = ct.exp(ct.sub(row, row_max))
        denominator = ct.sum(numerator, 1, keepdims=True)
        softmax_output = ct.truediv(numerator, denominator)

        softmax_output = ct.astype(softmax_output, input.dtype)
        ct.store(output, index=(row_idx, 0), tile=softmax_output)


#=============================================================================
# Chunked Softmax Kernel (3-pass with gather/scatter)
#=============================================================================

@ct.kernel(occupancy=4)
def softmax_chunked_kernel(output, input, n_rows: ct.Constant[int], n_cols: ct.Constant[int],
                           TILE_SIZE: ct.Constant[int]):
    pid = ct.bid(0)
    num_programs = ct.num_blocks(0)

    for row_idx in range(pid, n_rows, num_programs):
        num_chunks = (n_cols + TILE_SIZE - 1) // TILE_SIZE
        col_offsets_base = ct.arange(TILE_SIZE, dtype=ct.int32)

        row_max = ct.full((1,), -math.inf, dtype=ct.float32)
        denominator = ct.full((TILE_SIZE,), 0.0, dtype=ct.float32)

        # Pass 1: Find maximum
        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx * TILE_SIZE
            col_indices = ct.add(ct.full((TILE_SIZE,), chunk_start, dtype=ct.int32), col_offsets_base)
            chunk = ct.gather(input, (row_idx, col_indices), check_bounds=True, padding_value=-math.inf)
            chunk = ct.astype(chunk, ct.float32)
            chunk_max = ct.max(chunk, 0, keepdims=True)
            row_max = ct.maximum(row_max, chunk_max)

        # Pass 2: Compute denominator
        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx * TILE_SIZE
            col_indices = ct.add(ct.full((TILE_SIZE,), chunk_start, dtype=ct.int32), col_offsets_base)
            chunk = ct.gather(input, (row_idx, col_indices), check_bounds=True, padding_value=-math.inf)
            chunk = ct.astype(chunk, ct.float32)
            denominator = ct.add(denominator, ct.exp(ct.sub(chunk, row_max)))
        denom_sum = ct.sum(denominator, 0, keepdims=True)

        # Pass 3: Compute softmax and scatter
        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx * TILE_SIZE
            col_indices = ct.add(ct.full((TILE_SIZE,), chunk_start, dtype=ct.int32), col_offsets_base)
            chunk = ct.gather(input, (row_idx, col_indices), check_bounds=True, padding_value=-math.inf)
            chunk = ct.astype(chunk, ct.float32)
            softmax_output = ct.truediv(ct.exp(ct.sub(chunk, row_max)), denom_sum)
            softmax_output = ct.astype(softmax_output, input.dtype)
            ct.scatter(output, (row_idx, col_indices), softmax_output, check_bounds=True)


#=============================================================================
# Example harness
#=============================================================================

def next_power_of_2(n):
    if n <= 0:
        return 1
    p = 1
    while p < n:
        p <<= 1
    return p


def prepare(*, benchmark: bool = False, M: int = None, N: int = None, dtype=np.float32):
    if M is None:
        M = 4096 if benchmark else 256
    if N is None:
        N = 4096 if benchmark else 256
    input = cp.random.randn(M, N).astype(dtype)
    return {
        "input": input,
        "output_tma": cp.empty_like(input),
        "output_chunked": cp.empty_like(input),
        "M": M,
        "N": N,
    }


def run(data, *, tile_tma: int = None, tile_chunked: int = 1024, nruns: int = 1, warmup: int = 0):
    input = data["input"]
    output_tma = data["output_tma"]
    output_chunked = data["output_chunked"]
    M, N = data["M"], data["N"]

    if tile_tma is None:
        tile_tma = next_power_of_2(N)

    stream = cp.cuda.get_current_stream()
    NUM_SM = cp.cuda.runtime.getDeviceProperties(cp.cuda.runtime.getDevice())["multiProcessorCount"]

    def run_tma():
        num_programs = min(NUM_SM * 2, M)  # occupancy=2
        ct.launch(stream, (num_programs,), softmax_tma_kernel,
                  (output_tma, input, M, N, tile_tma))

    def run_chunked():
        num_programs = min(NUM_SM * 4, M)  # occupancy=4
        ct.launch(stream, (num_programs,), softmax_chunked_kernel,
                  (output_chunked, input, M, N, tile_chunked))

    # Warmup
    for _ in range(warmup):
        run_tma()
        run_chunked()
    cp.cuda.runtime.deviceSynchronize()

    # Timed TMA runs
    times_tma = []
    for _ in range(nruns):
        start = cp.cuda.Event()
        end = cp.cuda.Event()
        start.record(stream)
        run_tma()
        end.record(stream)
        end.synchronize()
        times_tma.append(cp.cuda.get_elapsed_time(start, end))

    # Timed chunked runs
    times_chunked = []
    for _ in range(nruns):
        start = cp.cuda.Event()
        end = cp.cuda.Event()
        start.record(stream)
        run_chunked()
        end.record(stream)
        end.synchronize()
        times_chunked.append(cp.cuda.get_elapsed_time(start, end))

    return {
        "output_tma": output_tma,
        "output_chunked": output_chunked,
        "times": {
            "cuTile TMA": times_tma,
            "cuTile Chunked": times_chunked,
        }
    }


def verify(data, result):
    x = cp.asnumpy(data["input"])
    M, N = data["M"], data["N"]
    # Reference softmax
    row_max = np.max(x, axis=1, keepdims=True)
    exps = np.exp(x - row_max)
    expected = exps / np.sum(exps, axis=1, keepdims=True)

    for label in ("output_tma", "output_chunked"):
        out = cp.asnumpy(result[label])
        assert np.allclose(out, expected, atol=1e-5, rtol=1e-4), \
            f"{label} mismatch! max diff: {np.max(np.abs(out - expected))}"


def metric(data):
    MN = data["M"] * data["N"] * 4  # sizeof(float32)
    return {
        # TMA: 1 read + 1 write
        "cuTile TMA": (2 * MN, "GB/s"),
        # Chunked: 3 reads (gather per pass) + 1 write (scatter)
        "cuTile Chunked": (4 * MN, "GB/s"),
    }


# No run_others for softmax - would need torch for a fair comparison


#=============================================================================
# Main
#=============================================================================

def test_softmax(M, N, tile_tma=None, tile_chunked=1024, dtype=np.float32, name=None):
    if tile_tma is None:
        tile_tma = next_power_of_2(N)
    name = name or f"softmax ({M}x{N}), tma_tile={tile_tma}, chunked_tile={tile_chunked}"
    print(f"--- {name} ---")
    data = prepare(M=M, N=N, dtype=dtype)
    result = run(data, tile_tma=tile_tma, tile_chunked=tile_chunked)
    verify(data, result)
    print("  tma passed, chunked passed")


def main():
    print("--- cuTile Softmax Examples ---\n")

    test_softmax(256, 256)
    test_softmax(1024, 1024)
    test_softmax(4096, 4096)

    print("\n--- All softmax examples completed ---")


if __name__ == "__main__":
    main()
