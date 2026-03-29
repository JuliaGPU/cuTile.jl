#!/usr/bin/env python3
# EXCLUDE FROM TESTING
#
# Generic benchmark runner for cuTile Python examples
# Discovers and benchmarks all examples in the examples/ directory

import os
import importlib.util
import cupy as cp

#=============================================================================
# Configuration
#=============================================================================

NRUNS = 20
WARMUP = 5

#=============================================================================
# Benchmark Utilities
#=============================================================================

class BenchmarkResult:
    def __init__(self, name: str, min_ms: float, mean_ms: float, throughput: str = ""):
        self.name = name
        self.min_ms = min_ms
        self.mean_ms = mean_ms
        self.throughput = throughput


def format_throughput(total, unit: str, time_ms: float) -> str:
    if unit == "GB/s":
        gbps = total / (time_ms / 1000) / 1e9
        return f"{gbps:.0f} GB/s"
    elif unit == "TFLOPS":
        tflops = total / (time_ms / 1000) / 1e12
        return f"{tflops:.1f} TFLOPS"
    elif unit == "μs":
        return f"{time_ms * 1000:.0f} μs"
    else:
        return ""


def print_table(title: str, results: list):
    """Print formatted benchmark results table."""
    print()
    print("=" * 72)
    print(f"  {title}")
    print("=" * 72)
    has_throughput = any(r.throughput for r in results)
    if has_throughput:
        print(f"{'Implementation':<20}{'Min (ms)':<12}{'Mean (ms)':<12}Throughput")
    else:
        print(f"{'Implementation':<20}{'Min (ms)':<12}Mean (ms)")
    print("-" * 72)
    for r in results:
        if has_throughput:
            print(f"{r.name:<20}{r.min_ms:<12.3f}{r.mean_ms:<12.3f}{r.throughput}")
        else:
            print(f"{r.name:<20}{r.min_ms:<12.3f}{r.mean_ms:.3f}")
    print("-" * 72)


#=============================================================================
# Benchmark Discovery & Execution
#=============================================================================

def discover_benchmarks():
    """Discover all benchmark-enabled examples in the examples directory."""
    examples = []
    examples_dir = os.path.dirname(__file__)
    for file in sorted(os.listdir(examples_dir)):
        if not file.endswith(".py"):
            continue
        if file == "benchmarks.py":
            continue
        name = file.replace(".py", "")
        examples.append(name)
    return examples


def run_benchmark(name: str):
    """Load and run benchmark for a given example."""
    examples_dir = os.path.dirname(__file__)
    file_path = os.path.join(examples_dir, f"{name}.py")

    # Import module dynamically
    spec = importlib.util.spec_from_file_location(name, file_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # Check required functions exist (unprefixed)
    prepare_fn = getattr(mod, "prepare", None)
    run_fn = getattr(mod, "run", None)
    if not prepare_fn or not run_fn:
        return None

    # Prepare data with benchmark=True for larger sizes
    data = prepare_fn(benchmark=True)

    # Get metric info if available
    metric_fn = getattr(mod, "metric", None)
    metric_total, metric_unit = (0, "") if not metric_fn else metric_fn(data)

    # Run cuTile
    result = run_fn(data, nruns=NRUNS, warmup=WARMUP)

    # Extract times (handle times_fwd/times_bwd for layernorm)
    if "times" in result:
        results = {"cuTile": result["times"]}
    elif "times_fwd" in result:
        results = {
            "cuTile Fwd": result["times_fwd"],
            "cuTile Bwd": result["times_bwd"]
        }
    else:
        return None

    # Run others if available
    run_others_fn = getattr(mod, "run_others", None)
    if run_others_fn:
        others = run_others_fn(data, nruns=NRUNS, warmup=WARMUP)
        results.update(others)

    return results, metric_total, metric_unit


#=============================================================================
# Main
#=============================================================================

def main():
    import torch  # For GPU name

    print("=" * 72)
    print("  cuTile Python Benchmarks")
    print("=" * 72)
    print()
    print("Configuration:")
    print(f"  Runs: {NRUNS} (+ {WARMUP} warmup)")
    print(f"  GPU: {torch.cuda.get_device_name()}")

    for name in discover_benchmarks():
        print(f"\nBenchmarking {name}...")

        ret = run_benchmark(name)
        if ret is None:
            print("  (skipped - no prepare/run functions)")
            continue

        results, metric_total, metric_unit = ret

        # Convert to BenchmarkResult for printing
        benchmark_results = []
        for impl_name, times in results.items():
            min_t = min(times)
            mean_t = sum(times) / len(times)
            tp = format_throughput(metric_total, metric_unit, min_t) if metric_unit else ""
            benchmark_results.append(BenchmarkResult(impl_name, min_t, mean_t, tp))

        # Sort by min time
        benchmark_results.sort(key=lambda r: r.min_ms)

        print_table(name, benchmark_results)

    print()
    print("=" * 72)
    print("  Benchmark Complete")
    print("=" * 72)


if __name__ == "__main__":
    main()
