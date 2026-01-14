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

NRUNS = 10
WARMUP = 3

#=============================================================================
# Benchmark Utilities
#=============================================================================

class BenchmarkResult:
    def __init__(self, name: str, min_ms: float, mean_ms: float):
        self.name = name
        self.min_ms = min_ms
        self.mean_ms = mean_ms


def print_table(title: str, results: list):
    """Print formatted benchmark results table."""
    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)
    print(f"{'Implementation':<20}{'Min (ms)':<12}Mean (ms)")
    print("-" * 60)
    for r in results:
        print(f"{r.name:<20}{r.min_ms:<12.3f}{r.mean_ms:.3f}")
    print("-" * 60)


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

    return results


#=============================================================================
# Main
#=============================================================================

def main():
    import torch  # For GPU name

    print("=" * 60)
    print("  cuTile Python Benchmarks")
    print("=" * 60)
    print()
    print("Configuration:")
    print(f"  Runs: {NRUNS} (+ {WARMUP} warmup)")
    print(f"  GPU: {torch.cuda.get_device_name()}")

    for name in discover_benchmarks():
        print(f"\nBenchmarking {name}...")

        results = run_benchmark(name)
        if results is None:
            print("  (skipped - no prepare/run functions)")
            continue

        # Convert to BenchmarkResult for printing
        benchmark_results = []
        for impl_name, times in results.items():
            min_t = min(times)
            mean_t = sum(times) / len(times)
            benchmark_results.append(BenchmarkResult(impl_name, min_t, mean_t))

        # Sort by min time
        benchmark_results.sort(key=lambda r: r.min_ms)

        print_table(name, benchmark_results)

    print()
    print("=" * 60)
    print("  Benchmark Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
