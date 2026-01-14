# EXCLUDE FROM TESTING
#
# Generic benchmark runner for cuTile.jl examples
# Discovers and benchmarks all examples in the examples/ directory

using CUDA

#=============================================================================
 Configuration
=============================================================================#

const NRUNS = 10
const WARMUP = 3

#=============================================================================
 Benchmark Utilities
=============================================================================#

struct BenchmarkResult
    name::String
    min_ms::Float64
    mean_ms::Float64
end

function print_table(title::String, results::Vector{BenchmarkResult})
    println()
    println("=" ^ 60)
    println("  ", title)
    println("=" ^ 60)
    println(rpad("Implementation", 20), rpad("Min (ms)", 12), "Mean (ms)")
    println("-" ^ 60)
    for r in results
        println(rpad(r.name, 20), rpad(round(r.min_ms, digits=3), 12),
                round(r.mean_ms, digits=3))
    end
    println("-" ^ 60)
end

#=============================================================================
 Benchmark Discovery & Execution
=============================================================================#

function discover_benchmarks()
    examples = String[]
    for file in readdir(@__DIR__)
        endswith(file, ".jl") || continue
        file == "benchmarks.jl" && continue
        name = replace(file, ".jl" => "")
        push!(examples, name)
    end
    return sort(examples)
end

function run_benchmark(name::String)
    file = joinpath(@__DIR__, name * ".jl")

    # Include file in anonymous module to avoid polluting namespace
    mod = Module()
    Base.include(mod, file)

    # Check required functions exist (unprefixed)
    isdefined(mod, :prepare) || return nothing
    isdefined(mod, :run) || return nothing

    # Prepare data with benchmark=true for larger sizes
    data = mod.prepare(; benchmark=true)

    # Run cuTile
    result = mod.run(data; nruns=NRUNS, warmup=WARMUP)

    # Extract times (handle times_fwd/times_bwd for layernorm)
    if hasproperty(result, :times)
        results = Dict{String, Vector{Float64}}("cuTile" => result.times)
    elseif hasproperty(result, :times_fwd)
        results = Dict{String, Vector{Float64}}(
            "cuTile Fwd" => result.times_fwd,
            "cuTile Bwd" => result.times_bwd
        )
    else
        return nothing
    end

    # Run others if available
    if isdefined(mod, :run_others)
        others = mod.run_others(data; nruns=NRUNS, warmup=WARMUP)
        merge!(results, others)
    end

    return results
end

#=============================================================================
 Main
=============================================================================#

function main()
    println("=" ^ 60)
    println("  cuTile.jl Benchmarks")
    println("=" ^ 60)
    println()
    println("Configuration:")
    println("  Runs: $NRUNS (+ $WARMUP warmup)")
    println("  GPU: ", CUDA.name(CUDA.device()))

    for name in discover_benchmarks()
        println("\nBenchmarking $name...")

        results = run_benchmark(name)
        if results === nothing
            println("  (skipped - no prepare/run functions)")
            continue
        end

        # Convert to BenchmarkResult for printing
        benchmark_results = BenchmarkResult[]
        for (impl_name, times) in results
            min_t = minimum(times)
            mean_t = sum(times) / length(times)
            push!(benchmark_results, BenchmarkResult(impl_name, min_t, mean_t))
        end

        # Sort by min time
        sort!(benchmark_results, by=r -> r.min_ms)

        print_table(name, benchmark_results)
    end

    println()
    println("=" ^ 60)
    println("  Benchmark Complete")
    println("=" ^ 60)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
