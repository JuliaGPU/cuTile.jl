# EXCLUDE FROM TESTING
#
# Generic benchmark runner for cuTile.jl examples
# Discovers and benchmarks all examples in the examples/ directory

using CUDA

#=============================================================================
 Configuration
=============================================================================#

const NRUNS = 20
const WARMUP = 5

#=============================================================================
 Benchmark Utilities
=============================================================================#

struct BenchmarkResult
    name::String
    min_ms::Float64
    mean_ms::Float64
    throughput::String  # e.g. "841 GB/s" or "43.1 TFLOPS" or ""
end

function format_throughput(total, unit::String, time_ms::Float64)
    if unit == "GB/s"
        gbps = total / (time_ms / 1000) / 1e9
        return "$(round(Int, gbps)) GB/s"
    elseif unit == "TFLOPS"
        tflops = total / (time_ms / 1000) / 1e12
        return "$(round(tflops, digits=1)) TFLOPS"
    elseif unit == "μs"
        return "$(round(Int, time_ms * 1000)) μs"
    else
        return ""
    end
end

function print_table(title::String, results::Vector{BenchmarkResult})
    println()
    println("=" ^ 72)
    println("  ", title)
    println("=" ^ 72)
    has_throughput = any(r -> !isempty(r.throughput), results)
    if has_throughput
        println(rpad("Implementation", 20), rpad("Min (ms)", 12), rpad("Mean (ms)", 12), "Throughput")
    else
        println(rpad("Implementation", 20), rpad("Min (ms)", 12), "Mean (ms)")
    end
    println("-" ^ 72)
    for r in results
        if has_throughput
            println(rpad(r.name, 20), rpad(round(r.min_ms, digits=3), 12),
                    rpad(round(r.mean_ms, digits=3), 12), r.throughput)
        else
            println(rpad(r.name, 20), rpad(round(r.min_ms, digits=3), 12),
                    round(r.mean_ms, digits=3))
        end
    end
    println("-" ^ 72)
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
    data = @invokelatest mod.prepare(; benchmark=true)

    # Get metric info if available
    metric_total, metric_unit = 0, ""
    if isdefined(mod, :metric)
        metric_total, metric_unit = @invokelatest mod.metric(data)
    end

    # Run cuTile
    result = @invokelatest mod.run(data; nruns=NRUNS, warmup=WARMUP)

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
        others = @invokelatest mod.run_others(data; nruns=NRUNS, warmup=WARMUP)
        merge!(results, others)
    end

    return results, metric_total, metric_unit
end

#=============================================================================
 Main
=============================================================================#

function main()
    println("=" ^ 72)
    println("  cuTile.jl Benchmarks")
    println("=" ^ 72)
    println()
    println("Configuration:")
    println("  Runs: $NRUNS (+ $WARMUP warmup)")
    println("  GPU: ", CUDA.name(CUDA.device()))

    for name in discover_benchmarks()
        println("\nBenchmarking $name...")

        ret = run_benchmark(name)
        if ret === nothing
            println("  (skipped - no prepare/run functions)")
            continue
        end

        results, metric_total, metric_unit = ret

        # Convert to BenchmarkResult for printing
        benchmark_results = BenchmarkResult[]
        for (impl_name, times) in results
            min_t = minimum(times)
            mean_t = sum(times) / length(times)
            tp = !isempty(metric_unit) ? format_throughput(metric_total, metric_unit, min_t) : ""
            push!(benchmark_results, BenchmarkResult(impl_name, min_t, mean_t, tp))
        end

        # Sort by min time
        sort!(benchmark_results, by=r -> r.min_ms)

        print_table(name, benchmark_results)
    end

    println()
    println("=" ^ 72)
    println("  Benchmark Complete")
    println("=" ^ 72)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
