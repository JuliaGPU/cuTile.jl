import cuTile, CUDA
using ParallelTestRunner
using Pkg
using InteractiveUtils: versioninfo

CUDA.functional() || error("CUDA.jl is not functional; cuTile tests require a working GPU")

# Load helpers, cuTileTestRecord, and ParallelTestRunner overrides on the main
# process. The same file is re-`include`d on every worker via `init_worker_code`
# below; the record type must exist on both sides of the Malt boundary.
include(joinpath(@__DIR__, "setup.jl"))

# Forcibly precompile the current environment in parallel: Pkg sometimes ignores
# dependencies pointed through via `[sources]` (IRStructurizer, FileCheck, …).
Pkg.precompile()

@info "Julia information:\n" * sprint(io -> versioninfo(io))
@info "CUDA information:\n" * sprint(io -> CUDA.versioninfo(io))
@info "cuTile information:\n" * sprint(io -> cuTile.versioninfo(io))


## test discovery

testsuite = find_tests(@__DIR__)
delete!(testsuite, "setup")

# Add examples to the test suite (only on Julia 1.12+, where they're supported)
examples_root = joinpath(@__DIR__, "..", "examples")
if VERSION >= v"1.12"
    for (name, body) in find_tests(examples_root)
        path = joinpath(examples_root, name * ".jl")
        readline(path) == "# EXCLUDE FROM TESTING" && continue
        dir = dirname(path)
        testsuite["examples/$name"] = quote
            cd($dir) do
                redirect_stdout(devnull) do
                    $body
                    @eval main()
                end
            end
        end
    end
end


## GPU-memory-based parallelism

args = parse_args(ARGS)

# Pick the first visible device and use its free memory to cap worker count.
# (Set `CUDA_VISIBLE_DEVICES` to choose which device is used.)
first_gpu = first(CUDA.devices())
gpu_free = CUDA.device!(first_gpu) do
    mem = CUDA.free_memory()
    CUDA.device_reset!()
    mem
end
gpu_jobs = max(1, Int(gpu_free) ÷ (2 * 2^30))

if args.jobs === nothing
    default_jobs = min(ParallelTestRunner.default_njobs(), gpu_jobs)
    args = ParallelTestRunner.ParsedArgs(
        Some(default_jobs), args.verbose, args.quickfail, args.list,
        args.custom, args.positionals,
    )
end


## worker setup

const init_worker_code = quote
    include($(joinpath(@__DIR__, "setup.jl")))
end

const init_code = quote
    using CUDA
    using cuTile
    import cuTile as ct

    using FileCheck
end


## run

runtests(cuTile, args;
         testsuite, init_code, init_worker_code,
         RecordType = cuTileTestRecord)
