import cuTile.Experimental: autotune_launch, clear_autotune_cache
using cuTile.Experimental: AbstractSearchSpace, CartesianSpace, FixedSpace

using Random

const AUTOTUNE_LOCK = ReentrantLock()
const AUTOTUNE_CACHE = Dict{Any, Dict{Any, Any}}()

struct VerificationError <: Exception
    msg::String
end

const TUNING_PRESETS = (
    fast     = (warmup=1, reps=3,  refine_topk=0, refine_reps=2),
    default  = (warmup=2, reps=5,  refine_topk=2, refine_reps=4),
    thorough = (warmup=2, reps=7, refine_topk=4, refine_reps=6),
)

function normalize_tuning(tuning::NamedTuple)
    preset = get(tuning, :preset, :default)
    preset isa Symbol || throw(ArgumentError("tuning.preset must be a Symbol"))
    hasproperty(TUNING_PRESETS, preset) ||
        throw(ArgumentError("Unknown preset `$preset`; use :fast, :default, or :thorough"))

    base = merge(getproperty(TUNING_PRESETS, preset),
        (seed=nothing, force=false, precompile_workers=Threads.nthreads()))

    # Apply user overrides (excluding :preset)
    overrides = NamedTuple(k => v for (k, v) in pairs(tuning) if k !== :preset)
    return merge(base, overrides)
end

# Extract hint fields (occupancy, num_ctas) from a config for launch kwargs
function hints_from_cfg(cfg)
    n = hasproperty(cfg, :num_ctas) ? cfg.num_ctas : nothing
    o = hasproperty(cfg, :occupancy) ? cfg.occupancy : nothing
    return (num_ctas=n, occupancy=o)
end

function time_ms(run_once::Function, get_args::Function;
                 warmup::Int, reps::Int, verify::Union{Nothing, Function}=nothing)
    CUDA.synchronize()
    for _ in 1:max(warmup, verify !== nothing ? 1 : 0)
        run_once(get_args())
    end

    if verify !== nothing
        CUDA.synchronize()
        verify() || throw(VerificationError("config produced incorrect output"))
    end

    best_ms = Inf32
    for _ in 1:reps
        args = get_args()
        CUDA.synchronize()
        elapsed_s = CUDA.@elapsed run_once(args)
        CUDA.synchronize()
        best_ms = min(best_ms, Float32(elapsed_s * 1000))
    end
    return best_ms
end

function eval_cfg(@nospecialize(f), cfg, grid_fn::Function, args_fn::Function;
                  sm_arch::String, opt_level::Int, warmup::Int, reps::Int,
                  verify::Union{Nothing, Function}=nothing)
    run_once = args -> cuTile.launch(f, grid_fn(cfg), args...;
        sm_arch, opt_level, hints_from_cfg(cfg)...)
    return time_ms(run_once, () -> args_fn(cfg); warmup, reps, verify)
end

function precompile_cfg(@nospecialize(f), cfg, grid_fn::Function, args_fn::Function;
                        sm_arch::String, opt_level::Int)
    grid_fn(cfg)
    args = args_fn(cfg)
    tile_args = map(to_tile_arg, args)

    # Mirror launch's Constant handling
    unwrapped_types = map(tile_args) do arg
        arg isa Constant ? constant_eltype(typeof(arg)) : typeof(arg)
    end
    argtypes = Tuple{unwrapped_types...}

    world = Base.get_world_counter()
    mi = method_instance(f, argtypes; world)
    mi === nothing && throw(MethodError(f, argtypes))

    has_consts = any(x -> x isa Constant, tile_args)
    const_argtypes = if has_consts
        cats = Any[CC.Const(f)]
        for arg in tile_args
            push!(cats, arg isa Constant ? CC.Const(arg[]) : typeof(arg))
        end
        cats
    else
        nothing
    end

    hints = hints_from_cfg(cfg)
    opts = (sm_arch=sm_arch, opt_level=opt_level, num_ctas=hints.num_ctas, occupancy=hints.occupancy)
    cache = CacheView{CuTileResults}((:cuTile, opts), world)
    emit_function(cache, mi; const_argtypes)
end

function precompile_candidates(@nospecialize(f), configs::Vector{Any},
                               grid_fn::Function, args_fn::Function;
                               sm_arch::String, opt_level::Int, workers::Int)
    isempty(configs) && return configs, nothing
    iszero(workers) && return configs, nothing

    workers = min(workers, Threads.nthreads(), length(configs))
    compiled = fill(true, length(configs))
    errors = Vector{Any}(nothing, length(configs))
    sem = Base.Semaphore(workers)

    @sync for (i, cfg) in enumerate(configs)
        Threads.@spawn Base.acquire(sem) do
            try
                precompile_cfg(f, cfg, grid_fn, args_fn; sm_arch, opt_level)
            catch err
                compiled[i] = false
                errors[i] = (cfg, err)
            end
        end
    end

    first_err = nothing
    for e in errors
        if e !== nothing
            first_err = e
            break
        end
    end

    return configs[compiled], first_err
end

function measure_candidates(@nospecialize(f), configs::Vector{Any},
                            grid_fn::Function, args_fn::Function;
                            sm_arch::String, opt_level::Int, warmup::Int, reps::Int,
                            verify::Union{Nothing, Function}=nothing)
    record = Tuple{Any, Float32}[]
    first_error = nothing
    for cfg in configs
        ms = try
            eval_cfg(f, cfg, grid_fn, args_fn; sm_arch, opt_level, warmup, reps, verify)
        catch err
            err isa VerificationError && @warn "Config $cfg failed verification, skipping"
            first_error === nothing && (first_error = (cfg, err))
            continue
        end
        push!(record, (cfg, ms))
    end
    return record, first_error
end

function find_or_tune(@nospecialize(f), space::AbstractSearchSpace, rng::AbstractRNG,
                      grid_fn::Function, args_fn::Function, tuning;
                      sm_arch::String, opt_level::Int, kernel_key, arg_key,
                      verify::Union{Nothing, Function}=nothing)
    if !tuning.force
        entry = lock(AUTOTUNE_LOCK) do
            per_kernel = get(AUTOTUNE_CACHE, kernel_key, nothing)
            per_kernel !== nothing ? get(per_kernel, arg_key, nothing) : nothing
        end
        entry !== nothing && return entry, true
    end

    checker = verify !== nothing ? verify() : nothing

    trials = collect(space)

    trials = Any[trials...]
    trials, precompile_error = precompile_candidates(f, trials, grid_fn, args_fn;
        sm_arch, opt_level, workers=tuning.precompile_workers)

    record, first_error = measure_candidates(f, trials, grid_fn, args_fn;
        sm_arch, opt_level, warmup=tuning.warmup, reps=tuning.reps, verify=checker)

    if isempty(record)
        # Prefer showing the precompile error (more informative) over the benchmark error
        err_info = first_error !== nothing ? first_error : precompile_error
        if err_info === nothing
            throw(ArgumentError("No valid config found in search space"))
        else
            cfg, err = err_info
            throw(ArgumentError(
                "No valid config found. First failure for cfg=$cfg: $(sprint(showerror, err))"))
        end
    end

    # Refinement: re-benchmark top K with more reps to stabilize the winner
    if tuning.refine_topk > 0 && length(record) > 1
        sort!(record, by=last)
        top_configs = Any[first(r) for r in record[1:min(tuning.refine_topk, length(record))]]
        refined, _ = measure_candidates(f, top_configs, grid_fn, args_fn;
            sm_arch, opt_level, warmup=tuning.warmup, reps=tuning.refine_reps)
        if !isempty(refined)
            record = refined
        end
    end

    _, best_idx = findmin(last, record)
    candidate = (; best_config=record[best_idx][1], tuning_record=record)

    return lock(AUTOTUNE_LOCK) do
        per_kernel = get!(Dict{Any,Any}, AUTOTUNE_CACHE, kernel_key)
        if !tuning.force && haskey(per_kernel, arg_key)
            per_kernel[arg_key], true
        else
            per_kernel[arg_key] = candidate
            candidate, false
        end
    end
end

function autotune_launch(@nospecialize(f), space::AbstractSearchSpace,
                         grid_fn::Function, args_fn::Function;
                         key=nothing,
                         key_fn::Union{Nothing, Function}=nothing,
                         launch_args_fn::Union{Nothing, Function}=nothing,
                         verify::Union{Nothing, Function}=nothing,
                         tuning::NamedTuple=NamedTuple(),
                         sm_arch::String=default_sm_arch(),
                         opt_level::Int=3)
    tuning = normalize_tuning(tuning)
    rng = tuning.seed !== nothing ? MersenneTwister(tuning.seed) : Random.default_rng()

    kernel_key = (f, sm_arch, opt_level)
    arg_key = key !== nothing ? key : (key_fn !== nothing ? key_fn() : nothing)

    entry, cache_hit = find_or_tune(f, space, rng, grid_fn, args_fn, tuning;
        sm_arch, opt_level, kernel_key, arg_key, verify)

    cfg = entry.best_config
    grid = grid_fn(cfg)
    args = launch_args_fn !== nothing ? launch_args_fn(cfg) : args_fn(cfg)

    cuTile.launch(f, grid, args...; sm_arch, opt_level, hints_from_cfg(cfg)...)

    return (; tuned_config=cfg, grid, tuning_record=copy(entry.tuning_record), cache_hit)
end

# Convenience: accept plain Vector (→ FixedSpace) or NamedTuple (→ CartesianSpace)
function autotune_launch(@nospecialize(f), configs, grid_fn::Function, args_fn::Function; kwargs...)
    space = configs isa NamedTuple ? CartesianSpace(configs) : FixedSpace(configs)
    return autotune_launch(f, space, grid_fn, args_fn; kwargs...)
end

function clear_autotune_cache(; kernel=nothing, key=nothing)
    lock(AUTOTUNE_LOCK) do
        if kernel === nothing
            key === nothing || throw(ArgumentError("`key` requires `kernel`"))
            empty!(AUTOTUNE_CACHE)
            return nothing
        end

        for kernel_key in collect(keys(AUTOTUNE_CACHE))
            kernel_key isa Tuple || continue
            kernel_key[1] === kernel || continue
            per_kernel = AUTOTUNE_CACHE[kernel_key]
            key === nothing ? empty!(per_kernel) : pop!(per_kernel, key, nothing)
            isempty(per_kernel) && delete!(AUTOTUNE_CACHE, kernel_key)
        end
    end
    return nothing
end
