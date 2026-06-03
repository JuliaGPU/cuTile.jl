public autotune_launch, clear_autotune_cache

const AUTOTUNE_CACHE = Base.Lockable(Dict{Any, Dict{Any, Any}}())

struct VerificationError <: Exception
    msg::String
end

Base.showerror(io::IO, err::VerificationError) =
    print(io, "VerificationError: ", err.msg)

const TUNING_PRESETS = (
    fast     = (warmup=1, reps=3, refine_topk=0, refine_reps=2),
    default  = (warmup=2, reps=5, refine_topk=2, refine_reps=4),
    thorough = (warmup=2, reps=7, refine_topk=4, refine_reps=6),
)

struct TuningOptions
    warmup::Int
    reps::Int
    refine_topk::Int
    refine_reps::Int
    seed::Union{Nothing, Int}
    force::Bool
    precompile_workers::Int
end

_tuning_defaults() = (
    seed=nothing,
    force=false,
    precompile_workers=Threads.nthreads(),
)

# Lower bound for each count field; the others have no minimum.
const _TUNING_MINIMA = (warmup=0, reps=1, refine_topk=0, refine_reps=1,
                        precompile_workers=0)

function normalize_tuning(tuning::NamedTuple)
    valid_keys = (:preset, fieldnames(TuningOptions)...)
    unknown = setdiff(keys(tuning), valid_keys)
    isempty(unknown) ||
        throw(ArgumentError("Unknown tuning option(s): $(join(unknown, ", "))"))

    preset = get(tuning, :preset, :default)
    preset isa Symbol && hasproperty(TUNING_PRESETS, preset) ||
        throw(ArgumentError("Unknown tuning preset `$preset`; use :fast, :default, or :thorough"))

    overrides = NamedTuple(k => v for (k, v) in pairs(tuning) if k !== :preset)
    values = merge(_tuning_defaults(), getproperty(TUNING_PRESETS, preset), overrides)

    # The struct's field types coerce/reject bad value types; we only enforce
    # the lower bounds that the types can't. Pull fields by name since `values`
    # is in merge order, not struct-field order.
    opts = TuningOptions((getproperty(values, f) for f in fieldnames(TuningOptions))...)
    for (name, lo) in pairs(_TUNING_MINIMA)
        getfield(opts, name) >= lo ||
            throw(ArgumentError("tuning.$name must be >= $lo, got $(getfield(opts, name))"))
    end
    return opts
end

@inline _hint_from_cfg(cfg, name::Symbol, fallback) =
    hasproperty(cfg, name) ? getproperty(cfg, name) : fallback

function hints_from_cfg(cfg; static_num_ctas=nothing, static_occupancy=nothing)
    return (
        num_ctas=_hint_from_cfg(cfg, :num_ctas, static_num_ctas),
        occupancy=_hint_from_cfg(cfg, :occupancy, static_occupancy),
    )
end

function _check_static_hint_conflicts(configs; static_num_ctas=nothing,
                                      static_occupancy=nothing)
    statics = (num_ctas=static_num_ctas, occupancy=static_occupancy)
    for (hint, static_value) in pairs(statics)
        static_value === nothing && continue
        any(cfg -> hasproperty(cfg, hint), configs) && throw(ArgumentError(
            "`$hint` is both a static kwarg and an axis in the search space. Pick one."))
    end
    return nothing
end

function _collect_trials(space::AbstractSearchSpace, seed)
    trials = Any[cfg for cfg in space]
    if seed !== nothing && length(trials) > 1
        shuffle!(MersenneTwister(seed), trials)
    end
    return trials
end

@inline _grid_dims(grid) = grid isa Integer ? (grid,) : grid
@inline _converted_args(args_fn, cfg) = map(cuTileconvert, args_fn(cfg))
@inline _argtypes(args) = Tuple{map(Core.Typeof, args)...}

function _compile_cfg(@nospecialize(f), cfg, args_fn;
                      sm_arch::VersionNumber, opt_level::Int,
                      static_num_ctas=nothing, static_occupancy=nothing)
    converted = _converted_args(args_fn, cfg)
    return temporary_cufunction(f, _argtypes(converted);
        sm_arch, opt_level,
        hints_from_cfg(cfg; static_num_ctas, static_occupancy)...)
end

function _time_ms(run_once, get_args;
                  warmup::Int, reps::Int, verify=nothing, reset=nothing)
    CUDACore.synchronize()
    for _ in 1:max(warmup, verify === nothing ? 0 : 1)
        reset !== nothing && reset()
        run_once(get_args())
    end

    if verify !== nothing
        CUDACore.synchronize()
        verify() || throw(VerificationError("config produced incorrect output"))
    end

    best_ms = Inf32
    for _ in 1:reps
        reset !== nothing && reset()
        args = get_args()
        CUDACore.synchronize()
        elapsed_s = CUDACore.@elapsed run_once(args)
        CUDACore.synchronize()
        best_ms = min(best_ms, Float32(elapsed_s * 1000))
    end
    return best_ms
end

function eval_cfg(@nospecialize(f), cfg, grid_fn, args_fn;
                  sm_arch::VersionNumber, opt_level::Int, warmup::Int, reps::Int,
                  static_num_ctas=nothing, static_occupancy=nothing,
                  verify=nothing, reset=nothing)
    grid = _grid_dims(grid_fn(cfg))
    kernel = _compile_cfg(f, cfg, args_fn; sm_arch, opt_level,
                          static_num_ctas, static_occupancy)

    run_once = converted -> kernel(converted...; blocks=grid)
    get_args = () -> _converted_args(args_fn, cfg)
    return _time_ms(run_once, get_args; warmup, reps, verify, reset)
end

function precompile_cfg(@nospecialize(f), cfg, args_fn;
                        sm_arch::VersionNumber, opt_level::Int,
                        static_num_ctas=nothing, static_occupancy=nothing)
    _compile_cfg(f, cfg, args_fn; sm_arch, opt_level,
                 static_num_ctas, static_occupancy)
    return nothing
end

const TimingRecord = Tuple{Any, Float32}

function _measure_cfg!(record::Vector{TimingRecord}, first_error::Base.RefValue,
                       @nospecialize(f), cfg, grid_fn, args_fn; kwargs...)
    ms = try
        eval_cfg(f, cfg, grid_fn, args_fn; kwargs...)
    catch err
        err isa InterruptException && rethrow()
        if err isa VerificationError
            @warn "Config failed verification; skipping" cfg
        else
            bt = catch_backtrace()
            @debug "Config failed during autotuning; skipping" cfg exception=(err, bt)
        end
        first_error[] === nothing && (first_error[] = (cfg, err))
        return nothing
    end
    push!(record, (cfg, ms))
    return nothing
end

function measure_candidates(@nospecialize(f), configs::Vector{Any}, grid_fn, args_fn;
                            sm_arch::VersionNumber, opt_level::Int,
                            warmup::Int, reps::Int,
                            static_num_ctas=nothing, static_occupancy=nothing,
                            verify=nothing, reset=nothing)
    record = TimingRecord[]
    first_error = Ref{Any}(nothing)
    for cfg in configs
        _measure_cfg!(record, first_error, f, cfg, grid_fn, args_fn;
                      sm_arch, opt_level, warmup, reps,
                      static_num_ctas, static_occupancy, verify, reset)
    end
    return record, first_error[]
end

"""
    pipelined_tune(f, configs, grid_fn, args_fn; ...) -> (record, precompile_error, first_error)

Compile candidate configurations on worker tasks while the caller task measures
completed candidates. Measurement stays on the caller task because CUDA state is
task-local; workers only run the untimed temporary compile path.
"""
function pipelined_tune(@nospecialize(f), configs::Vector{Any}, grid_fn, args_fn;
                        sm_arch::VersionNumber, opt_level::Int,
                        warmup::Int, reps::Int, workers::Int,
                        static_num_ctas=nothing, static_occupancy=nothing,
                        verify=nothing, reset=nothing)
    isempty(configs) && return TimingRecord[], nothing, nothing

    if iszero(workers) || length(configs) == 1
        record, first_error = measure_candidates(f, configs, grid_fn, args_fn;
            sm_arch, opt_level, warmup, reps,
            static_num_ctas, static_occupancy, verify, reset)
        return record, nothing, first_error
    end

    workers = min(workers, Threads.nthreads(), length(configs))
    ready = Channel{Any}(length(configs))
    jobs = Channel{Any}(length(configs))
    foreach(cfg -> put!(jobs, cfg), configs)
    close(jobs)
    
    cancelled = Threads.Atomic{Bool}(false)
    precompile_error = Ref{Any}(nothing)
    error_lock = ReentrantLock()    
    ctx = CUDACore.context()

    producer = Threads.@spawn try
        @sync for _ in 1:workers
            Threads.@spawn begin
                CUDACore.context!(ctx)
                for cfg in jobs
                    cancelled[] && break
                    try
                        precompile_cfg(f, cfg, args_fn; sm_arch, opt_level,
                                        static_num_ctas, static_occupancy)
                        cancelled[] || put!(ready, cfg)
                    catch err
                        if err isa InterruptException
                            cancelled[] = true
                            rethrow()
                        end
                        lock(error_lock) do
                            precompile_error[] === nothing &&
                                (precompile_error[] = (cfg, err))
                        end
                    end
                end
            end
        end
    finally
        close(ready)
    end

    record = TimingRecord[]
    first_error = Ref{Any}(nothing)
    try
        for cfg in ready
            _measure_cfg!(record, first_error, f, cfg, grid_fn, args_fn;
                          sm_arch, opt_level, warmup, reps,
                          static_num_ctas, static_occupancy, verify, reset)
        end
        wait(producer)
    catch
        cancelled[] = true
        while isready(ready)
            take!(ready)
        end
        try
            wait(producer)
        catch
        end
        rethrow()
    end

    return record, precompile_error[], first_error[]
end

@inline _entry_in_trials(entry, trials) =
    any(cfg -> cfg == entry.best_config, trials)

function _cached_entry(kernel_key, arg_key, trials)
    entry = Base.@lock AUTOTUNE_CACHE begin
        per_kernel = get(AUTOTUNE_CACHE[], kernel_key, nothing)
        per_kernel === nothing ? nothing : get(per_kernel, arg_key, nothing)
    end
    entry !== nothing && _entry_in_trials(entry, trials) ? entry : nothing
end

function _cache_candidate!(candidate, kernel_key, arg_key, trials; force::Bool)
    Base.@lock AUTOTUNE_CACHE begin
        per_kernel = get!(AUTOTUNE_CACHE[], kernel_key) do
            Dict{Any, Any}()
        end
        if !force
            entry = get(per_kernel, arg_key, nothing)
            if entry !== nothing && _entry_in_trials(entry, trials)
                return entry, true
            end
        end
        per_kernel[arg_key] = candidate
        return candidate, false
    end
end

function _no_valid_config_error(first_error, precompile_error)
    err_info = first_error !== nothing ? first_error : precompile_error
    if err_info === nothing
        throw(ArgumentError("No valid config found in search space"))
    end

    cfg, err = err_info
    throw(ArgumentError(
        "No valid config found. First failure for cfg=$cfg: $(sprint(showerror, err))"))
end

function _refine_record(@nospecialize(f), record::Vector{TimingRecord}, tuning::TuningOptions,
                        grid_fn, args_fn;
                        sm_arch::VersionNumber, opt_level::Int,
                        static_num_ctas=nothing, static_occupancy=nothing,
                        verify=nothing, reset=nothing)
    (tuning.refine_topk > 0 && length(record) > 1) || return record

    sort!(record, by=last)
    top = Any[first(r) for r in record[1:min(tuning.refine_topk, length(record))]]
    refined, _ = measure_candidates(f, top, grid_fn, args_fn;
        sm_arch, opt_level, warmup=tuning.warmup, reps=tuning.refine_reps,
        static_num_ctas, static_occupancy, verify, reset)
    return isempty(refined) ? record : refined
end

function _best_candidate(record::Vector{TimingRecord})
    _, best_idx = findmin(last, record)
    return (; best_config=record[best_idx][1], tuning_record=record)
end

function find_or_tune(@nospecialize(f), space::AbstractSearchSpace,
                      grid_fn, args_fn, tuning::TuningOptions;
                      sm_arch::VersionNumber, opt_level::Int, kernel_key, arg_key,
                      static_num_ctas=nothing, static_occupancy=nothing,
                      verify=nothing, setup=nothing)
    trials = _collect_trials(space, tuning.seed)
    isempty(trials) && throw(ArgumentError("No valid config found in search space"))
    _check_static_hint_conflicts(trials; static_num_ctas, static_occupancy)

    if !tuning.force
        entry = _cached_entry(kernel_key, arg_key, trials)
        entry !== nothing && return entry, true, nothing
    end

    checker = verify !== nothing ? verify() : nothing
    reset = setup !== nothing ? setup() : nothing

    record, precompile_error, first_error =
        with(_SCOPED_INF_CACHE => _fresh_inf_cache()) do
            pipelined_tune(f, trials, grid_fn, args_fn;
                sm_arch, opt_level,
                warmup=tuning.warmup, reps=tuning.reps,
                workers=tuning.precompile_workers,
                static_num_ctas, static_occupancy,
                verify=checker, reset)
        end

    isempty(record) && _no_valid_config_error(first_error, precompile_error)

    record = _refine_record(f, record, tuning, grid_fn, args_fn;
        sm_arch, opt_level,
        static_num_ctas, static_occupancy,
        verify=checker, reset)

    candidate = _best_candidate(record)
    entry, cache_hit = _cache_candidate!(candidate, kernel_key, arg_key, trials;
                                         force=tuning.force)
    return entry, cache_hit, reset
end

@inline _as_cfg_fn(f::Function) = f
@inline _as_cfg_fn(x) = Returns(x)

"""
    autotune_launch(f, space, grid, args; key, launch_args, verify, setup,
                    tuning, sm_arch, opt_level,
                    num_ctas=nothing, occupancy=nothing)

Tune `f` over `space` and launch the fastest valid config.

`space` can be an `AbstractSearchSpace`, a `NamedTuple` of cartesian axes, or
an iterable of `NamedTuple` configs. `grid`, `args`, and `launch_args` can be
plain values or `cfg -> value` functions. Results are cached per
`(f, sm_arch, opt_level, num_ctas, occupancy)` and user `key`.
"""
function autotune_launch(@nospecialize(f), space::AbstractSearchSpace,
                         grid, args;
                         key=nothing,
                         launch_args=nothing,
                         verify=nothing,
                         setup=nothing,
                         tuning::NamedTuple=NamedTuple(),
                         sm_arch::VersionNumber=default_sm_arch(),
                         opt_level::Int=3,
                         num_ctas=nothing,
                         occupancy=nothing)
    tuning = normalize_tuning(tuning)

    grid_fn = _as_cfg_fn(grid)
    args_fn = _as_cfg_fn(args)
    launch_args_fn = launch_args === nothing ? args_fn : _as_cfg_fn(launch_args)

    kernel_key = (f, sm_arch, opt_level, num_ctas, occupancy)
    entry, cache_hit, reset = find_or_tune(f, space, grid_fn, args_fn, tuning;
        sm_arch, opt_level, kernel_key, arg_key=key,
        static_num_ctas=num_ctas, static_occupancy=occupancy,
        verify, setup)

    cfg = entry.best_config
    grid = grid_fn(cfg)
    launched_args = launch_args_fn(cfg)

    reset !== nothing && reset()

    cuTile.launch(f, grid, launched_args...; sm_arch, opt_level,
                  hints_from_cfg(cfg; static_num_ctas=num_ctas,
                                      static_occupancy=occupancy)...)

    return (; tuned_config=cfg, grid, tuning_record=copy(entry.tuning_record), cache_hit)
end

function autotune_launch(@nospecialize(f), configs, grid, args; kwargs...)
    space = configs isa NamedTuple ? CartesianSpace(configs) : FixedSpace(configs)
    return autotune_launch(f, space, grid, args; kwargs...)
end

function clear_autotune_cache(; kernel=nothing, key=nothing)
    Base.@lock AUTOTUNE_CACHE begin
        cache = AUTOTUNE_CACHE[]
        if kernel === nothing
            key === nothing || throw(ArgumentError("`key` requires `kernel`"))
            empty!(cache)
            return nothing
        end

        for kernel_key in collect(keys(cache))
            kernel_key isa Tuple || continue
            kernel_key[1] === kernel || continue
            per_kernel = cache[kernel_key]
            key === nothing ? empty!(per_kernel) : pop!(per_kernel, key, nothing)
            isempty(per_kernel) && delete!(cache, kernel_key)
        end
    end
    return nothing
end
