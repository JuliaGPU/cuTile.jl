const AUTOTUNE_LOCK = ReentrantLock()
const AUTOTUNE_CACHE = Dict{Any, Dict{Any, Any}}()

struct VerificationError <: Exception
    msg::String
end

const TUNING_PRESETS = (
    fast     = (warmup=1, reps=3,  refine_topk=0, refine_reps=2),
    default  = (warmup=2, reps=5,  refine_topk=2, refine_reps=4),
    thorough = (warmup=2, reps=7,  refine_topk=4, refine_reps=6),
)

function normalize_tuning(tuning::NamedTuple)
    preset = get(tuning, :preset, :default)
    preset isa Symbol || throw(ArgumentError("tuning.preset must be a Symbol"))
    hasproperty(TUNING_PRESETS, preset) ||
        throw(ArgumentError("Unknown preset `$preset`; use :fast, :default, or :thorough"))

    base = merge(getproperty(TUNING_PRESETS, preset),
        (seed=nothing, force=false, precompile_workers=Threads.nthreads()))

    overrides = NamedTuple(k => v for (k, v) in pairs(tuning) if k !== :preset)
    return merge(base, overrides)
end

# Extract hint fields (num_ctas, occupancy) from a config, falling back to
# the static defaults supplied by the caller. cfg takes precedence; the
# caller is expected to have rejected the both-supplied case upstream
# (see `autotune_launch`).
function hints_from_cfg(cfg; static_num_ctas=nothing, static_occupancy=nothing)
    n = hasproperty(cfg, :num_ctas) ? cfg.num_ctas : static_num_ctas
    o = hasproperty(cfg, :occupancy) ? cfg.occupancy : static_occupancy
    return (num_ctas=n, occupancy=o)
end

function time_ms(run_once, get_args;
                 warmup::Int, reps::Int, verify=nothing, reset=nothing)
    CUDACore.synchronize()
    for _ in 1:max(warmup, verify !== nothing ? 1 : 0)
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
    grid = grid_fn(cfg)
    grid_dims = grid isa Integer ? (grid,) : grid

    # Compile once, then convert + call each rep. We `cufunction` outside the
    # timed loop so JIT cost doesn't pollute the measurement.
    sample_converted = map(cuTileconvert, args_fn(cfg))
    tt = Tuple{map(Core.Typeof, sample_converted)...}
    kernel = cufunction(f, tt; sm_arch, opt_level,
                        hints_from_cfg(cfg; static_num_ctas, static_occupancy)...)

    run_once = converted -> kernel(converted...; blocks=grid_dims)
    get_args = () -> map(cuTileconvert, args_fn(cfg))
    return time_ms(run_once, get_args; warmup, reps, verify, reset)
end

function precompile_cfg(@nospecialize(f), cfg, args_fn;
                        sm_arch::VersionNumber, opt_level::Int,
                        static_num_ctas=nothing, static_occupancy=nothing)
    converted = map(cuTileconvert, args_fn(cfg))
    tt = Tuple{map(Core.Typeof, converted)...}
    cufunction(f, tt; sm_arch, opt_level,
               hints_from_cfg(cfg; static_num_ctas, static_occupancy)...)
    return nothing
end

function precompile_candidates(@nospecialize(f), configs::Vector{Any}, args_fn;
                               sm_arch::VersionNumber, opt_level::Int, workers::Int,
                               static_num_ctas=nothing, static_occupancy=nothing)
    isempty(configs) && return configs, nothing
    iszero(workers) && return configs, nothing

    workers = min(workers, Threads.nthreads(), length(configs))
    compiled = fill(true, length(configs))
    errors = Vector{Any}(nothing, length(configs))
    sem = Base.Semaphore(workers)
    cancelled = Threads.Atomic{Bool}(false)

    try
        @sync for (i, cfg) in enumerate(configs)
            Threads.@spawn begin
                cancelled[] && return
                Base.acquire(sem) do
                    cancelled[] && return
                    try
                        precompile_cfg(f, cfg, args_fn; sm_arch, opt_level,
                                       static_num_ctas, static_occupancy)
                    catch err
                        compiled[i] = false
                        errors[i] = (cfg, err)
                    end
                end
            end
        end
    catch e
        cancelled[] = true
        e isa InterruptException || rethrow()
        @warn "Precompilation interrupted, waiting for in-flight workers…"
        rethrow()
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

function measure_candidates(@nospecialize(f), configs::Vector{Any}, grid_fn, args_fn;
                            sm_arch::VersionNumber, opt_level::Int, warmup::Int, reps::Int,
                            static_num_ctas=nothing, static_occupancy=nothing,
                            verify=nothing, reset=nothing)
    record = Tuple{Any, Float32}[]
    first_error = nothing
    for cfg in configs
        ms = try
            eval_cfg(f, cfg, grid_fn, args_fn; sm_arch, opt_level, warmup, reps,
                     static_num_ctas, static_occupancy, verify, reset)
        catch err
            if err isa InterruptException
                @warn "Benchmarking interrupted after $(length(record)) configs"
                break
            end
            err isa VerificationError && @warn "Config $cfg failed verification, skipping"
            first_error === nothing && (first_error = (cfg, err))
            continue
        end
        push!(record, (cfg, ms))
    end
    return record, first_error
end

"""
    pipelined_tune(f, configs, grid_fn, args_fn; ...) -> (record, precompile_error, first_error)

Overlap parallel compile with sequential measure. Compile workers (`Threads.@spawn`)
push each finished cfg onto a `Channel`; the master task pulls them off in
arrival order and runs `eval_cfg` on each. Net wall time is roughly
`max(parallel_compile_time, total_measure_time + first_compile)` instead of
the all-compile-then-all-measure sum.

The master task is the consumer by design: it inherits the CUDA context from
the caller, which the timed `eval_cfg` needs (CUDA state is task-local).
`workers=0` skips parallel compile entirely and falls back to a
compile-then-measure cycle on the master.

`record` is in completion order, not trial order — callers that care about
deterministic ordering should sort the result.
"""
function pipelined_tune(@nospecialize(f), configs::Vector{Any}, grid_fn, args_fn;
                        sm_arch::VersionNumber, opt_level::Int,
                        warmup::Int, reps::Int, workers::Int,
                        static_num_ctas=nothing, static_occupancy=nothing,
                        verify=nothing, reset=nothing)
    record = Tuple{Any, Float32}[]
    if isempty(configs)
        return record, nothing, nothing
    end

    # Serial fallback: avoids channel + extra task overhead when there's
    # nothing to overlap (workers=0) or only one cfg to evaluate.
    if iszero(workers) || length(configs) == 1
        rec, ferr = measure_candidates(f, configs, grid_fn, args_fn;
            sm_arch, opt_level, warmup, reps,
            static_num_ctas, static_occupancy, verify, reset)
        return rec, nothing, ferr
    end

    workers = min(workers, Threads.nthreads(), length(configs))
    cancelled = Threads.Atomic{Bool}(false)
    sem = Base.Semaphore(workers)
    precompile_error = Ref{Any}(nothing)
    err_lock = ReentrantLock()

    # Buffer == n: producers never block on put!. Channel carries
    # (trial_index, cfg) pairs; the index is preserved so callers that want
    # trial-order can recover it.
    ch = Channel{Tuple{Int, Any}}(length(configs))

    # Producer driver: runs in its own task so the master can start consuming
    # the moment the first cfg lands. `@sync` inside ensures we don't close
    # the channel until every spawned compiler has either pushed or recorded
    # an error.
    producer = Threads.@spawn begin
        try
            @sync for (i, cfg) in enumerate(configs)
                cancelled[] && break
                Threads.@spawn begin
                    cancelled[] && return
                    Base.acquire(sem) do
                        cancelled[] && return
                        try
                            precompile_cfg(f, cfg, args_fn; sm_arch, opt_level,
                                           static_num_ctas, static_occupancy)
                            cancelled[] || put!(ch, (i, cfg))
                        catch err
                            lock(err_lock) do
                                precompile_error[] === nothing &&
                                    (precompile_error[] = (cfg, err))
                            end
                        end
                    end
                end
            end
        finally
            close(ch)
        end
    end

    # Master consumes (and times) on this task — keeps the CUDA context for
    # `eval_cfg` consistent with the caller's.
    first_error = nothing
    try
        for (_, cfg) in ch
            ms = try
                eval_cfg(f, cfg, grid_fn, args_fn; sm_arch, opt_level, warmup, reps,
                         static_num_ctas, static_occupancy, verify, reset)
            catch err
                if err isa InterruptException
                    @warn "Benchmarking interrupted after $(length(record)) configs"
                    cancelled[] = true
                    break
                end
                err isa VerificationError &&
                    @warn "Config $cfg failed verification, skipping"
                first_error === nothing && (first_error = (cfg, err))
                continue
            end
            push!(record, (cfg, ms))
        end
    catch err
        cancelled[] = true
        # Drain any items already in the channel so producers don't block on
        # put! while we wait for them to notice the cancel flag.
        while isready(ch); take!(ch); end
        wait(producer)
        rethrow()
    end

    wait(producer)
    return record, precompile_error[], first_error
end

function find_or_tune(@nospecialize(f), space::AbstractSearchSpace, rng::AbstractRNG,
                      grid_fn, args_fn, tuning;
                      sm_arch::VersionNumber, opt_level::Int, kernel_key, arg_key,
                      static_num_ctas=nothing, static_occupancy=nothing,
                      verify=nothing, setup=nothing)
    if !tuning.force
        entry = lock(AUTOTUNE_LOCK) do
            per_kernel = get(AUTOTUNE_CACHE, kernel_key, nothing)
            per_kernel !== nothing ? get(per_kernel, arg_key, nothing) : nothing
        end
        entry !== nothing && return entry, true, nothing
    end

    checker = verify !== nothing ? verify() : nothing
    reset = setup !== nothing ? setup() : nothing

    trials = Any[collect(space)...]

    # Conflict check: if the cfg carries a `num_ctas`/`occupancy` field AND
    # the caller also provided a static value, error rather than silently
    # ignoring one. (Handles the case where `space` is opaque to the macro
    # — a user-built `CartesianSpace(...)` or `FixedSpace([(...),...])`.)
    if !isempty(trials)
        sample = first(trials)
        if static_num_ctas !== nothing && hasproperty(sample, :num_ctas)
            throw(ArgumentError(
                "`num_ctas` is both a static kwarg and an axis in the search space. " *
                "Pick one."))
        end
        if static_occupancy !== nothing && hasproperty(sample, :occupancy)
            throw(ArgumentError(
                "`occupancy` is both a static kwarg and an axis in the search space. " *
                "Pick one."))
        end
    end

    # Share the inference cache across all per-cfg const-seeded compiles.
    # Each cfg differs only in `Constant{T,V}` values, so the generic
    # inference graph is identical — without sharing, kernels with slow
    # inference paths (e.g. `ct.load(..., order=…)`) pay that cost N times.
    #
    # `pipelined_tune` overlaps the parallel compile fan-out with sequential
    # GPU measurement: as soon as the first compiler finishes, the master
    # starts timing while the remaining compilers continue in the background.
    record, precompile_error, first_error =
        with(_SCOPED_INF_CACHE => _fresh_inf_cache()) do
            pipelined_tune(f, trials, grid_fn, args_fn;
                sm_arch, opt_level,
                warmup=tuning.warmup, reps=tuning.reps,
                workers=tuning.precompile_workers,
                static_num_ctas, static_occupancy,
                verify=checker, reset)
        end

    if isempty(record)
        err_info = first_error !== nothing ? first_error : precompile_error
        if err_info === nothing
            throw(ArgumentError("No valid config found in search space"))
        else
            cfg, err = err_info
            throw(ArgumentError(
                "No valid config found. First failure for cfg=$cfg: $(sprint(showerror, err))"))
        end
    end

    if tuning.refine_topk > 0 && length(record) > 1
        sort!(record, by=last)
        top_configs = Any[first(r) for r in record[1:min(tuning.refine_topk, length(record))]]
        refined, _ = measure_candidates(f, top_configs, grid_fn, args_fn;
            sm_arch, opt_level, warmup=tuning.warmup, reps=tuning.refine_reps,
            static_num_ctas, static_occupancy, reset)
        if !isempty(refined)
            record = refined
        end
    end

    _, best_idx = findmin(last, record)
    candidate = (; best_config=record[best_idx][1], tuning_record=record)

    # Race: another thread may have populated the cache while we were
    # tuning. If so, return their result and report `cache_hit=true` so
    # the caller's accounting stays accurate.
    entry, cache_hit = lock(AUTOTUNE_LOCK) do
        per_kernel = get!(Dict{Any,Any}, AUTOTUNE_CACHE, kernel_key)
        if !tuning.force && haskey(per_kernel, arg_key)
            per_kernel[arg_key], true
        else
            per_kernel[arg_key] = candidate
            candidate, false
        end
    end
    return entry, cache_hit, reset
end

# Normalize `grid`/`args` to a callable. A `Function` is used as-is; any
# other value is wrapped in `Returns(...)`. Lets `autotune_launch` accept
# `blocks = 1024` or `args = (a, b, c)` directly instead of forcing the
# caller to write `cfg -> 1024` for grids that don't depend on the cfg.
@inline _as_cfg_fn(f::Function) = f
@inline _as_cfg_fn(x) = Returns(x)

"""
    autotune_launch(f, space, grid, args; key, launch_args, verify, setup,
                    tuning, sm_arch, opt_level,
                    num_ctas=nothing, occupancy=nothing)

Tune `f` over `space` (an [`AbstractSearchSpace`](@ref) or a `Vector`/`NamedTuple`
shorthand) and launch the best config.

`grid` and `args` are either functions of the form `cfg -> grid` /
`cfg -> args_tuple` (for grids/args that depend on the cfg), or plain
values (for grids/args that don't). Internally both are normalized to
callables. `launch_args` follows the same convention; defaults to `args`.

Results are cached per `(f, sm_arch, opt_level, num_ctas, occupancy) ⇒ key`.

`num_ctas` and `occupancy` may be supplied as **static** kwargs (applied
uniformly to every cfg — useful for `ByTarget(...)`-style per-arch dispatch)
OR as **axes** inside `space` (tuned per cfg), but not both. Specifying
both throws an `ArgumentError`.
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
    rng = tuning.seed !== nothing ? MersenneTwister(tuning.seed) : Random.default_rng()

    grid_fn = _as_cfg_fn(grid)
    args_fn = _as_cfg_fn(args)
    launch_args_fn = launch_args === nothing ? args_fn : _as_cfg_fn(launch_args)

    kernel_key = (f, sm_arch, opt_level, num_ctas, occupancy)

    entry, cache_hit, reset = find_or_tune(f, space, rng, grid_fn, args_fn, tuning;
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
