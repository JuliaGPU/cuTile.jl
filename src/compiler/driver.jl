
#=============================================================================
 Meta nodes and compilation hints
=============================================================================#

# Codegen options, as consumed by `emit_tile`.
# Hint fields (opt_level, num_ctas, occupancy, num_worker_warps) represent explicit
# overrides only; `nothing` means "consult @compiler_options meta nodes in the IR
# during compilation."
const CGOpts = @NamedTuple{
    sm_arch::Union{VersionNumber, Nothing},
    opt_level::Union{Int, Nothing},
    num_ctas::Union{Int, Nothing},
    occupancy::Union{Int, Nothing},
    num_worker_warps::Union{Int, Nothing},
    bytecode_version::VersionNumber
}

#=============================================================================
 Compiler jobs

 A cuTile compilation is a `TileJob`: a MethodInstance and world, a target
 (architecture and bytecode version), params (hints and const-seeded argument
 types) and a kernel name. The job names a compilation: it keys the results
 cache (see `compile_or_lookup` in launch.jl), is what the `@device_code_*`
 hook reports, and is what the reflection entry points take.
=============================================================================#

struct TileCompilerTarget
    sm_arch::Union{VersionNumber, Nothing}
    bytecode_version::VersionNumber
end

struct TileCompilerParams
    opt_level::Union{Int, Nothing}
    num_ctas::Union{Int, Nothing}
    occupancy::Union{Int, Nothing}
    num_worker_warps::Union{Int, Nothing}
    # `(Const(f), arg2, …)` for const-seeded inference; a Tuple, not the driver's
    # Vector, so that jobs compare structurally (`===` on immutables).
    const_argtypes::Union{Tuple, Nothing}
end

# Everything about a compilation except what is being compiled: keys the codegen
# results of a kernel's CodeInstance, which outlives any one world.
struct TileConfig
    target::TileCompilerTarget
    params::TileCompilerParams
    name::String
end

"""
    TileJob

A cuTile compilation: a `MethodInstance` and world, and a configuration — the
target architecture and bytecode version, the compilation hints and const-seeded
argument types, and the kernel name. Built by [`tile_job`](@ref); taken by the
reflection entry points and reported to the `@device_code_*` macros.
"""
struct TileJob
    source::MethodInstance
    world::UInt
    config::TileConfig
end

# Compilation hook for `@device_code_*`: called with every launch's `TileJob`.
const compile_hook = Ref{Union{Nothing,Function}}(nothing)

# Inference is partitioned by the owner stamped on every CodeInstance; codegen
# results are stored per job on that CodeInstance (`TileResults`). cuTile's
# inference depends on neither the target nor the hints — the interpreter takes
# only a world — so every Tile job shares one partition: one inference per
# kernel, N codegens. Const-seeded arguments are `const_entries` on that shared
# CodeInstance. A `Symbol` is interned, so the owner is identity-stable for
# Julia's inference engine without any interning of our own.
const TILE_CACHE_OWNER = :cuTile

# CompilerCaching's handle on the inference partition. The results type is
# `Nothing`: codegen artifacts are stored per job by `compile_or_lookup`, not here.
inference_cache(world::UInt) = CacheView{Any, Nothing}(TILE_CACHE_OWNER, world)
inference_cache(job::TileJob) = inference_cache(job.world)

function tile_job(mi::MethodInstance, world::UInt;
                  sm_arch::Union{VersionNumber, Nothing}=nothing,
                  opt_level::Union{Int, Nothing}=nothing,
                  num_ctas::Union{Int, Nothing}=nothing,
                  occupancy::Union{Int, Nothing}=nothing,
                  num_worker_warps::Union{Int, Nothing}=nothing,
                  bytecode_version::VersionNumber=cuTile.bytecode_version(),
                  const_argtypes=nothing,
                  name::Union{String, Nothing}=nothing)
    target = TileCompilerTarget(sm_arch, bytecode_version)
    params = TileCompilerParams(opt_level, num_ctas, occupancy, num_worker_warps,
                                const_argtypes === nothing ? nothing : Tuple(const_argtypes))
    config = TileConfig(target, params, something(name, sanitize_name(string(mi.def.name))))
    return TileJob(mi, world, config)
end

# The same job for another architecture.
with_target(job::TileJob, sm_arch::VersionNumber) =
    TileJob(job.source, job.world,
            TileConfig(TileCompilerTarget(sm_arch, job.config.target.bytecode_version),
                       job.config.params, job.config.name))

# The job's const-seeded argument types in the driver's `Vector{Any}` form.
function job_const_argtypes(job::TileJob)
    cats = job.config.params.const_argtypes
    cats === nothing ? nothing : collect(Any, cats)
end

job_opts(job::TileJob) = CGOpts((sm_arch=job.config.target.sm_arch,
                                 opt_level=job.config.params.opt_level,
                                 num_ctas=job.config.params.num_ctas,
                                 occupancy=job.config.params.occupancy,
                                 num_worker_warps=job.config.params.num_worker_warps,
                                 bytecode_version=job.config.target.bytecode_version))

"""
    process_meta!(ir::CC.IRCode) -> ir

Move `:meta` expression nodes from `ir.stmts` into `ir.meta`, mirroring
Julia's `process_meta!` in `Compiler/src/optimize.jl`. This normalizes IR
from `inflate_ir` (which leaves meta as stmts) to match the `typeinf_ircode`
path (which already extracts meta via `convert_to_ircode`).
"""
function process_meta!(ir::CC.IRCode)
    for i in 1:length(ir.stmts)
        stmt = ir.stmts[i][:stmt]
        if stmt isa Expr && stmt.head === :meta
            push!(ir.meta, stmt)
            @static if VERSION >= v"1.12-"
                ir.stmts[i][:stmt] = nothing
            else
                CC.setindex!(ir.stmts[i], nothing, :stmt)
            end
        end
    end
    return ir
end

"""
    extract_meta(ir::CC.IRCode) -> Dict{Symbol, Any}

Extract cuTile meta nodes from IRCode. Meta nodes are inserted by `@compiler_options`
and survive through lowering/optimization. After `process_meta!` normalization,
all meta nodes reside in `ir.meta`.
"""
function extract_meta(ir::CC.IRCode)
    meta = Dict{Symbol, Any}()
    for expr in ir.meta
        if expr isa Expr && expr.head === :meta && length(expr.args) >= 3 && expr.args[1] === :cuTile
            meta[expr.args[2]::Symbol] = expr.args[3]
        end
    end
    return meta
end

"""
    resolve_hint(explicit, kernel_meta, key, sm_arch)

Resolve a hint value with precedence: explicit kwarg > @compiler_options meta > nothing.
"""
function resolve_hint(explicit, kernel_meta::Dict{Symbol, Any}, key::Symbol,
                      sm_arch::Union{VersionNumber, Nothing})
    val = if explicit !== nothing
        explicit
    elseif haskey(kernel_meta, key) && sm_arch !== nothing
        resolve(kernel_meta[key], sm_arch)
    else
        nothing
    end
    validate_hint(key, val)
    return val
end


#=============================================================================
 Compilation phases
=============================================================================#

"""
    get_ci(cache, mi; const_argtypes=nothing) -> CodeInstance

Ensure inference is done and return the CodeInstance. Runs `typeinf!` which is
a no-op when already cached. When `const_argtypes` is provided, also ensures
the const-specialized entry exists.
"""
function get_ci(cache::CacheView, mi::Core.MethodInstance;
                const_argtypes::Union{Vector{Any}, Nothing}=nothing)
    # Ensure CI exists
    ci = get(cache, mi, nothing)
    if ci === nothing
        interp = cuTileInterpreter(cache)
        ci = typeinf!(interp, mi)
        ci === nothing && error("Inference failed for $mi")
    end

    # Run const-prop inference, if needed
    if const_argtypes !== nothing
        interp = cuTileInterpreter(cache)
        typeinf!(cache, interp, mi, const_argtypes)
    end

    return ci
end


# Get the inferred source and return type from a CodeInstance.
function get_inferred(cache::CacheView{K,V}, ci::Core.CodeInstance,
                      mi::Core.MethodInstance; const_argtypes::Union{Vector{Any},
                      Nothing}=nothing) where {K,V}
    rettype = CC.widenconst(ci.rettype)
    if const_argtypes === nothing
        src = @something get_source(ci)
    else
        # Read our const-specialized entry directly rather than through
        # `get_source(ci, argtypes)`: that takes the first `CachedResult` on the
        # CI whatever its results type, and `TileResults` is attached to it too.
        cached = CC.traverse_analysis_results(ci) do @nospecialize(result)
            result isa CompilerCaching.CachedResult{V} ? result : nothing
        end
        cached === nothing && error("No const-specialized inference results for $mi")
        i = findfirst(entry -> entry.argtypes == const_argtypes, cached.const_entries)
        i === nothing && error("No const-specialized inference result for $mi with $const_argtypes")
        entry = cached.const_entries[i]
        src = entry.src
        src isa Core.CodeInfo || error("No const-specialized source for $mi")
        rettype = CC.widenconst(entry.rettype)
    end
    ir = CC.inflate_ir(src, mi)
    return ir, rettype
end

"""
    emit_julia(cache, mi; const_argtypes=nothing) -> (IRCode, rettype)

Julia phase: run inference and return IRCode.
"""
function emit_julia(cache::CacheView, mi::Core.MethodInstance;
                    const_argtypes::Union{Vector{Any}, Nothing}=nothing)
    ci = get_ci(cache, mi; const_argtypes)
    get_inferred(cache, ci, mi; const_argtypes)
end

"""
    emit_structured(ir::IRCode, rettype) -> (StructuredIRCode, rettype, kernel_meta)

Structurize IRCode into StructuredIRCode. Pure transformation, no caching.
"""
function emit_structured(ir::CC.IRCode, rettype)
    process_meta!(ir)
    kernel_meta = extract_meta(ir)
    sci = StructuredIRCode(ir)
    return (sci, rettype, kernel_meta)
end

"""
    emit_tile(sci, rettype, kernel_meta; name, opts, cache, const_argtypes) -> Vector{UInt8}

Generate Tile IR bytecode from StructuredIRCode. Pure computation, no caching.
`cache` is needed for subprogram compilation inside `emit_kernel!`.
"""
function emit_tile(sci::StructuredIRCode, rettype, kernel_meta::Dict{Symbol,Any};
                   name::String,
                   opts::CGOpts,
                   cache::CacheView,
                   const_argtypes::Union{Vector{Any}, Nothing}=nothing)
    # Resolve hints: launch()/code_tiled() kwargs > @compiler_options meta > defaults
    resolved_num_ctas = resolve_hint(opts.num_ctas, kernel_meta, :num_ctas, opts.sm_arch)
    resolved_occupancy = resolve_hint(opts.occupancy, kernel_meta, :occupancy, opts.sm_arch)
    resolved_num_worker_warps = resolve_hint(opts.num_worker_warps, kernel_meta,
                                             :num_worker_warps, opts.sm_arch)

    # Generate Tile IR bytecode
    bytecode = write_bytecode!(1; version=opts.bytecode_version) do writer, func_buf
        emit_kernel!(writer, func_buf, sci, rettype;
            name,
            sm_arch = opts.sm_arch,
            num_ctas = resolved_num_ctas,
            occupancy = resolved_occupancy,
            num_worker_warps = resolved_num_worker_warps,
            cache,
            const_argtypes
        )
    end

    return bytecode
end


"""
    emit_bytecode(job::TileJob) -> (; bytecode, kernel_meta)

Julia phase through Tile IR for a job: inference (cached in the job's
partition), structurization and bytecode emission (recomputed on every call).
"""
function emit_bytecode(job::TileJob)
    cache = inference_cache(job)
    const_argtypes = job_const_argtypes(job)
    ir, rettype = emit_julia(cache, job.source; const_argtypes)
    sci, rettype, kernel_meta = emit_structured(ir, rettype)
    bytecode = emit_tile(sci, rettype, kernel_meta;
                         name=job.config.name, opts=job_opts(job), cache, const_argtypes)
    return (; bytecode, kernel_meta)
end

# Dump bytecode to `$JULIA_CUTILE_DUMP_BYTECODE/<file>.ln<line>[.n].cutile`, if set.
function dump_bytecode(mi::MethodInstance, bytecode::Vector{UInt8})
    dump_dir = get(ENV, "JULIA_CUTILE_DUMP_BYTECODE", nothing)
    dump_dir === nothing && return
    mkpath(dump_dir)
    base_filename = first(splitext(basename(string(mi.def.file))))
    dump_path = joinpath(dump_dir, "$(base_filename).ln$(mi.def.line).cutile")
    counter = 1
    while isfile(dump_path)
        counter += 1
        dump_path = joinpath(dump_dir, "$(base_filename).ln$(mi.def.line).$(counter).cutile")
    end
    println(stderr, "Dumping TILEIR bytecode to file: $dump_path")
    write(dump_path, bytecode)
end
