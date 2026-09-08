#=============================================================================
 Compiler jobs

 TileJob identifies the source and configuration used by compilation,
 caching, and reflection.
=============================================================================#

public TileJob, tile_job

struct TileCompilerTarget
    # `nothing` for a job created without a CUDA device: the stages before
    # `tileiras` still run, but architecture-dependent `@compiler_options`
    # hints cannot be resolved and are ignored.
    sm_arch::Union{VersionNumber, Nothing}
    bytecode_version::VersionNumber
end

# Compilation hints; `nothing` defers to the kernel's `@compiler_options`.
struct TileCompilerParams
    opt_level::Union{Int, Nothing}
    num_ctas::Union{Int, Nothing}
    occupancy::Union{Int, Nothing}
    num_worker_warps::Union{Int, Nothing}
end

# Compilation results are keyed by config within each inference result.
struct TileConfig
    target::TileCompilerTarget
    params::TileCompilerParams
    name::String
end

"""
    TileJob

A compilation request containing a `MethodInstance`, its world age and any
constant arguments, plus the target architecture, bytecode version, compilation
hints, and kernel name. Construct jobs with [`tile_job`](@ref) and pass them to
the reflection functions to inspect that configuration.

Jobs are immutable and compare structurally, so equal jobs are `===`.
"""
struct TileJob
    source::MethodInstance
    # `(Const(f), arg2, …)` seeding const-propagating inference, or `nothing`
    # for the generic inferred source. A Tuple rather than CompilerCaching's
    # Vector so that jobs compare structurally.
    const_argtypes::Union{Tuple, Nothing}
    world::UInt
    config::TileConfig
end

"""
    tile_job(f, argtypes; world=Base.get_world_counter(), kwargs...) -> TileJob
    tile_job(mi::MethodInstance, world; const_argtypes=nothing, kwargs...) -> TileJob

Create a [`TileJob`](@ref) for `f` and `argtypes` (which may contain
`Constant{T,V}` types), or for a method instance with the given const-seeded
argument types. Keyword arguments configure the compilation:

- `sm_arch`: the target architecture. Defaults to the active CUDA device's, or
  `nothing` without one; then only the stages before `tileiras` are available
  and architecture-dependent `@compiler_options` hints are ignored.
- `bytecode_version`: the Tile IR bytecode version to emit.
- `opt_level`, `num_ctas`, `occupancy`, `num_worker_warps`: compilation hints
  overriding the kernel's `@compiler_options`.
- `name`: the kernel's name in the bytecode; defaults to the method's.
"""
function tile_job(mi::MethodInstance, world::UInt;
                  const_argtypes::Union{Tuple, Nothing}=nothing,
                  sm_arch::Union{VersionNumber, Nothing}=nothing,
                  bytecode_version::VersionNumber=cuTile.bytecode_version(),
                  opt_level::Union{Int, Nothing}=nothing,
                  num_ctas::Union{Int, Nothing}=nothing,
                  occupancy::Union{Int, Nothing}=nothing,
                  num_worker_warps::Union{Int, Nothing}=nothing,
                  name::Union{String, Nothing}=nothing)
    sm_arch = @something sm_arch device_sm_arch() Some(nothing)
    sm_arch === nothing || validate_tile_ir_target(sm_arch, bytecode_version)
    target = TileCompilerTarget(sm_arch, bytecode_version)
    params = TileCompilerParams(opt_level, num_ctas, occupancy, num_worker_warps)
    config = TileConfig(target, params, @something name sanitize_name(string(mi.def.name)))
    return TileJob(mi, const_argtypes, world, config)
end

# The architecture `tileiras` assembles a job for.
function target_arch(job::TileJob)
    sm_arch = job.config.target.sm_arch
    sm_arch === nothing && throw(ArgumentError(
        "the job has no target architecture (it was created without a CUDA device); " *
        "pass `sm_arch` explicitly"))
    return sm_arch
end

# The job's const-seeded argument types in CompilerCaching's `Vector{Any}` form.
const_argtypes_vector(job::TileJob) =
    job.const_argtypes === nothing ? nothing : collect(Any, job.const_argtypes)

# `(f, tt)` with `Constant` argument types restored.
function job_signature(job::TileJob)
    mi = job.source
    ftype = mi.specTypes.parameters[1]
    f = isdefined(ftype, :instance) ? ftype.instance : ftype
    arg_types = collect(Any, mi.specTypes.parameters[2:end])
    if job.const_argtypes !== nothing
        # const_argtypes is (Const(f), arg2, ...); arg_types omits f.
        for i in eachindex(arg_types)
            cat = job.const_argtypes[i+1]
            cat isa CC.Const && (arg_types[i] = typeof(Constant(cat.val)))
        end
    end
    return f, Tuple{arg_types...}
end

function Base.show(io::IO, job::TileJob)
    f, tt = job_signature(job)
    (; target, params, name) = job.config
    print(io, "TileJob(", f, "(", join(tt.parameters, ", "), ")")
    print(io, "; sm_arch=", something(target.sm_arch, "nothing"),
              ", bytecode_version=v\"", target.bytecode_version, "\"")
    for field in fieldnames(TileCompilerParams)
        hint = getfield(params, field)
        hint === nothing || print(io, ", ", field, "=", hint)
    end
    name == sanitize_name(string(job.source.def.name)) || print(io, ", name=", repr(name))
    print(io, ")")
end


#=============================================================================
 Compilation hook

 `@device_code_*` macros, cuTile's and GPUCompiler's alike, observe
 compilations through `GPUCompiler.compile_hook`, called with the job of every
 kernel that is compiled or launched while it is set.
=============================================================================#

# Launches run in the frozen world (`invoke_frozen`), but the hook closure
# lives in the user's latest one, hence `invokelatest`.
function run_compile_hook(job::TileJob)
    hook = GPUCompiler.compile_hook[]
    hook === nothing || Base.invokelatest(hook, job)
    return
end


#=============================================================================
 Inference

 Inference is shared across targets and hints. Constant arguments select a
 SpecializedResult on the generic CodeInstance. CompilerCaching attaches
 JobResults to the corresponding inference result.
=============================================================================#

const TILE_CACHE_OWNER = :cuTile

inference_cache(world::UInt) = CacheView{JobResults}(TILE_CACHE_OWNER, world)
inference_cache(job::TileJob) = inference_cache(job.world)

"""
    infer(cache, mi) -> CodeInstance
    infer(cache, mi, argtypes::Vector{Any}) -> SpecializedResult
    infer(job::TileJob) -> Union{CodeInstance, SpecializedResult}

The inference result of a method instance in `cache`, or of a job: its
CodeInstance, or the const-seeded entry on it for `argtypes`. Runs inference on
a miss.
"""
function infer(cache::CacheView, mi::MethodInstance)
    ci = get(cache, mi, nothing)
    ci === nothing || return ci
    ci = typeinf!(cuTileInterpreter(cache), mi)
    ci === nothing && error("Inference failed for $mi")
    return ci
end

function infer(cache::CacheView, mi::MethodInstance, argtypes::Vector{Any})
    ci = infer(cache, mi)
    entry = specialization(cache, ci, argtypes)
    entry === nothing || return entry
    entry = typeinf!(cache, cuTileInterpreter(cache), mi, argtypes)
    entry === nothing && error("Inference failed for $mi on $argtypes")
    return entry
end

function infer(job::TileJob)
    cache = inference_cache(job)
    job.const_argtypes === nothing && return infer(cache, job.source)
    return infer(cache, job.source, const_argtypes_vector(job))
end

inferred_rettype(ci::Core.CodeInstance) = CC.widenconst(ci.rettype)
inferred_rettype(entry::SpecializedResult) = CC.widenconst(entry.rettype)


#=============================================================================
 Stages

 emit_tile(job) runs inference, structurization, and bytecode generation.
 compile(job) assembles the result with tileiras. Only inference is cached
 within these stages; launch.jl caches the resulting CUBIN.
=============================================================================#

"""
    emit_julia(job::TileJob) -> (IRCode, rettype)
    emit_julia(cache, mi::MethodInstance) -> (IRCode, rettype)

Julia phase: the inferred, optimized IR of a job, or of a callee (subprogram)
method instance in `cache`.
"""
function emit_julia(mi::MethodInstance, inferred)
    src = @something get_source(inferred) error("No inferred source for $mi")
    return CC.inflate_ir(src, mi), inferred_rettype(inferred)
end
emit_julia(job::TileJob) = emit_julia(job.source, infer(job))
emit_julia(cache::CacheView, mi::MethodInstance) = emit_julia(mi, infer(cache, mi))

"""
    emit_structured(ir::IRCode, rettype) -> (StructuredIRCode, rettype, kernel_meta)

Structurize IRCode into StructuredIRCode.
"""
function emit_structured(ir::CC.IRCode, rettype)
    process_meta!(ir)
    kernel_meta = extract_meta(ir)
    sci = StructuredIRCode(ir)
    return (sci, rettype, kernel_meta)
end

"""
    emit_tile(job::TileJob, sci, rettype, kernel_meta) -> (; bytecode, opt_level)
    emit_tile(job::TileJob) -> (; bytecode, opt_level)

Tile IR phase: generate bytecode from StructuredIRCode, resolving the job's
hints against the kernel's `@compiler_options`. Also returns the resolved
`tileiras` optimization level, which is a flag to the assembler rather than
part of the bytecode. The one-argument form runs the preceding stages first.
"""
function emit_tile(job::TileJob, sci::StructuredIRCode, rettype, kernel_meta::Dict{Symbol,Any})
    (; target, params, name) = job.config
    hint(key, explicit) = resolve_hint(explicit, kernel_meta, key, target.sm_arch)
    num_ctas = hint(:num_ctas, params.num_ctas)
    occupancy = hint(:occupancy, params.occupancy)
    num_worker_warps = hint(:num_worker_warps, params.num_worker_warps)
    opt_level = something(hint(:opt_level, params.opt_level), 3)

    bytecode = write_bytecode!(1; version=target.bytecode_version) do writer, func_buf
        emit_kernel!(writer, func_buf, sci, rettype;
                     name, sm_arch=target.sm_arch, num_ctas, occupancy, num_worker_warps,
                     cache=inference_cache(job),
                     const_argtypes=const_argtypes_vector(job))
    end
    return (; bytecode, opt_level)
end

function emit_tile(job::TileJob)
    ir, rettype = emit_julia(job)
    sci, rettype, kernel_meta = emit_structured(ir, rettype)
    return emit_tile(job, sci, rettype, kernel_meta)
end

"""
    compile(job::TileJob) -> Vector{UInt8}

Compile a job to a CUBIN: Tile IR bytecode, assembled with `tileiras` (through
the object cache). Reports the job to the `@device_code_*` hook. Uncached;
`compile_or_lookup` caches for launches.
"""
function compile(job::TileJob)
    sm_arch = target_arch(job)
    validate_tileiras_target(job.config.target.bytecode_version)
    run_compile_hook(job)
    (; bytecode, opt_level) = emit_tile(job)
    dump_bytecode(job.source, bytecode)
    return assemble(bytecode, sm_arch, opt_level)
end


#=============================================================================
 Meta nodes and compilation hints
=============================================================================#

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
Meta hints depend on the architecture, so they are skipped without one.
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

# Dump bytecode to `$JULIA_CUTILE_DUMP_BYTECODE/<file>.ln<line>[.n].cutile`, if set.
const bytecode_dump_lock = ReentrantLock()

function dump_bytecode(mi::MethodInstance, bytecode::Vector{UInt8})
    dump_dir = get(ENV, "JULIA_CUTILE_DUMP_BYTECODE", nothing)
    dump_dir === nothing && return
    Base.@lock bytecode_dump_lock begin
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
    return
end
