#=============================================================================
 Reflection utilities
=============================================================================#

export code_tiled
public code_typed, code_ircode, code_structured
public code_ptx, code_sass

function disassemble_tileir(bytecode::Vector{UInt8}, version::VersionNumber;
                            debuginfo::Bool=false)::String
    validate_bytecode_version(version)
    disassembler_version = tileir_disassembler_version()
    version <= disassembler_version || throw(ArgumentError(
        "Tile IR bytecode v$version cannot be decoded by the selected v$disassembler_version " *
        "disassembler"))
    disasm = tileir_disassembler(; debuginfo)
    mktempdir() do dir
        input_path = joinpath(dir, "kernel.tile")
        write(input_path, bytecode)
        read(`$disasm $input_path`, String)
    end
end

"""
    code_ircode(mi::MethodInstance; world, always_inline=true) -> (IRCode, rettype)

Get optimized IRCode for a MethodInstance using cuTile's overlay method table.
If always_inline=true (default), forces all functions to be inlined.
"""
function code_ircode(mi::MethodInstance; world::UInt=Base.get_world_counter(),
                     always_inline::Bool=true)
    interp = cuTileInterpreter(inference_cache(world); always_inline)
    result = CC.typeinf_ircode(interp, mi, nothing)

    if result === nothing
        throw(ErrorException("Type inference failed for $mi"))
    end

    ir, rettype = result
    return ir, rettype
end

"""
    process_const_argtypes(f, argtypes) -> (stripped, const_argtypes)

Split `Constant{T,V}` types from argtypes for method lookup, and build a
`const_argtypes` vector with `CC.Const(V)` entries for const-seeded inference.

Returns `(stripped, nothing)` when no Constant types are present.
"""
function process_const_argtypes(@nospecialize(f), @nospecialize(argtypes))
    params = argtypes isa DataType ? argtypes.parameters :
             argtypes isa Tuple ? argtypes : fieldtypes(argtypes)
    has_consts = any(T -> T <: Constant || CC.isconstType(T), params)
    stripped_params = map(params) do T
        T <: Constant ? constant_eltype(T) : T
    end
    stripped = Tuple{stripped_params...}
    const_argtypes = if has_consts
        cats = Any[CC.Const(f)]
        for T in params
            if T <: Constant
                push!(cats, CC.Const(constant_value(T)))
            elseif CC.isconstType(T)
                push!(cats, CC.Const(T.parameters[1]))
            else
                push!(cats, T)
            end
        end
        cats
    else
        nothing
    end
    return stripped, const_argtypes
end

constant_eltype(::Type{Constant{T,V}}) where {T,V} = T
constant_value(::Type{Constant{T,V}}) where {T,V} = V


#=============================================================================
 Jobs
=============================================================================#

public tile_job, TileJob

"""
    tile_job(f, argtypes; sm_arch, opt_level, num_ctas, occupancy, num_worker_warps,
             bytecode_version, world) -> TileJob

The [`TileJob`](@ref) for compiling `f` on `argtypes` (which may contain
`Constant{T,V}` types) with the given compilation options. `sm_arch` may be
`nothing` for stages that do not run `tileiras`.
"""
function tile_job(@nospecialize(f), @nospecialize(argtypes);
                  world::UInt=Base.get_world_counter(), kwargs...)
    stripped, const_argtypes = process_const_argtypes(f, argtypes)
    mi = lookup_method_instance(f, stripped; world)
    tile_job(mi, world; const_argtypes, kwargs...)
end

# `(f, tt)` with `Constant` argument types restored, for printing headers.
function job_signature(job::TileJob)
    mi = job.source
    ftype = mi.specTypes.parameters[1]
    f = isdefined(ftype, :instance) ? ftype.instance : ftype
    arg_types = collect(Any, mi.specTypes.parameters[2:end])
    cats = job.config.params.const_argtypes
    if cats !== nothing
        # cats is [Const(f), arg2, ...]; arg_types omits f.
        for i in eachindex(arg_types)
            cats[i+1] isa CC.Const && (arg_types[i] = typeof(Constant(cats[i+1].val)))
        end
    end
    return f, Tuple{arg_types...}
end

# The architecture tileiras will target: the job's, or the active device's.
function resolve_target(job::TileJob, purpose::String)
    sm_arch = job.config.target.sm_arch
    sm_arch === nothing || return sm_arch
    try
        default_sm_arch()
    catch
        throw(ArgumentError("sm_arch must be specified when $purpose without a CUDA device"))
    end
end

# tileiras's -O level: the job's, else the kernel's hint, else 3.
resolve_opt_level(job::TileJob, kernel_meta, target) =
    something(resolve_hint(job.config.params.opt_level, kernel_meta, :opt_level, target), 3)


#=============================================================================
 Stages
=============================================================================#

"""
    code_typed(f, argtypes; world, kwargs...) -> Vector{Any}

Return typed code for a cuTile function. Analogous to `Base.code_typed`.
"""
function code_typed(job::TileJob)
    ir, rettype = emit_julia(inference_cache(job), job.source;
                             const_argtypes=job_const_argtypes(job))
    [ir => rettype]
end
code_typed(@nospecialize(f), @nospecialize(argtypes); kwargs...) =
    code_typed(tile_job(f, argtypes; kwargs...))

"""
    code_structured(f, argtypes; kwargs...) -> Vector{Pair{StructuredIRCode, DataType}}

Return the structured IR for a cuTile function.
"""
function code_structured(job::TileJob; optimize::Bool=true)
    ir, rettype = emit_julia(inference_cache(job), job.source;
                             const_argtypes=job_const_argtypes(job))
    sci, rettype, _ = emit_structured(ir, rettype)
    if optimize
        sci = copy(sci)
        run_passes!(sci)
    end
    [sci => rettype]
end
code_structured(@nospecialize(f), @nospecialize(argtypes); optimize::Bool=true, kwargs...) =
    code_structured(tile_job(f, argtypes; kwargs...); optimize)

"""
    code_tiled([io::IO], f, argtypes; sm_arch, opt_level, num_ctas, occupancy,
               num_worker_warps, remarks=false)

Print the CUDA Tile IR for a Julia function as a textual MLIR representation.
Analogous to `code_llvm`.

Set `remarks=true` to also run `tileiras` and print its optimization remarks.
This requires `tileiras` 13.4 or newer. When no GPU is available, pass `sm_arch`
explicitly.
"""
function code_tiled(io::IO, job::TileJob; debuginfo::Bool=false, remarks::Bool=false)
    (; bytecode, kernel_meta) = emit_bytecode(job)
    bytecode_version = job.config.target.bytecode_version
    print(io, disassemble_tileir(bytecode, bytecode_version; debuginfo))
    if remarks
        tileiras_version() >= v"13.4" || throw(ArgumentError(
            "tileiras optimization remarks require tileiras 13.4 or newer"))
        validate_tileiras_target(bytecode_version)
        target = resolve_target(job, "requesting remarks")
        _, text = run_tileiras(bytecode, target, resolve_opt_level(job, kernel_meta, target);
                               remarks=true)
        if !isempty(text)
            println(io)
            println(io, "// tileiras optimization remarks")
            for line in eachline(IOBuffer(text); keep=true)
                print(io, "// ", line)
            end
        end
    end
end
code_tiled(io::IO, @nospecialize(f), @nospecialize(argtypes);
           debuginfo::Bool=false, remarks::Bool=false, kwargs...) =
    code_tiled(io, tile_job(f, argtypes; kwargs...); debuginfo, remarks)
code_tiled(@nospecialize(f), @nospecialize(argtypes); kwargs...) =
    code_tiled(stdout, f, argtypes; kwargs...)

"""
    compile(job::TileJob) -> Vector{UInt8}

Compile a job to a CUBIN: Tile IR bytecode, assembled with `tileiras` (through
the disk cache). Uncached otherwise; `compile_or_lookup` caches for launches.
"""
function compile(job::TileJob)
    # Report to `@device_code_*`. Launches get here through `invoke_frozen`, but
    # the hook closure lives in the user's latest world — `invokelatest` it.
    if compile_hook[] !== nothing
        Base.invokelatest(compile_hook[], job)
    end
    job = with_target(job, resolve_target(job, "compiling"))
    validate_tileiras_target(job.config.target.bytecode_version)
    (; bytecode, kernel_meta) = emit_bytecode(job)
    dump_bytecode(job.source, bytecode)
    sm_arch = job.config.target.sm_arch
    return assemble(bytecode, sm_arch, resolve_opt_level(job, kernel_meta, sm_arch))
end

"""
    code_ptx([io::IO], f, argtypes; sm_arch, opt_level, num_ctas, occupancy,
             num_worker_warps)

Print the PTX that `tileiras` generates for a Julia function. This shows the
thread-level SIMT program the tile-level kernel is lowered to, with every
compiler decision (thread mapping, CTA size, pipelining, synchronization)
already made. When no GPU is available, pass `sm_arch` explicitly.

!!! warning "Unstable"
    The PTX is recorded by `tileiras` in an undocumented CUBIN section and may
    go away.
"""
code_ptx(io::IO, job::TileJob) = print(io, extract_ptx(compile(job)))
code_ptx(io::IO, @nospecialize(f), @nospecialize(argtypes); kwargs...) =
    code_ptx(io, tile_job(f, argtypes; kwargs...))
code_ptx(@nospecialize(f), @nospecialize(argtypes); kwargs...) =
    code_ptx(stdout, f, argtypes; kwargs...)

"""
    code_sass([io::IO], f, argtypes; sm_arch, opt_level, num_ctas, occupancy,
              num_worker_warps)

Print the SASS machine code that a Julia function compiles to, by assembling
the Tile IR with `tileiras` and disassembling the resulting CUBIN with
`nvdisasm`. When no GPU is available, pass `sm_arch` explicitly. For the
binary a launch actually loaded, use `CUDA.@device_code_sass`.
"""
code_sass(io::IO, job::TileJob) = print(io, disassemble_cubin(compile(job)))
code_sass(io::IO, @nospecialize(f), @nospecialize(argtypes); kwargs...) =
    code_sass(io, tile_job(f, argtypes; kwargs...))
code_sass(@nospecialize(f), @nospecialize(argtypes); kwargs...) =
    code_sass(stdout, f, argtypes; kwargs...)


#=============================================================================
 Device code reflection macros
=============================================================================#

export @device_code_tiled
public @device_code_typed, @device_code_structured
public @device_code_ptx

# Install `inner_hook` as the compile hook for the duration of the expression,
# called once per distinct job.
function emit_hooked_compilation(inner_hook, ex...)
    user_code = ex[end]
    user_kwargs = ex[1:end-1]
    quote
        jobs = Set{TileJob}()
        function outer_hook(job::TileJob)
            job in jobs && return
            push!(jobs, job)
            # the user hook might invoke the compiler again, so disable the hook
            old_hook = $compile_hook[]
            try
                $compile_hook[] = nothing
                $inner_hook(job; $(map(esc, user_kwargs)...))
            finally
                $compile_hook[] = old_hook
            end
        end

        try
            $compile_hook[] = outer_hook
            $(esc(user_code))
        finally
            $compile_hook[] = nothing
        end

        if isempty(jobs)
            error("no kernels executed while evaluating the given expression")
        end
        nothing
    end
end

function tile_hook(inner)
    function (job::TileJob; io::IO=stdout, kwargs...)
        f, tt = job_signature(job)
        println(io, "// $f($(join(tt.parameters, ", ")))")
        println(io)
        inner(io, job; kwargs...)
        println(io)
    end
end

"""
    @device_code_tiled [io=stdout] [remarks=false] expression

Print the Tile IR (MLIR) for all kernels compiled while evaluating the expression.
With `remarks=true`, also print `tileiras` optimization remarks for each kernel.

# Example
```julia
@device_code_tiled @cuda backend=cuTile blocks=grid vadd(a, b, c)
```
"""
macro device_code_tiled(ex...)
    hook = tile_hook((io, job; kwargs...) -> code_tiled(io, job; kwargs...))
    emit_hooked_compilation(hook, ex...)
end

"""
    @device_code_structured [io=stdout] expression

Print the StructuredIRCode for all kernels compiled while evaluating the expression.

# Example
```julia
@device_code_structured @cuda backend=cuTile blocks=grid vadd(a, b, c)
```
"""
macro device_code_structured(ex...)
    hook = tile_hook((io, job; kwargs...) -> println(io, first(only(code_structured(job; kwargs...)))))
    emit_hooked_compilation(hook, ex...)
end

"""
    @device_code_typed [io=stdout] expression

Print the typed Julia IR for all kernels compiled while evaluating the expression.

# Example
```julia
@device_code_typed @cuda backend=cuTile blocks=grid vadd(a, b, c)
```
"""
macro device_code_typed(ex...)
    hook = tile_hook((io, job; kwargs...) -> println(io, first(only(code_typed(job)))))
    emit_hooked_compilation(hook, ex...)
end

"""
    @device_code_ptx [io=stdout] expression

Print the PTX generated by `tileiras` for all kernels compiled while
evaluating the expression. Unstable, like [`code_ptx`](@ref).

# Example
```julia
@device_code_ptx @cuda backend=cuTile blocks=grid vadd(a, b, c)
```
"""
macro device_code_ptx(ex...)
    hook = tile_hook((io, job; kwargs...) -> code_ptx(io, job))
    emit_hooked_compilation(hook, ex...)
end
