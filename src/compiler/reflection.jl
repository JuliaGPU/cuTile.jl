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

Split `Constant{T,V}` types from argtypes for method lookup, and build the
`(Const(f), args...)` tuple with `CC.Const(V)` entries seeding const-prop
inference (see `TileJob`).

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
    has_consts || return stripped, nothing
    const_argtypes = map(params) do T
        if T <: Constant
            CC.Const(constant_value(T))
        elseif CC.isconstType(T)
            CC.Const(T.parameters[1])
        else
            T
        end
    end
    return stripped, (CC.Const(f), const_argtypes...)
end

constant_eltype(::Type{Constant{T,V}}) where {T,V} = T
constant_value(::Type{Constant{T,V}}) where {T,V} = V

function tile_job(@nospecialize(f), @nospecialize(argtypes);
                  world::UInt=Base.get_world_counter(), kwargs...)
    stripped, const_argtypes = process_const_argtypes(f, argtypes)
    mi = lookup_method_instance(f, stripped; world)
    tile_job(mi, world; const_argtypes, kwargs...)
end


#=============================================================================
 Stages
=============================================================================#

"""
    code_typed(job::TileJob) -> Vector{Pair{IRCode, DataType}}
    code_typed(f, argtypes; kwargs...) -> Vector{Pair{IRCode, DataType}}

Return typed code for a cuTile function. Analogous to `Base.code_typed`.
Keyword arguments are those of [`tile_job`](@ref).
"""
function code_typed(job::TileJob)
    ir, rettype = emit_julia(job)
    [ir => rettype]
end
code_typed(@nospecialize(f), @nospecialize(argtypes); kwargs...) =
    code_typed(tile_job(f, argtypes; kwargs...))

GPUCompiler.code_typed(job::TileJob) = code_typed(job)
function GPUCompiler.code_warntype(io::IO, job::TileJob; debuginfo::Symbol=:default)
    inferred = infer(job)
    src = @something get_source(inferred) error("No inferred source for $(job.source)")
    GPUCompiler.code_warntype(io, src, inferred_rettype(inferred); debuginfo)
end


"""
    code_structured(job::TileJob; optimize=true) -> Vector{Pair{StructuredIRCode, DataType}}
    code_structured(f, argtypes; optimize=true, kwargs...)

Return the structured IR for a cuTile function, after the optimization passes
unless `optimize=false`. Keyword arguments are those of [`tile_job`](@ref).
"""
function code_structured(job::TileJob; optimize::Bool=true)
    ir, rettype = emit_julia(job)
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
    code_tiled([io::IO], job::TileJob; debuginfo=false, remarks=false)
    code_tiled([io::IO], f, argtypes; debuginfo=false, remarks=false, kwargs...)

Print the CUDA Tile IR for a Julia function as a textual MLIR representation.
Analogous to `code_llvm`. Keyword arguments are those of [`tile_job`](@ref):
without a CUDA device, pass `sm_arch` explicitly to resolve
architecture-dependent `@compiler_options` hints as a launch would.

Set `remarks=true` to also run `tileiras` and print its optimization remarks.
This requires `tileiras` 13.4 or newer, and a target architecture.
"""
function code_tiled(io::IO, job::TileJob; debuginfo::Bool=false, remarks::Bool=false)
    (; bytecode, opt_level) = emit_tile(job)
    bytecode_version = job.config.target.bytecode_version
    print(io, disassemble_tileir(bytecode, bytecode_version; debuginfo))
    if remarks
        tileiras_version() >= v"13.4" || throw(ArgumentError(
            "tileiras optimization remarks require tileiras 13.4 or newer"))
        validate_tileiras_target(bytecode_version)
        _, text = run_tileiras(bytecode, target_arch(job), opt_level; remarks=true)
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
code_tiled(job::TileJob; kwargs...) = code_tiled(stdout, job; kwargs...)
code_tiled(@nospecialize(f), @nospecialize(argtypes); kwargs...) =
    code_tiled(stdout, f, argtypes; kwargs...)

"""
    code_ptx([io::IO], job::TileJob)
    code_ptx([io::IO], f, argtypes; kwargs...)

Print the PTX that `tileiras` generates for a Julia function. This shows the
thread-level SIMT program the tile-level kernel is lowered to, with every
compiler decision (thread mapping, CTA size, pipelining, synchronization)
already made. Keyword arguments are those of [`tile_job`](@ref); when no GPU is
available, pass `sm_arch` explicitly.

!!! warning "Unstable"
    The PTX is recorded by `tileiras` in an undocumented CUBIN section and may
    go away.
"""
code_ptx(io::IO, job::TileJob) = print(io, extract_ptx(compile(job)))
code_ptx(io::IO, @nospecialize(f), @nospecialize(argtypes); kwargs...) =
    code_ptx(io, tile_job(f, argtypes; kwargs...))
code_ptx(job::TileJob) = code_ptx(stdout, job)
GPUCompiler.code_native(io::IO, job::TileJob) = code_ptx(io, job)
code_ptx(@nospecialize(f), @nospecialize(argtypes); kwargs...) =
    code_ptx(stdout, f, argtypes; kwargs...)

"""
    code_sass([io::IO], job::TileJob)
    code_sass([io::IO], f, argtypes; kwargs...)

Print the SASS machine code that a Julia function compiles to, by assembling
the Tile IR with `tileiras` and disassembling the resulting CUBIN with
`nvdisasm`. Keyword arguments are those of [`tile_job`](@ref); when no GPU is
available, pass `sm_arch` explicitly. For the binary a launch actually loaded,
use `CUDA.@device_code_sass`.
"""
code_sass(io::IO, job::TileJob) = print(io, disassemble_cubin(compile(job)))
code_sass(io::IO, @nospecialize(f), @nospecialize(argtypes); kwargs...) =
    code_sass(io, tile_job(f, argtypes; kwargs...))
code_sass(job::TileJob) = code_sass(stdout, job)
code_sass(@nospecialize(f), @nospecialize(argtypes); kwargs...) =
    code_sass(stdout, f, argtypes; kwargs...)


#=============================================================================
 Device code reflection macros
=============================================================================#

export @device_code_tiled
public @device_code_structured, @device_code_ptx

# The Julia-level stages are inspected with GPUCompiler's macros, which cover
# every backend through the shared hook; cuTile only adds its own stages.
using GPUCompiler: @device_code_typed, @device_code_warntype
public @device_code_typed, @device_code_warntype

# Tile-specific stages ignore other back-ends sharing the compile hook.
emit_hooked_compilation(hook, ex...) =
    GPUCompiler.emit_hooked_compilation(hook, ex...; job_filter=job -> job isa TileJob)

# A hook printing a signature header around `inner(io, job; kwargs...)`.
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
    @device_code_structured [io=stdout] [optimize=true] expression

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
    @device_code_ptx [io=stdout] expression

Print the PTX generated by `tileiras` for all kernels compiled while
evaluating the expression. Unstable, like [`code_ptx`](@ref).

# Example
```julia
@device_code_ptx @cuda backend=cuTile blocks=grid vadd(a, b, c)
```
"""
macro device_code_ptx(ex...)
    hook = tile_hook((io, job) -> code_ptx(io, job))
    emit_hooked_compilation(hook, ex...)
end
