# Host-side kernel launch.
#
# Compiles a Julia function with `TileArray` arguments to Tile IR bytecode,
# runs `tileiras` to lower bytecode → CUBIN, loads the cubin into the active
# CUDA context, and launches it via `cudacall`. Compilation is cached per
# `TileJob` on the kernel's CodeInstance.

using ObjectFile: ObjectFile, readmeta, Sections
using CUDACore: CUDACore, CuArray, CuModule, CuFunction, CuContext, context, cudacall,
                device, capability, AbstractBackend, AbstractKernel, kernel_convert,
                kernel_compile
using CUDA_Compiler_jll
using GPUToolbox: LazyInitialized
using Preferences: @load_preference
using CompilerCaching: ObjCache

using Adapt: Adapt, adapt

"""
    KernelAdaptor

`Adapt.jl` adaptor used to convert host-side launch arguments into their
kernel-side counterparts. `AbstractArray`s become `TileArray`s; `Type`
values become `Constant`s. User-defined structs containing arrays compose
naturally via `Adapt.adapt_structure`.

This is the cuTile analogue of `CUDACore.KernelAdaptor`.
"""
struct KernelAdaptor end

Adapt.adapt_storage(::KernelAdaptor, arr::AbstractArray) = TileArray(arr)
Adapt.adapt_storage(::KernelAdaptor, t::Type) = Constant(t)

# Adapt's defaults for `PermutedDimsArray` and `SubArray` recurse by
# rebuilding the wrapper around `adapt(parent)`. We can't follow that
# pattern because `TileArray` isn't `<:AbstractArray` — strided-wrapper
# state is absorbed into its `sizes`/`strides` fields directly. Short-circuit
# the recursion so the whole wrapper becomes a single TileArray.
Adapt.adapt_structure(::KernelAdaptor, arr::PermutedDimsArray) = TileArray(arr)
Adapt.adapt_structure(::KernelAdaptor, arr::SubArray) = TileArray(arr)

"""
    cuTileconvert(x)

Convert a launch argument to its kernel-side form via `Adapt.adapt` with
`KernelAdaptor()`. Mirrors `CUDACore.cudaconvert`.
"""
cuTileconvert(x) = adapt(KernelAdaptor(), x)


#=============================================================================
 Backend registration — plugs cuTile into CUDACore's `@cuda` dispatch protocol.
=============================================================================#

"""
    TileBackend()

cuTile backend for `@cuda backend=...`. Routes the call through
[`cuTile.cufunction`](@ref) (Tile IR bytecode → tileiras → CUBIN) and
returns a [`TileKernel`](@ref) for launch.

```julia
@cuda backend=cuTile blocks=N my_kernel(a, b, c)        # via DefaultBackend()
@cuda backend=cuTile.TileBackend() blocks=N my_kernel(a, b, c)
```
"""
struct TileBackend <: AbstractBackend end

"""
    DefaultBackend() -> TileBackend

The default cuTile backend, looked up by `@cuda backend=cuTile`. Returns a
[`TileBackend`](@ref). Provided as the module-level resolution hook for
`CUDACore`'s `@cuda` dispatch.
"""
DefaultBackend() = TileBackend()

CUDACore.kernel_convert(::TileBackend, x) = cuTileconvert(x)

CUDACore.kernel_compile(::TileBackend, f::F, tt::TT=Tuple{}; kwargs...) where {F,TT} =
    cufunction(f, tt; kwargs...)


#=============================================================================
 Toolchain and target validation.
=============================================================================#

# User overrides from `LocalPreferences.toml`.
const tileiras_override = @load_preference("tileiras", nothing)
const bytecode_version_override = let s = @load_preference("bytecode_version", nothing)
    s === nothing ? nothing : VersionNumber(s)
end

function parse_compiler_timeout(value)
    value === nothing && return nothing
    value isa Real && !(value isa Bool) ||
        throw(ArgumentError("compiler_timeout_seconds must be a positive number"))
    timeout = Float64(value)
    isfinite(timeout) && timeout > 0 ||
        throw(ArgumentError("compiler_timeout_seconds must be a positive number"))
    return timeout
end

const compiler_timeout =
    parse_compiler_timeout(@load_preference("compiler_timeout_seconds", nothing))

# Cross-session CUBIN cache (see `emit_binary!`). The `disk_cache` preference gates it;
# location and capacity follow Julia's object cache (`JULIA_OBJCACHE_PATH`,
# `JULIA_OBJCACHE_CAPACITY`, `JULIA_OBJCACHE=0`).
const disk_cache_enabled = let setting = @load_preference("disk_cache", true)
    setting isa Bool || throw(ArgumentError("disk_cache must be a boolean"))
    setting
end
# Preferences of the former private disk cache; no longer honoured (warned about in `__init__`).
const stale_cache_preferences = filter(!isnothing, [
    @load_preference("cache_dir", nothing) === nothing ? nothing : "cache_dir",
    @load_preference("cache_size_bytes", nothing) === nothing ? nothing : "cache_size_bytes"])

"Namespace of cuTile's CUBIN entries in the object cache."
const CUBIN_CACHE_NS = "cuTile/cubin"

"""
    CUBIN_CACHE_SCHEMA

Schema of the CUBIN cache key. Bump when the key framing or the meaning of the value
changes incompatibly; no existing entry matches afterwards.
"""
const CUBIN_CACHE_SCHEMA = 4

"""
    cubin_cache_fields(bytecode, sm_arch, opt_level)

Key fields of the CUBIN cache entry for compiling `bytecode` with `tileiras` for
`sm_arch` at `opt_level`: the full `tileiras --version` output, the target, the
optimization level and the bytecode. Hash with
`ObjCache.keyhash(CUBIN_CACHE_SCHEMA, cubin_cache_fields(...)...)`.
"""
cubin_cache_fields(bytecode::Vector{UInt8}, sm_arch::VersionNumber, opt_level::Integer) =
    (tileir_toolchain().compiler_identity, sm_arch, opt_level, bytecode)

"""
    TileCompilerTimeoutError

The external Tile IR compiler exceeded the configured timeout.
"""
struct TileCompilerTimeoutError <: Exception
    seconds::Float64
end

function Base.showerror(io::IO, err::TileCompilerTimeoutError)
    print(io, "tileiras exceeded the ", err.seconds, " second compiler timeout; ",
          "reducing the tile size may reduce compilation time")
end

public TileCompilerTimeoutError

"""
    tileiras_available() -> Bool

Whether a `tileiras` binary is available, either through the `tileiras`
preference or through `CUDA_Compiler_jll`. The JLL lacks `tileiras` when its
selected CUDA version predates the compiler (CUDA 13.2) or when no artifact is
available for the platform.
"""
tileiras_available() =
    tileiras_override !== nothing ||
    (CUDA_Compiler_jll.is_available() && isdefined(CUDA_Compiler_jll, :tileiras))

function check_tileiras_available()
    tileiras_available() && return
    error("The selected CUDA_Compiler_jll (CUDA $(CUDA_Compiler_jll.cuda_version)) " *
          "does not provide `tileiras`; select CUDA 13.2 or newer, or set the " *
          "`tileiras` preference to a local binary.")
end

"""
    tileiras_path() -> String

Path to the `tileiras` binary. Honors the `tileiras` preference when set,
otherwise falls back to `CUDA_Compiler_jll.tileiras_path`.
"""
function tileiras_path()
    tileiras_override === nothing || return tileiras_override
    check_tileiras_available()
    return CUDA_Compiler_jll.tileiras_path
end

"""
    tileiras_root() -> String

CUDA toolkit root passed to `tileiras` as `CUDA_ROOT`. With the `tileiras`
preference set, derived as the parent of the override binary's `bin/`
directory; otherwise the JLL's `artifact_dir`.
"""
tileiras_root() =
    tileiras_override === nothing ? CUDA_Compiler_jll.artifact_dir :
                                    dirname(dirname(tileiras_override))

"""
    tileiras_cmd(args...) -> Cmd

Construct a Cmd to invoke `tileiras` with `args`, with `CUDA_ROOT` set
to [`tileiras_root`](@ref).
"""
function tileiras_cmd(args...)
    check_tileiras_available()
    cmd = tileiras_override === nothing ?
        `$(CUDA_Compiler_jll.tileiras()) $args` :
        Cmd([tileiras_override, args...])
    return addenv(cmd, "CUDA_ROOT" => tileiras_root())
end

"""
    tileir_disassembler(; debuginfo=false) -> Cmd

Return the lazily-resolved Tile IR disassembler. An override requires
`tileirdisasm` from the same toolkit.
"""
function tileir_disassembler(; debuginfo::Bool=false)
    info = get!(discover_tileir_disassembler, tileir_disassembler_cache)
    cmd = info.command
    return debuginfo ? `$cmd $(info.debuginfo_flag)` : cmd
end

tileir_disassembler_version() =
    get!(discover_tileir_disassembler, tileir_disassembler_cache).version

struct TileIRDisassembler
    command::Cmd
    debuginfo_flag::String
    version::VersionNumber
end

const tileir_disassembler_cache = LazyInitialized{TileIRDisassembler}()

function discover_tileir_disassembler()
    if tileiras_override !== nothing
        disasm = joinpath(dirname(tileiras_override), "tileirdisasm")
        isfile(disasm) || error("no `tileirdisasm` next to $(tileiras_path())")
        proc, log = run_and_collect(`$disasm --version`)
        success(proc) || error("tileirdisasm --version failed with exit code " *
                               "$(proc.exitcode):\n$log")
        return TileIRDisassembler(`$disasm`, "--print-debug-info",
                                  parse_tileiras_version(log))
    end
    # Prefer the disassembler shipped next to `tileiras` (CUDA 13.4+), which is
    # version-matched to the compiler and thus decodes any bytecode we emit.
    if CUDA_Compiler_jll.is_available() && isdefined(CUDA_Compiler_jll, :tileirdisasm)
        proc, log = run_and_collect(`$(CUDA_Compiler_jll.tileirdisasm()) --version`)
        success(proc) || error("tileirdisasm --version failed with exit code " *
                               "$(proc.exitcode):\n$log")
        return TileIRDisassembler(CUDA_Compiler_jll.tileirdisasm(), "--print-debug-info",
                                  parse_tileiras_version(log))
    end
    CUDA_Tile_jll.is_available() || error("CUDA_Tile_jll is not available")
    translate = `$(CUDA_Tile_jll.cuda_tile_translate()) --cudatilebc-to-mlir`
    return TileIRDisassembler(translate, "--mlir-print-debuginfo",
                              Base.pkgversion(CUDA_Tile_jll))
end

function run_and_collect(cmd; timeout=compiler_timeout)
    stdout = Pipe()
    proc = run(pipeline(ignorestatus(cmd); stdout, stderr=stdout), wait=false)
    close(stdout.in)
    reader = Threads.@spawn String(read(stdout))
    timed_out = Ref(false)
    timer = if timeout === nothing
        nothing
    else
        Timer(timeout) do _
            if process_running(proc)
                timed_out[] = true
                kill(proc, Base.SIGKILL)
            end
        end
    end
    try
        Base.wait(proc)
    finally
        timer === nothing || close(timer)
    end
    log = strip(fetch(reader))
    timed_out[] && throw(TileCompilerTimeoutError(timeout))
    return proc, log
end

function cleanup_tileiras_remarks(remarks::AbstractString)
    starts = findall(r"(?m)^---(?:[ \t].*)?\r?$", remarks)
    if length(starts) >= 2
        prefix = remarks[firstindex(remarks):prevind(remarks, first(starts).start)]
        documents = String[]
        seen = Set{String}()
        for (i, start) in enumerate(starts)
            stop = i == length(starts) ? lastindex(remarks) :
                   prevind(remarks, starts[i + 1].start)
            document = String(SubString(remarks, start.start, stop))
            if document ∉ seen
                push!(seen, document)
                push!(documents, document)
            end
        end
        remarks = prefix * join(documents)
    end

    lines = readlines(IOBuffer(remarks); keep=true)
    folded = String[]
    i = 1
    while i <= length(lines)
        line = lines[i]
        push!(folded, line)
        if startswith(line, "  - Reason:")
            i += 1
            while i <= length(lines) && lines[i] == line
                i += 1
            end
        else
            i += 1
        end
    end
    return join(folded)
end

function run_tileiras(bytecode::Vector{UInt8}, sm_arch::VersionNumber,
                      opt_level::Int; remarks::Bool=false)
    input_path = tempname() * ".tile"
    output_path = tempname() * ".cubin"
    remarks_path = tempname() * ".remarks.yaml"
    compiled = false
    try
        write(input_path, bytecode)
        args = String[input_path, "-o", output_path,
                      "--gpu-name", format_sm_arch(sm_arch),
                      "-O$(opt_level)", "--lineinfo"]
        if remarks
            append!(args, ["--remark-format=yaml", "--remarks=all",
                           "--remarks-output-file=$(remarks_path)"])
        end
        proc, log = run_and_collect(tileiras_cmd(args...))
        if !success(proc)
            reason = proc.termsignal > 0 ? "tileiras received signal $(proc.termsignal)" :
                                           "tileiras exited with code $(proc.exitcode)"
            msg = "Failed to compile Tile IR ($reason)"
            isempty(log) || (msg *= "\n" * log)
            msg *= "\nIf you think this is a bug, please file an issue and attach $(input_path)"
            if parse(Bool, get(ENV, "BUILDKITE", "false"))
                run(`buildkite-agent artifact upload $(input_path)`)
            end
            error(msg)
        end
        compiled = true
        cubin = read(output_path)
        remark_text = remarks && isfile(remarks_path) ?
                      cleanup_tileiras_remarks(read(remarks_path, String)) : ""
        return cubin, remark_text
    finally
        compiled && rm(input_path, force=true)
        rm(output_path, force=true)
        rm(remarks_path, force=true)
    end
end

"""
    extract_ptx(io::IO) -> String
    extract_ptx(cubin::AbstractVector{UInt8}) -> String

The PTX recorded in a CUBIN. [`run_tileiras`](@ref) compiles with
`--lineinfo`, which makes the assembler embed the PTX it consumed in a
`.nv_debug_ptx_txt` section (one NUL-terminated entry per line, used by
`cuda-gdb` for PTX-level source display).
"""
function extract_ptx(io::IO)
    # CUBINs are ELF64 files with a machine type binutils does not know;
    # ObjectFile.jl only needs the section headers.
    handle = only(readmeta(io))
    section = findfirst(Sections(handle), ".nv_debug_ptx_txt")
    section === nothing && error(
        "CUBIN does not embed PTX (no .nv_debug_ptx_txt section); " *
        "it was compiled without line info")
    lines = split(String(read(section)), '\0'; keepempty=false)
    return join(lines, '\n') * "\n"
end
extract_ptx(cubin::AbstractVector{UInt8}) = extract_ptx(IOBuffer(cubin))

"""
    nvdisasm_cmd(args...) -> Cmd

Construct a Cmd to invoke `nvdisasm` with `args`. With the `tileiras`
preference set, requires `nvdisasm` from the same toolkit; otherwise uses
`CUDA_Compiler_jll`.
"""
function nvdisasm_cmd(args...)
    if tileiras_override !== nothing
        disasm = joinpath(dirname(tileiras_override), "nvdisasm")
        isfile(disasm) || error("no `nvdisasm` next to $(tileiras_path())")
        return `$disasm $args`
    end
    CUDA_Compiler_jll.is_available() && isdefined(CUDA_Compiler_jll, :nvdisasm) ||
        error("CUDA_Compiler_jll does not provide `nvdisasm`")
    return `$(CUDA_Compiler_jll.nvdisasm()) $args`
end

"""
    disassemble_cubin(cubin::Vector{UInt8}) -> String

Disassemble a CUBIN to SASS using `nvdisasm`.
"""
function disassemble_cubin(cubin::Vector{UInt8})
    mktempdir() do dir
        path = joinpath(dir, "kernel.cubin")
        write(path, cubin)
        read(nvdisasm_cmd(path), String)
    end
end

const TILEIRAS_VERSION_REGEX = r"V(\d+\.\d+\.\d+)"

function parse_tileiras_version(log::AbstractString)
    m = match(TILEIRAS_VERSION_REGEX, log)
    m === nothing && throw(ArgumentError(
        "version output does not contain a V<major>.<minor>.<patch> token:\n$log"))
    return VersionNumber(m.captures[1])
end

function select_bytecode_version(max_version::VersionNumber,
                                 requested::Union{VersionNumber, Nothing})
    validate_bytecode_version(max_version)
    requested === nothing && return max_version
    validate_bytecode_version(requested)
    requested <= max_version || throw(ArgumentError(
        "bytecode_version=v$requested was requested, but the selected tileiras only " *
        "accepts through v$max_version"))
    return requested
end

function probe_max_bytecode_version()
    last_log = ""
    for version in reverse(SUPPORTED_BYTECODE_VERSIONS)
        # Empty bytecode (no functions) — just a header + section terminator.
        probe = write_bytecode!(0; version) do writer, func_buf
        end
        input_path = tempname() * ".tile"
        output_path = tempname() * ".cubin"
        try
            write(input_path, probe)
            proc, log = run_and_collect(tileiras_cmd(input_path, "-o", output_path,
                                                     "--gpu-name", "sm_100"))
            success(proc) && return version
            last_log = log
        finally
            rm(input_path, force=true)
            rm(output_path, force=true)
        end
    end
    error("tileiras rejected every supported bytecode version " *
          "($(join(reverse(SUPPORTED_BYTECODE_VERSIONS), ", "))); last log:\n$last_log")
end

struct TileIRToolchain
    tileiras_version::VersionNumber
    compiler_identity::String
    max_bytecode_version::VersionNumber
    bytecode_version::VersionNumber
end

const tileir_toolchain_cache = LazyInitialized{TileIRToolchain}()

function discover_tileir_toolchain()
    bytecode_version_override === nothing ||
        validate_bytecode_version(bytecode_version_override)
    check_tileiras_available()

    proc, identity = run_and_collect(tileiras_cmd("--version"))
    success(proc) || error("tileiras --version failed with exit code " *
                           "$(proc.exitcode):\n$identity")
    version = parse_tileiras_version(identity)
    max_version = probe_max_bytecode_version()
    selected_version = select_bytecode_version(max_version, bytecode_version_override)
    return TileIRToolchain(version, identity, max_version, selected_version)
end

tileir_toolchain() = get!(discover_tileir_toolchain, tileir_toolchain_cache)

"""
    tileiras_version() -> VersionNumber

Version of the selected `tileiras` executable. The executable is resolved and
queried lazily, once per process.
"""
tileiras_version() = tileir_toolchain().tileiras_version

"""
    bytecode_version() -> VersionNumber

The validated Tile IR bytecode version emitted by default. This is either the
highest version accepted by the selected `tileiras`, or the `bytecode_version`
preference after checking that both cuTile and `tileiras` support it.
"""
bytecode_version() = tileir_toolchain().bytecode_version

function validate_tileiras_target(version::VersionNumber)
    validate_bytecode_version(version)
    max_version = tileir_toolchain().max_bytecode_version
    version <= max_version || throw(ArgumentError(
        "Tile IR bytecode v$version cannot be compiled by the selected tileiras, which " *
        "accepts through v$max_version"))
    return nothing
end

"""
    tile_ir_requirement(cap::VersionNumber) -> Union{Tuple{String,VersionNumber}, Nothing}

The architecture-family name and the minimum bytecode version Tile IR requires
on a device of compute capability `cap`, or `nothing` if Tile IR is not
supported on that capability at all. Pure (no device access) so the gate logic
in [`check_tile_ir_support`] can be unit-tested without a GPU.
"""
function tile_ir_requirement(cap::VersionNumber)
    if cap >= v"10.0"       # Blackwell
        return ("Blackwell", v"13.1")
    elseif cap >= v"9.0"    # Hopper
        return ("Hopper", v"13.3")
    elseif cap >= v"8.0"    # Ampere / Ada
        return ("Ampere/Ada", v"13.2")
    else
        return nothing
    end
end

"""
    check_tile_ir_support(sm_arch)

Validate that the selected bytecode version supports Tile IR on `sm_arch`.
Returns the bytecode version cuTile should emit, provided it meets the target's
minimum requirement (Blackwell ≥ v13.1, Hopper ≥ v13.3, Ampere/Ada ≥ v13.2).
"""
function check_tile_ir_support(sm_arch::VersionNumber)
    version = bytecode_version()
    validate_tile_ir_target(sm_arch, version)
    return version
end

function validate_tile_ir_target(sm_arch::VersionNumber, version::VersionNumber)
    validate_bytecode_version(version)
    requirement = tile_ir_requirement(sm_arch)
    requirement === nothing && throw(ArgumentError(
        "Tile IR is not supported on compute capability $sm_arch ($(format_sm_arch(sm_arch)))"))
    arch, min_version = requirement
    version >= min_version || throw(ArgumentError(
        "Tile IR on $arch ($(format_sm_arch(sm_arch))) requires bytecode v$min_version+, " *
        "got v$version"))
    return nothing
end

#=============================================================================
 Argument-type unwrapping for cufunction.
=============================================================================#

"""
    unwrap_argtypes(f, tt) -> (argtypes::Type{<:Tuple}, const_argtypes::Union{Vector{Any},Nothing})

Compile-time-specialized derivation of:
- `argtypes::Type{<:Tuple}` — concrete dispatch tuple for `method_instance(f, argtypes)`,
  with `Constant{T,V}` slots unwrapped to `T`.
- `const_argtypes::Vector{Any}` — `[CC.Const(f), ...args]` with `Constant{T,V}` slots
  replaced by `CC.Const(V)`, for const-prop inference. `nothing` when no `Constant`
  arguments are present (skips the const-seeding pipeline entirely).

`@generated` so the unwrapped `Tuple` type and the `Constant`-vs-not branching
fold to constants at the call site. Only the `Vector{Any}` allocation and the
`CC.Const(...)` boxes for runtime values survive to runtime.
"""
@generated function unwrap_argtypes(@nospecialize(f), ::Type{TT}) where TT <: Tuple
    unwrapped = map(t -> t <: Constant ? constant_eltype(t) : t, TT.parameters)
    argtypes_T = Tuple{unwrapped...}
    has_consts = any(t -> t <: Constant, TT.parameters)
    has_consts || return :(($argtypes_T, nothing))

    cats_exprs = Any[:(CC.Const(f))]
    for t in TT.parameters
        if t <: Constant
            push!(cats_exprs, :(CC.Const($(t.parameters[2]))))
        else
            push!(cats_exprs, t)
        end
    end
    return :(($argtypes_T, Any[$(cats_exprs...)]))
end


#=============================================================================
 Compilation: bytecode → CUBIN → CuFunction.
=============================================================================#

"""
    assemble(bytecode, sm_arch, opt_level) -> Vector{UInt8}

Assemble Tile IR bytecode to a CUBIN with `tileiras`, through the disk cache.
"""
function assemble(bytecode::Vector{UInt8}, sm_arch::VersionNumber, opt_level::Int)
    # Cross-session cache of the tileiras output. The key covers every input
    # that changes the CUBIN: bytecode, sm_arch, opt_level and the tileiras
    # identity, so different compiler builds never collide. `bytecode_version`
    # is encoded in the bytecode itself, so it's covered transitively. CUBIN is
    # address-free, so it is always persistable.
    disk_cache_enabled || return first(run_tileiras(bytecode, sm_arch, opt_level))
    return ObjCache.get!(CUBIN_CACHE_NS, cubin_cache_fields(bytecode, sm_arch, opt_level)...;
                         schema=CUBIN_CACHE_SCHEMA, persistable=true) do
        first(run_tileiras(bytecode, sm_arch, opt_level))
    end
end

# The compilation results of a job: its CUBIN, shared by every context, and the
# session-local kernels linked from it.
mutable struct CuTileResults
    cuda_bin::Union{Nothing, Vector{UInt8}}
    # linear-scanned by context; usually holds a single entry
    kernels::Vector{Tuple{CuContext, Any}}
    CuTileResults() = new(nothing, Tuple{CuContext, Any}[])
end

# Results for every configuration of a kernel, attached to its CodeInstance by
# CompilerCaching (and persisted with it into package images). One CodeInstance
# serves every target, hint and constant configuration of a kernel, so entries are
# keyed by config; the CodeInstance itself outlives any one world.
mutable struct TileResults
    entries::Vector{Pair{TileConfig, CuTileResults}}
    TileResults() = new(Pair{TileConfig, CuTileResults}[])
end

# The results struct for `job`, or `nothing` while its kernel has not been inferred.
# Once it has, an empty struct is created on first access and returned ever after.
function cached_results(job::TileJob)
    ci = get(inference_cache(job), job.source, nothing)
    ci === nothing && return nothing
    results = CompilerCaching.results(TileResults, ci)
    for (config, res) in results.entries
        # configs are immutable, so `===` compares them structurally
        config === job.config && return res
    end
    res = CuTileResults()
    push!(results.entries, job.config => res)
    return res
end

const compile_lock = ReentrantLock()

"""
    compile_or_lookup(job::TileJob) -> CuTileResults

The cached compilation results for `job`, running the compiler on a miss. The
`compile_hook` check forces the compile path so that `@device_code_*` observe
the compilation even on a hit. No CUDA context required.
"""
function compile_or_lookup(job::TileJob)::CuTileResults
    # A targetless job is useful for pre-tileiras reflection, but a cached CUBIN
    # must be identified by the architecture it was assembled for. Resolve the
    # target before looking up results so it becomes part of the cache identity.
    job = with_target(job, resolve_target(job, "compiling"))
    Base.@lock compile_lock begin
        res = cached_results(job)
        if res === nothing || res.cuda_bin === nothing || compile_hook[] !== nothing
            cubin = compile(job)
            res = @something res cached_results(job)
            res.cuda_bin = cubin
        end
        return res
    end
end

"""
    link(job::TileJob, res::CuTileResults) -> CuFunction

Load the job's CUBIN onto the active CUDA context.
"""
function link(job::TileJob, res::CuTileResults)
    cumod = CuModule(res.cuda_bin::Vector{UInt8})
    return CuFunction(cumod, job.config.name)
end


#=============================================================================
 TileKernel + cufunction: hoisted compilation step.

 Mirrors the `cufunction(f, tt) -> HostKernel` pattern in CUDACore. Once
 obtained, calling `(::TileKernel)(args...; blocks=…)` skips the MI lookup
 and cache dispatch — only argument flatten + `cudacall` runs.
=============================================================================#

"""
    TileKernel{F, TT}

A compiled cuTile kernel. Returned by [`cuTile.cufunction`](@ref) and the
target of `(::TileKernel)(args...; blocks, …)` calls. Concrete subtype of
`CUDACore.AbstractKernel`.
"""
struct TileKernel{F, TT} <: AbstractKernel{F, TT}
    f::F
    fun::CuFunction
end

"""
    cuTile.cufunction(f, tt=Tuple{}; sm_arch=nothing, opt_level=nothing,
                      num_ctas=nothing, occupancy=nothing, num_worker_warps=nothing,
                      name=nothing) -> TileKernel

Compile `f` for the cuTile backend. `tt` is the tuple of *converted*
argument types (i.e. after `cuTileconvert`/`Adapt.adapt(KernelAdaptor(), …)`).
Compilation is cached; calling `cufunction` repeatedly with the same
`(f, tt, opts)` is O(1) after the first compile.

Mirrors `CUDACore.cufunction` but produces a [`TileKernel`](@ref). Results are
stored on the kernel's Julia `CodeInstance`, so invalidation rides on Julia's
normal CI lifecycle.
"""
function cufunction(@nospecialize(f), tt::Type{<:Tuple}=Tuple{};
                    sm_arch::Union{VersionNumber, Nothing}=nothing,
                    opt_level::Union{Int, Nothing}=nothing,
                    num_ctas::Union{Int, Nothing}=nothing,
                    occupancy::Union{Int, Nothing}=nothing,
                    num_worker_warps::Union{Int, Nothing}=nothing,
                    name::Union{String, Nothing}=nothing)
    resolved_sm_arch = sm_arch !== nothing ? sm_arch : default_sm_arch()
    bytecode_version = check_tile_ir_support(resolved_sm_arch)

    # Single pass over `tt.parameters`: build the unwrapped argtypes tuple
    # (Constant{T,V} → T for MI lookup) and the const_argtypes vector
    # (Constant{T,V} → CC.Const(V) for inference) together. cufunction
    # specializes on `tt`, so this loop unrolls per kernel signature.
    argtypes, const_argtypes = unwrap_argtypes(f, tt)

    # The compilation pipeline (typeinf!, codegen, bytecode emission) gets
    # invalidated by any package that defines methods on Base.Compiler hooks
    # like `OptimizationParams(::AbstractInterpreter)`. To reuse precompiled
    # native code, run the pipeline in the world captured at __init__.
    opts = (; sm_arch=resolved_sm_arch, bytecode_version, opt_level, num_ctas, occupancy,
              num_worker_warps, name)
    invoke_frozen(cufunction_compile, f, tt, argtypes, const_argtypes, opts)::TileKernel{Core.Typeof(f), tt}
end

# The job of a launch: the kernel's MethodInstance for the unwrapped argument
# types, in the current world, with the launch's options.
function launch_job(@nospecialize(f), @nospecialize(argtypes),
                    const_argtypes::Union{Vector{Any}, Nothing}; kwargs...)
    world = Base.get_world_counter()
    mi = method_instance(f, argtypes; world)
    mi === nothing && throw(MethodError(f, argtypes))
    if !Base.isdispatchtuple(mi.specTypes)
        sig = Base.signature_type(f, argtypes)
        mi = CC.specialize_method(mi.def, sig, mi.sparam_vals)::Core.MethodInstance
    end
    return tile_job(mi, world; const_argtypes, kwargs...)
end

# Inner compilation routine; called via `invoke_frozen` so its method dispatches
# happen in the world captured at __init__, reusing precompiled native code
# even when later-loaded packages would otherwise have invalidated it.
function cufunction_compile(@nospecialize(f), @nospecialize(tt), @nospecialize(argtypes),
                             const_argtypes::Union{Vector{Any}, Nothing}, opts::NamedTuple)
    validate_tile_ir_target(opts.sm_arch, opts.bytecode_version)
    job = launch_job(f, argtypes, const_argtypes; opts...)
    res = compile_or_lookup(job)

    # Resolve the kernel for the active context. `CuFunction`s are session-local
    # handles, so they live in the results struct's linear cache rather than being
    # persisted; the scan is almost always over a single entry.
    ctx = context()
    for (cached_ctx, cached_kernel) in res.kernels
        cached_ctx === ctx && return cached_kernel::TileKernel{Core.Typeof(f), tt}
    end
    kernel = TileKernel{Core.Typeof(f), tt}(f, link(job, res))
    # don't cache session-local handles while generating output: the results struct
    # is serialized into the package image along with its CodeInstance.
    if ccall(:jl_generating_output, Cint, ()) != 1
        push!(res.kernels, (ctx, kernel))
    end
    return kernel
end

# Tile IR has a 24-bit grid limit per dimension.
const _MAX_GRID_DIM = (1 << 24) - 1

# Recursively expand `val_expr::T` into a flat list of (expr, type) pairs that
# match the kernel's flat scalar parameter signature: TileArray expands to
# (ptr, sizes..., strides...), ghost types contribute nothing, primitives pass
# through, structs recurse field-by-field. Used by the `@generated` launch path
# to fold the flatten step into compile-time call construction.
function _flatten_static!(arg_exprs, type_exprs, @nospecialize(T), val_expr)
    if T <: TileArray
        push!(arg_exprs, :($val_expr.ptr))
        push!(type_exprs, fieldtype(T, :ptr))
        sizes_T = fieldtype(T, :sizes)
        for i in 1:fieldcount(sizes_T)
            push!(arg_exprs, :($val_expr.sizes[$i]))
            push!(type_exprs, fieldtype(sizes_T, i))
        end
        strides_T = fieldtype(T, :strides)
        for i in 1:fieldcount(strides_T)
            push!(arg_exprs, :($val_expr.strides[$i]))
            push!(type_exprs, fieldtype(strides_T, i))
        end
    elseif is_ghost_type(T)
        # contribute nothing
    elseif isprimitivetype(T)
        push!(arg_exprs, val_expr)
        push!(type_exprs, T)
    else
        for i in 1:fieldcount(T)
            field_T = fieldtype(T, i)
            _flatten_static!(arg_exprs, type_exprs, field_T,
                             :(getfield($val_expr, $i)))
        end
    end
    return
end

# `convert=Val(...)` is the AbstractKernel callable convention from CUDACore;
# `@cuda` passes `convert=Val(false)` because args were already converted at
# expansion time. We always treat args as already-converted — direct
# `kernel(args...)` calls without the macro should pass converted args.
#
# `@generated` so the flatten/typeof work folds to a direct cudacall expression
# at compile time. Mirrors the LLVM `HostKernel` generated callable in CUDACore;
# without it, runtime `Iterators.flatten` + `map(typeof, ...)` + tuple splatting
# costs ~400 ns per launch even for trivial kernels.
@generated function (k::TileKernel)(args::Vararg{Any, N}; blocks=1, threads=1,
                                    convert=Val(false), kwargs...) where {N}
    arg_exprs = Any[]
    type_exprs = Any[]
    for i in 1:N
        _flatten_static!(arg_exprs, type_exprs, args[i], :(args[$i]))
    end
    # Trailing implicit KernelState slot — matches the bytecode kernel signature.
    push!(arg_exprs, :(state.seed))
    push!(type_exprs, UInt32)

    quote
        state = KernelState()
        grid_dims = blocks isa Integer ? (blocks,) : blocks
        for (i, dim) in enumerate(grid_dims)
            if dim > _MAX_GRID_DIM
                error("Grid[$i] exceeds 24-bit limit: max=$_MAX_GRID_DIM, got=$dim. " *
                      "Use multiple kernel launches for larger workloads.")
            end
        end
        # Note: threads=1 lets the driver use the cubin's EIATTR_REQNTID metadata
        # which specifies the actual thread count (typically 128 for Tile kernels).
        cudacall(k.fun, Tuple{$(type_exprs...)}, $(arg_exprs...);
                 blocks=grid_dims, threads, kwargs...)
        return nothing
    end
end


#=============================================================================
 launch: high-level convenience wrapper, retained as the function-call entry
 point alongside `@cuda backend=cuTile …`.
=============================================================================#

"""
    launch(f, grid, args...; sm_arch=nothing, opt_level=nothing,
           num_ctas=nothing, occupancy=nothing, num_worker_warps=nothing,
           dependent=false, stream=CUDACore.stream(), name=nothing)

Compile and launch a Tile IR kernel. `args` are converted via
`cuTileconvert` (CuArray → TileArray, Type → Constant). Equivalent to
`@cuda backend=cuTile blocks=grid f(args...)` modulo
slight kwarg naming.

Set `dependent=true` on a consumer kernel to allow it to overlap with the
preceding kernel in `stream`. The producer should call
[`grid_dependency_control_launch_dependents`](@ref), and the consumer must call
[`grid_dependency_control_wait`](@ref) before accessing the producer's results.

# Example
```julia
using CUDA, cuTile

a = CUDA.zeros(Float32, 1024); b = CUDA.ones(Float32, 1024); c = similar(a)

function vadd_kernel(a::cuTile.TileArray{Float32,1}, b::cuTile.TileArray{Float32,1},
                     c::cuTile.TileArray{Float32,1})
    pid = cuTile.bid(1)
    ta = cuTile.load(a, (pid,), (16,))
    tb = cuTile.load(b, (pid,), (16,))
    cuTile.store(c, (pid,), ta + tb)
    return
end

cuTile.launch(vadd_kernel, 64, a, b, c)
```
"""
function launch(@nospecialize(f), grid, args...;
                sm_arch::Union{VersionNumber, Nothing}=nothing,
                opt_level::Union{Int, Nothing}=nothing,
                num_ctas::Union{Int, Nothing}=nothing,
                occupancy::Union{Int, Nothing}=nothing,
                num_worker_warps::Union{Int, Nothing}=nothing,
                dependent::Bool=false,
                stream=CUDACore.stream(),
                name::Union{String, Nothing}=nothing)
    converted = map(cuTileconvert, args)
    tt = Tuple{map(Core.Typeof, converted)...}
    kernel = cufunction(f, tt; sm_arch, opt_level, num_ctas, occupancy, num_worker_warps, name)
    kernel(converted...; blocks=grid, dependent, stream)
    return nothing
end

"""
    default_sm_arch() -> VersionNumber

Get the compute capability of the current CUDA device as a VersionNumber.
Returns e.g. `v"12.0"` for compute capability 12.0.
"""
default_sm_arch() = capability(device())


#=============================================================================
 Version reporting
=============================================================================#

"""
    versioninfo([io::IO=stdout])

Print information about the active `tileiras`, the bytecode version
cuTile.jl will emit for it, and any user overrides set via
`LocalPreferences.toml`.
"""
function versioninfo(io::IO=stdout)
    println(io, "cuTile toolchain:")

    toolchain = tileir_toolchain()
    install = tileiras_override === nothing ? "artifact installation" : "local installation"
    println(io, "- tileiras $(toolchain.tileiras_version), $install")

    bv = toolchain.bytecode_version
    bv_src = bytecode_version_override === nothing ? "auto-detected" : "set via preference"
    max_bv = toolchain.max_bytecode_version
    max_suffix = bv == max_bv ? "" : " (tileiras accepts through v$(max_bv.major).$(max_bv.minor))"
    println(io, "- bytecode v$(bv.major).$(bv.minor), $bv_src$max_suffix")
end
