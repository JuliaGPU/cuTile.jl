# Host-side kernel launch.
#
# Compiles a Julia function with `TileArray` arguments to Tile IR bytecode,
# runs `tileiras` to lower bytecode → CUBIN, loads the cubin into the active
# CUDA context, and launches it via `cudacall`. Compilation is cached per
# `(MethodInstance, sm_arch, opt_level, num_ctas, occupancy, num_worker_warps, bytecode_version)`.

using CUDACore: CUDACore, CuArray, CuModule, CuFunction, cudacall, device, capability,
                AbstractBackend, AbstractKernel, kernel_convert, kernel_compile
using CUDA_Compiler_jll
using GPUToolbox: LazyInitialized
using Preferences: @load_preference

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
 Cache sharding key.
=============================================================================#

# Pack a `VersionNumber` into a `UInt16` as `(major << 8) | minor`. Lossy: drops
# patch/prerelease/build (none of which we need for SM architectures or Tile IR
# bytecode versions). Used by `TileCacheKey` to keep the owner isbits.
@inline pack_version(v::VersionNumber) = (UInt16(v.major) << 8) | UInt16(v.minor)
@inline unpack_version(x::UInt16) = VersionNumber(Int(x >> 8), Int(x & 0xff))

# isbits sentinel codec for `Union{Int, Nothing}` hint fields (`opt_level`,
# `num_ctas`, `occupancy`, `num_worker_warps`). `-1` is unused as a value, so we
# use it for `nothing`.
const _UNSET = -1
@inline pack_hint(x::Union{Int, Nothing}) = x === nothing ? _UNSET : x
@inline unpack_hint(x::Int) = x == _UNSET ? nothing : x

"""
    TileCacheKey

Owner stamped onto every launch-cached `CodeInstance`. `lookup` (called once
per `cufunction` via `ensure_compiled`) ccalls `jl_rettype_inferred` with the
owner as `Any`, which forces a heap box; making the owner `isbits` shrinks the
box from ~80 B (the previous `Tuple{Symbol, NamedTuple{…}}` shape, dominated by
`VersionNumber`'s non-isbits prerelease/build tuples) to ~32 B.

`VersionNumber` fields are packed into `UInt16` and `Union{Int, Nothing}`
hint fields use `_UNSET` (`-1`) as the `nothing` sentinel. Decoding back to
the original types happens once per cache miss in `emit_binary!` /
`emit_tile!`, never on the hot lookup path.
"""
struct TileCacheKey
    sm_arch::UInt16
    bytecode_version::UInt16
    opt_level::Int
    num_ctas::Int
    occupancy::Int
    num_worker_warps::Int
end
TileCacheKey(sm_arch::VersionNumber, bytecode_version::VersionNumber,
             opt_level::Union{Int, Nothing}, num_ctas::Union{Int, Nothing},
             occupancy::Union{Int, Nothing}, num_worker_warps::Union{Int, Nothing}) =
    TileCacheKey(pack_version(sm_arch), pack_version(bytecode_version),
                 pack_hint(opt_level), pack_hint(num_ctas), pack_hint(occupancy),
                 pack_hint(num_worker_warps))


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

"""
    reflection_bytecode_version() -> VersionNumber

The default Tile IR bytecode version for reflection (`code_tiled`): the emitted
[`bytecode_version`](@ref), clamped to the newest version the selected
disassembler can decode. The kernel bytecode passed to `tileiras` may thus be
newer than what reflection displays.
"""
function reflection_bytecode_version()
    version = bytecode_version()
    disasm = tileir_disassembler_version()
    version <= disasm && return version
    supported = filter(<=(disasm), SUPPORTED_BYTECODE_VERSIONS)
    isempty(supported) && return version  # let disassemble_tileir raise the error
    return maximum(supported)
end

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
    emit_binary!(cache, mi, ci, res; const_argtypes=nothing) -> Vector{UInt8}

Cached binary phase: compile Tile IR bytecode to CUBIN using tileiras.
"""
function emit_binary!(cache::CacheView, mi::Core.MethodInstance,
                      ci::Core.CodeInstance, res::CuTileResults;
                      const_argtypes::Union{Vector{Any}, Nothing}=nothing)
    # Recurse first — emit_structured! at the bottom of the chain fires
    # `compile_hook` for `@device_code_*` reflection, which must run on every
    # launch even when downstream artifacts are fully cached.
    bytecode = emit_tile!(cache, mi, ci, res; const_argtypes)

    res.cuda_bin !== nothing && return res.cuda_bin

    sm_arch = unpack_version(cache.owner.sm_arch)

    # Resolve opt_level here (not in emit_tile) because it's a tileiras flag, not bytecode.
    # num_ctas/occupancy/num_worker_warps are resolved in emit_tile because they're encoded in bytecode.
    _, _, kernel_meta = res.julia_ir
    opt_level = something(resolve_hint(unpack_hint(cache.owner.opt_level),
                                       kernel_meta, :opt_level, sm_arch), 3)

    # Disk cache lookup. The hash covers every input that changes the CUBIN
    # — bytecode + sm_arch + opt_level + tileiras identity — so different
    # compiler builds never collide. `bytecode_version` is encoded
    # in the bytecode itself, so it's covered transitively by the bytecode hash.
    dc = DiskCache.global_cache()
    cache_key = nothing
    if dc !== nothing
        identity = tileir_toolchain().compiler_identity
        cache_key = DiskCache.compute_key(bytecode, sm_arch, opt_level, identity)
        cubin = try
            DiskCache.get(dc, cache_key)
        catch err
            @debug "cuTile disk cache lookup failed" exception=(err, catch_backtrace())
            nothing
        end
        if cubin !== nothing
            res.cuda_bin = cubin
            return cubin
        end
    end

    # Run tileiras to produce CUBIN
    res.cuda_bin, _ = run_tileiras(bytecode, sm_arch, opt_level)

    if cache_key !== nothing
        try
            DiskCache.put!(dc, cache_key, res.cuda_bin)
        catch err
            @debug "cuTile disk cache store failed" exception=(err, catch_backtrace())
        end
    end

    return res.cuda_bin
end

"""
    link(cache, mi, ci, res) -> CuFunction

GPU-side link phase: load the CUBIN cached in `res.cuda_bin` (produced by
[`compile`](@ref)) onto the active CUDA device and return the resulting
`CuFunction`.
"""
function link(cache::CacheView, mi::Core.MethodInstance,
              ci::Core.CodeInstance, res::CuTileResults)
    res.cuda_func !== nothing && return res.cuda_func

    kernel_name = sanitize_name(string(mi.def.name))
    cumod = CuModule(res.cuda_bin::Vector{UInt8})
    cufunc = CuFunction(cumod, kernel_name)
    res.cuda_func = cufunc
    return cufunc
end


#=============================================================================
 TileKernel + cufunction: hoisted compilation step.

 Mirrors the `cufunction(f, tt) -> HostKernel` pattern in CUDACore. Once
 obtained, calling `(::TileKernel)(args...; blocks=…)` skips the MI lookup
 and CompilerCaching dispatch — only argument flatten + `cudacall` runs.
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

Mirrors `CUDACore.cufunction` but produces a [`TileKernel`](@ref). Caching
is delegated to CompilerCaching: the resulting `TileKernel` is stored in
the `CuTileResults` attached to the underlying Julia `CodeInstance`, so
invalidation rides on Julia's normal CI lifecycle.
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

    key = TileCacheKey(resolved_sm_arch, bytecode_version, opt_level, num_ctas, occupancy,
                       num_worker_warps)

    # Single pass over `tt.parameters`: build the unwrapped argtypes tuple
    # (Constant{T,V} → T for MI lookup) and the const_argtypes vector
    # (Constant{T,V} → CC.Const(V) for inference) together. cufunction
    # specializes on `tt`, so this loop unrolls per kernel signature.
    argtypes, const_argtypes = unwrap_argtypes(f, tt)

    # The compilation pipeline (typeinf!, codegen, bytecode emission) gets
    # invalidated by any package that defines methods on Base.Compiler hooks
    # like `OptimizationParams(::AbstractInterpreter)`. To reuse precompiled
    # native code, run the pipeline in the world captured at __init__.
    invoke_frozen(cufunction_compile, f, tt, argtypes, const_argtypes, key)::TileKernel{Core.Typeof(f), tt}
end

"""
    compile(f, argtypes, const_argtypes, key) -> (cache, mi, ci, res)

Host-side compile phase: run inference, codegen, bytecode emission, and
`tileiras` to produce a CUBIN. Returns the compilation cache state needed
by [`link`](@ref) to load the result onto the GPU. No CUDA context required.
"""
function compile(@nospecialize(f), @nospecialize(argtypes),
                 const_argtypes::Union{Vector{Any}, Nothing},
                 key::TileCacheKey)
    validate_tile_ir_target(unpack_version(key.sm_arch),
                            unpack_version(key.bytecode_version))
    world = Base.get_world_counter()
    mi = method_instance(f, argtypes; world)
    mi === nothing && throw(MethodError(f, argtypes))
    if !Base.isdispatchtuple(mi.specTypes)
        sig = Base.signature_type(f, argtypes)
        mi = CC.specialize_method(mi.def, sig, mi.sparam_vals)::Core.MethodInstance
    end

    cache = CacheView{CuTileResults}(key, world)

    # Single resolution of (ci, res) up front — threaded through the emit_*!
    # chain so each phase only does its own short-circuit, not redundant
    # cache lookups. The cached compilation results are attached to the
    # underlying `CodeInstance` via CompilerCaching; the `TileKernel` wrapper
    # rides along in the same `CuTileResults`, so kernel-instance lifecycle
    # follows the CI's instead of needing a separate global Dict.
    ci, res = ensure_compiled(cache, mi, const_argtypes)

    # Always walk the emit chain (each phase short-circuits on its own cached
    # field, but `emit_structured!` also fires `compile_hook` for reflection,
    # which has to run on every launch even when the cube/cufunc is cached).
    emit_binary!(cache, mi, ci, res; const_argtypes)
    return cache, mi, ci, res
end

# Inner compilation routine; called via `invoke_frozen` so its method dispatches
# happen in the world captured at __init__, reusing precompiled native code
# even when later-loaded packages would otherwise have invalidated it.
function cufunction_compile(@nospecialize(f), @nospecialize(tt), @nospecialize(argtypes),
                             const_argtypes::Union{Vector{Any}, Nothing},
                             key::TileCacheKey)
    cache, mi, ci, res = compile(f, argtypes, const_argtypes, key)

    cufunc = link(cache, mi, ci, res)

    res.tile_kernel !== nothing && return res.tile_kernel::TileKernel{Core.Typeof(f), tt}
    kernel = TileKernel{Core.Typeof(f), tt}(f, cufunc)
    res.tile_kernel = kernel
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
           num_ctas=nothing, occupancy=nothing, num_worker_warps=nothing, name=nothing)

Compile and launch a Tile IR kernel. `args` are converted via
`cuTileconvert` (CuArray → TileArray, Type → Constant). Equivalent to
`@cuda backend=cuTile blocks=grid f(args...)` modulo
slight kwarg naming.

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
                name::Union{String, Nothing}=nothing)
    converted = map(cuTileconvert, args)
    tt = Tuple{map(Core.Typeof, converted)...}
    kernel = cufunction(f, tt; sm_arch, opt_level, num_ctas, occupancy, num_worker_warps, name)
    kernel(converted...; blocks=grid)
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
