# Host-side kernel launch.
#
# Compiles a Julia function with `TileArray` arguments to Tile IR bytecode,
# runs `tileiras` to lower bytecode → CUBIN, loads the cubin into the active
# CUDA context, and launches it via `cudacall`. Compilation is cached per
# `(MethodInstance, sm_arch, opt_level, num_ctas, occupancy, num_worker_warps, bytecode_version)`.

using CUDACore: CUDACore, CuArray, CuModule, CuFunction, cudacall, device, capability,
                AbstractBackend, AbstractKernel, kernel_convert, kernel_compile, PerDevice
using CUDA_Compiler_jll
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
 Toolkit / device validation (cached: once per `(capability, cuda_version)`).
=============================================================================#

# User overrides from `LocalPreferences.toml`.
const tileiras_override = @load_preference("tileiras", nothing)
const bytecode_version_override = let s = @load_preference("bytecode_version", nothing)
    s === nothing ? nothing : VersionNumber(s)
end

"""
    tileiras_path() -> String

Path to the `tileiras` binary. Honors the `tileiras` preference when set,
otherwise falls back to `CUDA_Compiler_jll.tileiras_path`.
"""
tileiras_path() = something(tileiras_override, CUDA_Compiler_jll.tileiras_path)

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
    cmd = tileiras_override === nothing ?
        `$(CUDA_Compiler_jll.tileiras()) $args` :
        Cmd([tileiras_override, args...])
    return addenv(cmd, "CUDA_ROOT" => tileiras_root())
end

function run_and_collect(cmd)
    stdout = Pipe()
    proc = run(pipeline(ignorestatus(cmd); stdout, stderr=stdout), wait=false)
    close(stdout.in)
    reader = Threads.@spawn String(read(stdout))
    Base.wait(proc)
    log = strip(fetch(reader))
    return proc, log
end

const tile_ir_support = PerDevice{Union{Nothing, VersionNumber}}()

const toolkit_version_cache = Base.Lockable(Base.RefValue{Union{Nothing, String}}(nothing))

# Bytecode versions cuTile.jl can emit, in ascending order. Each version listed
# here is one we have explicit `cb.version >= v"X.Y"` handling for in
# `bytecode/encodings.jl`. `bytecode_version()` probes `tileiras` to find the
# highest entry it accepts (or returns the user's preference override).
const SUPPORTED_BYTECODE_VERSIONS = (v"13.1", v"13.2", v"13.3")

const max_bytecode_version_cache = Base.Lockable(Base.RefValue{Union{Nothing, VersionNumber}}(nothing))

# Matches the `V<major>.<minor>.<patch>` component of `tileiras --version`,
# e.g. `V13.2.78`. That's the part that actually identifies the compiler;
# the surrounding `Built on …` / `Build local.local.…` lines vary across
# rebuilds of the same logical version and would over-invalidate the cache.
const TILEIRAS_VERSION_REGEX = r"V(\d+\.\d+\.\d+)"

"""
    toolkit_version() -> String

Lazy-cached `tileiras` toolkit identity, used as the toolkit-identity
component of the disk-cache key so distinct CUDA Toolkit patches
(e.g. `13.2.51` vs `13.2.78`) produce distinct cubins.

We invoke `tileiras --version` once per process and pull out the
`V<major>.<minor>.<patch>` token (the only thing that actually
identifies the compiler — `Built on …` and `Build local.local.…`
lines also vary across rebuilds of the same logical version, which
would over-invalidate the cache).

`CUDA_Compiler_jll.cuda_version` is not enough on its own: it only
exposes major.minor, so different patches would alias. cuTile Python
sidesteps the parsing problem by hashing the entire `--version`
stdout (`_get_compiler_version_string` in `_compile.py`); we extract
the V-token instead and fall back to the full stdout only if the
regex misses, so a future format change degrades to Python-style
over-invalidation rather than aliasing distinct compilers.

Failure to invoke `tileiras` returns the string `"unknown"` so the
cache still functions (all entries collapse into a single toolkit
bucket); it's the caller's job to decide whether that's acceptable.
"""
function toolkit_version()
    Base.@lock toolkit_version_cache begin
        ref = toolkit_version_cache[]
        ref[] === nothing || return ref[]::String

        proc, log = run_and_collect(tileiras_cmd("--version"))
        success(proc) ||
            error("tileiras --version failed: $(proc.exitcode), log: $log")

        m = match(TILEIRAS_VERSION_REGEX, log)
        isnothing(m) && error("tileiras --version output did not match expected format: $log")

        ref[] = m.captures[1]
        return ref[]::String
    end
end

"""
    bytecode_version() -> VersionNumber

The Tile IR bytecode version that `cuTile` will emit by default. Either
the highest version the current `tileiras` binary accepts (probed by
emitting a minimal empty bytecode buffer at each entry of
`SUPPORTED_BYTECODE_VERSIONS` newest-first and picking the first that
compiles cleanly), or, if the `bytecode_version` preference is set, that
value.

Result is cached for the lifetime of the process.

We probe rather than reading `CUDA_Compiler_jll.cuda_version` because
users can override `tileiras` via JLL preferences, in which case the
JLL's static `cuda_version` no longer reflects the actual binary's
capabilities. Mirrors `_get_max_supported_bytecode_version` in cuTile
Python's `_compile.py`.
"""
function bytecode_version()
    Base.@lock max_bytecode_version_cache begin
        ref = max_bytecode_version_cache[]
        ref[] === nothing || return ref[]::VersionNumber
        if bytecode_version_override !== nothing
            bytecode_version_override in SUPPORTED_BYTECODE_VERSIONS ||
                error("preference bytecode_version=v$bytecode_version_override " *
                      "is not in $SUPPORTED_BYTECODE_VERSIONS")
            ref[] = bytecode_version_override
        else
            ref[] = probe_max_bytecode_version()
        end
        return ref[]::VersionNumber
    end
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
    check_tile_ir_support()

Validate that the current `tileiras` toolkit supports Tile IR on the active
device. Returns the bytecode version cuTile should emit for this device
(per [`bytecode_version`]), provided it meets the device's minimum
requirement (Blackwell ≥ v13.1, Hopper ≥ v13.3, Ampere/Ada ≥ v13.2).
"""
function check_tile_ir_support()
    if tileiras_override === nothing && !CUDA_Compiler_jll.is_available()
        error("CUDA_Compiler_jll is not available and no `tileiras` preference is set; " *
              "cannot compile Tile IR kernels")
    end

    dev = device()
    ver = get!(tile_ir_support, dev) do
        ver = bytecode_version()

        cap = capability(dev)
        sm_str = format_sm_arch(cap)
        req = tile_ir_requirement(cap)
        if req === nothing
            @error "Tile IR is not supported on compute capability $cap ($sm_str)"
            return nothing
        end
        arch, min_ver = req
        if ver < min_ver
            @error "Tile IR on $arch ($sm_str) requires bytecode ≥ v$min_ver, detected v$ver"
            return nothing
        end

        return ver
    end

    if ver === nothing
        error("CUDA Tile is not supported on the current device")
    end
    return ver::VersionNumber
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

# Serializes the Julia-side codegen pipeline (inference, structured IR,
# tile-IR emission). `emit_tile!` and everything it calls into mutates shared
# state: `CacheView` entries, `CuTileResults` fields, the inference cache,
# and CompilerCaching's per-CI const_entries vector. None of that is
# thread-safe. We hold the lock only across `emit_tile!`; the tileiras
# subprocess below runs unlocked so concurrent `cufunction` calls (e.g.
# from autotuning's precompile fan-out) can still overlap their tileiras
# shell-outs.
const EMIT_TILE_LOCK = ReentrantLock()

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
    bytecode = Base.@lock EMIT_TILE_LOCK emit_tile!(cache, mi, ci, res; const_argtypes)

    res.cuda_bin !== nothing && return res.cuda_bin

    sm_arch = unpack_version(cache.owner.sm_arch)

    # Resolve opt_level here (not in emit_tile) because it's a tileiras flag, not bytecode.
    # num_ctas/occupancy/num_worker_warps are resolved in emit_tile because they're encoded in bytecode.
    _, _, kernel_meta = res.julia_ir
    opt_level = something(resolve_hint(unpack_hint(cache.owner.opt_level),
                                       kernel_meta, :opt_level, sm_arch), 3)

    # Disk cache lookup. The hash covers every input that changes the CUBIN
    # — bytecode + sm_arch + opt_level + tileiras toolkit version — so
    # different toolkit versions never collide. `bytecode_version` is encoded
    # in the bytecode itself, so it's covered transitively by the bytecode hash.
    dc = DiskCache.global_cache()
    cache_key = nothing
    if dc !== nothing
        cache_key = DiskCache.compute_key(bytecode, sm_arch, opt_level, toolkit_version())
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
    input_path = tempname() * ".tile"
    output_path = tempname() * ".cubin"
    compiled = false
    try
        write(input_path, bytecode)
        cmd = tileiras_cmd(input_path, "-o", output_path,
                           "--gpu-name", format_sm_arch(sm_arch),
                           "-O$(opt_level)", "--lineinfo")
        proc, log = run_and_collect(cmd)
        if !success(proc)
            reason = proc.termsignal > 0 ? "tileiras received signal $(proc.termsignal)" :
                                           "tileiras exited with code $(proc.exitcode)"
            msg = "Failed to compile Tile IR ($reason)"
            if !isempty(log)
                msg *= "\n" * log
            end
            msg *= "\nIf you think this is a bug, please file an issue and attach $(input_path)"
            if parse(Bool, get(ENV, "BUILDKITE", "false"))
                run(`buildkite-agent artifact upload $(input_path)`)
            end
            error(msg)
        end
        compiled = true
        res.cuda_bin = read(output_path)
    finally
        compiled && rm(input_path, force=true)
        rm(output_path, force=true)
    end

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
    bytecode_version = check_tile_ir_support()
    resolved_sm_arch = sm_arch !== nothing ? sm_arch : default_sm_arch()

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
    #
    # Held under EMIT_TILE_LOCK so concurrent compiles (e.g. autotuning's
    # precompile fan-out) don't race on inference / CompilerCaching state.
    # The lock-protected region also includes the lookup fast path; that
    # path is just a hashtable read, so brief contention here is fine.
    ci, res = Base.@lock EMIT_TILE_LOCK ensure_compiled(cache, mi, const_argtypes)

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
    pid = cuTile.bid(0)
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

Print information about the active `tileiras` toolkit, the bytecode version
cuTile.jl will emit for it, and any user overrides set via
`LocalPreferences.toml`.
"""
function versioninfo(io::IO=stdout)
    println(io, "cuTile toolchain:")

    install = tileiras_override === nothing ? "artifact installation" : "local installation"
    println(io, "- tileiras $(toolkit_version()), $install")

    bv = bytecode_version()
    bv_src = bytecode_version_override === nothing ? "auto-detected" : "set via preference"
    println(io, "- bytecode v$(bv.major).$(bv.minor), $bv_src")
end
