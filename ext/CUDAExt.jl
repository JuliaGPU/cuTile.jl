module CUDAExt

using cuTile
using cuTile: TileArray, Constant, CuTileResults,
              emit_tile!, sanitize_name, constant_eltype, flatten,
              resolve_hint, format_sm_arch

using CompilerCaching: CacheView, method_instance, results

import Core.Compiler as CC

using CUDACore: CuArray, CuModule, CuFunction, cudacall, device, capability
using CUDA_Compiler_jll

import Base.Broadcast: BroadcastStyle
import CUDACore: CuArrayStyle

public launch

function run_and_collect(cmd)
    stdout = Pipe()
    proc = run(pipeline(ignorestatus(cmd); stdout, stderr=stdout), wait=false)
    close(stdout.in)
    reader = Threads.@spawn String(read(stdout))
    Base.wait(proc)
    log = strip(fetch(reader))
    return proc, log
end

"""
    check_tile_ir_support()

Validate that the current CUDA toolkit version supports Tile IR on the active device.
"""
function check_tile_ir_support()
    if !CUDA_Compiler_jll.is_available()
        error("CUDA_Compiler_jll is not available; cannot compile Tile IR kernels")
    end

    cuda_ver = CUDA_Compiler_jll.cuda_version
    cap = capability(device())
    sm_str = format_sm_arch(cap)

    if cap >= v"10.0"       # Blackwell
        cuda_ver >= v"13.1" ||
            error("Tile IR on Blackwell ($sm_str) requires CUDA ≥ 13.1, got $cuda_ver")
    elseif cap >= v"9.0"    # Hopper — not supported
        error("Tile IR is not supported on Hopper ($sm_str)")
    elseif cap >= v"8.0"    # Ampere / Ada
        cuda_ver >= v"13.2" ||
            error("Tile IR on Ampere/Ada ($sm_str) requires CUDA ≥ 13.2, got $cuda_ver")
    else
        error("Tile IR is not supported on compute capability $cap ($sm_str)")
    end

    # Return bytecode version matching the toolkit
    return VersionNumber(cuda_ver.major, cuda_ver.minor)
end

"""
    emit_binary!(cache, mi; const_argtypes=nothing) -> Vector{UInt8}

Cached binary phase: compile Tile IR bytecode to CUBIN using tileiras.
"""
function emit_binary!(cache::CacheView, mi::Core.MethodInstance;
                      const_argtypes::Union{Vector{Any}, Nothing}=nothing)
    bytecode = emit_tile!(cache, mi; const_argtypes)

    ci = get(cache, mi)
    res = const_argtypes !== nothing ? results(cache, ci, const_argtypes) : results(cache, ci)
    res.cuda_bin !== nothing && return res.cuda_bin

    opts = cache.owner[2]

    # Resolve opt_level here (not in emit_tile) because it's a tileiras flag, not bytecode.
    # num_ctas/occupancy are resolved in emit_tile because they're encoded in bytecode.
    _, _, kernel_meta = res.julia_ir
    opt_level = something(resolve_hint(opts.opt_level, kernel_meta, :opt_level, opts.sm_arch), 3)

    # Run tileiras to produce CUBIN
    input_path = tempname() * ".tile"
    output_path = tempname() * ".cubin"
    compiled = false
    try
        write(input_path, bytecode)
        cmd = addenv(`$(CUDA_Compiler_jll.tileiras()) $input_path -o $output_path --gpu-name $(format_sm_arch(opts.sm_arch)) -O$(opt_level) --lineinfo`,
                     "CUDA_ROOT" => CUDA_Compiler_jll.artifact_dir)
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

    return res.cuda_bin
end

"""
    emit_function!(cache, mi; const_argtypes=nothing) -> CuFunction

Cached function phase: load CUBIN into GPU memory as a CuFunction.
"""
function emit_function!(cache::CacheView, mi::Core.MethodInstance;
                        const_argtypes::Union{Vector{Any}, Nothing}=nothing)
    cubin = emit_binary!(cache, mi; const_argtypes)

    ci = get(cache, mi)
    res = const_argtypes !== nothing ? results(cache, ci, const_argtypes) : results(cache, ci)
    res.cuda_func !== nothing && return res.cuda_func

    kernel_name = sanitize_name(string(mi.def.name))
    cumod = CuModule(cubin)
    cufunc = CuFunction(cumod, kernel_name)
    res.cuda_func = cufunc
    return cufunc
end

"""
    launch(f, grid, args...; name=nothing, sm_arch=default_sm_arch(), opt_level=3, num_ctas=nothing, occupancy=nothing)

Compile and launch a kernel function with the given grid size and arguments.

Arguments are automatically flattened using `flatten()` - TileArray arguments
are expanded to their constituent ptr, sizes, and strides parameters.

# Arguments
- `f`: The kernel function to launch
- `grid`: Grid size as Int or NTuple for multi-dimensional grids
- `args...`: Kernel arguments (may include TileArray)
- `name`: Optional kernel name for debugging
- `sm_arch`: Target GPU architecture (default: current device's capability)
- `opt_level`: Optimization level 0-3 (default: 3)
- `num_ctas`: Number of CTAs in a CGA, 1-16, must be power of 2 (default: nothing)
- `occupancy`: Expected active CTAs per SM, 1-32 (default: nothing)

# Example
```julia
using CUDA, cuTile

a = CUDA.zeros(Float32, 1024)
b = CUDA.ones(Float32, 1024)
c = CUDA.zeros(Float32, 1024)

function vadd_kernel(a::cuTile.TileArray{Float32,1}, b::cuTile.TileArray{Float32,1},
                     c::cuTile.TileArray{Float32,1})
    pid = cuTile.bid(0)
    ta = cuTile.load(a, (pid,), (16,))
    tb = cuTile.load(b, (pid,), (16,))
    cuTile.store(c, (pid,), ta + tb)
    return nothing
end

# CuArrays are automatically converted to TileArray
cuTile.launch(vadd_kernel, 64, a, b, c)
```
"""
function cuTile.launch(@nospecialize(f), grid, args...;
                       name::Union{String, Nothing}=nothing,
                       sm_arch::Union{VersionNumber, Nothing}=nothing,
                       opt_level::Union{Int, Nothing}=nothing,
                       num_ctas::Union{Int, Nothing}=nothing,
                       occupancy::Union{Int, Nothing}=nothing)
    bytecode_version = check_tile_ir_support()

    # Resolve sm_arch: nothing → device capability
    resolved_sm_arch = sm_arch !== nothing ? sm_arch : default_sm_arch()

    # Convert CuArray -> TileArray (and other conversions)
    tile_args = map(to_tile_arg, args)

    # Unwrap Constant{T,V} → T for MI lookup (kernel sees plain types)
    unwrapped_types = map(tile_args) do arg
        arg isa Constant ? constant_eltype(typeof(arg)) : typeof(arg)
    end
    argtypes = Tuple{unwrapped_types...}

    # Get world age and method instance
    # Don't pass method_table - kernel functions are in the global table
    # The overlay table is only used by the interpreter during inference
    world = Base.get_world_counter()
    mi = method_instance(f, argtypes; world)
    mi === nothing && throw(MethodError(f, argtypes))

    # Build const_argtypes for const-seeded inference
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

    # Create cache view with compilation options as sharding keys
    opts = (sm_arch=resolved_sm_arch, opt_level=opt_level,
            num_ctas=num_ctas, occupancy=occupancy,
            bytecode_version=bytecode_version)
    cache = CacheView{CuTileResults}((:cuTile, opts), world)

    # Run cached compilation
    cufunc = emit_function!(cache, mi; const_argtypes)

    # Flatten arguments for cudacall - Constant returns () so ghost types disappear
    flat_args = Tuple(Iterators.flatten(map(flatten, tile_args)))
    flat_types = Tuple{map(typeof, flat_args)...}

    # Get grid dimensions
    grid_dims = grid isa Integer ? (grid,) : grid

    # Validate grid dimensions - Tile IR has a 24-bit limit
    max_grid_dim = (1 << 24) - 1  # 16,777,215
    for (i, dim) in enumerate(grid_dims)
        if dim > max_grid_dim
            error("Grid[$i] exceeds 24-bit limit: max=$max_grid_dim, got=$dim. " *
                  "Use multiple kernel launches for larger workloads.")
        end
    end

    # Launch with cached CuFunction
    # Note: threads=1 allows the driver to use the cubin's EIATTR_REQNTID metadata
    # which specifies the actual thread count (typically 128 for Tile kernels)
    cudacall(cufunc, flat_types, flat_args...; blocks=grid_dims, threads=1)

    return nothing
end

"""
    default_sm_arch() -> VersionNumber

Get the compute capability of the current CUDA device as a VersionNumber.
Returns e.g. `v"12.0"` for compute capability 12.0.
"""
default_sm_arch() = capability(device())

"""
    to_tile_arg(x)

Convert launch arguments to their kernel argument form.
AbstractArrays (like CuArray) are converted to TileArray for metadata.
Other values pass through unchanged.
"""
to_tile_arg(x) = x
to_tile_arg(arr::AbstractArray) = TileArray(arr)
to_tile_arg(t::Type) = Constant(t)

# Tiled Broadcast — TiledStyle wins over CuArrayStyle
BroadcastStyle(::cuTile.TiledStyle{N}, ::CuArrayStyle{M}) where {N,M} = cuTile.TiledStyle{max(N,M)}()

end
