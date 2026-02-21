module CUDAExt

using cuTile
using cuTile: TileArray, Constant, CGOpts, CuTileResults, emit_code, sanitize_name,
              constant_eltype, constant_value

using CompilerCaching: CacheView, method_instance, results

import Core.Compiler as CC

using CUDA: CUDA, CuModule, CuFunction, cudacall, device, capability
using CUDA_Compiler_jll

public launch

const EMIT_CODE_LOCK = ReentrantLock()

"""
    emit_binary(cache, mi; const_argtypes=nothing) -> Vector{UInt8}

Binary phase: compile Tile IR bytecode to CUBIN using tileiras.
"""
function emit_binary(cache::CacheView, mi::Core.MethodInstance;
                     const_argtypes::Union{Vector{Any}, Nothing}=nothing)
    bytecode = lock(EMIT_CODE_LOCK) do
        emit_code(cache, mi; const_argtypes)
    end

    ci = get(cache, mi)
    res = const_argtypes !== nothing ? results(cache, ci, const_argtypes) : results(cache, ci)
    res.cuda_bin !== nothing && return res.cuda_bin

    opts = cache.owner[2]

    # Run tileiras to produce CUBIN
    input_path = tempname() * ".tile"
    output_path = tempname() * ".cubin"
    try
        write(input_path, bytecode)
        run(`$(CUDA_Compiler_jll.tileiras()) $input_path -o $output_path --gpu-name $(opts.sm_arch) -O$(opts.opt_level)`)
        res.cuda_bin = read(output_path)
    finally
        rm(input_path, force=true)
        rm(output_path, force=true)
    end

    return res.cuda_bin
end

"""
    emit_function(cache, mi; const_argtypes=nothing) -> CuFunction

Function phase: load CUBIN into GPU memory as a CuFunction.
"""
function emit_function(cache::CacheView, mi::Core.MethodInstance;
                       const_argtypes::Union{Vector{Any}, Nothing}=nothing)
    cubin = emit_binary(cache, mi; const_argtypes)

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
                       sm_arch::String=default_sm_arch(),
                       opt_level::Int=3,
                       num_ctas::Union{Int, Nothing}=nothing,
                       occupancy::Union{Int, Nothing}=nothing)
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
    opts = (sm_arch=sm_arch, opt_level=opt_level, num_ctas=num_ctas, occupancy=occupancy)
    cache = CacheView{CuTileResults}((:cuTile, opts), world)

    # Run cached compilation
    cufunc = emit_function(cache, mi; const_argtypes)

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
    default_sm_arch() -> String

Get the SM architecture string for the current CUDA device.
Returns e.g. "sm_120" for compute capability 12.0.
"""
function default_sm_arch()
    cap = capability(device())
    "sm_$(cap.major)$(cap.minor)"
end

"""
    flatten(x)

Flatten a value into a tuple of its leaf fields for kernel launch.
Scalars return themselves wrapped in a tuple. Structs like TileArray
return their fields in order.

This is used by the launch helper to splat arguments to cudacall.
"""
flatten(x) = (x,)
flatten(arr::TileArray{T, N}) where {T, N} = (arr.ptr, arr.sizes..., arr.strides...)
flatten(::Constant) = ()  # Ghost types are not passed to cudacall

"""
    to_tile_arg(x)

Convert launch arguments to their kernel argument form.
AbstractArrays (like CuArray) are converted to TileArray for metadata.
Other values pass through unchanged.
"""
to_tile_arg(x) = x
to_tile_arg(arr::AbstractArray) = TileArray(arr)

include("autotune/autotune.jl")

end
