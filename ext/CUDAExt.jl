module CUDAExt

using cuTile
using cuTile: TileArray, Constant, emit_tileir

using CUDA: CuModule, CuFunction, cudacall, device, capability
using CUDA_Compiler_jll

public launch

# Compilation cache - stores CuFunction directly to avoid re-loading CuModule
const _compilation_cache = Dict{Any, Any}()  # (method, argtypes, sm_arch, opt_level, num_ctas, occupancy) => CuFunction

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

    # Compute argument types from the converted arguments
    argtypes = Tuple{map(typeof, tile_args)...}

    # Determine kernel name
    kernel_name = name !== nothing ? name : string(nameof(f))

    # Use method instance in case of a redefinition
    method = which(f, argtypes)

    # Check compilation cache - returns CuFunction directly
    cache_key = (method, argtypes, sm_arch, opt_level, num_ctas, occupancy)
    cufunc = get(_compilation_cache, cache_key, nothing)
    if cufunc === nothing || cuTile.compile_hook[] !== nothing
        cubin = compile(f, argtypes; name, sm_arch, opt_level, num_ctas, occupancy)
        if cufunc === nothing
            cumod = CuModule(cubin)
            cufunc = CuFunction(cumod, kernel_name)
            _compilation_cache[cache_key] = cufunc
        end
    end

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
    compile(f, argtypes; name=nothing, sm_arch=default_sm_arch(), opt_level=3, num_ctas=nothing, occupancy=nothing) -> Vector{UInt8}

Compile a Julia kernel function to a CUDA binary.
"""
function compile(@nospecialize(f), @nospecialize(argtypes);
                 name::Union{String, Nothing}=nothing,
                 sm_arch::String=default_sm_arch(),
                 opt_level::Int=3,
                 num_ctas::Union{Int, Nothing}=nothing,
                 occupancy::Union{Int, Nothing}=nothing)
    tile_bytecode = emit_tileir(f, argtypes; name, sm_arch,
                                 num_ctas, occupancy)

    # Dump bytecode if JULIA_CUTILE_DUMP_BYTECODE is set
    dump_dir = get(ENV, "JULIA_CUTILE_DUMP_BYTECODE", nothing)
    if dump_dir !== nothing
        mkpath(dump_dir)
        # Get source location from the function's method
        mi = first(methods(f, Base.to_tuple_type(argtypes)))
        base_filename = basename(string(mi.file))
        base_filename = first(splitext(base_filename))
        # Find unique filename, adding counter if file exists
        dump_path = joinpath(dump_dir, "$(base_filename).ln$(mi.line).cutile")
        counter = 1
        while isfile(dump_path)
            counter += 1
            dump_path = joinpath(dump_dir, "$(base_filename).ln$(mi.line).$(counter).cutile")
        end
        println(stderr, "Dumping TILEIR bytecode to file: $dump_path")
        write(dump_path, tile_bytecode)
    end

    input_path = tempname() * ".tile"
    output_path = tempname() * ".cubin"

    try
        write(input_path, tile_bytecode)
        run(`$(CUDA_Compiler_jll.tileiras()) $input_path -o $output_path --gpu-name $sm_arch -O$opt_level`)
        return read(output_path)
    finally
        rm(input_path, force=true)
        rm(output_path, force=true)
    end
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

end
