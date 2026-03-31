#=============================================================================
 Reflection utilities
=============================================================================#

export code_tiled
public code_typed, code_ircode, code_structured

function disassemble_tileir(bytecode::Vector{UInt8})::String
    mktempdir() do dir
        input_path = joinpath(dir, "kernel.tile")
        output_path = joinpath(dir, "kernel.disasm")
        write(input_path, bytecode)
        read(`$(cuda_tile_translate()) --cudatilebc-to-mlir $input_path`, String)
    end
end

"""
    code_typed(f, argtypes; world, kwargs...) -> Vector{Any}

Return typed code for a cuTile function. Analogous to `Base.code_typed`.
"""
function code_typed(@nospecialize(f), @nospecialize(argtypes);
                    world::UInt=Base.get_world_counter(), kwargs...)
    stripped, const_argtypes = process_const_argtypes(f, argtypes)
    mi = lookup_method_instance(f, stripped; world)
    cache = CacheView{CuTileResults}(:cuTile, world)
    ir, rettype = emit_julia(cache, mi; const_argtypes)
    [ir => rettype]
end

"""
    code_ircode(mi::MethodInstance; world, always_inline=true) -> (IRCode, rettype)

Get optimized IRCode for a MethodInstance using cuTile's overlay method table.
If always_inline=true (default), forces all functions to be inlined.
"""
function code_ircode(mi::MethodInstance; world::UInt=Base.get_world_counter(),
                     always_inline::Bool=true)
    cache = CacheView{CuTileResults}(:cuTile, world)
    interp = cuTileInterpreter(cache; always_inline)
    result = CC.typeinf_ircode(interp, mi, nothing)

    if result === nothing
        throw(ErrorException("Type inference failed for $mi"))
    end

    ir, rettype = result
    return ir, rettype
end

"""
    code_structured(f, argtypes; kwargs...) -> Vector{Pair{StructuredIRCode, DataType}}

Return the structured IR for a cuTile function.
"""
function code_structured(@nospecialize(f), @nospecialize(argtypes);
                         world::UInt=Base.get_world_counter(),
                         optimize::Bool=true)
    stripped, const_argtypes = process_const_argtypes(f, argtypes)
    mi = lookup_method_instance(f, stripped; world)
    cache = CacheView{CuTileResults}(:cuTile, world)
    ir, rettype = emit_julia(cache, mi; const_argtypes)
    sci, rettype, _ = emit_structured(ir, rettype)
    if optimize
        sci = deepcopy(sci)
        run_passes!(sci)
    end
    [sci => rettype]
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
    has_consts = any(T -> T <: Constant, params)
    stripped_params = map(params) do T
        T <: Constant ? constant_eltype(T) : T
    end
    stripped = Tuple{stripped_params...}
    const_argtypes = if has_consts
        cats = Any[CC.Const(f)]
        for T in params
            push!(cats, T <: Constant ? CC.Const(constant_value(T)) : T)
        end
        cats
    else
        nothing
    end
    return stripped, const_argtypes
end

constant_eltype(::Type{Constant{T,V}}) where {T,V} = T
constant_value(::Type{Constant{T,V}}) where {T,V} = V

"""
    code_tiled([io::IO], f, argtypes; sm_arch, opt_level, num_ctas, occupancy)

Print the CUDA Tile IR for a Julia function as a textual MLIR representation.
Analogous to `code_llvm`/`code_native`. Calls the driver directly without
caching in CuTileResults, so reflection never pollutes the compilation cache.
"""
function code_tiled(io::IO, @nospecialize(f), @nospecialize(argtypes);
                    sm_arch::Union{VersionNumber, Nothing}=nothing,
                    opt_level::Union{Int, Nothing}=nothing,
                    num_ctas::Union{Int, Nothing}=nothing,
                    occupancy::Union{Int, Nothing}=nothing,
                    bytecode_version::VersionNumber=DEFAULT_BYTECODE_VERSION,
                    world::UInt=Base.get_world_counter())
    stripped, const_argtypes = process_const_argtypes(f, argtypes)
    mi = lookup_method_instance(f, stripped; world)

    opts = CGOpts((sm_arch=sm_arch, opt_level=opt_level, num_ctas=num_ctas, occupancy=occupancy,
                    bytecode_version=bytecode_version))
    cache = CacheView{CuTileResults}(:cuTile, world)
    ir, rettype = emit_julia(cache, mi; const_argtypes)
    sci, rettype, kernel_meta = emit_structured(ir, rettype)
    bytecode = emit_tile(sci, rettype, kernel_meta;
                         name=sanitize_name(string(mi.def.name)),
                         opts, cache, const_argtypes)
    print(io, disassemble_tileir(bytecode))
end
code_tiled(@nospecialize(f), @nospecialize(argtypes); kwargs...) =
    code_tiled(stdout, f, argtypes; kwargs...)


#=============================================================================
 Device code reflection macros
=============================================================================#

export @device_code_tiled
public @device_code_typed, @device_code_structured

# Following GPUCompiler's pattern for @device_code_* macros
function emit_hooked_compilation(inner_hook, ex...)
    user_code = ex[end]
    user_kwargs = ex[1:end-1]
    quote
        seen = Set{Tuple{Any,Any}}()
        function outer_hook(f, tt)
            if !in((f, tt), seen)
                old_hook = $compile_hook[]
                try
                    $compile_hook[] = nothing
                    $inner_hook(f, tt; $(map(esc, user_kwargs)...))
                finally
                    $compile_hook[] = old_hook
                end
                push!(seen, (f, tt))
            end
        end

        try
            $compile_hook[] = outer_hook
            $(esc(user_code))
        finally
            $compile_hook[] = nothing
        end

        if isempty(seen)
            error("no kernels executed while evaluating the given expression")
        end

        nothing
    end
end

"""
    @device_code_tiled [io=stdout] expression

Print the Tile IR (MLIR) for all kernels compiled while evaluating the expression.

# Example
```julia
@device_code_tiled launch(vadd, grid, a, b, c)
```
"""
macro device_code_tiled(ex...)
    function hook(f, tt; io::IO=stdout, kwargs...)
        println(io, "// $f($(join(tt.parameters, ", ")))")
        println(io)
        code_tiled(io, f, Tuple(tt.parameters); kwargs...)
        println(io)
    end
    emit_hooked_compilation(hook, ex...)
end

"""
    @device_code_structured [io=stdout] expression

Print the StructuredIRCode for all kernels compiled while evaluating the expression.

# Example
```julia
@device_code_structured launch(vadd, grid, a, b, c)
```
"""
macro device_code_structured(ex...)
    function hook(f, tt; io::IO=stdout, kwargs...)
        println(io, "// $f($(join(tt.parameters, ", ")))")
        println(io)
        sci, _ = only(code_structured(f, Tuple(tt.parameters); kwargs...))
        println(io, sci)
    end
    emit_hooked_compilation(hook, ex...)
end

"""
    @device_code_typed [io=stdout] expression

Print the typed Julia IR for all kernels compiled while evaluating the expression.

# Example
```julia
@device_code_typed launch(vadd, grid, a, b, c)
```
"""
macro device_code_typed(ex...)
    function hook(f, tt; io::IO=stdout, kwargs...)
        println(io, "// $f($(join(tt.parameters, ", ")))")
        println(io)
        ci, _ = only(code_typed(f, Tuple(tt.parameters); kwargs...))
        println(io, ci)
    end
    emit_hooked_compilation(hook, ex...)
end
