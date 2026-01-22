# Compilation interface for cuTile
#
# This file provides the public compilation API:
# - cuTileInterpreter: custom interpreter with overlay method table
# - emit_ir/emit_code: three-phase compilation callbacks (emit_executable in CUDAExt)
# - code_tiled/@code_tiled: reflection utilities

export code_tiled, @code_tiled

using CompilerCaching: CacheView, @setup_caching, compile_hook,
                       compile_hook!, method_instance, typeinf!

#=============================================================================
 Interpreter
=============================================================================#

Base.Experimental.@MethodTable cuTileMethodTable

function get_method_table_view(world::UInt)
    CC.CachedMethodTable(CC.OverlayMethodTable(world, cuTileMethodTable))
end

"""
Custom interpreter that supports overlay method tables for cuTile compilation.
This is necessary because NativeInterpreter has a fixed method_table type parameter.
"""
struct cuTileInterpreter <: CC.AbstractInterpreter
    cache::CacheView
    method_table::CC.CachedMethodTable{CC.OverlayMethodTable}
    inf_cache::Vector{CC.InferenceResult}
    inf_params::CC.InferenceParams
    opt_params::CC.OptimizationParams
end

function cuTileInterpreter(cache::CacheView; always_inline::Bool=true)
    method_table = get_method_table_view(cache.world)
    inf_cache = Vector{CC.InferenceResult}()
    inf_params = CC.InferenceParams()
    opt_params = if always_inline
        CC.OptimizationParams(; inline_cost_threshold=typemax(Int))
    else
        CC.OptimizationParams()
    end
    return cuTileInterpreter(cache, method_table, inf_cache, inf_params, opt_params)
end

# Required AbstractInterpreter interface methods
CC.InferenceParams(interp::cuTileInterpreter) = interp.inf_params
CC.OptimizationParams(interp::cuTileInterpreter) = interp.opt_params
CC.get_inference_cache(interp::cuTileInterpreter) = interp.inf_cache

# World age
@static if isdefined(CC, :get_inference_world)
    CC.get_inference_world(interp::cuTileInterpreter) = interp.cache.world
else
    CC.get_world_counter(interp::cuTileInterpreter) = interp.cache.world
end

# Method table - this enables the overlays
CC.method_table(interp::cuTileInterpreter) = interp.method_table

# Locking - not needed for non-cached compilation
CC.lock_mi_inference(::cuTileInterpreter, ::MethodInstance) = nothing
CC.unlock_mi_inference(::cuTileInterpreter, ::MethodInstance) = nothing

# Setup caching - generates cache_owner and ipo_dataflow_analysis! methods
@setup_caching cuTileInterpreter.cache

# Optimization flags
CC.may_optimize(::cuTileInterpreter) = true
CC.may_compress(::cuTileInterpreter) = true
CC.may_discard_trees(::cuTileInterpreter) = true

# Disable semi-concrete interpretation (broken with overlays per JuliaLang/julia#47349)
function CC.concrete_eval_eligible(interp::cuTileInterpreter,
    @nospecialize(f), result::CC.MethodCallResult, arginfo::CC.ArgInfo, sv::CC.InferenceState)
    ret = @invoke CC.concrete_eval_eligible(interp::CC.AbstractInterpreter,
        f::Any, result::CC.MethodCallResult, arginfo::CC.ArgInfo, sv::CC.InferenceState)
    if ret === :semi_concrete_eval
        return :none
    end
    return ret
end

"""
    code_ircode(mi::MethodInstance; world, always_inline=true) -> (IRCode, rettype)

Get optimized IRCode for a MethodInstance using cuTile's overlay method table.
If always_inline=true (default), forces all functions to be inlined.
"""
function code_ircode(mi::MethodInstance; world::UInt=Base.get_world_counter(),
                     always_inline::Bool=true)
    cache = CacheView(:cuTile, world)
    interp = cuTileInterpreter(cache; always_inline)
    result = CC.typeinf_ircode(interp, mi, nothing)

    if result === nothing
        error("Type inference failed for $mi")
    end

    ir, rettype = result
    return ir, rettype
end

#=============================================================================
 Compilation phases
=============================================================================#

# Compilation options for cache sharding
const CGOpts = @NamedTuple{
    sm_arch::Union{String, Nothing},
    opt_level::Int,
    num_ctas::Union{Int, Nothing},
    occupancy::Union{Int, Nothing}
}

"""
    emit_ir(cache, mi) -> (StructuredIRCode, rettype)

IR phase: populate code cache with dependencies and return structured IR.
This phase uses cuTile's overlay method table for intrinsic substitution.
"""
function emit_ir(cache::CacheView, mi::Core.MethodInstance)
    interp = cuTileInterpreter(cache)
    codeinfos = typeinf!(cache, interp, mi)

    # Get IRCode from the CodeInfo - no second inference needed
    ci, codeinfo = first(codeinfos)

    # Get the MethodInstance from the CodeInstance for safety
    ci_mi = @static if VERSION >= v"1.12-"
        CC.get_ci_mi(ci)
    else
        ci.def::MethodInstance
    end

    ir = CC.inflate_ir(codeinfo, ci_mi)

    sci = StructuredIRCode(ir)
    return (sci, ci.rettype)
end

"""
    emit_code(cache, mi) -> Vector{UInt8}

Code phase: generate Tile IR bytecode from StructuredIRCode.
This phase is deterministic and does not require CUDA.

Returns bytecode that can be compiled to CUBIN by tileiras in the emit_executable phase.
"""
function emit_code(cache::CacheView, mi::Core.MethodInstance)
    sci, rettype = get!(emit_ir, cache, mi, :ir)
    opts = cache.keys

    # Generate Tile IR bytecode
    bytecode = write_bytecode!(1) do writer, func_buf
        emit_kernel!(writer, func_buf, sci, rettype;
            name = string(mi.def.name),
            sm_arch = opts.sm_arch,
            num_ctas = opts.num_ctas,
            occupancy = opts.occupancy
        )
    end

    # Dump bytecode if JULIA_CUTILE_DUMP_BYTECODE is set
    dump_dir = get(ENV, "JULIA_CUTILE_DUMP_BYTECODE", nothing)
    if dump_dir !== nothing
        mkpath(dump_dir)
        base_filename = basename(string(mi.def.file))
        base_filename = first(splitext(base_filename))
        dump_path = joinpath(dump_dir, "$(base_filename).ln$(mi.def.line).cutile")
        counter = 1
        while isfile(dump_path)
            counter += 1
            dump_path = joinpath(dump_dir, "$(base_filename).ln$(mi.def.line).$(counter).cutile")
        end
        println(stderr, "Dumping TILEIR bytecode to file: $dump_path")
        write(dump_path, bytecode)
    end

    return bytecode
end

#=============================================================================
 Reflection utilities
=============================================================================#

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

Return typed code for a cuTile function.. Analogous to `Base.code_typed`.
"""
function code_typed(@nospecialize(f), @nospecialize(argtypes);
                    world::UInt=Base.get_world_counter(), kwargs...)
    cache = CacheView(:cuTile, world)
    interp = cuTileInterpreter(cache)
    Base.code_typed(f, argtypes; world, interp, kwargs...)
end

"""
    code_structured(f, argtypes; kwargs...) -> StructuredIRCode

Return the structured IR for a cuTile function.
"""
function code_structured(@nospecialize(f), @nospecialize(argtypes);
                         sm_arch::Union{String, Nothing}=nothing,
                         opt_level::Int=3,
                         num_ctas::Union{Int, Nothing}=nothing,
                         occupancy::Union{Int, Nothing}=nothing)
    world = Base.get_world_counter()
    mi = @something(method_instance(f, argtypes; world, method_table=cuTileMethodTable),
                    method_instance(f, argtypes; world),
                    throw(MethodError(f, argtypes)))

    opts = (sm_arch=sm_arch, opt_level=opt_level, num_ctas=num_ctas, occupancy=occupancy)
    cache = CacheView{CGOpts}(:cuTile, world, opts)

    sci, rettype = get!(emit_ir, cache, mi, :ir)
    return sci
end

"""
    code_tiled(f, argtypes; sm_arch, opt_level, num_ctas, occupancy) -> String

Return the CUDA Tile IR for a Julia function as a textual MLIR representation.
Analogous to `code_typed` or `code_structured`.

Uses the same caching infrastructure as `launch`, benefiting from cached IR
and code results.
"""
function code_tiled(@nospecialize(f), @nospecialize(argtypes);
                    sm_arch::Union{String, Nothing}=nothing,
                    opt_level::Int=3,
                    num_ctas::Union{Int, Nothing}=nothing,
                    occupancy::Union{Int, Nothing}=nothing)
    world = Base.get_world_counter()
    mi = @something(method_instance(f, argtypes; world, method_table=cuTileMethodTable),
                    method_instance(f, argtypes; world),
                    throw(MethodError(f, argtypes)))

    opts = (sm_arch=sm_arch, opt_level=opt_level, num_ctas=num_ctas, occupancy=occupancy)
    cache = CacheView{CGOpts}(:cuTile, world, opts)

    bytecode = get!(emit_code, cache, mi, :code)
    disassemble_tileir(bytecode)
end

# compilation hooking: uses CompilerCaching's global hook
function emit_hooked_compilation(inner_hook, ex...)
    user_code = ex[end]
    user_kwargs = ex[1:end-1]
    quote
        # we only want to invoke the hook once for every compilation
        seen = Set()
        function outer_hook(cache, mi)
            key = (cache, mi)
            if !in(key, seen)
                # the user hook might invoke the compiler again, so disable the hook
                old_hook = $compile_hook()
                try
                    $compile_hook!(nothing)
                    opts = cache.keys
                    $inner_hook(cache, mi; $(map(esc, user_kwargs)...))
                finally
                    $compile_hook!(old_hook)
                end
                push!(seen, key)
            end
        end

        # now invoke the user code with this hook in place
        try
            $compile_hook!(outer_hook)
            $(esc(user_code))
        finally
            $compile_hook!(nothing)
        end

        if isempty(seen)
            error("no kernels executed while evaluating the given expression")
        end

        nothing
    end
end

macro code_tiled(ex...)
    function hook(cache, mi; io::IO=stdout)
        # The hook fires during get!, so bytecode is being cached
        # at this moment - retrieve it via get!
        bytecode = get!(emit_code, cache, mi, :code)
        println(io, "// $(mi.def.name)($(join(map(string, mi.specTypes.parameters[2:end]), ", ")))")
        println(io)
        println(io, disassemble_tileir(bytecode))
    end
    emit_hooked_compilation(hook, ex...)
end
