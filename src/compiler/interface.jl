# Compilation interface for cuTile
#
# This file provides the public compilation API:
# - cuTileInterpreter: custom interpreter with overlay method table
# - emit_ir/emit_code: compilation callbacks (emit_binary/emit_function in CUDAExt)
# - code_tiled/@code_tiled: reflection utilities

export code_tiled, @device_code_tiled

using CompilerCaching: CacheView, @setup_caching, method_instance, typeinf!, results, get_source

# Compilation hook for @device_code_* macros - intercepts compilations for reflection
const compile_hook = Ref{Union{Nothing,Function}}(nothing)

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
CC.may_discard_trees(::cuTileInterpreter) = false

#=============================================================================
 Custom return-type inference (tfuncs) for intrinsics
=============================================================================#

# Per-intrinsic return type overrides using multiple dispatch.
# Returns nothing when no override applies (fallback).
# Concrete per-intrinsic methods are defined in intrinsics/ (after the
# Intrinsics module exists).
tfunc(@nospecialize(f), argtypes::Vector{Any}) = nothing

#=============================================================================
 Subprogram inference for reduce/scan
=============================================================================#

# Intrinsics.reduce and Intrinsics.scan accept a subprogram function `f` that
# is never called in their bodies — inference treats it as dead. We intercept
# abstract_call_known to trigger a synthetic inference of `f(T, T)`,
# front-loading subprogram inference and establishing proper invalidation edges.

# On 1.12+, compute_edges! walks stmt_info and calls add_edges_impl.
# We need a custom CallInfo that propagates both the reduce/scan call's edges
# and the subprogram's edges.
# TODO: switch to IndirectCallInfo when JuliaLang/julia#59221 lands
@static if isdefined(CC, :add_edges_impl)  # 1.12+
    struct SubprogramCallInfo <: CC.CallInfo
        call::CC.CallInfo
        subprogram::CC.CallInfo
    end
    CC.add_edges_impl(edges::Vector{Any}, info::SubprogramCallInfo) =
        (CC.add_edges!(edges, info.call); CC.add_edges!(edges, info.subprogram))
    CC.nsplit_impl(info::SubprogramCallInfo) = CC.nsplit_impl(info.call)
    CC.getsplit_impl(info::SubprogramCallInfo, idx::Int) = CC.getsplit_impl(info.call, idx)
    CC.getresult_impl(info::SubprogramCallInfo, idx::Int) = CC.getresult_impl(info.call, idx)
end

# Version-portable StmtInfo constructor for subprogram inference
@static if hasfield(CC.StmtInfo, :saw_latestworld)  # 1.12+
    _subprogram_si(si) = CC.StmtInfo(true, si.saw_latestworld)
else  # 1.11
    _subprogram_si(si) = CC.StmtInfo(true)
end

# Detect vtypes parameter (1.14+)
const _HAS_VTYPES = hasmethod(CC.abstract_call,
    Tuple{CC.AbstractInterpreter, CC.ArgInfo, CC.StmtInfo,
          Union{Vector{CC.VarState},Nothing}, CC.AbsIntState, Int})

"""
Trigger a synthetic `abstract_call` for the subprogram function `f(T, T)`
so that inference discovers the subprogram callee and establishes invalidation edges.
Returns the result of `abstract_call` (Future{CallMeta} on 1.12+, CallMeta on 1.11),
or `nothing` if inapplicable.
"""
function _infer_subprogram(interp::cuTileInterpreter, @nospecialize(f),
                           arginfo::CC.ArgInfo, si, vtypes, sv)
    (f === Intrinsics.reduce || f === Intrinsics.scan) || return nothing
    argtypes = arginfo.argtypes
    length(argtypes) >= 4 || return nothing

    tile_type = CC.widenconst(argtypes[2])
    f_type = argtypes[4]

    # Build body arg types: [f_type, T₁, T₁, T₂, T₂, ...] for each operand
    body_argtypes = Any[f_type]
    if tile_type <: Tuple && all(p -> p <: Tile, tile_type.parameters)
        # always-tuple interface — Tuple{Tile{T1,S1}, ...}
        for p in tile_type.parameters
            T = p.parameters[1]
            push!(body_argtypes, T, T)
        end
    else
        return nothing
    end

    csi = _subprogram_si(si)
    cargs = CC.ArgInfo(nothing, body_argtypes)

    @static if _HAS_VTYPES
        CC.abstract_call(interp, cargs, csi, vtypes, sv, 1)
    else
        CC.abstract_call(interp, cargs, csi, sv, 1)
    end
end

# Override abstract_call_known for custom return-type inference (tfuncs) and
# subprogram inference for reduce/scan.
#
# On 1.12+, abstract_call_known returns Future{CallMeta}. The caller uses the
# CallMeta.info to populate stmt_info[pc], which compute_edges! later walks.
# We return a new Future that wraps the original result's info with
# SubprogramCallInfo, so the subprogram's edges end up in stmt_info and thus
# in the CodeInstance's edge list.
@static if _HAS_VTYPES   # 1.14+
    function CC.abstract_call_known(interp::cuTileInterpreter, @nospecialize(f),
            arginfo::CC.ArgInfo, si::CC.StmtInfo, vtypes::Union{CC.VarTable,Nothing},
            sv::CC.InferenceState, max_methods::Int = CC.get_max_methods(interp, f, sv))
        result = @invoke CC.abstract_call_known(interp::CC.AbstractInterpreter, f::Any,
            arginfo::CC.ArgInfo, si::CC.StmtInfo, vtypes::Union{CC.VarTable,Nothing},
            sv::CC.InferenceState, max_methods::Int)
        rt_override = tfunc(f, arginfo.argtypes)
        subprog = _infer_subprogram(interp, f, arginfo, si, vtypes, sv)
        rt_override === nothing && subprog === nothing && return result
        wrapped = CC.Future{CC.CallMeta}()
        push!(sv.tasks, function (interp′, sv′)
            isready(result) || return false
            subprog !== nothing && !isready(subprog) && return false
            cm = result[]
            sp = subprog !== nothing ? subprog[] : nothing
            rt = rt_override !== nothing ? rt_override : cm.rt
            info = sp !== nothing ? SubprogramCallInfo(cm.info, sp.info) : cm.info
            wrapped[] = CC.CallMeta(rt, cm.exct, cm.effects, info, cm.refinements)
            return true
        end)
        return wrapped
    end
elseif isdefined(CC, :Future)   # 1.12–1.13
    function CC.abstract_call_known(interp::cuTileInterpreter, @nospecialize(f),
            arginfo::CC.ArgInfo, si::CC.StmtInfo,
            sv::CC.InferenceState, max_methods::Int = CC.get_max_methods(interp, f, sv))
        result = @invoke CC.abstract_call_known(interp::CC.AbstractInterpreter, f::Any,
            arginfo::CC.ArgInfo, si::CC.StmtInfo,
            sv::CC.InferenceState, max_methods::Int)
        rt_override = tfunc(f, arginfo.argtypes)
        subprog = _infer_subprogram(interp, f, arginfo, si, nothing, sv)
        rt_override === nothing && subprog === nothing && return result
        wrapped = CC.Future{CC.CallMeta}()
        push!(sv.tasks, function (interp′, sv′)
            isready(result) || return false
            subprog !== nothing && !isready(subprog) && return false
            cm = result[]
            sp = subprog !== nothing ? subprog[] : nothing
            rt = rt_override !== nothing ? rt_override : cm.rt
            info = sp !== nothing ? SubprogramCallInfo(cm.info, sp.info) : cm.info
            wrapped[] = CC.CallMeta(rt, cm.exct, cm.effects, info, cm.refinements)
            return true
        end)
        return wrapped
    end
else   # 1.11: synchronous, edges auto-tracked via stmt_edges
    function CC.abstract_call_known(interp::cuTileInterpreter, @nospecialize(f),
            arginfo::CC.ArgInfo, si::CC.StmtInfo,
            sv::CC.AbsIntState, max_methods::Int = CC.get_max_methods(interp, f, sv))
        result = @invoke CC.abstract_call_known(interp::CC.AbstractInterpreter, f::Any,
            arginfo::CC.ArgInfo, si::CC.StmtInfo,
            sv::CC.AbsIntState, max_methods::Int)
        _infer_subprogram(interp, f, arginfo, si, nothing, sv)  # side-effect only
        rt_override = tfunc(f, arginfo.argtypes)
        if rt_override !== nothing
            return CC.CallMeta(rt_override, result.exct, result.effects,
                               result.info)
        end
        return result
    end
end

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
    cache = CacheView{CuTileResults}(:cuTile, world)
    interp = cuTileInterpreter(cache; always_inline)
    result = CC.typeinf_ircode(interp, mi, nothing)

    if result === nothing
        throw(ErrorException("Type inference failed for $mi"))
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

# Results struct for caching compilation phases
mutable struct CuTileResults
    julia_ir::Any    # (StructuredIRCode, rettype)
    tile_bc::Any     # Vector{UInt8} bytecode
    cuda_bin::Any    # Vector{UInt8} cubin (populated by CUDAExt)
    cuda_func::Any   # CuFunction (populated by CUDAExt)
    CuTileResults() = new(nothing, nothing, nothing, nothing)
end

"""
    emit_ir(cache, mi) -> (StructuredIRCode, rettype)

IR phase: run inference if needed and return structured IR.
"""
function emit_ir(cache::CacheView, mi::Core.MethodInstance)
    # Invoke compile hook if set (for @device_code_* reflection)
    # Pass (f, tt) tuple to enable direct use with reflection utilities
    if compile_hook[] !== nothing
        ftype = mi.specTypes.parameters[1]
        f = isdefined(ftype, :instance) ? ftype.instance : ftype
        tt = Tuple{mi.specTypes.parameters[2:end]...}
        compile_hook[](f, tt)
    end

    # Ensure CI exists
    ci = get(cache, mi, nothing)
    if ci === nothing
        interp = cuTileInterpreter(cache)
        typeinf!(cache, interp, mi)
        ci = get(cache, mi)
    end

    # Check IR cache
    res = results(cache, ci)
    res.julia_ir !== nothing && return res.julia_ir

    # Compute IR from CodeInfo
    src = @something get_source(ci)
    ir = CC.inflate_ir(src, mi)
    sci = StructuredIRCode(ir)

    res.julia_ir = (sci, ci.rettype)
    return res.julia_ir
end

"""
    emit_code(cache, mi) -> Vector{UInt8}

Code phase: generate Tile IR bytecode from StructuredIRCode.
"""
function emit_code(cache::CacheView, mi::Core.MethodInstance)
    # Delegate to previous phase (handles CI + IR)
    sci, rettype = emit_ir(cache, mi)

    # Check code cache
    ci = get(cache, mi)
    res = results(cache, ci)
    res.tile_bc !== nothing && return res.tile_bc

    # Compute bytecode
    opts = cache.owner[2]

    # Generate Tile IR bytecode
    bytecode = write_bytecode!(1) do writer, func_buf
        emit_kernel!(writer, func_buf, sci, rettype;
            name = string(mi.def.name),
            sm_arch = opts.sm_arch,
            num_ctas = opts.num_ctas,
            occupancy = opts.occupancy,
            cache = cache
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

    res.tile_bc = bytecode
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

Return typed code for a cuTile function. Analogous to `Base.code_typed`.
"""
function code_typed(@nospecialize(f), @nospecialize(argtypes);
                    world::UInt=Base.get_world_counter(), kwargs...)
    cache = CacheView{CuTileResults}(:cuTile, world)
    interp = cuTileInterpreter(cache)
    Base.code_typed(f, argtypes; world, interp, kwargs...)
end

"""
    code_structured(f, argtypes; kwargs...) -> Vector{Pair{StructuredIRCode, DataType}}

Return the structured IR for a cuTile function.
"""
function code_structured(@nospecialize(f), @nospecialize(argtypes);
                         world::UInt=Base.get_world_counter(),
                         validate::Bool=true)
    cache = CacheView{CuTileResults}(:cuTile, world)
    interp = cuTileInterpreter(cache)
    map(Base.code_ircode(f, argtypes; world, interp)) do (ir, ret_type)
        StructuredIRCode(ir; validate) => ret_type
    end
end

"""
    code_tiled([io::IO], f, argtypes; sm_arch, opt_level, num_ctas, occupancy)

Print the CUDA Tile IR for a Julia function as a textual MLIR representation.
Analogous to `code_llvm`/`code_native`. Uses the compilation cache for
consistency with `launch`.
"""
function code_tiled(io::IO, @nospecialize(f), @nospecialize(argtypes);
                    sm_arch::Union{String, Nothing}=nothing,
                    opt_level::Int=3,
                    num_ctas::Union{Int, Nothing}=nothing,
                    occupancy::Union{Int, Nothing}=nothing,
                    world::UInt=Base.get_world_counter())
    tt = Base.signature_type(f, argtypes)
    if !Base.isdispatchtuple(tt)
        throw(ArgumentError("code_tiled requires a dispatch tuple, got non-concrete signature"))
    end
    mi = @something(method_instance(f, argtypes; world, method_table=cuTileMethodTable),
                    method_instance(f, argtypes; world),
                    throw(MethodError(f, argtypes)))
    opts = (sm_arch=sm_arch, opt_level=opt_level, num_ctas=num_ctas, occupancy=occupancy)
    cache = CacheView{CuTileResults}((:cuTile, opts), world)
    bytecode = emit_code(cache, mi)
    print(io, disassemble_tileir(bytecode))
end
code_tiled(@nospecialize(f), @nospecialize(argtypes); kwargs...) =
    code_tiled(stdout, f, argtypes; kwargs...)


#=============================================================================
 Device code reflection macros
=============================================================================#

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
