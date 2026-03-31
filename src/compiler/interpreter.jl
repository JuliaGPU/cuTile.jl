# Integration with Julia's abstract interpreter


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
 Custom inference for intrinsics
=============================================================================#

# Per-intrinsic return type overrides.
# Returns nothing when no override applies (fallback).
tfunc(𝕃, @nospecialize(f), @nospecialize args...) = nothing

# Per-intrinsic effect overrides.
# Returns nothing when no override applies (fallback).
efunc(@nospecialize(f), effects::CC.Effects) = nothing

# Predicate for functions defined in the Intrinsics module.
# These get NoCallInfo() so they stay as Expr(:call) rather than Expr(:invoke).
isintrinsic(@nospecialize(f)) = isa(f, Function) && parentmodule(f) === Intrinsics


#=============================================================================
 Subprogram inference
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
    if tile_type isa DataType && tile_type <: Tuple &&
            all(p -> p isa DataType && p <: Tile, tile_type.parameters)
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
        is_intr = isintrinsic(f)
        𝕃 = CC.typeinf_lattice(interp)
        rt_override = tfunc(𝕃, f, arginfo.argtypes[2:end]...)
        subprog = _infer_subprogram(interp, f, arginfo, si, vtypes, sv)
        !is_intr && rt_override === nothing && subprog === nothing && return result
        wrapped = CC.Future{CC.CallMeta}()
        push!(sv.tasks, function (interp′, sv′)
            isready(result) || return false
            subprog !== nothing && !isready(subprog) && return false
            cm = result[]
            sp = subprog !== nothing ? subprog[] : nothing
            rt = rt_override !== nothing ? rt_override : cm.rt
            efunc_override = is_intr ? efunc(f, cm.effects) : nothing
            effects = efunc_override !== nothing ? efunc_override : cm.effects
            info = is_intr ? CC.NoCallInfo() : cm.info
            info = sp !== nothing ? SubprogramCallInfo(info, sp.info) : info
            wrapped[] = CC.CallMeta(rt, cm.exct, effects, info, cm.refinements)
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
        is_intr = isintrinsic(f)
        𝕃 = CC.typeinf_lattice(interp)
        rt_override = tfunc(𝕃, f, arginfo.argtypes[2:end]...)
        subprog = _infer_subprogram(interp, f, arginfo, si, nothing, sv)
        !is_intr && rt_override === nothing && subprog === nothing && return result
        wrapped = CC.Future{CC.CallMeta}()
        push!(sv.tasks, function (interp′, sv′)
            isready(result) || return false
            subprog !== nothing && !isready(subprog) && return false
            cm = result[]
            sp = subprog !== nothing ? subprog[] : nothing
            rt = rt_override !== nothing ? rt_override : cm.rt
            efunc_override = is_intr ? efunc(f, cm.effects) : nothing
            effects = efunc_override !== nothing ? efunc_override : cm.effects
            info = is_intr ? CC.NoCallInfo() : cm.info
            info = sp !== nothing ? SubprogramCallInfo(info, sp.info) : info
            wrapped[] = CC.CallMeta(rt, cm.exct, effects, info, cm.refinements)
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
        is_intr = isintrinsic(f)
        𝕃 = CC.typeinf_lattice(interp)
        rt_override = tfunc(𝕃, f, arginfo.argtypes[2:end]...)
        rt = rt_override !== nothing ? rt_override : result.rt
        efunc_override = is_intr ? efunc(f, result.effects) : nothing
        effects = efunc_override !== nothing ? efunc_override : result.effects
        info = is_intr ? CC.NoCallInfo() : result.info
        if is_intr || rt_override !== nothing
            return CC.CallMeta(rt, result.exct, effects, info)
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
