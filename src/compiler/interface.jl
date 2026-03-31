# Compilation interface for cuTile
#
# This file provides the public compilation API:
# - cuTileInterpreter: custom interpreter with overlay method table
# - emit_ir/emit_code: compilation callbacks (emit_binary/emit_function in CUDAExt)
# - code_tiled/@code_tiled: reflection utilities

export code_tiled, @device_code_tiled

import CompilerCaching
using CompilerCaching: CacheView, @setup_caching, method_instance, match_method_instance, typeinf!, results, get_source

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

# Compilation options for cache sharding.
# Hint fields (opt_level, num_ctas, occupancy) represent explicit overrides only;
# `nothing` means "consult @compiler_options meta nodes in the IR during compilation."
const CGOpts = @NamedTuple{
    sm_arch::Union{VersionNumber, Nothing},
    opt_level::Union{Int, Nothing},
    num_ctas::Union{Int, Nothing},
    occupancy::Union{Int, Nothing},
    bytecode_version::VersionNumber
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
    process_meta!(ir::CC.IRCode) -> ir

Move `:meta` expression nodes from `ir.stmts` into `ir.meta`, mirroring
Julia's `process_meta!` in `Compiler/src/optimize.jl`. This normalizes IR
from `inflate_ir` (which leaves meta as stmts) to match the `typeinf_ircode`
path (which already extracts meta via `convert_to_ircode`).
"""
function process_meta!(ir::CC.IRCode)
    for i in 1:length(ir.stmts)
        stmt = ir.stmts[i][:stmt]
        if stmt isa Expr && stmt.head === :meta
            push!(ir.meta, stmt)
            @static if VERSION >= v"1.12-"
                ir.stmts[i][:stmt] = nothing
            else
                CC.setindex!(ir.stmts[i], nothing, :stmt)
            end
        end
    end
    return ir
end

"""
    extract_meta(ir::CC.IRCode) -> Dict{Symbol, Any}

Extract cuTile meta nodes from IRCode. Meta nodes are inserted by `@compiler_options`
and survive through lowering/optimization. After `process_meta!` normalization,
all meta nodes reside in `ir.meta`.
"""
function extract_meta(ir::CC.IRCode)
    meta = Dict{Symbol, Any}()
    for expr in ir.meta
        if expr isa Expr && expr.head === :meta && length(expr.args) >= 3 && expr.args[1] === :cuTile
            meta[expr.args[2]::Symbol] = expr.args[3]
        end
    end
    return meta
end

"""
    resolve_hint(explicit, kernel_meta, key, sm_arch)

Resolve a hint value with precedence: explicit kwarg > @compiler_options meta > nothing.
"""
function resolve_hint(explicit, kernel_meta::Dict{Symbol, Any}, key::Symbol,
                      sm_arch::Union{VersionNumber, Nothing})
    val = if explicit !== nothing
        explicit
    elseif haskey(kernel_meta, key) && sm_arch !== nothing
        resolve(kernel_meta[key], sm_arch)
    else
        nothing
    end
    validate_hint(key, val)
    return val
end

"""
    get_ci(cache, mi; const_argtypes=nothing) -> CodeInstance

Ensure inference is done and return the CodeInstance. Runs `typeinf!` which is
a no-op when already cached. When `const_argtypes` is provided, also ensures
the const-specialized entry exists.
"""
function get_ci(cache::CacheView, mi::Core.MethodInstance;
                const_argtypes::Union{Vector{Any}, Nothing}=nothing)
    # Ensure CI exists
    ci = get(cache, mi, nothing)
    if ci === nothing
        interp = cuTileInterpreter(cache)
        typeinf!(cache, interp, mi)
        ci = get(cache, mi)
    end

    # Run const-prop inference, if needed
    if const_argtypes !== nothing
        interp = cuTileInterpreter(cache)
        typeinf!(cache, interp, mi, const_argtypes)
    end

    return ci
end

# Get the inferred source and return type from a CodeInstance.
function get_inferred(cache::CacheView{K,V}, ci::Core.CodeInstance,
                      mi::Core.MethodInstance; const_argtypes::Union{Vector{Any},
                      Nothing}=nothing) where {K,V}
    rettype = CC.widenconst(ci.rettype)
    if const_argtypes === nothing
        src = @something get_source(ci)
    else
        src = @something get_source(ci, const_argtypes)

        # Extract the return type from a const-specialized entry.
        cached = CC.traverse_analysis_results(ci) do @nospecialize(result)
            result isa CompilerCaching.CachedResult{V} ? result : nothing
        end
        for entry in cached.const_entries
            if entry.argtypes == const_argtypes
                rettype = CC.widenconst(entry.rettype)
            end
        end
    end
    ir = CC.inflate_ir(src, mi)
    return ir, rettype
end

"""
    emit_julia(cache, mi; const_argtypes=nothing) -> (IRCode, rettype)

Julia phase: run inference and return IRCode.
"""
function emit_julia(cache::CacheView, mi::Core.MethodInstance;
                    const_argtypes::Union{Vector{Any}, Nothing}=nothing)
    ci = get_ci(cache, mi; const_argtypes)
    get_inferred(cache, ci, mi; const_argtypes)
end

"""
    emit_ir(cache, mi; const_argtypes=nothing) -> (StructuredIRCode, rettype, kernel_meta)

IR phase: structurize and optimize IRCode. Caches the result.
"""
function emit_ir(cache::CacheView, mi::Core.MethodInstance;
                 const_argtypes::Union{Vector{Any}, Nothing}=nothing)
    # Invoke compile hook if set (for @device_code_* reflection)
    # Pass (f, tt) tuple to enable direct use with reflection utilities.
    # Reconstruct Constant{T,V} types from const_argtypes so that code_tiled
    # can recover const-seeded arguments (MI specTypes have them unwrapped).
    if compile_hook[] !== nothing
        ftype = mi.specTypes.parameters[1]
        f = isdefined(ftype, :instance) ? ftype.instance : ftype
        arg_types = collect(Any, mi.specTypes.parameters[2:end])
        if const_argtypes !== nothing
            # const_argtypes is [Const(f), arg2, ...]; arg_types omits f,
            # so arg_types[i] corresponds to const_argtypes[i+1].
            for i in eachindex(arg_types)
                if const_argtypes[i+1] isa CC.Const
                    val = const_argtypes[i+1].val
                    arg_types[i] = Constant{typeof(val), val}
                end
            end
        end
        tt = Tuple{arg_types...}
        compile_hook[](f, tt)
    end

    ci = get_ci(cache, mi; const_argtypes)

    # Check result cache
    res = const_argtypes !== nothing ? results(cache, ci, const_argtypes) : results(cache, ci)
    res.julia_ir !== nothing && return res.julia_ir

    # Inflate and structurize
    ir, rettype = get_inferred(cache, ci, mi; const_argtypes)
    process_meta!(ir)
    kernel_meta = extract_meta(ir)
    sci = StructuredIRCode(ir)
    res.julia_ir = (sci, rettype, kernel_meta)
    return res.julia_ir
end

# Encode characters outside [a-zA-Z0-9_] as _XX hex escapes for PTX/MLIR compatibility.
sanitize_name(name::String) = replace(name, r"[^a-zA-Z0-9_]" => c -> "_$(string(UInt8(only(c)); base=16, pad=2))")

"""
    emit_code(cache, mi; const_argtypes=nothing) -> Vector{UInt8}

Code phase: generate Tile IR bytecode from StructuredIRCode.
"""
function emit_code(cache::CacheView, mi::Core.MethodInstance;
                   const_argtypes::Union{Vector{Any}, Nothing}=nothing)
    # Delegate to previous phase (handles CI + IR)
    sci, rettype, kernel_meta = emit_ir(cache, mi; const_argtypes)

    # Check code cache
    ci = get(cache, mi)
    res = const_argtypes !== nothing ? results(cache, ci, const_argtypes) : results(cache, ci)
    res.tile_bc !== nothing && return res.tile_bc

    # Compute bytecode
    opts = cache.owner[2]

    # Resolve hints: launch()/code_tiled() kwargs > @compiler_options meta > defaults
    resolved_num_ctas = resolve_hint(opts.num_ctas, kernel_meta, :num_ctas, opts.sm_arch)
    resolved_occupancy = resolve_hint(opts.occupancy, kernel_meta, :occupancy, opts.sm_arch)

    # Generate Tile IR bytecode
    bytecode = write_bytecode!(1; version=opts.bytecode_version) do writer, func_buf
        emit_kernel!(writer, func_buf, sci, rettype;
            name = sanitize_name(string(mi.def.name)),
            sm_arch = opts.sm_arch,
            num_ctas = resolved_num_ctas,
            occupancy = resolved_occupancy,
            cache = cache,
            const_argtypes = const_argtypes
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
    lookup_method_instance(f, argtypes; world) -> MethodInstance

Look up a MethodInstance for a cuTile function, checking the overlay method table first.
"""
function lookup_method_instance(@nospecialize(f), @nospecialize(argtypes);
                                world::UInt=Base.get_world_counter())
    tt = Base.signature_type(f, argtypes)
    if !Base.isdispatchtuple(tt)
        throw(ArgumentError("requires a dispatch tuple, got non-concrete signature"))
    end
    @something(match_method_instance(f, argtypes; world, method_table=cuTileMethodTable),
               match_method_instance(f, argtypes; world),
               throw(MethodError(f, argtypes)))
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
    code_structured(f, argtypes; kwargs...) -> Vector{Pair{StructuredIRCode, DataType}}

Return the structured IR for a cuTile function.
"""
function code_structured(@nospecialize(f), @nospecialize(argtypes);
                         world::UInt=Base.get_world_counter(),
                         optimize::Bool=true)
    stripped, const_argtypes = process_const_argtypes(f, argtypes)
    mi = lookup_method_instance(f, stripped; world)
    cache = CacheView{CuTileResults}(:cuTile, world)
    sci, rettype, _ = emit_ir(cache, mi; const_argtypes)
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
Analogous to `code_llvm`/`code_native`. Uses the compilation cache for
consistency with `launch`.
"""
function code_tiled(io::IO, @nospecialize(f), @nospecialize(argtypes);
                    sm_arch::Union{VersionNumber, Nothing}=nothing,
                    opt_level::Union{Int, Nothing}=nothing,
                    num_ctas::Union{Int, Nothing}=nothing,
                    occupancy::Union{Int, Nothing}=nothing,
                    bytecode_version::VersionNumber=DEFAULT_BYTECODE_VERSION,
                    world::UInt=Base.get_world_counter())
    # Strip Constant types from argtypes for MI lookup, build const_argtypes
    stripped, const_argtypes = process_const_argtypes(f, argtypes)
    mi = lookup_method_instance(f, stripped; world)

    opts = (sm_arch=sm_arch, opt_level=opt_level, num_ctas=num_ctas, occupancy=occupancy,
            bytecode_version=bytecode_version)
    cache = CacheView{CuTileResults}((:cuTile, opts), world)
    bytecode = emit_code(cache, mi; const_argtypes)
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
