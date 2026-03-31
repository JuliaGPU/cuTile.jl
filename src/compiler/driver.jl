# Compilation hook for @device_code_* macros - intercepts compilations for reflection
const compile_hook = Ref{Union{Nothing,Function}}(nothing)


#=============================================================================
 Meta nodes and compilation hints
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


#=============================================================================
 Compilation phases
=============================================================================#

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
    emit_structured(ir::IRCode, rettype) -> (StructuredIRCode, rettype, kernel_meta)

Structurize IRCode into StructuredIRCode. Pure transformation, no caching.
"""
function emit_structured(ir::CC.IRCode, rettype)
    process_meta!(ir)
    kernel_meta = extract_meta(ir)
    sci = StructuredIRCode(ir)
    return (sci, rettype, kernel_meta)
end

"""
    emit_tile(sci, rettype, kernel_meta; name, opts, cache, const_argtypes) -> Vector{UInt8}

Generate Tile IR bytecode from StructuredIRCode. Pure computation, no caching.
`cache` is needed for subprogram compilation inside `emit_kernel!`.
"""
function emit_tile(sci::StructuredIRCode, rettype, kernel_meta::Dict{Symbol,Any};
                   name::String,
                   opts::CGOpts,
                   cache::CacheView,
                   const_argtypes::Union{Vector{Any}, Nothing}=nothing)
    # Resolve hints: launch()/code_tiled() kwargs > @compiler_options meta > defaults
    resolved_num_ctas = resolve_hint(opts.num_ctas, kernel_meta, :num_ctas, opts.sm_arch)
    resolved_occupancy = resolve_hint(opts.occupancy, kernel_meta, :occupancy, opts.sm_arch)

    # Generate Tile IR bytecode
    bytecode = write_bytecode!(1; version=opts.bytecode_version) do writer, func_buf
        emit_kernel!(writer, func_buf, sci, rettype;
            name,
            sm_arch = opts.sm_arch,
            num_ctas = resolved_num_ctas,
            occupancy = resolved_occupancy,
            cache,
            const_argtypes
        )
    end

    return bytecode
end


#=============================================================================
 Cached compilation
=============================================================================#

# Results struct for caching compilation phases
mutable struct CuTileResults
    julia_ir::Any    # (StructuredIRCode, rettype)
    tile_bc::Any     # Vector{UInt8} bytecode
    cuda_bin::Any    # Vector{UInt8} cubin (populated by CUDAExt)
    cuda_func::Any   # CuFunction (populated by CUDAExt)
    CuTileResults() = new(nothing, nothing, nothing, nothing)
end

# Cached wrappers around the driver's emit_* functions. These check/populate
# CuTileResults and are used by the production pipeline (launch → CUDAExt).
# Reflection APIs (code_tiled, code_structured, etc.) bypass this layer and
# call the driver directly, so they never pollute the compilation cache.

"""
    emit_structured!(cache, mi; const_argtypes=nothing) -> (StructuredIRCode, rettype, kernel_meta)

Cached IR phase. Invokes the compile hook (for `@device_code_*` macros),
checks `CuTileResults.julia_ir`, and delegates to `emit_structured` on cache miss.
"""
function emit_structured!(cache::CacheView, mi::Core.MethodInstance;
                  const_argtypes::Union{Vector{Any}, Nothing}=nothing)
    # Invoke compile hook if set (for @device_code_* reflection).
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

    # Compute fresh via driver
    ir, rettype = emit_julia(cache, mi; const_argtypes)
    result = emit_structured(ir, rettype)
    res.julia_ir = result
    return result
end

"""
    emit_tile!(cache, mi; const_argtypes=nothing) -> Vector{UInt8}

Cached code phase. Delegates to `emit_structured!` for the IR phase, checks
`CuTileResults.tile_bc`, and calls the driver's `emit_tile` on cache miss.
"""
function emit_tile!(cache::CacheView, mi::Core.MethodInstance;
                    const_argtypes::Union{Vector{Any}, Nothing}=nothing)
    # Delegate to cached IR phase
    ir_result = emit_structured!(cache, mi; const_argtypes)

    # Check code cache
    ci = get(cache, mi)
    res = const_argtypes !== nothing ? results(cache, ci, const_argtypes) : results(cache, ci)
    res.tile_bc !== nothing && return res.tile_bc

    # Compute bytecode via driver
    sci, rettype, kernel_meta = ir_result
    opts = CGOpts(cache.owner[2])
    bytecode = emit_tile(sci, rettype, kernel_meta;
                         name=sanitize_name(string(mi.def.name)),
                         opts, cache, const_argtypes)

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
