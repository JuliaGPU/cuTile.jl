# Tile IR intrinsics
#
# Organized according to https://docs.nvidia.com/cuda/tile-ir/latest/sections/operations.html

module Intrinsics

using Base: compilerbarrier, inferencebarrier
using ..cuTile: Tile, TileArray, Constant, TensorView, PartitionView
using ..cuTile: Signedness, ComparisonPredicate, ComparisonOrdering
using ..cuTile: IdentityVal, FloatIdentityVal, IntegerIdentityVal

end

"""
    @intrinsic signature

Define a Tile IR intrinsic in the `Intrinsics` module. These intrinsics are
defined to return `Any`, so need additional `tfunc` and `efunc` definitions
to specify their behavior.

A docstring can be attached to the intrinsic in the usual way:

    \"\"\"
        myop(x, y)

    Description of `myop`.
    \"\"\"
    @intrinsic myop(x, y)
"""
macro intrinsic(ex)
    body = quote
        compilerbarrier(:type, nothing)
    end
    funcdef = Expr(:function, ex, body)
    funcdef = Expr(:macrocall, Symbol("@noinline"), __source__, funcdef)
    name = ex isa Expr && ex.head === :where ? ex.args[1].args[1] : ex.args[1]
    # Define in the Intrinsics module via Core.eval, then expose a Base.@__doc__
    # marker on the binding so a leading docstring (handled by @doc) attaches to
    # the function inside Intrinsics. The two statements live in their own
    # top-level expressions (Expr(:toplevel, ...)) so each runs in a fresh world
    # age — without this, the binding access in the second statement would
    # trigger Julia 1.12+'s "access to binding in a world prior to its
    # definition world" warning.
    return esc(Expr(:toplevel,
        :(Core.eval(Intrinsics, $(QuoteNode(funcdef)))),
        :(Base.@__doc__ Intrinsics.$name),
    ))
end

"""
    instanceof_tfunc(lat) -> Type or nothing

Extract `T` from a lattice element representing `Type{T}`.
Simplified version of `Base.Compiler.instanceof_tfunc` that handles `Const(T)`
and `Type{T}` lattice elements. Returns `nothing` when `T` cannot be determined.
"""
function instanceof_tfunc(@nospecialize(lat))
    if isa(lat, CC.Const)
        val = lat.val
        return val isa Type ? val : nothing
    end
    tgt = CC.widenconst(lat)
    return tgt isa DataType && tgt <: Type && !isempty(tgt.parameters) ? tgt.parameters[1] : nothing
end

# Shared helper for creating load/store optimization hints
function create_optimization_hints(ctx::CGCtx, latency::Union{Int, Nothing}, allow_tma::Bool=true)
    isnothing(latency) && allow_tma && return nothing
    isnothing(latency) || 1 <= latency <= 10 || throw(ArgumentError("latency must be between 1 and 10, got $latency"))
    hints = LoadStoreHints(; latency, allow_tma)
    return make_load_store_hints(ctx.sm_arch, hints)
end

# Check if an intrinsic argument is a mask (Tile{Bool}) or nothing.
# Returns (CGVal, true) for a real mask, (nothing, false) for nothing.
function emit_optional_mask(ctx::CGCtx, args, idx::Int)
    idx > length(args) && return (nothing, false)
    tv = emit_value!(ctx, args[idx])
    has_mask = tv !== nothing && CC.widenconst(tv.jltype) !== Nothing
    return (has_mask ? tv : nothing, has_mask)
end

emit_intrinsic!(ctx::CGCtx, @nospecialize(func), args) = missing

include("intrinsics/core.jl")
include("intrinsics/conversions.jl")
include("intrinsics/arithmetic.jl")
include("intrinsics/math.jl")
include("intrinsics/memory.jl")
include("intrinsics/atomics.jl")
include("intrinsics/views.jl")
include("intrinsics/misc.jl")
include("intrinsics/fpmode.jl")
include("intrinsics/kernel_state.jl")

include("intrinsics/julia.jl")
