# Tile IR intrinsics
#
# Organized according to https://docs.nvidia.com/cuda/tile-ir/latest/sections/operations.html

module Intrinsics

using Base: compilerbarrier, inferencebarrier
using ..cuTile: Tile, TileArray, Constant, TensorView, PartitionView
using ..cuTile: Signedness, SignednessSigned, SignednessUnsigned
using ..cuTile: ComparisonPredicate, CmpLessThan, CmpLessThanOrEqual, CmpGreaterThan, CmpGreaterThanOrEqual, CmpEqual, CmpNotEqual
using ..cuTile: IdentityVal, FloatIdentityVal, IntegerIdentityVal

end

# NOTE: Intrinsics use bare signatures with dummy bodies (compilerbarrier(:type, nothing)).
#       Return types are provided by tfunc overrides in the interpreter.
#       Const-prop for overlay callers happens via @assume_effects :foldable at the
#       overlay level, not through intrinsic bodies.

"""
    @intrinsic signature
    @intrinsic function_definition

Define a Tile IR intrinsic in the `Intrinsics` module.

A bare signature (e.g. `@intrinsic foo(x)`) creates a dummy body using
`compilerbarrier(:type, nothing)` so body inference returns `Any`. Actual
return types come from `tfunc` overrides in the interpreter.

A function definition (e.g. `@intrinsic foo(x) = expr`) preserves the body,
providing a callable implementation for concrete evaluation. This is needed
when overlay callers with `@assume_effects :foldable` cause the compiler to
evaluate through intrinsic bodies (JuliaLang/julia#60583). The body should
provide a correct scalar implementation using `Core.Intrinsics`, or return
`nothing` for side-effect-only intrinsics.
"""
macro intrinsic(ex)
    body = quote
        compilerbarrier(:type, nothing)
    end
    funcdef = Expr(:function, ex, body)
    funcdef = Expr(:macrocall, Symbol("@noinline"), nothing, funcdef)
    return esc(:(Core.eval(Intrinsics, $(QuoteNode(funcdef)))))
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

emit_intrinsic!(ctx::CGCtx, @nospecialize(func), args) = missing

include("intrinsics/core.jl")
include("intrinsics/conversions.jl")
include("intrinsics/arithmetic.jl")
include("intrinsics/math.jl")
include("intrinsics/memory.jl")
include("intrinsics/atomics.jl")
include("intrinsics/views.jl")
include("intrinsics/misc.jl")

include("intrinsics/julia.jl")
