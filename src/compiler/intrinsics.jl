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

using ExprTools: splitdef, combinedef

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
    funcdef = if ex isa Expr && ex.head in (:function, :(=))
        combinedef(splitdef(ex))
    else
        body = quote
            compilerbarrier(:type, nothing)
        end
        Expr(:function, ex, body)
    end
    funcdef = Expr(:macrocall, Symbol("@noinline"), nothing, funcdef)
    return esc(:(Core.eval(Intrinsics, $(QuoteNode(funcdef)))))
end

emit_intrinsic!(ctx::CGCtx, @nospecialize(func), args) = missing

# Shared helper for creating load/store optimization hints
function create_optimization_hints(ctx::CGCtx, latency::Union{Int, Nothing}, allow_tma::Bool=true)
    isnothing(latency) && allow_tma && return nothing
    isnothing(latency) || 1 <= latency <= 10 || throw(ArgumentError("latency must be between 1 and 10, got $latency"))
    hints = LoadStoreHints(; latency, allow_tma)
    return make_load_store_hints(ctx.sm_arch, hints)
end

include("intrinsics/core.jl")
include("intrinsics/conversions.jl")
include("intrinsics/arithmetic.jl")
include("intrinsics/math.jl")
include("intrinsics/memory.jl")
include("intrinsics/atomics.jl")
include("intrinsics/views.jl")
include("intrinsics/misc.jl")

include("intrinsics/julia.jl")
