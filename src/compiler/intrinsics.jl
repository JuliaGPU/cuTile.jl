# Tile IR intrinsics
#
# Organized according to https://docs.nvidia.com/cuda/tile-ir/latest/sections/operations.html

module Intrinsics

using Base: compilerbarrier, donotdelete
using ..cuTile: Tile, TileArray, Constant, TensorView, PartitionView
using ..cuTile: Signedness, SignednessSigned, SignednessUnsigned
using ..cuTile: ComparisonPredicate, CmpLessThan, CmpLessThanOrEqual, CmpGreaterThan, CmpGreaterThanOrEqual, CmpEqual, CmpNotEqual
using ..cuTile: IdentityVal, FloatIdentityVal, IntegerIdentityVal

end

# NOTE: Due to JuliaLang/julia#60583, intrinsics may be called during constant evaluation.
#       Because of that, such intrinsics (such as basic arithmetic) need to provide an
#       implementation that actually computes a valid result using Julia intrinsics.
#
#       Sometimes that's not possible, e.g., because the functionality required for that is
#       overlayed by methods calling back into the intrinsic (e.g. `sin`), so for those
#       intrinsics we disable constant folding using a `compilerbarrier(:const)`
#
# NOTE: Side-effectful intrinsics (stores, atomics) use `donotdelete(args...)` in their
#       bodies to prevent the optimizer from DCE'ing calls. `donotdelete` is a Julia builtin
#       with `effect_free=ALWAYS_FALSE`, which inference propagates through the function body.
#       `@assume_effects !:effect_free` does NOT work — `override_effects` can only strengthen
#       effects (set ALWAYS_TRUE), not weaken them. Spoofing `ipo_effects` via a custom
#       `CC.finish!` override is possible but fragile (must race against `finishinfer!` setting
#       `use_const_api` based on pre-override effects). `donotdelete` is the simplest correct
#       approach.

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
