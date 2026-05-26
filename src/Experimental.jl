module Experimental

using ..cuTile
using ..cuTile: cuTileconvert, cufunction, default_sm_arch, _SCOPED_INF_CACHE

using CUDACore: CUDACore

using Base.ScopedValues: with
import Core.Compiler as CC
using Random

# Builds a fresh inference cache compatible with the running Julia version.
# Used to wrap an autotune pass in `with(_SCOPED_INF_CACHE => …)` so all the
# per-config const-seeded inference calls share results instead of paying
# the slow paths (e.g. `ct.load(..., order=…)`) once per config.
@inline _fresh_inf_cache() = @static if isdefined(CC, :InferenceCache)
    CC.InferenceCache()
else
    Vector{CC.InferenceResult}()
end

abstract type AbstractSearchSpace end

Base.length(s::AbstractSearchSpace) = count(_ -> true, s)

struct FixedSpace{names,NT<:NamedTuple{names}} <: AbstractSearchSpace
    elements::Vector{NT}
end

Base.iterate(space::FixedSpace, args...) = iterate(space.elements, args...)

struct CartesianSpace{names,NT<:NamedTuple{names,<:Tuple{Vararg{Tuple}}}} <: AbstractSearchSpace
    constraint::Function
    axes::NT
end

CartesianSpace(axes::NamedTuple) = CartesianSpace(Returns(true), axes)
CartesianSpace(; axes...) = CartesianSpace(NamedTuple(axes))
CartesianSpace(constraint::Function; axes...) = CartesianSpace(constraint, NamedTuple(axes))

function Base.iterate(space::CartesianSpace{names}, state=nothing) where names
    to_cfg = vals -> NamedTuple{names}(vals)
    inner = state === nothing ?
        Iterators.filter(space.constraint ∘ to_cfg,
            Iterators.product(map(Tuple, values(space.axes))...)) :
        state.inner
    result = isnothing(state) ? iterate(inner) : iterate(inner, state.cursor)
    isnothing(result) && return nothing
    vals, cursor = result
    cfg = to_cfg(vals)
    return cfg, (; inner, cursor)
end

include("autotune.jl")

end
