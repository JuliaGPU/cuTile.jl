module Experimental

using ..cuTile
using ..cuTile: cuTileconvert, cufunction, default_sm_arch

using CUDACore: CUDACore

using Random

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
