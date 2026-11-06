public AbstractSearchSpace, FixedSpace, CartesianSpace

abstract type AbstractSearchSpace end

Base.length(space::AbstractSearchSpace) = count(Returns(true), space)

struct FixedSpace{T<:NamedTuple} <: AbstractSearchSpace
    elements::Vector{T}
end

FixedSpace(elements::AbstractVector{T}) where {T<:NamedTuple} =
    FixedSpace{T}(collect(elements))

function FixedSpace(configs)
    elements = collect(configs)
    all(config -> config isa NamedTuple, elements) ||
        throw(ArgumentError("FixedSpace requires NamedTuple configs"))
    return FixedSpace(NamedTuple[elements...])
end

Base.eltype(::Type{<:FixedSpace{T}}) where {T} = T
Base.length(space::FixedSpace) = length(space.elements)
Base.iterate(space::FixedSpace, args...) = iterate(space.elements, args...)

struct CartesianSpace{names,F,Axes<:NamedTuple{names}} <: AbstractSearchSpace
    constraint::F
    axes::Axes
end

_axis_tuple(axis::Tuple) = axis
_axis_tuple(axis) = Tuple(axis)

function _cartesian_space(constraint::F, axes::NamedTuple) where {F}
    tuple_axes = map(_axis_tuple, axes)
    return CartesianSpace{keys(tuple_axes),F,typeof(tuple_axes)}(constraint, tuple_axes)
end

CartesianSpace(axes::NamedTuple) = _cartesian_space(Returns(true), axes)
CartesianSpace(; axes...) = CartesianSpace(NamedTuple(axes))
CartesianSpace(constraint::Function; axes...) =
    _cartesian_space(constraint, NamedTuple(axes))

Base.eltype(::Type{<:CartesianSpace{names}}) where {names} = NamedTuple{names}

function Base.iterate(space::CartesianSpace{names}, state=nothing) where {names}
    product, cursor = state === nothing ?
        (Iterators.product(values(space.axes)...), nothing) :
        (state.product, state.cursor)

    result = cursor === nothing ? iterate(product) : iterate(product, cursor)
    while result !== nothing
        values, cursor = result
        cfg = NamedTuple{names}(values)
        space.constraint(cfg) && return cfg, (; product, cursor)
        result = iterate(product, cursor)
    end
    return nothing
end
