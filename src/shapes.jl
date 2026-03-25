# Type-safe shape wrappers: Julia (column-major) ↔ Tile IR (row-major)
#
# Tile IR is natively row-major: shapes are stored with the slowest-varying dimension first.
# Julia is column-major: shapes are stored with the fastest-varying dimension first.
# Converting between them is a simple reversal. The Shape{O} wrapper ensures we don't
# accidentally mix up conventions — IR operations accept only RowMajorShape, while
# user-facing shapes from Julia are ColMajorShape.

abstract type ShapeKind end
struct RowMajor <: ShapeKind end
struct ColMajor <: ShapeKind end
struct Scalar <: ShapeKind end

struct Shape{O<:ShapeKind}
    dims::Vector{Int}
end

const ScalarShape = Shape{Scalar}
const RowMajorShape = Shape{RowMajor}
const ColMajorShape = Shape{ColMajor}
const TileShape = Shape{<:Union{RowMajor, Scalar}}

ScalarShape() = Shape{Scalar}(Int[])

RowMajorShape(t::Tuple) = RowMajorShape(collect(Int, t))
RowMajorShape(s::ScalarShape) = RowMajorShape(s.dims)
RowMajorShape(s::RowMajorShape) = s
RowMajorShape(s::ColMajorShape) = RowMajorShape(reverse(s.dims))

ColMajorShape(t::Tuple) = ColMajorShape(collect(Int, t))
ColMajorShape(s::ScalarShape) = ColMajorShape(s.dims)
ColMajorShape(s::ColMajorShape) = s
ColMajorShape(s::RowMajorShape) = ColMajorShape(reverse(s.dims))

# Forward common operations to .dims
Base.length(s::Shape) = length(s.dims)
Base.isempty(s::Shape) = isempty(s.dims)
Base.getindex(s::Shape, i) = s.dims[i]
Base.setindex!(s::Shape, v, i) = (s.dims[i] = v; s)
Base.copy(s::Shape{O}) where O = Shape{O}(copy(s.dims))
Base.:(==)(a::Shape{O}, b::Shape{O}) where O = a.dims == b.dims
Base.iterate(s::Shape, state...) = iterate(s.dims, state...)
Base.eachindex(s::Shape) = eachindex(s.dims)
Base.collect(s::Shape) = s.dims
TupleType(s::Shape) = Tuple{s.dims...}
