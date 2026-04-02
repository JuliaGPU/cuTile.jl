# Type-safe shape wrappers: Julia (column-major) ↔ Tile IR (row-major)
#
# Tile IR is natively row-major: shapes are stored with the slowest-varying dimension first.
# Julia is column-major: shapes are stored with the fastest-varying dimension first.
# Converting between them is a simple reversal. The Shape{O} wrapper ensures we don't
# accidentally mix up conventions — IR operations accept only RowMajorShape, while
# user-facing shapes from Julia are ColMajorShape.
# Scalar (0D) shapes are represented as RowMajorShape(Int[]) — no separate type needed.

abstract type ShapeKind end
struct RowMajor <: ShapeKind end
struct ColMajor <: ShapeKind end

struct Shape{O<:ShapeKind}
    dims::Vector{Int}
end

const RowMajorShape = Shape{RowMajor}
const ColMajorShape = Shape{ColMajor}
# All shapes in codegen/bytecode are row-major. Scalars are 0D: RowMajorShape(()).
# ColMajorShape is only used at the Julia-facing boundary and converted before storage in CGVal.
const TileShape = RowMajorShape

RowMajorShape(t::Tuple) = RowMajorShape(collect(Int, t))
RowMajorShape(s::RowMajorShape) = s
RowMajorShape(s::ColMajorShape) = RowMajorShape(reverse(s.dims))

ColMajorShape(t::Tuple) = ColMajorShape(collect(Int, t))
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
