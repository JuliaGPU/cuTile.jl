# Method Overlay Infrastructure for cuTile Compilation
#
# Defines the @overlay macro and type conversion overlays.
# Arithmetic and math overlays are defined in arithmetic.jl and math.jl.

macro overlay(ex)
    esc(:(Base.Experimental.@consistent_overlay cuTileMethodTable Base.@assume_effects :foldable $ex))
end


#=============================================================================
 StepRange Construction
=============================================================================#

# GPU-safe replacement for Base.steprange_last to enable `for i in start:step:stop`.
# The original pulls in ArgumentError, @noinline overflow_case, and checked_srem_int.
# This overlay uses unsigned arithmetic (bitcast → unsigned rem → bitcast) which
# produces identical results and maps cleanly to Tile IR (signless integers make
# signed↔unsigned bitcasts no-ops).
@overlay function Base.steprange_last(start::T, step::T, stop::T) where {T<:Base.BitInteger}
    stop == start && return stop
    if step > zero(step)
        stop < start && return start - oneunit(step)  # empty range
        remain = signed(unsigned(stop - start) % unsigned(step))
        return stop - remain
    else
        stop > start && return start + oneunit(step)  # empty range
        remain = signed(unsigned(start - stop) % unsigned(-step))
        return stop + remain
    end
end


#=============================================================================
 Type Conversions
=============================================================================#

# Type tuples for metaprogramming specific overlays
# Generic overlays don't take precedence over Core's Int64(x::BuiltinInts) etc.
const SignedInts = (Int8, Int16, Int32, Int64)
const UnsignedInts = (UInt8, UInt16, UInt32, UInt64)
const Floats = (Float16, BFloat16, Float32, TFloat32, Float64)

# Integer to integer (specific type pairs for promotion/truncation)
for T in SignedInts, S in SignedInts
    T === S && continue
    if sizeof(T) > sizeof(S)
        @eval @overlay $T(x::$S) = Intrinsics.exti(x, $T, Signedness.Signed)
    else
        @eval @overlay $T(x::$S) = Intrinsics.trunci(x, $T)
    end
end

for T in UnsignedInts, S in UnsignedInts
    T === S && continue
    if sizeof(T) > sizeof(S)
        @eval @overlay $T(x::$S) = Intrinsics.exti(x, $T, Signedness.Unsigned)
    else
        @eval @overlay $T(x::$S) = Intrinsics.trunci(x, $T)
    end
end

# Bool to integer (zero-extend: false→0, true→1)
for T in (SignedInts..., UnsignedInts...)
    @eval @overlay $T(x::Bool) = Intrinsics.exti(x, $T, Signedness.Unsigned)
end

# Integer extension/truncation (via rem) - T and S both used in body
@overlay Base.rem(x::T, ::Type{S}) where {T <: Signed, S <: Signed} =
    sizeof(S) > sizeof(T) ? Intrinsics.exti(x, S, Signedness.Signed) :
    sizeof(S) < sizeof(T) ? Intrinsics.trunci(x, S) : x

@overlay Base.rem(x::T, ::Type{S}) where {T <: Unsigned, S <: Unsigned} =
    sizeof(S) > sizeof(T) ? Intrinsics.exti(x, S, Signedness.Unsigned) :
    sizeof(S) < sizeof(T) ? Intrinsics.trunci(x, S) : x

# Float to float
for T in Floats, S in Floats
    T === S && continue
    @eval @overlay $T(x::$S) = Intrinsics.ftof(x, $T)
end

# Integer to float
for F in Floats
    for I in SignedInts
        @eval @overlay $F(x::$I) = Intrinsics.itof(x, $F, Signedness.Signed)
    end
    for I in UnsignedInts
        @eval @overlay $F(x::$I) = Intrinsics.itof(x, $F, Signedness.Unsigned)
    end
    @eval @overlay $F(x::Bool) = Intrinsics.itof(x, $F, Signedness.Unsigned)
end

# Float to integer (via unsafe_trunc)
for F in Floats
    for I in SignedInts
        @eval @overlay Base.unsafe_trunc(::Type{$I}, x::$F) = Intrinsics.ftoi(x, $I, Signedness.Signed)
    end
    for I in UnsignedInts
        @eval @overlay Base.unsafe_trunc(::Type{$I}, x::$F) = Intrinsics.ftoi(x, $I, Signedness.Unsigned)
    end
end

# Float to integer (round with RoundToZero)
for F in Floats, I in (SignedInts..., UnsignedInts...)
    @eval @overlay function Base.round(::Type{$I}, x::$F, ::Base.Rounding.RoundingMode{:ToZero})
        # TODO: assert that x is within bounds etc
        unsafe_trunc($I, x)
    end
end

# Float to integer (direct constructor)
for F in Floats
    for I in SignedInts
        @eval @overlay function $I(x::$F)
            # TODO: assert that x is within bounds etc
            unsafe_trunc($I, x)
        end
    end
    for I in UnsignedInts
        @eval @overlay function $I(x::$F)
            # TODO: assert that x is within bounds etc
            unsafe_trunc($I, x)
        end
    end
end


#=============================================================================
 Printing
=============================================================================#

# Override all print/println entry points from coreio.jl to bypass stdout
# and route directly to the print_tko Tile IR instruction.
# Uses @consistent_overlay (not @overlay) because print has side effects.
Base.Experimental.@consistent_overlay cuTileMethodTable Base.print(x) =
    Intrinsics.print_tko(x)
Base.Experimental.@consistent_overlay cuTileMethodTable Base.print(x1, x2) =
    Intrinsics.print_tko(x1, x2)
Base.Experimental.@consistent_overlay cuTileMethodTable Base.print(xs...) =
    Intrinsics.print_tko(xs...)
Base.Experimental.@consistent_overlay cuTileMethodTable Base.println() =
    Intrinsics.print_tko("\n")
Base.Experimental.@consistent_overlay cuTileMethodTable Base.println(x) =
    Intrinsics.print_tko(x, "\n")
Base.Experimental.@consistent_overlay cuTileMethodTable Base.println(x1, x2) =
    Intrinsics.print_tko(x1, x2, "\n")
Base.Experimental.@consistent_overlay cuTileMethodTable Base.println(xs...) =
    Intrinsics.print_tko(xs..., "\n")

# String interpolation support: route string() to format_string intrinsic.
# For all-constant args, the interpreter constant-folds via :foldable effects.
# For args containing Tiles, the format_string intrinsic is emitted in the IR
# and later fused into print_tko by the print fusion pass.
@overlay Base.string(xs...) = Intrinsics.format_string(xs...)


#=============================================================================
 Tile Constructors
=============================================================================#

Base.Experimental.@consistent_overlay cuTileMethodTable Base.fill(v, dims::NTuple{N, Int}) where {N} =
    _full(v, typeof(v), dims)
Base.Experimental.@consistent_overlay cuTileMethodTable Base.zeros(::Type{T}, dims::NTuple{N, Int}) where {T, N} =
    _full(zero(T), T, dims)
Base.Experimental.@consistent_overlay cuTileMethodTable Base.ones(::Type{T}, dims::NTuple{N, Int}) where {T, N} =
    _full(one(T), T, dims)


#=============================================================================
 Array construction
=============================================================================#

## empty literals

Base.Experimental.@consistent_overlay cuTileMethodTable Base.getindex(::Type{T}) where {T<:Number} =
    throw(ArgumentError("empty tile literals are not supported"))
Base.Experimental.@consistent_overlay cuTileMethodTable Base.vect() =
    throw(ArgumentError("empty tile literals are not supported"))

## tile and scalar concatenation

const TileElem = Union{Tile, Number}

@generated function promote_tile_elements(xs::Tuple)
    types = xs.parameters
    T = mapreduce(type -> type <: Tile ? eltype(type) : type, promote_type, types)
    elements = Any[]
    for i in eachindex(types)
        push!(elements, :(convert(Tile{$T}, xs[$i])))
    end
    Expr(:tuple, elements...)
end

typed_tile_elements(::Type{T}, xs) where {T} =
    map(x -> convert(Tile{T}, x), xs)

function pad_tile(tile::Tile, rank)
    shape = size(tile)
    length(shape) == rank && return tile
    Intrinsics.broadcast(tile, (shape..., ntuple(_ -> 1, rank - length(shape))...))
end

function cat_split(widths, target, i=1, total=0)
    i == length(widths) && return length(widths) ÷ 2
    total += widths[i]
    total == target ? i : cat_split(widths, target, i + 1, total)
end

cat_tree(tiles, dim) = cat_tree(tiles, Val(dim))

@generated function cat_tree(tiles::Tuple, ::Val{dim}) where {dim}
    widths = map(type -> size(type, dim), tiles.parameters)

    function tree(first, last)
        first == last && return :(tiles[$first])
        split = first - 1 + cat_split(widths[first:last], sum(widths[first:last]) ÷ 2)
        :(Intrinsics.cat(($(tree(first, split)), $(tree(split + 1, last))), $(dim - 1)))
    end

    tree(1, length(widths))
end

function cat_tiles(tiles, dim)
    dim > 0 || throw(ArgumentError("concatenation dimension must be positive"))
    rank = max(dim, maximum(ndims, tiles))
    cat_tree(map(tile -> pad_tile(tile, rank), tiles), dim)
end

cat_layout(tiles, layout, row_first) = cat_layout(tiles, Val(layout), Val(row_first))

@generated function cat_layout(tiles::Tuple, ::Val{layout}, ::Val{row_first}) where {layout, row_first}
    isempty(layout) && return :(throw(ArgumentError("hvncat layout cannot be empty")))
    shaped = first(layout) isa Tuple
    rank = max(length(layout) == 1 && shaped && row_first ? 2 : length(layout),
               maximum(ndims, tiles.parameters))
    nodes = Any[]
    for i in eachindex(tiles.parameters)
        push!(nodes, :(pad_tile(tiles[$i], $rank)))
    end
    counts = ones(Int, length(nodes))

    function fixed_groups(nodes, counts, n, dim)
        length(nodes) % n == 0 || return nothing
        grouped = Any[]
        grouped_counts = Int[]
        for first in 1:n:length(nodes)
            indices = first:first + n - 1
            push!(grouped, :(cat_tree(($(nodes[indices]...),), $dim)))
            push!(grouped_counts, sum(counts[indices]))
        end
        grouped, grouped_counts
    end

    function shaped_groups(nodes, counts, targets, dim)
        grouped = Any[]
        grouped_counts = Int[]
        first = 1
        for target in targets
            total = 0
            last = first - 1
            while last < length(counts) && total < target
                last += 1
                total += counts[last]
            end
            total == target || return nothing
            push!(grouped, :(cat_tree(($(nodes[first:last]...),), $dim)))
            push!(grouped_counts, total)
            first = last + 1
        end
        first == length(nodes) + 1 || return nothing
        grouped, grouped_counts
    end

    if shaped
        all(level -> level isa Tuple && all(x -> x isa Int && x > 0, level), layout) ||
            return :(throw(ArgumentError("`shape` argument must consist of positive integers")))
        for (level, targets) in enumerate(layout)
            dim = row_first && level < 3 ? 3 - level : level
            result = shaped_groups(nodes, counts, targets, dim)
            result === nothing && return :(throw(DimensionMismatch("hvncat shape levels do not nest evenly")))
            nodes, counts = result
        end
    else
        all(x -> x isa Int && x > 0, layout) ||
            return :(throw(ArgumentError("`dims` argument must contain positive integers")))
        prod(layout) == length(nodes) ||
            return :(throw(ArgumentError("argument count does not match specified shape")))
        axes = length(layout) == 1 ? (1,) :
               row_first ? (2, 1, (3:length(layout))...) : ntuple(identity, length(layout))
        for dim in axes
            result = fixed_groups(nodes, counts, layout[dim], dim)
            result === nothing && return :(throw(DimensionMismatch("hvncat shape levels do not nest evenly")))
            nodes, counts = result
        end
    end

    length(nodes) == 1 || return :(throw(DimensionMismatch("hvncat layout does not form one tile")))
    only(nodes)
end

function cat_dimensions(tiles, dim::Integer)
    dim > 0 || throw(ArgumentError("All cat dimensions must be positive integers, but got $dim"))
    cat_tiles(tiles, Int(dim))
end

cat_dimensions(tiles, dims::Tuple{Vararg{Integer}}) =
    cat_dimensions(tiles, Val(dims))

@generated function cat_dimensions(tiles, ::Val{dims}) where {dims}
    isempty(dims) && return :(throw(ArgumentError("cat dimensions cannot be empty")))
    any(<=(0), dims) && return :(throw(ArgumentError(
        "All cat dimensions must be positive integers, but got $dims")))
    canonical = Tuple(unique(Int[dims...]))
    length(canonical) == 1 && return :(cat_tiles(tiles, $(only(canonical))))

    types = tiles.parameters
    shapes = map(size, types)
    rank = max(maximum(canonical), maximum(length, shapes))
    shapes = map(shape -> (shape..., ntuple(_ -> 1, rank - length(shape))...), shapes)
    T = mapreduce(eltype, promote_type, types)

    coords = Any[()]
    for _ in canonical
        next = Any[]
        for i in eachindex(types), prefix in coords
            push!(next, (prefix..., i))
        end
        coords = next
    end

    cells = Any[]
    for coord in coords
        shape = ntuple(rank) do dim
            position = findfirst(==(dim), canonical)
            position === nothing ? shapes[1][dim] : shapes[coord[position]][dim]
        end
        if all(==(first(coord)), coord)
            push!(cells, :(pad_tile(tiles[$(first(coord))], $rank)))
        else
            push!(cells, :(zeros($T, $(shape...))))
        end
    end

    layout = ntuple(dim -> dim in canonical ? length(types) : 1, rank)
    :(cat_layout(($(cells...),), $layout, false))
end


cat_dimensions(tiles, dims) =
    throw(ArgumentError("cat dimensions must be integers, got $dims"))

cat_dims_value(dims) = dims
cat_dims_value(::Val{D}) where {D} = D

Base.Experimental.@consistent_overlay cuTileMethodTable Base.vcat(x::TileElem, xs::TileElem...) =
    cat_tiles(promote_tile_elements((x, xs...)), 1)
Base.Experimental.@consistent_overlay cuTileMethodTable Base.hcat(x::TileElem, xs::TileElem...) =
    cat_tiles(promote_tile_elements((x, xs...)), 2)
Base.Experimental.@consistent_overlay cuTileMethodTable Base.hvcat(rows::Tuple{Vararg{Int}}, x::TileElem, xs::TileElem...) =
    cat_layout(promote_tile_elements((x, xs...)), Base.rows_to_dimshape(rows), true)
Base.Experimental.@consistent_overlay cuTileMethodTable Base.hvncat(dim::Int, x::TileElem, xs::TileElem...) =
    cat_tiles(promote_tile_elements((x, xs...)), dim)
Base.Experimental.@consistent_overlay cuTileMethodTable Base.hvncat(layout::Tuple, row_first::Bool, x::TileElem, xs::TileElem...) =
    cat_layout(promote_tile_elements((x, xs...)), layout, row_first)
Base.Experimental.@consistent_overlay cuTileMethodTable Base.cat(x::TileElem, xs::TileElem...; dims) =
    cat_dimensions(promote_tile_elements((x, xs...)), cat_dims_value(dims))

Base.Experimental.@consistent_overlay cuTileMethodTable Base.typed_vcat(::Type{T}, x::TileElem, xs::TileElem...) where {T<:Number} =
    cat_tiles(typed_tile_elements(T, (x, xs...)), 1)
Base.Experimental.@consistent_overlay cuTileMethodTable Base.typed_hcat(::Type{T}, x::TileElem, xs::TileElem...) where {T<:Number} =
    cat_tiles(typed_tile_elements(T, (x, xs...)), 2)
Base.Experimental.@consistent_overlay cuTileMethodTable Base.typed_hvcat(::Type{T}, rows::Tuple{Vararg{Int}}, x::TileElem, xs::TileElem...) where {T<:Number} =
    cat_layout(typed_tile_elements(T, (x, xs...)), Base.rows_to_dimshape(rows), true)
Base.Experimental.@consistent_overlay cuTileMethodTable Base.typed_hvncat(::Type{T}, dim::Int, x::TileElem, xs::TileElem...) where {T<:Number} =
    cat_tiles(typed_tile_elements(T, (x, xs...)), dim)
Base.Experimental.@consistent_overlay cuTileMethodTable Base.typed_hvncat(::Type{T}, layout::Tuple, row_first::Bool, x::TileElem, xs::TileElem...) where {T<:Number} =
    cat_layout(typed_tile_elements(T, (x, xs...)), layout, row_first)

Base.Experimental.@consistent_overlay cuTileMethodTable Base.vect(x::Number, xs::Number...) =
    cat_tiles(promote_tile_elements((x, xs...)), 1)
Base.Experimental.@consistent_overlay cuTileMethodTable Base.getindex(::Type{T}, x::Number, xs::Number...) where {T<:Number} =
    cat_tiles(typed_tile_elements(T, (x, xs...)), 1)
Base.Experimental.@consistent_overlay cuTileMethodTable Base.vect(x::TileElem, xs::TileElem...) =
    throw(ArgumentError("comma-separated tile literals are not supported; use concatenation syntax"))
