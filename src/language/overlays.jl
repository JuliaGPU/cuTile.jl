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

# Base.fill/zeros/ones return Tiles in kernel context, matching Julia's standard API.
# Marked non-foldable because they return differently-typed objects.
Base.Experimental.@consistent_overlay cuTileMethodTable @inline Base.fill(v, dims::NTuple{N, Int}) where {N} =
    _full(v, typeof(v), dims)
Base.Experimental.@consistent_overlay cuTileMethodTable @inline Base.zeros(::Type{T}, dims::NTuple{N, Int}) where {T, N} =
    _full(zero(T), T, dims)
Base.Experimental.@consistent_overlay cuTileMethodTable @inline Base.ones(::Type{T}, dims::NTuple{N, Int}) where {T, N} =
    _full(one(T), T, dims)


#=============================================================================
 Array construction syntax

 Bracket syntax builds Tiles in kernel context:

     [1, 2, 3, 4]     -> Base.vect    -> Tile{Int64, Tuple{4}}
     [1; 2; 3; 4]     -> Base.vcat    -> Tile{Int64, Tuple{4}}
     [1 2 3 4]        -> Base.hcat    -> Tile{Int64, Tuple{1, 4}}
     [1 3; 2 4]       -> Base.hvcat   -> Tile{Int64, Tuple{2, 2}}
     [1;; 2]          -> Base.hvncat  -> Tile{Int64, Tuple{1, 2}}   (dim::Int form)
     [1; 2;; 3; 4]    -> Base.hvncat  -> Tile{Int64, Tuple{2, 2}}   (dims::Tuple form)

 plus the `T[...]` typed variants, and the same bracket forms over Tile
 elements (`[q1; q2]`, `[t1 t2; t3 t4]`, …), which concatenate.

 Scalar elements funnel into the construction intrinsics (see
 intrinsics/core.jl): compile-time-constant elements emit one dense
 ConstantOp, runtime scalars a balanced cat tree. Tile elements lower to
 balanced `Intrinsics.cat` trees right here at the language level.
 Every dimension of the result (and of every intermediate concatenation)
 must be a power of two; violations are compile-time errors.
 Marked non-foldable like fill/zeros/ones because they return Tiles.

 Every overlay requires at least one element (`x, xs...` rather than plain
 `xs...`): Base code reachable from kernels calls `Base.vect()` on dead
 error paths, and hijacking the zero-element case turns its concrete
 `Vector{Any}` return type into `Any`, derailing downstream inference.
=============================================================================#

## scalar elements

Base.Experimental.@consistent_overlay cuTileMethodTable @inline Base.vect(x::Number, xs::Number...) =
    Intrinsics.vect(promote(x, xs...))
Base.Experimental.@consistent_overlay cuTileMethodTable @inline Base.getindex(::Type{T}, x::Number, xs::Number...) where {T<:Number} =
    Intrinsics.vect(map(T, (x, xs...)))

Base.Experimental.@consistent_overlay cuTileMethodTable @inline Base.vcat(x::Number, xs::Number...) =
    Intrinsics.vect(promote(x, xs...))
Base.Experimental.@consistent_overlay cuTileMethodTable @inline Base.typed_vcat(::Type{T}, x::Number, xs::Number...) where {T<:Number} =
    Intrinsics.vect(map(T, (x, xs...)))

Base.Experimental.@consistent_overlay cuTileMethodTable @inline Base.hcat(x::Number, xs::Number...) =
    Intrinsics.hvncat((1, 1 + length(xs)), true, promote(x, xs...))
Base.Experimental.@consistent_overlay cuTileMethodTable @inline Base.typed_hcat(::Type{T}, x::Number, xs::Number...) where {T<:Number} =
    Intrinsics.hvncat((1, 1 + length(xs)), true, map(T, (x, xs...)))

Base.Experimental.@consistent_overlay cuTileMethodTable @inline Base.hvcat(rows::Tuple{Vararg{Int}}, x::Number, xs::Number...) =
    Intrinsics.hvcat(rows, promote(x, xs...))
Base.Experimental.@consistent_overlay cuTileMethodTable @inline Base.typed_hvcat(::Type{T}, rows::Tuple{Vararg{Int}}, x::Number, xs::Number...) where {T<:Number} =
    Intrinsics.hvcat(rows, map(T, (x, xs...)))

# `dims::Tuple` deliberately also matches the tuple-of-tuples shape descriptor
# that ragged/mixed N-D syntax lowers to ([1 2; 3;;; 4 5; 6]); the intrinsic
# decodes it by probing Base's own hvncat on the host, so invalid literals
# get Julia's exact error instead of Base's array code derailing the compiler.
Base.Experimental.@consistent_overlay cuTileMethodTable @inline Base.hvncat(dims::Tuple, row_first::Bool, x::Number, xs::Number...) =
    Intrinsics.hvncat(dims, row_first, promote(x, xs...))
Base.Experimental.@consistent_overlay cuTileMethodTable @inline Base.typed_hvncat(::Type{T}, dims::Tuple, row_first::Bool, x::Number, xs::Number...) where {T<:Number} =
    Intrinsics.hvncat(dims, row_first, map(T, (x, xs...)))

# Single-dim syntax like [1;; 2] lowers to hvncat(dim::Int, xs...): the result
# has `dim` dimensions, all singleton except the last.
Base.Experimental.@consistent_overlay cuTileMethodTable @inline Base.hvncat(dim::Int, x::Number, xs::Number...) =
    Intrinsics.hvncat(ntuple(i -> i == dim ? 1 + length(xs) : 1, dim), false, promote(x, xs...))
Base.Experimental.@consistent_overlay cuTileMethodTable @inline Base.typed_hvncat(::Type{T}, dim::Int, x::Number, xs::Number...) where {T<:Number} =
    Intrinsics.hvncat(ntuple(i -> i == dim ? 1 + length(xs) : 1, dim), false, map(T, (x, xs...)))

## tile (and mixed tile/scalar) elements: bracket syntax concatenates

# Tile/scalar element for the concatenating overlays below. Scalars are
# lifted to unit tiles, so block forms like [x [y z]; [u, v] M] work (Base
# semantics; though under Tile IR's power-of-two rule mixed-size blocks
# only fit in dimensions where all blocks are the same size).
const TileElem = Union{Tile, Number}

# Build a balanced pairwise `Intrinsics.cat` tree over `leaves` along the
# 0-indexed Julia axis `axis0`. Balanced (rather than a linear fold) so that
# equal-sized inputs keep every intermediate dimension a power of two.
function _cat_tree_expr(leaves::Vector{Any}, axis0::Int)
    function tree(lo::Int, hi::Int)
        lo == hi && return leaves[lo]
        mid = (lo + hi) >> 1
        :(Intrinsics.cat(($(tree(lo, mid)), $(tree(mid + 1, hi))), $axis0))
    end
    tree(1, length(leaves))
end

@generated function _cat_tiles(::Val{axis0}, xs::Tile...) where {axis0}
    Expr(:block, Expr(:meta, :inline),
         _cat_tree_expr(Any[:(xs[$i]) for i in 1:length(xs)], axis0))
end

# Concatenation requires a common element type; promote like Base's cat
@generated function _promote_tiles(xs::Tile...)
    T = mapreduce(eltype, promote_type, xs)
    elems = Any[eltype(x) === T ? :(xs[$i]) : :(convert(Tile{$T}, xs[$i]))
                for (i, x) in enumerate(xs)]
    Expr(:block, Expr(:meta, :inline), Expr(:tuple, elems...))
end

# vcat treats scalars as length-1 vectors
@inline _lift_1d(x::Number) = _full(x, typeof(x), (1,))
@inline _lift_1d(t::Tile) = t

# hcat/hvcat treat scalars as 1x1 blocks and vectors as columns
@inline _lift_2d(x::Number) = _full(x, typeof(x), (1, 1))
@inline _lift_2d(t::Tile{T, Tuple{N}}) where {T, N} = Intrinsics.broadcast(t, (N, 1))
@inline _lift_2d(t::Tile) = t

@generated function _hvcat_tiles(::Val{rows}, xs::Tile...) where {rows}
    off = 0
    row_exprs = Any[]
    for k in rows
        push!(row_exprs, _cat_tree_expr(Any[:(xs[$(off + j)]) for j in 1:k], 1))
        off += k
    end
    off == length(xs) ||
        error("hvcat: row layout $rows does not match $(length(xs)) elements")
    Expr(:block, Expr(:meta, :inline), _cat_tree_expr(row_exprs, 0))
end

# hvncat(dim, ...) concatenates along `dim`, lifting inputs to at least
# `dim` dimensions with trailing singletons (e.g. [t1;;; t2] stacks to 3-D).
@inline _lift_nd(::Val{D}, x::Number) where {D} =
    _full(x, typeof(x), ntuple(_ -> 1, Val(D)))
@inline _lift_nd(::Val{D}, t::Tile) where {D} = t
@generated function _hvncat_tiles(::Val{dim}, xs::Tile...) where {dim}
    shapes = [Tuple(x.parameters[2].parameters) for x in xs]
    rank = max(dim, maximum(length, shapes))
    leaves = Any[
        length(s) == rank ? :(xs[$i]) :
            :(Intrinsics.broadcast(xs[$i], $((s..., ntuple(_ -> 1, rank - length(s))...))))
        for (i, s) in enumerate(shapes)
    ]
    Expr(:block, Expr(:meta, :inline), _cat_tree_expr(leaves, dim - 1))
end

Base.Experimental.@consistent_overlay cuTileMethodTable @inline Base.vcat(x::TileElem, xs::TileElem...) =
    _cat_tiles(Val(0), _promote_tiles(map(_lift_1d, (x, xs...))...)...)
Base.Experimental.@consistent_overlay cuTileMethodTable @inline Base.hcat(x::TileElem, xs::TileElem...) =
    _cat_tiles(Val(1), _promote_tiles(map(_lift_2d, (x, xs...))...)...)
Base.Experimental.@consistent_overlay cuTileMethodTable @inline Base.hvcat(rows::Tuple{Vararg{Int}}, x::TileElem, xs::TileElem...) =
    _hvcat_tiles(Val(rows), _promote_tiles(map(_lift_2d, (x, xs...))...)...)
Base.Experimental.@consistent_overlay cuTileMethodTable @inline Base.hvncat(dim::Int, x::TileElem, xs::TileElem...) =
    _hvncat_tiles(Val(dim), _promote_tiles(map(Base.Fix1(_lift_nd, Val(dim)), (x, xs...))...)...)

# Base.cat with the dims keyword — the generic entry point behind vcat/hcat.
# Scalars lift to unit tiles as in bracket syntax; `dims` must fold to a
# compile-time constant (a literal or Val). No @constprop needed: @inline
# frames with a literal kwarg const-prop through the kwcall wrapper on
# their own (verified by the construction tests).
Base.Experimental.@consistent_overlay cuTileMethodTable @inline Base.cat(x::TileElem, xs::TileElem...; dims) =
    _cat_dims(dims, x, xs...)

@inline _cat_dims(::Val{D}, xs::TileElem...) where {D} = _cat_dims(Int(D), xs...)
@inline function _cat_dims(dims::Integer, xs::TileElem...)
    d = Int(dims)
    _hvncat_tiles(Val(d), _promote_tiles(map(Base.Fix1(_lift_nd, Val(d)), xs)...)...)
end

# T[...] typed concatenation converts every element (scalar or Tile) to T
@inline _convert_elem(::Type{T}, x::Number) where {T} = T(x)
@inline _convert_elem(::Type{T}, t::Tile) where {T} = convert(Tile{T}, t)

Base.Experimental.@consistent_overlay cuTileMethodTable @inline Base.typed_vcat(::Type{T}, x::TileElem, xs::TileElem...) where {T<:Number} =
    _cat_tiles(Val(0), map(e -> _lift_1d(_convert_elem(T, e)), (x, xs...))...)
Base.Experimental.@consistent_overlay cuTileMethodTable @inline Base.typed_hcat(::Type{T}, x::TileElem, xs::TileElem...) where {T<:Number} =
    _cat_tiles(Val(1), map(e -> _lift_2d(_convert_elem(T, e)), (x, xs...))...)
Base.Experimental.@consistent_overlay cuTileMethodTable @inline Base.typed_hvcat(::Type{T}, rows::Tuple{Vararg{Int}}, x::TileElem, xs::TileElem...) where {T<:Number} =
    _hvcat_tiles(Val(rows), map(e -> _lift_2d(_convert_elem(T, e)), (x, xs...))...)
Base.Experimental.@consistent_overlay cuTileMethodTable @inline Base.typed_hvncat(::Type{T}, dim::Int, x::TileElem, xs::TileElem...) where {T<:Number} =
    _hvncat_tiles(Val(dim), map(e -> _lift_nd(Val(dim), _convert_elem(T, e)), (x, xs...))...)

# Commas collect into a Vector on the host, which has no tile equivalent —
# route tile-containing [a, b] / hvncat-shape-form calls into the scalar
# intrinsics so they fail with a clear compile-time error rather than
# derailing on Base's array code.
Base.Experimental.@consistent_overlay cuTileMethodTable @inline Base.vect(x::TileElem, xs::TileElem...) =
    Intrinsics.vect((x, xs...))
Base.Experimental.@consistent_overlay cuTileMethodTable @inline Base.hvncat(dims::Tuple, row_first::Bool, x::TileElem, xs::TileElem...) =
    Intrinsics.hvncat(dims, row_first, (x, xs...))
Base.Experimental.@consistent_overlay cuTileMethodTable @inline Base.typed_hvncat(::Type{T}, dims::Tuple, row_first::Bool, x::TileElem, xs::TileElem...) where {T<:Number} =
    Intrinsics.hvncat(dims, row_first, map(Base.Fix1(_convert_elem, T), (x, xs...)))


