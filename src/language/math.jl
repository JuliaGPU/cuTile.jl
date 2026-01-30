# Math operations

public rsqrt


## scalar math

# unary
rsqrt(x::T) where {T <: AbstractFloat} = Intrinsics.rsqrt(x)
for fn in (:abs, :ceil, :floor, :exp, :exp2, :log, :log2, :sqrt,
           :sin, :cos, :tan, :sinh, :cosh, :tanh)
    @eval @overlay Base.$fn(x::T) where {T <: ScalarFloat} = Intrinsics.$fn(x)
end

@overlay Base.fma(x::T, y::T, z::T) where {T <: ScalarFloat} = Intrinsics.fma(x, y, z)

# max/min
@overlay Base.max(x::T, y::T) where {T <: Signed} = Intrinsics.maxi(x, y, SignednessSigned)
@overlay Base.max(x::T, y::T) where {T <: Unsigned} = Intrinsics.maxi(x, y, SignednessUnsigned)
@overlay Base.min(x::T, y::T) where {T <: Signed} = Intrinsics.mini(x, y, SignednessSigned)
@overlay Base.min(x::T, y::T) where {T <: Unsigned} = Intrinsics.mini(x, y, SignednessUnsigned)
@overlay Base.max(x::T, y::T) where {T <: ScalarFloat} = Intrinsics.maxf(x, y)
@overlay Base.min(x::T, y::T) where {T <: ScalarFloat} = Intrinsics.minf(x, y)


## tile math

# element-wise unary
for fn in (:exp, :exp2, :log, :log2, :sqrt, :rsqrt, :ceil, :floor,
           :sin, :cos, :tan, :sinh, :cosh, :tanh)
    @eval @inline Base.Broadcast.broadcasted(::TileStyle, ::typeof($fn), a::Tile{T}) where {T<:AbstractFloat} =
        Intrinsics.$fn(a)
end
@inline Base.Broadcast.broadcasted(::TileStyle, ::typeof(abs), a::Tile{T}) where {T<:AbstractFloat} =
    Intrinsics.absf(a)
@inline Base.Broadcast.broadcasted(::TileStyle, ::typeof(abs), a::Tile{T}) where {T<:Integer} =
    Intrinsics.absi(a)

# element-wise binary
@inline function Base.Broadcast.broadcasted(::TileStyle, ::typeof(rem), a::Tile{T,S1}, b::Tile{T,S2}) where {T<:AbstractFloat,S1,S2}
    S = broadcast_shape(S1, S2)
    Intrinsics.remf(broadcast_to(a, S), broadcast_to(b, S))
end
## max/min
for (op, intrinsic) in ((:max, :maxf), (:min, :minf))
    @eval @inline function Base.Broadcast.broadcasted(::TileStyle, ::typeof($op),
            a::Tile{T,S1}, b::Tile{T,S2}) where {T<:AbstractFloat,S1,S2}
        S = broadcast_shape(S1, S2)
        Intrinsics.$intrinsic(broadcast_to(a, S), broadcast_to(b, S))
    end
end
for (op, intrinsic) in ((:max, :maxi), (:min, :mini))
    @eval begin
        @inline function Base.Broadcast.broadcasted(::TileStyle, ::typeof($op),
                a::Tile{T,S1}, b::Tile{T,S2}) where {T<:Signed,S1,S2}
            S = broadcast_shape(S1, S2)
            Intrinsics.$intrinsic(broadcast_to(a, S), broadcast_to(b, S), SignednessSigned)
        end
        @inline function Base.Broadcast.broadcasted(::TileStyle, ::typeof($op),
                a::Tile{T,S1}, b::Tile{T,S2}) where {T<:Unsigned,S1,S2}
            S = broadcast_shape(S1, S2)
            Intrinsics.$intrinsic(broadcast_to(a, S), broadcast_to(b, S), SignednessUnsigned)
        end
    end
end

# element-wise ternary
## fma
@inline function Base.Broadcast.broadcasted(::TileStyle, ::typeof(fma),
        a::Tile{T,S1}, b::Tile{T,S2}, c::Tile{T,S3}) where {T<:AbstractFloat,S1,S2,S3}
    S = broadcast_shape(broadcast_shape(S1, S2), S3)
    Intrinsics.fma(broadcast_to(a, S), broadcast_to(b, S), broadcast_to(c, S))
end


## mixed math

# broadcasted max/min (float, mixed tile-scalar)
for (op, intrinsic) in ((:max, :maxf), (:min, :minf))
    @eval begin
        @inline Base.Broadcast.broadcasted(::TileStyle, ::typeof($op), a::Tile{T,S}, b::Number) where {T<:AbstractFloat,S} =
            Intrinsics.$intrinsic(a, broadcast_to(Tile(T(b)), S))
        @inline Base.Broadcast.broadcasted(::TileStyle, ::typeof($op), a::Number, b::Tile{T,S}) where {T<:AbstractFloat,S} =
            Intrinsics.$intrinsic(broadcast_to(Tile(T(a)), S), b)
    end
end

# broadcasted max/min (integer, mixed tile-scalar)
for (op, intrinsic) in ((:max, :maxi), (:min, :mini))
    @eval begin
        @inline Base.Broadcast.broadcasted(::TileStyle, ::typeof($op), a::Tile{T,S}, b::Number) where {T<:Signed,S} =
            Intrinsics.$intrinsic(a, broadcast_to(Tile(T(b)), S), SignednessSigned)
        @inline Base.Broadcast.broadcasted(::TileStyle, ::typeof($op), a::Number, b::Tile{T,S}) where {T<:Signed,S} =
            Intrinsics.$intrinsic(broadcast_to(Tile(T(a)), S), b, SignednessSigned)
        @inline Base.Broadcast.broadcasted(::TileStyle, ::typeof($op), a::Tile{T,S}, b::Number) where {T<:Unsigned,S} =
            Intrinsics.$intrinsic(a, broadcast_to(Tile(T(b)), S), SignednessUnsigned)
        @inline Base.Broadcast.broadcasted(::TileStyle, ::typeof($op), a::Number, b::Tile{T,S}) where {T<:Unsigned,S} =
            Intrinsics.$intrinsic(broadcast_to(Tile(T(a)), S), b, SignednessUnsigned)
    end
end

# Float remainder (rem.(tile, scalar), rem.(scalar, tile))
@inline Base.Broadcast.broadcasted(::TileStyle, ::typeof(rem), a::Tile{T,S}, b::Number) where {T<:AbstractFloat,S} =
    Intrinsics.remf(a, broadcast_to(Tile(T(b)), S))
@inline Base.Broadcast.broadcasted(::TileStyle, ::typeof(rem), a::Number, b::Tile{T,S}) where {T<:AbstractFloat,S} =
    Intrinsics.remf(broadcast_to(Tile(T(a)), S), b)


## fma with scalar c (most common: fma.(a, b, bias))
@inline function Base.Broadcast.broadcasted(::TileStyle, ::typeof(fma),
        a::Tile{T,S1}, b::Tile{T,S2}, c::Number) where {T<:AbstractFloat,S1,S2}
    S = broadcast_shape(S1, S2)
    Intrinsics.fma(broadcast_to(a, S), broadcast_to(b, S), broadcast_to(Tile(T(c)), S))
end
## fma with scalar a
@inline function Base.Broadcast.broadcasted(::TileStyle, ::typeof(fma),
        a::Number, b::Tile{T,S1}, c::Tile{T,S2}) where {T<:AbstractFloat,S1,S2}
    S = broadcast_shape(S1, S2)
    Intrinsics.fma(broadcast_to(Tile(T(a)), S), broadcast_to(b, S), broadcast_to(c, S))
end
## fma with scalar b
@inline function Base.Broadcast.broadcasted(::TileStyle, ::typeof(fma),
        a::Tile{T,S1}, b::Number, c::Tile{T,S2}) where {T<:AbstractFloat,S1,S2}
    S = broadcast_shape(S1, S2)
    Intrinsics.fma(broadcast_to(a, S), broadcast_to(Tile(T(b)), S), broadcast_to(c, S))
end
