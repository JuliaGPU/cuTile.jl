# Math operations

# `exp2` and friends are plain `Base` overlays with no cuTile-owned docstring, so they are
# not marked `public`: `Base` already exports them.
public rsqrt


## scalar math

# unary
"""
    rsqrt(x)

Reciprocal square root, `1 / sqrt(x)`, computed as a single Tile IR operation
rather than a division followed by a square root. Broadcast it (`rsqrt.(tile)`)
to apply it element-wise over a tile.
"""
rsqrt(x::T) where {T <: AbstractFloat} = Intrinsics.rsqrt(x)
for fn in (:ceil, :floor, :exp, :exp2, :log, :log2, :sqrt,
           :sin, :cos, :tan, :sinh, :cosh, :tanh)
    @eval @overlay Base.$fn(x::T) where {T <: ScalarFloat} = Intrinsics.$fn(x)
end

@overlay Base.abs(x::T) where {T <: ScalarFloat} = Intrinsics.absf(x)
@overlay Base.abs(x::T) where {T <: Signed} = Intrinsics.absi(x)

@overlay Base.isnan(x::T) where {T <: ScalarFloat} = Intrinsics.cmpf(x, x, ComparisonPredicate.NotEqual, ComparisonOrdering.Unordered)

@overlay Base.rem(x::T, y::T) where {T <: ScalarFloat} = Intrinsics.remf(x, y)

@overlay Base.fma(x::T, y::T, z::T) where {T <: ScalarFloat} = Intrinsics.fma(x, y, z)

# Julia's `atan(y, x)` = `atan(y/x)`; Tile IR's `atan2(a, b)` = `atan(a/b)`.
# Arguments map first-to-first.
@overlay Base.atan(y::T, x::T) where {T <: ScalarFloat} = Intrinsics.atan2(y, x)
@overlay Base.atan(x::T) where {T <: ScalarFloat} = Intrinsics.atan2(x, one(T))

# Tile IR has no copysign; combine the magnitude bits of `x` with the sign bit of `y`.
@overlay function Base.copysign(x::T, y::T) where {T <: Base.IEEEFloat}
    U = Base.uinttype(T)
    mask = Base.sign_mask(T)
    reinterpret(T, (reinterpret(U, x) & ~mask) | (reinterpret(U, y) & mask))
end

# A quotient as `a * (1 / b)` (`div.approx.f32`, within 2 ulp), without the
# range check and slow-path call IEEE division makes for every element. For a
# divisor in a narrow range whose quotient only corrects a larger term.
@inline function approx_div(a::Float32, b::Float32)
    Intrinsics.fpmode_begin(Rounding.Approx, nothing)
    q = a / b
    Intrinsics.fpmode_end()
    q
end

# Base's `log1p` and `expm1` look up tables with a per-element index. Float32
# uses the table-free algorithms of fdlibm (as in musl's `log1pf` and `expm1f`,
# within 0.86 ulp), written without early returns so that every element
# computes each path and selects its own. Float16 and BFloat16 round the
# Float32 result. Other types use Kahan's formulations on top of `log` and
# `exp`: the rounding error of `u = 1 + x` (or `u = exp(x)`) cancels in the
# quotient, leaving a few ulp (2.4 in Float32). Like the other overlays,
# out-of-domain arguments give NaN rather than throwing.
@overlay function Base.log1p(x::Float32)
    ix = reinterpret(UInt32, x)
    # For 1 + x in [sqrt(2)/2, sqrt(2)), use x directly. Otherwise split
    # 1 + x = 2^k * (1 + f) and carry the rounding error of 1 + x in c.
    direct = (ix < 0x3ed413d0) | ((ix >= 0x80000000) & (ix <= 0xbe95f619))
    u = 1f0 + x
    iu = reinterpret(UInt32, u) + (0x3f800000 - 0x3f3504f3)
    k = reinterpret(Int32, iu >> UInt32(23)) - Int32(0x7f)
    c = ifelse(k < Int32(25), approx_div(ifelse(k >= Int32(2), 1f0 - (u - x), x - (u - 1f0)), u), 0f0)
    f = reinterpret(Float32, (iu & 0x007fffff) + 0x3f3504f3) - 1f0
    k = ifelse(direct, Int32(0), k)
    c = ifelse(direct, 0f0, c)
    f = ifelse(direct, x, f)

    # log(1 + f) = f - f^2/2 + s*(f^2/2 + R(s^2)), with s = f / (2 + f)
    s = approx_div(f, 2f0 + f)
    z = s * s
    w = z * z
    t1 = w * (reinterpret(Float32, 0x3eccce13) + w * reinterpret(Float32, 0x3e789e26))
    t2 = z * (reinterpret(Float32, 0x3f2aaaaa) + w * reinterpret(Float32, 0x3e91e9ee))
    hfsq = 0.5f0 * f * f
    dk = Float32(k)
    ln2_hi = reinterpret(Float32, 0x3f317180)
    ln2_lo = reinterpret(Float32, 0x3717f7d1)
    r = s * (hfsq + (t2 + t1)) + (dk * ln2_lo + c) - hfsq + f + dk * ln2_hi

    # |x| < 2^-24, +Inf and NaN give x; -1 gives -Inf; x < -1 gives NaN.
    r = ifelse((ix << UInt32(1)) < (0x33800000 << UInt32(1)), x, r)
    r = ifelse((ix >= 0x7f800000) & (ix < 0x80000000), x, r)
    ifelse(ix >= 0xbf800000, ifelse(x == -1f0, -Inf32, NaN32), r)
end
@overlay function Base.expm1(x::Float32)
    ix = reinterpret(UInt32, x)
    hx = ix & 0x7fffffff
    neg = ix >= 0x80000000

    # For |x| > ln2/2, reduce to hi - lo = x - k*ln2, with c the rounding error.
    ln2_hi = reinterpret(Float32, 0x3f317180)
    ln2_lo = reinterpret(Float32, 0x3717f7d1)
    kf = reinterpret(Float32, 0x3fb8aa3b) * x + ifelse(neg, -0.5f0, 0.5f0)
    k = ifelse(hx < 0x3f851592, ifelse(neg, Int32(-1), Int32(1)), unsafe_trunc(Int32, kf))
    t = Float32(k)
    hi = x - t * ln2_hi  # exact for these k
    lo = t * ln2_lo
    reduced = hx > 0x3eb17218
    xr = ifelse(reduced, hi - lo, x)
    c = ifelse(reduced, (hi - xr) - lo, 0f0)
    k = ifelse(reduced, k, Int32(0))

    # expm1(xr) from a rational approximation, then scaled by 2^k.
    hfx = 0.5f0 * xr
    hxs = xr * hfx
    r1 = 1f0 + hxs * (reinterpret(Float32, 0xbd088868) + hxs * reinterpret(Float32, 0x3acf3010))
    t3 = 3f0 - r1 * hfx
    e = hxs * approx_div(r1 - t3, 6f0 - xr * t3)
    r0 = xr - (xr * e - hxs)
    e = xr * (e - c) - c - hxs
    rm1 = 0.5f0 * (xr - e) - 0.5f0
    rp1 = ifelse(xr < -0.25f0, -2f0 * (e - (xr + 0.5f0)), 1f0 + 2f0 * (xr - e))
    twopk = reinterpret(Float32, reinterpret(UInt32, Int32(0x7f) + k) << UInt32(23))
    twomk = reinterpret(Float32, reinterpret(UInt32, Int32(0x7f) - k) << UInt32(23))
    y = xr - e + 1f0
    # 2^128 is not a Float32, so scale by 2 * 2^127 for k = 128.
    rfar = ifelse(k == Int32(128), y * 2f0 * reinterpret(Float32, 0x7f000000), y * twopk) - 1f0
    rmid = ifelse(k < Int32(23), (xr - e + (1f0 - twomk)) * twopk,
                  (xr - (e + twomk) + 1f0) * twopk)
    r = ifelse(k == Int32(0), r0,
        ifelse(k == Int32(-1), rm1,
        ifelse(k == Int32(1), rp1,
        ifelse((k < Int32(0)) | (k > Int32(56)), rfar, rmid))))

    # |x| < 2^-25 and NaN give x; below -27 ln2, -1; above the overflow
    # threshold (higher than fdlibm's 0x42b17180, which overflows early), Inf.
    r = ifelse(hx < 0x33000000, x, r)
    far = hx >= 0x4195b844
    r = ifelse(far & neg, -1f0, r)
    r = ifelse(far & !neg & (x > reinterpret(Float32, 0x42b17217)), Inf32, r)
    ifelse(hx > 0x7f800000, x, r)
end
@overlay Base.log1p(x::T) where {T <: Union{Float16, BFloat16}} = T(Base.log1p(Float32(x)))
@overlay Base.expm1(x::T) where {T <: Union{Float16, BFloat16}} = T(Base.expm1(Float32(x)))
@overlay function Base.log1p(x::T) where {T <: ScalarFloat}
    u = one(T) + x
    r = Intrinsics.log(u) * (x / (u - one(T)))
    ifelse(u == one(T), x, ifelse(u == T(Inf), u, r))
end
@overlay function Base.expm1(x::T) where {T <: ScalarFloat}
    u = Intrinsics.exp(x)
    um1 = u - one(T)
    r = um1 * (x / Intrinsics.log(u))
    ifelse(u == one(T), x, ifelse(um1 == -one(T) || u == T(Inf), um1, r))
end

# max/min
@overlay Base.max(x::T, y::T) where {T <: Signed} = Intrinsics.maxi(x, y, Signedness.Signed)
@overlay Base.max(x::T, y::T) where {T <: Unsigned} = Intrinsics.maxi(x, y, Signedness.Unsigned)
@overlay Base.min(x::T, y::T) where {T <: Signed} = Intrinsics.mini(x, y, Signedness.Signed)
@overlay Base.min(x::T, y::T) where {T <: Unsigned} = Intrinsics.mini(x, y, Signedness.Unsigned)
@overlay Base.max(x::T, y::T) where {T <: ScalarFloat} = Intrinsics.maxf(x, y)
@overlay Base.min(x::T, y::T) where {T <: ScalarFloat} = Intrinsics.minf(x, y)


## tile math

# All tile broadcasting (unary, binary, ternary) is handled by the generic
# Broadcast.copy → map path: scalar @overlay methods provide the element-wise
# implementation, and map handles broadcasting + to_scalar/from_scalar.
# No explicit broadcasted methods needed here.
