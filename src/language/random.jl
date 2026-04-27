# Device-side random number generation
#
# Uniform Philox2x32-7 RNG, tile-vectorized. Sits on top of the four
# placeholder state intrinsics (`rng_counter`, `rng_advance`, `rng_seed`,
# `rng_set_seed`) in `compiler/intrinsics/random.jl`, which the
# `rng_state_pass!` lowering pass rewrites to concrete SSA — threading two
# per-stream UInt32 slots (counter and seed) through structured control
# flow.
#
# === Pipeline =============================================================
#
#   user code: Random.rand(T, dims)        # overlay → rand_tile
#                       │
#                       ▼
#   rand_tile(stream, T, dims) =
#       finalize_tile(T, underlying_rand(T, stream, dims))
#                       │
#       ┌───────────────┴────────────────────┐
#       ▼                                    ▼
#   Layer 1: width-matched unsigned core     Layer 2: same-width type
#   ─ "use all the bits per Philox call"     ─ "give it the right Julia type"
#
#   underlying_rand dispatch (only place      finalize_tile dispatch (no bit
#   that knows about bit-packing):            packing here):
#     Int8/UInt8        → uint8_tile            Tile{UIntᵢ}  → identity
#     Int16/UInt16/F16/BF16 → uint16_tile       Tile{UIntᵢ}  → bitcast (Int)
#     Int32/UInt32/F32  → uint32_tile           Tile{UInt32} → u01(Float32)
#     Int64/UInt64/F64  → uint64_tile           Tile{UInt16} → u01(Float16)
#                                               Tile{UInt16} → u01(BFloat16)
#                                               Tile{UInt64} → u01(Float64)
#
# `rand_uint32_tile` is the primitive (calls philox2x_rounds). The other
# three derive from it: `_uint64` packs (c1, c2) into UInt64s, `_uint16`
# splits each UInt32 into 2 chunks via shri+trunci+cat, `_uint8` splits
# 4 ways. The cyclic tile distribution makes every cat/split a
# register-level no-op (verified on SASS — zero added warp shuffles).
#
# Adding a same-width output type: one line in `underlying_rand` + one
# method on `finalize_tile`. The Philox core never needs to change.
#
# === In-kernel API ========================================================
#
#   rand()                              # → Float32 scalar
#   rand(::Type{T})                     # → T scalar         (T ∈ RandTypes)
#   rand(::Type{T}, dims)               # → Tile{T, dims}    (T ∈ RandTypes)
#   rand(dims)                          # → Tile{Float32, dims}
#   Random.seed!(Random.default_rng(), seed)   # re-seed threaded stream
#
# Per-stream key derivation (`rng_key`) hashes the linear block id with the
# Weyl constant and XORs in the threaded seed — different seeds give
# uncorrelated Philox streams across blocks; different blocks under the
# same seed remain disjoint.
#
# Loops: `rng_state_pass!` threads (counter, seed) as loop-carried SSA, so
# each iteration's `rand` sees a fresh counter; in-loop `Random.seed!`
# similarly threads the seed.
#
# Limitations:
#  - Output types: `RandTypes` (8/16/32/64-bit ints + F16/BF16/F32/F64).
#  - No `randn` / `randexp` yet. The two-layer split above extends to
#    `randn` cleanly: `underlying_normal` overrides narrow floats to
#    UInt32 (Box-Muller's polynomial approximations need ≥23-bit input);
#    a `randn_finalize_tile` does pair → boxmuller → cat. Same Philox
#    core, no new intrinsics.

using Random

public rand, DeviceRNG

# `KernelState` (carrying the per-launch `seed`) is defined in
# `language/kernel_state.jl`; `lower_rng_state!` seeds the default stream from
# `KernelState.seed`, while `cuTile.RNG`-driven fill kernels still take a
# `(seed, counter)` pair as explicit kernel args.

#=============================================================================
 Philox2x32 constants (matching CUDA.jl / Random123.jl)
=============================================================================#

const PHILOX_M = 0xd256d193 % UInt32
const PHILOX_W = 0x9E3779B9 % UInt32   # Weyl step (bumpkey)

#=============================================================================
 Philox round (tile-polymorphic — works for 0D and ND shapes alike)
=============================================================================#

function philox2x_round(c1::Tile{UInt32, S}, c2::Tile{UInt32, S},
                        k::Tile{UInt32, S}) where {S}
    m = fill(PHILOX_M, size(c1))
    hi = Intrinsics.mulhii(c1, m, Signedness.Unsigned)
    lo = Intrinsics.muli(c1, m)
    (hi .⊻ k .⊻ c2, lo)
end

philox2x_bumpkey(k::Tile{UInt32, S}) where {S} =
    k .+ fill(PHILOX_W, size(k))

# Unrolled R-round Philox, for R ∈ 1..7.
function philox2x_rounds(::Val{R}, c1::Tile{UInt32, S}, c2::Tile{UInt32, S},
                         k::Tile{UInt32, S}) where {R, S}
    if R > 0;                             c1, c2 = philox2x_round(c1, c2, k); end
    if R > 1; k = philox2x_bumpkey(k);   c1, c2 = philox2x_round(c1, c2, k); end
    if R > 2; k = philox2x_bumpkey(k);   c1, c2 = philox2x_round(c1, c2, k); end
    if R > 3; k = philox2x_bumpkey(k);   c1, c2 = philox2x_round(c1, c2, k); end
    if R > 4; k = philox2x_bumpkey(k);   c1, c2 = philox2x_round(c1, c2, k); end
    if R > 5; k = philox2x_bumpkey(k);   c1, c2 = philox2x_round(c1, c2, k); end
    if R > 6; k = philox2x_bumpkey(k);   c1, c2 = philox2x_round(c1, c2, k); end
    return (c1, c2)
end

#=============================================================================
 Uniform-float conversion — matches CUDA.jl's `u01`
=============================================================================#

# Map u ∈ [0, 2^32) to a Float32 via `(u * 2^-32) + 2^-33`. Output range is
# `[2^-33, 1.0f0]`: the +2^-33 bias keeps the lower bound strictly positive
# (log-safe for Box-Muller in a future randn), while Float32 rounding makes
# the upper bound saturate to exactly 1.0 for u ≥ 0xFFFFFF80 (Float32 ulp
# near 1 is 2^-23, which absorbs the bias). Matches CUDA.jl's `u01`. Note
# this differs from Julia stdlib's `[0, 1)` convention — log(0) avoidance
# was prioritized over interval shape.
function u01(::Type{Float32}, u::Tile{UInt32, S}) where {S}
    shape = size(u)
    uf    = convert(Tile{Float32}, u)
    scale = fill(Float32(2)^-32, shape)
    bias  = fill(Float32(2)^-33, shape)
    uf .* scale .+ bias
end

# Float16 / BFloat16 from a UInt16 sample — bit-pattern construction. Top
# (mantissa-width) bits become the mantissa, exponent is set for [1, 2),
# subtract 1. The low mantissa bit is forced set so the result lands in
# (0, 1) — preserves the Float32 path's "no zero" property and is log-safe.
function u01(::Type{Float16}, u::Tile{UInt16, S}) where {S}
    shape = size(u)
    # Float16: 10 mantissa bits, exponent for [1, 2) is 0x3C00.
    m    = Intrinsics.shri(u, fill(UInt16(6), shape), Signedness.Unsigned)
    bits = Intrinsics.ori(m, fill(UInt16(0x3C01), shape))
    Intrinsics.bitcast(bits, Float16) - fill(Float16(1), shape)
end
function u01(::Type{BFloat16}, u::Tile{UInt16, S}) where {S}
    shape = size(u)
    # BFloat16: 7 mantissa bits, exponent for [1, 2) is 0x3F80.
    m    = Intrinsics.shri(u, fill(UInt16(9), shape), Signedness.Unsigned)
    bits = Intrinsics.ori(m, fill(UInt16(0x3F81), shape))
    Intrinsics.bitcast(bits, BFloat16) - fill(BFloat16(1), shape)
end

# Float64 from a UInt64 sample — bit-pattern construction matching
# GPUArrays / CUDA.jl. Avoids the costly `Float64(::UInt64)` conversion on
# consumer GPUs. The low mantissa bit is forced set so the result lands in
# (0, 1), preserving the Float32 path's (0, 1] convention.
function u01(::Type{Float64}, u::Tile{UInt64, S}) where {S}
    shape = size(u)
    shifted = Intrinsics.shri(u, fill(UInt64(12), shape), Signedness.Unsigned)
    bits    = Intrinsics.ori(shifted, fill(0x3ff0000000000001, shape))
    Intrinsics.bitcast(bits, Float64) - fill(1.0, shape)
end

#=============================================================================
 Per-block key derivation
=============================================================================#

# Linearize the 3D block id, hash with the Weyl constant, and XOR in the
# stream's threaded RNG seed (defaults to 0 when no seed has been set).
# Different seeds produce mathematically uncorrelated Philox streams across
# blocks, while different blocks under the same seed remain disjoint.
#
# A Weyl-hashed stream ID is also mixed in so two `DeviceRNG()`s in the same
# kernel produce uncorrelated streams even without explicit seeding. Stream 0
# (the shared `default_rng`) contributes nothing, preserving the default
# stream's existing key.
#
# `lower_rng_state!` seeds *every* stream's initial seed slot from
# `KernelState.seed`, so consecutive launches of a kernel using only
# `DeviceRNG()` (no `Random.seed!`) still diverge via the per-launch host
# seed. Cross-stream divergence within a single launch comes from the
# stream-ID mix above; cross-launch divergence comes from the host seed.

function rng_key(stream)
    b1 = Intrinsics.get_tile_block_id(Int32(0))
    b2 = Intrinsics.get_tile_block_id(Int32(1))
    b3 = Intrinsics.get_tile_block_id(Int32(2))
    nb1 = Intrinsics.get_num_tile_blocks(Int32(0))
    nb2 = Intrinsics.get_num_tile_blocks(Int32(1))
    b = b1 + b2 * nb1 + b3 * nb1 * nb2
    # Signless-int bitcast: signed/unsigned representations share bytes.
    # `stream` is an Int literal after `assign_rng_streams!`, so the stream-mix
    # term folds to a compile-time constant. Narrow to Int32 first (Tile IR has
    # no cross-sign narrowing overlay) and bitcast into UInt32 for the multiply.
    stream_u32 = reinterpret(UInt32, Int32(stream))
    scalar_key = reinterpret(UInt32, b) * PHILOX_W ⊻ stream_u32 * PHILOX_M
    # Wrap the scalar base as a 0D tile so the XOR with `rng_seed(stream)`
    # (which returns `Tile{UInt32, Tuple{}}`) dispatches to `Intrinsics.xori`'s
    # tile-tile method. `⊻` on (scalar, tile) has no Base method.
    Intrinsics.xori(Tile(scalar_key), Intrinsics.rng_seed(stream))
end

#=============================================================================
 Tile-rand implementation
=============================================================================#

# Per-element counter: for N-D tiles, base + flat iota so each element sees a
# unique counter. For 0D (scalar) tiles, `iota` would be ill-defined (it
# requires a 1-d result) — just use the base counter directly.

counter_tile(c_base::Tile{UInt32, Tuple{}}, ::Tuple{}) = c_base

function counter_tile(c_base::Tile{UInt32, Tuple{}}, dims::NTuple{N, Int}) where {N}
    # Build a 1-d iota of length prod(dims), then reshape to dims. Works
    # uniformly for 1-d and higher-rank shapes since the flat linear index
    # is all we care about.
    flat = Intrinsics.iota((prod(dims),), UInt32)
    idx  = reshape(flat, dims)
    fill(c_base, dims) .+ idx
end

# Produce a tile of UInt32 randoms. `stream` is an `Int` ID returned by
# `rng_stream`/`rng_default`; `rng_state_pass!` keys per-stream state on it.
# Each Philox call produces two UInt32s (`c1`, `c2`). For `n ≥ 2` we run
# `n/2` Philox calls and `cat` the two lanes — halving Philox compute. The
# cyclic tile distribution makes `cat` a register-level no-op (each thread's
# c1 and c2 shares fall into its own output positions; verified via SASS:
# zero added warp shuffles, zero shared-mem traffic). For `n == 1` (0D
# scalar) we keep the single-call form since there's no second lane to
# place. The host counter still advances by `n` (logical element count)
# rather than `n/2` (actual Philox calls), wasting half the counter range
# (2^31 still usable) in exchange for keeping the host advance scheme
# T-agnostic.

function rand_uint32_tile(stream, dims::NTuple{N, Int}) where {N}
    n = prod(dims)
    k_scalar = rng_key(stream)
    c_base   = Intrinsics.rng_counter(stream)
    Intrinsics.rng_advance(stream, n)
    if n == 1
        # 0D / scalar: one Philox call, take c1 only.
        k_tile   = fill(k_scalar, dims)
        counters = counter_tile(c_base, dims)
        ctr2     = fill(UInt32(0), dims)
        c1, _c2  = philox2x_rounds(Val(7), counters, ctr2, k_tile)
        return c1
    else
        # n ≥ 2: do n/2 Philox calls and use both lanes.
        half = n >> 1
        k_tile   = fill(k_scalar, half)
        counters = counter_tile(c_base, (half,))
        ctr2     = fill(UInt32(0), half)
        c1, c2   = philox2x_rounds(Val(7), counters, ctr2, k_tile)
        return reshape(cat((c1, c2), 1), dims)
    end
end

# UInt64 variant: same Philox cost as the UInt32 path (1 round per element)
# by packing both Philox lanes into a UInt64 — `low | (high << 32)`.
function rand_uint64_tile(stream, dims::NTuple{N, Int}) where {N}
    n = prod(dims)
    k_scalar = rng_key(stream)
    c_base   = Intrinsics.rng_counter(stream)
    Intrinsics.rng_advance(stream, n)

    k_tile   = fill(k_scalar, dims)
    counters = counter_tile(c_base, dims)
    ctr2     = fill(UInt32(0), dims)

    c1, c2 = philox2x_rounds(Val(7), counters, ctr2, k_tile)
    lo = convert(Tile{UInt64}, c1)
    hi = convert(Tile{UInt64}, c2)
    Intrinsics.ori(lo, Intrinsics.shli(hi, fill(UInt64(32), dims)))
end

# UInt16 / UInt8 derive from rand_uint32_tile by splitting each UInt32 into
# 2 / 4 chunks and concatenating. The cyclic tile distribution makes both
# the cat and the per-chunk shri/trunci stay in-thread, so packing more
# outputs per Philox call doesn't add warp shuffles or shared-mem traffic.
# For `n < fan_out` (0D scalar UInt16; 0D / 2-elt UInt8) the natural split
# produces more chunks than needed and we `extract` the prefix.

function rand_uint16_tile(stream, dims::NTuple{N, Int}) where {N}
    n     = prod(dims)
    raw_n = max(1, n >> 1)
    raw   = rand_uint32_tile(stream, (raw_n,))
    lo    = Intrinsics.trunci(raw, UInt16)
    hi    = Intrinsics.trunci(Intrinsics.shri(raw, fill(UInt32(16), raw_n),
                                              Signedness.Unsigned), UInt16)
    full  = cat((lo, hi), 1)
    n < 2 * raw_n ? reshape(extract(full, (1,), (n,)), dims) : reshape(full, dims)
end

function rand_uint8_tile(stream, dims::NTuple{N, Int}) where {N}
    n     = prod(dims)
    raw_n = max(1, n >> 2)
    raw   = rand_uint32_tile(stream, (raw_n,))
    shr(amt) = Intrinsics.shri(raw, fill(UInt32(amt), raw_n), Signedness.Unsigned)
    b0 = Intrinsics.trunci(raw,    UInt8)
    b1 = Intrinsics.trunci(shr(8),  UInt8)
    b2 = Intrinsics.trunci(shr(16), UInt8)
    b3 = Intrinsics.trunci(shr(24), UInt8)
    full = cat((cat((b0, b1), 1), cat((b2, b3), 1)), 1)
    n < 4 * raw_n ? reshape(extract(full, (1,), (n,)), dims) : reshape(full, dims)
end

# Map each output type to its underlying unsigned RNG core (matched in
# width). This is the only place that decides how wide the Philox output
# tile is; all bit-packing logic stays inside rand_uint{8,16,32,64}_tile.
underlying_rand(::Type{T}, stream, dims) where {T<:Union{Int8,  UInt8}}                       = rand_uint8_tile(stream, dims)
underlying_rand(::Type{T}, stream, dims) where {T<:Union{Int16, UInt16, Float16, BFloat16}}   = rand_uint16_tile(stream, dims)
underlying_rand(::Type{T}, stream, dims) where {T<:Union{Int32, UInt32, Float32}}             = rand_uint32_tile(stream, dims)
underlying_rand(::Type{T}, stream, dims) where {T<:Union{Int64, UInt64, Float64}}             = rand_uint64_tile(stream, dims)

# Convert the Philox sample to the requested type, matched in width to the
# underlying unsigned RNG core. Tile IR integers are signless, so the
# unsigned→signed conversion is just a `bitcast` relabel.

const RandTypes = Union{Int8, UInt8, Int16, UInt16, Int32, UInt32, Int64, UInt64,
                        Float16, BFloat16, Float32, Float64}

# Identity for unsigned RNG output types.
finalize_tile(::Type{T}, u::Tile{T}) where {T<:Union{UInt8, UInt16, UInt32, UInt64}} = u
# Same-width signed: signless bitcast.
finalize_tile(::Type{Int8},  u::Tile{UInt8})  = Intrinsics.bitcast(u, Int8)
finalize_tile(::Type{Int16}, u::Tile{UInt16}) = Intrinsics.bitcast(u, Int16)
finalize_tile(::Type{Int32}, u::Tile{UInt32}) = Intrinsics.bitcast(u, Int32)
finalize_tile(::Type{Int64}, u::Tile{UInt64}) = Intrinsics.bitcast(u, Int64)
# Floats: u01 conversion (each float type matches its raw unsigned in width).
finalize_tile(::Type{Float32},  u::Tile{UInt32}) = u01(Float32, u)
finalize_tile(::Type{T}, u::Tile{UInt16}) where {T<:Union{Float16, BFloat16}} = u01(T, u)
finalize_tile(::Type{Float64},  u::Tile{UInt64}) = u01(Float64, u)


rand_tile(stream, ::Type{T}, dims::NTuple{N, Int}) where {T<:RandTypes, N} =
    finalize_tile(T, underlying_rand(T, stream, dims))

# Scalar form: produce a 0D tile, extract via to_scalar.

rand_scalar(stream, ::Type{T}) where {T} =
    Intrinsics.to_scalar(rand_tile(stream, T, ()))

#=============================================================================
 Handle + DeviceRNG types; Random API overlays
=============================================================================#

"""
    cuTile.DeviceRNG()
    Random.default_rng()   # inside a kernel

Kernel-scope RNG. Two ways to obtain one:

- `DeviceRNG()` — opens a fresh independent stream. Distinct construction
  sites yield streams with their own `(counter, seed)` slots. Use this
  when you want multiple uncorrelated RNGs in a kernel.
- `Random.default_rng()` — returns the shared default stream. All
  `default_rng()` call sites in a kernel share one `(counter, seed)`
  pair, matching the CPU `Random` convention that bare `rand()` uses an
  implicit global RNG.

Seeding one `DeviceRNG` does not affect another.

    a = cuTile.DeviceRNG()
    b = cuTile.DeviceRNG()
    Random.seed!(a, 1); Random.seed!(b, 2)
    x = rand(a, Float32, (16,))
    y = rand(b, Float32, (16,))   # uncorrelated with x

The wrapped `stream::Int` is an opaque compile-time ID: `0` for the
default stream, `1..N` assigned by `rng_assign_ids_pass!` per call site.
Codegen is unaware of RNG semantics — the state intrinsics lower to
plain arithmetic on integer-keyed state slots, nothing else.
"""
struct DeviceRNG <: Random.AbstractRNG
    stream::Int
end

# Outer ctor — opens a fresh stream via the intrinsic (rewritten to a
# literal Int at the IR level by `rng_assign_ids_pass!`).
DeviceRNG() = DeviceRNG(Intrinsics.rng_stream())

# Overlays for methods that shadow stdlib's `Random.default_rng` / `Random.rand`.
# Non-foldable, state mutation is a real effect.

Base.Experimental.@consistent_overlay cuTileMethodTable Random.default_rng() =
    DeviceRNG(Intrinsics.rng_default())

# Bare rand / rand(T) / rand(T, dims) — route through the default stream.
Base.Experimental.@consistent_overlay cuTileMethodTable Random.rand(::Type{T}) where {T<:RandTypes} =
    rand_scalar(Intrinsics.rng_default(), T)
Base.Experimental.@consistent_overlay cuTileMethodTable Random.rand() =
    rand_scalar(Intrinsics.rng_default(), Float32)
Base.Experimental.@consistent_overlay cuTileMethodTable Random.rand(::Type{T}, dims::NTuple{N, Int}) where {T<:RandTypes, N} =
    rand_tile(Intrinsics.rng_default(), T, dims)
Base.Experimental.@consistent_overlay cuTileMethodTable Random.rand(dims::NTuple{N, Int}) where {N} =
    rand_tile(Intrinsics.rng_default(), Float32, dims)

# Untyped variadic — default to `Float32` instead of stdlib's `Float64`.
Base.Experimental.@consistent_overlay cuTileMethodTable Random.rand(dims::Integer...) =
    Random.rand(Dims(dims))

function Random.seed!(rng::DeviceRNG, seed::Integer)
    Intrinsics.rng_set_seed(rng.stream, UInt32(seed))
    return rng
end

# Explicit-stream rand(rng, ...) variants — unwrap and pass the stream ID.
Random.rand(rng::DeviceRNG, ::Type{T}) where {T<:RandTypes} =
    rand_scalar(rng.stream, T)
Random.rand(rng::DeviceRNG) = rand_scalar(rng.stream, Float32)
Random.rand(rng::DeviceRNG, ::Type{T}, dims::NTuple{N, Int}) where {T<:RandTypes, N} =
    rand_tile(rng.stream, T, dims)
Random.rand(rng::DeviceRNG, dims::NTuple{N, Int}) where {N} =
    rand_tile(rng.stream, Float32, dims)
Random.rand(rng::DeviceRNG, dims::Integer...) =
    Random.rand(rng, Dims(dims))

#=============================================================================
 Host-level RNG wrapper
=============================================================================#

public RNG, rand

"""
    cuTile.RNG([seed::Integer])
    cuTile.RNG(seed::UInt32, counter::UInt32)

Host-side handle for generating random numbers on the device via the cuTile
Philox2x32-7 implementation. Mirrors `Random.AbstractRNG` — use with
`Random.rand!` / `Random.rand` or the `cuTile.rand` / `cuTile.rand!` helpers.

At launch time the fill kernel calls `Random.seed!(Random.default_rng(), seed)`
and advances the threaded counter by `counter`, so two `RNG`s with
different seeds produce uncorrelated Philox streams. The counter is
auto-advanced by `length(A)` after each `rand!` call, giving disjoint
streams across consecutive calls on the same RNG.
"""
mutable struct RNG <: Random.AbstractRNG
    seed::UInt32
    counter::UInt32

    RNG(seed::UInt32, counter::UInt32) = new(seed, counter)
end

RNG(seed::Integer) = RNG(seed % UInt32, UInt32(0))
RNG() = RNG(Base.rand(Random.RandomDevice(), UInt32))

Base.copy(rng::RNG) = RNG(rng.seed, rng.counter)
Base.hash(rng::RNG, h::UInt) = hash(rng.seed, hash(rng.counter, h))
Base.:(==)(a::RNG, b::RNG) = (a.seed == b.seed) && (a.counter == b.counter)

function Random.seed!(rng::RNG, seed::Integer)
    rng.seed = seed % UInt32
    rng.counter = UInt32(0)
    rng
end
Random.seed!(rng::RNG) = Random.seed!(rng, Base.rand(Random.RandomDevice(), UInt32))

# Bump the counter by `n` and propagate the carry into the seed so wrapping
# the UInt32 counter doesn't reuse the previous Philox stream. Mirrors
# GPUArrays.advance_counter! (host/random.jl).
function advance_counter!(rng::RNG, n::UInt32)
    new_counter = rng.counter + n
    new_counter < rng.counter && (rng.seed += UInt32(1))
    rng.counter = new_counter
    return rng
end

# Fixed tile size used by the host-level fill kernels. Chosen from the perf
# sweep (see bench results) — 512 is the throughput sweet spot at mid-size
# arrays and converges with the others at ≥64 MiB.
const RAND_FILL_TILE = 512

function rand_fill_kernel(out::TileArray{T, 1}, seed::UInt32, counter::UInt32) where {T}
    # `RNG` plumbs its `(seed, counter)` through as explicit kernel args
    # rather than via `KernelState`: seed re-keys the default stream and
    # `rng_advance` shifts the per-element counter window so consecutive
    # `rand!` calls on the same `RNG` produce disjoint output.
    Random.seed!(Random.default_rng(), seed)
    Intrinsics.rng_advance(Intrinsics.rng_default(), counter)
    pid = bid(1)
    t = Random.rand(T, (RAND_FILL_TILE,))
    store(out, pid, t)
    return
end

# Global RNG handle lazily created on first use. The methods that actually
# dispatch on `CuArray` live in `ext/CUDAExt.jl` because `CuArray` is a weak
# dependency.

const global_rng = Ref{Union{Nothing, RNG}}(nothing)

function get_global_rng()
    global_rng[] === nothing && (global_rng[] = RNG())
    global_rng[]
end

"""
    cuTile.seed!([seed])

Re-seed the global `cuTile.rand` / `cuTile.rand!` RNG.
"""
function seed!(seed::Integer=Base.rand(Random.RandomDevice(), UInt32))
    Random.seed!(get_global_rng(), seed)
    return
end

# `cuTile.rand` / `cuTile.rand!` stubs — the real implementations live in the
# CUDAExt extension. Calling without CUDA.jl loaded raises a helpful error.
function rand end
function rand! end
