# Device-side random number generation
#
# Uniform Philox2x32-7 on top of existing tile-aware arithmetic intrinsics
# and four placeholder state intrinsics (`rng_counter`, `rng_advance`,
# `rng_seed`, `rng_set_seed`) from `compiler/intrinsics/random.jl`. Those
# intrinsics are rewritten to concrete SSA values by `rng_state_pass!`,
# which threads two UInt32 state slots — counter and seed — through
# structured control flow.
#
# In-kernel API:
#   rand()                              -> Float32 scalar
#   rand(::Type{T})                     -> T scalar, T ∈ {Float32, UInt32}
#   rand(::Type{T}, dims)               -> Tile{T, dims}
#   rand(dims)                          -> Tile{Float32, dims}
#   Random.seed!(Random.default_rng(), seed)   # re-seed the threaded state
#
# The key is derived per block as `bid_linear * PHILOX_W ⊻ rng_seed()`.
# With the default (unseeded) state, `rng_seed() = 0` and the key reduces
# to the bid-derived value — two launches under the same grid produce
# identical output. Seeding via `Random.seed!` changes the threaded seed,
# producing a mathematically independent Philox stream.
#
# v1 limitations:
#  - Output types: `Float32`, `UInt32`. No Float16/Float64/Bool.
#  - No `randn`, `randexp`.
#
# Loops: the counter is threaded as a loop-carried SSA value by
# `rng_state_pass!`, so each iteration's `rand` call sees a distinct
# counter. The seed is carried similarly if set inside a loop body.

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

# Broadcast a scalar UInt32 constant to match a tile's shape.
bc_const(val::UInt32, shape::NTuple{N, Int}) where {N} =
    broadcast_to(Tile(val), shape)

function philox2x_round(c1::Tile{UInt32, S}, c2::Tile{UInt32, S},
                        k::Tile{UInt32, S}) where {S}
    shape = size(c1)
    m = bc_const(PHILOX_M, shape)
    hi = Intrinsics.mulhii(c1, m, Signedness.Unsigned)
    lo = Intrinsics.muli(c1, m)
    (hi .⊻ k .⊻ c2, lo)
end

philox2x_bumpkey(k::Tile{UInt32, S}) where {S} =
    k .+ bc_const(PHILOX_W, size(k))

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
    uf = convert(Tile{Float32}, u)
    scale = bc_const_f32(Float32(2)^-32, shape)
    bias  = bc_const_f32(Float32(2)^-33, shape)
    uf .* scale .+ bias
end


bc_const_f32(val::Float32, shape::NTuple{N, Int}) where {N} =
    broadcast_to(Tile(val), shape)

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
    broadcast_to(c_base, dims) .+ idx
end

# Produce a tile of UInt32 randoms. `stream` is an `Int` ID returned by
# `rng_stream`/`rng_default`; `rng_state_pass!` keys per-stream state on it.

function rand_uint32_tile(stream, dims::NTuple{N, Int}) where {N}
    n = prod(dims)
    k_scalar = rng_key(stream)
    c_base   = Intrinsics.rng_counter(stream)
    Intrinsics.rng_advance(stream, n)

    k_tile   = broadcast_to(Tile(k_scalar), dims)
    counters = counter_tile(c_base, dims)
    ctr2     = bc_const(UInt32(0), dims)

    c1, _c2 = philox2x_rounds(Val(7), counters, ctr2, k_tile)
    c1
end

# Convert the UInt32 output to the requested type.

finalize_tile(::Type{UInt32}, u::Tile{UInt32}) = u

finalize_tile(::Type{Float32}, u::Tile{UInt32}) = u01(Float32, u)


rand_tile(stream, ::Type{T}, dims::NTuple{N, Int}) where {T, N} =
    finalize_tile(T, rand_uint32_tile(stream, dims))

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

# Overlays — non-foldable, state mutation is a real effect.

Base.Experimental.@consistent_overlay cuTileMethodTable Random.default_rng() =
    DeviceRNG(Intrinsics.rng_default())

Base.Experimental.@consistent_overlay cuTileMethodTable function Random.seed!(rng::DeviceRNG, seed::Integer)
    Intrinsics.rng_set_seed(rng.stream, UInt32(seed))
    return rng
end

# Bare rand / rand(T) / rand(T, dims) — route through the default stream.
Base.Experimental.@consistent_overlay cuTileMethodTable Random.rand(::Type{Float32}) =
    rand_scalar(Intrinsics.rng_default(), Float32)
Base.Experimental.@consistent_overlay cuTileMethodTable Random.rand(::Type{UInt32}) =
    rand_scalar(Intrinsics.rng_default(), UInt32)
Base.Experimental.@consistent_overlay cuTileMethodTable Random.rand() =
    rand_scalar(Intrinsics.rng_default(), Float32)
Base.Experimental.@consistent_overlay cuTileMethodTable Random.rand(::Type{T}, dims::NTuple{N, Int}) where {T<:Union{Float32, UInt32}, N} =
    rand_tile(Intrinsics.rng_default(), T, dims)
Base.Experimental.@consistent_overlay cuTileMethodTable Random.rand(dims::NTuple{N, Int}) where {N} =
    rand_tile(Intrinsics.rng_default(), Float32, dims)

# Explicit-stream rand(rng, ...) variants — unwrap and pass the stream ID.
Base.Experimental.@consistent_overlay cuTileMethodTable Random.rand(rng::DeviceRNG, ::Type{T}) where {T<:Union{Float32, UInt32}} =
    rand_scalar(rng.stream, T)
Base.Experimental.@consistent_overlay cuTileMethodTable Random.rand(rng::DeviceRNG) =
    rand_scalar(rng.stream, Float32)
Base.Experimental.@consistent_overlay cuTileMethodTable Random.rand(rng::DeviceRNG, ::Type{T}, dims::NTuple{N, Int}) where {T<:Union{Float32, UInt32}, N} =
    rand_tile(rng.stream, T, dims)
Base.Experimental.@consistent_overlay cuTileMethodTable Random.rand(rng::DeviceRNG, dims::NTuple{N, Int}) where {N} =
    rand_tile(rng.stream, Float32, dims)

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

for T in (Float32, UInt32)
    kname = Symbol("rand_fill_", nameof(T))
    @eval function $kname(out::TileArray{$T, 1}, seed::UInt32, counter::UInt32)
        # `RNG` plumbs its `(seed, counter)` through as explicit kernel args
        # rather than via `KernelState`: seed re-keys the default stream and
        # `rng_advance` shifts the per-element counter window so consecutive
        # `rand!` calls on the same `RNG` produce disjoint output.
        Random.seed!(Random.default_rng(), seed)
        Intrinsics.rng_advance(Intrinsics.rng_default(), counter)
        pid = bid(1)
        t = Random.rand($T, (RAND_FILL_TILE,))
        store(out, pid, t)
        return
    end
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
