"""
    DiskCache

Content-addressable disk cache for compiled Tile IR → CUBIN artifacts.

cuTile's in-memory cache (on `CodeInstance` via `CompilerCaching`) only
covers a single session. This submodule adds an LMDB-backed cache that
persists across sessions, so the second run of a kernel skips `tileiras`
entirely.

# Layout

- A single LMDB env at `\$(scratchspace)/disk_cache/`. The directory
  contains `data.mdb` + `lock.mdb`; wiping the cache means `rm -rf` of
  that directory or `Scratch.delete_scratch!`.
- Set `JULIA_CUTILE_CACHE_DIR` to override the cache directory. Values
  `0`, `off`, `none`, and the empty string disable the disk cache. Set
  `JULIA_CUTILE_CACHE_SIZE` to override the default 1 GiB LMDB map size.
- Keys: `sha256(SCHEMA_VERSION || toolkit_version || sm_arch || opt_level || bytecode)`.
  Any input change produces a fresh key, so old-toolkit entries never
  match on lookup. Bump [`SCHEMA_VERSION`](@ref) to invalidate every
  existing entry on the next access (e.g. after a value-framing change).
- Values: 8 bytes little-endian `atime_ns` followed by the CUBIN bytes.
  The atime drives the LRU eviction policy.

# Eviction

LMDB provides no built-in cache replacement; `put!` returns
`MDB_MAP_FULL` when the map is exhausted. We prune *before* hitting that
point. Deletes are also copy-on-write and need free pages, so draining
a fully-saturated map is unreliable. On every `put!`:

1. Compute env utilization via `LMDB.info` + the cached page size.
2. If the current usage plus incoming entry would cross
   [`HIGH_WATER`](@ref) (90%), call [`evict_lru!`](@ref) targeting
   [`LOW_WATER`](@ref) (75%) with enough room for the incoming entry.
3. Then write the new entry.

`evict_lru!` cursor-walks all entries collecting `(key, atime, size)`,
sorts by atime ascending, and deletes the oldest in batches of ~100 per
write txn. The small batches matter: LMDB's COW semantics need a few
generations of headroom before freed pages become available again.

Atime refresh on a hit is throttled to once per session per key. Without
that, a hot kernel would write back on every cache lookup.
"""
module DiskCache

import LMDB
using SHA: sha256
using Scratch: @get_scratch!

# ===========================================================================
# Value framing — `atime_ns ‖ payload`
# ===========================================================================

const ATIME_PREFIX = 8  # bytes for a UInt64 little-endian timestamp

# `tryget(txn, dbi, key, AtimedBlob)` returns the payload (atime stripped),
# or `nothing` for both NOTFOUND and malformed entries. Both surface as a
# cache miss. Single copy out of the mmap via `read(io, Vector{UInt8})`.
struct AtimedBlob end

function Base.read(io::IO, ::Type{AtimedBlob})
    bytesavailable(io) < ATIME_PREFIX && return nothing
    skip(io, ATIME_PREFIX)
    return read(io, Vector{UInt8})
end

# `walk(cur, Vector{UInt8}, AtimeMeta)` yields per-entry `(atime, full_size)`
# pairs without copying out the payload. Eviction uses it to sort by atime
# and approximate freed-byte counts.
struct AtimeMeta end

function Base.read(io::IO, ::Type{AtimeMeta})
    sz = bytesavailable(io)
    atime = sz >= ATIME_PREFIX ? ltoh(read(io, UInt64)) : UInt64(0)
    return (atime = atime, size = sz)
end

# ===========================================================================
# Cache handle
# ===========================================================================

"""
    Cache

Opaque handle to an opened LMDB-backed disk cache. Created via [`open`](@ref),
released via [`close`](@ref). Safe to share across threads (LMDB serializes
writers internally; readers use `MDB_NOTLS` so they're decoupled from
the OS thread).
"""
mutable struct Cache
    env::LMDB.Environment
    dbi::LMDB.Database
    psize::Int                       # LMDB page size in bytes (cached at open)
    path::String
    refreshed::Set{Vector{UInt8}}    # keys whose atime we already bumped this session
    state_lock::ReentrantLock        # guards `refreshed` + serializes evictions

    function Cache(env, dbi, psize::Integer, path::AbstractString)
        new(env, dbi, Int(psize), String(path),
            Set{Vector{UInt8}}(), ReentrantLock())
    end
end

isopen(cache::Cache) = LMDB.isopen(cache.env)

const DEFAULT_MAPSIZE = Csize_t(1) << 30

"""
    close(cache::Cache)

Release the underlying LMDB environment. Idempotent.
"""
function close(cache::Cache)
    LMDB.isopen(cache.env) || return
    LMDB.close(cache.env, cache.dbi)
    LMDB.close(cache.env)
    return
end

"""
    open(path; mapsize = 1<<30, maxreaders = 510) -> Cache

Open or create the disk cache at `path`. The directory is created if
missing. `mapsize` is the maximum on-disk size in bytes (LMDB grows the
map sparsely up to this limit). `maxreaders` caps concurrent reader
transactions.
"""
function open(path::AbstractString; mapsize::Integer = DEFAULT_MAPSIZE,
              maxreaders::Integer = 510)
    mkpath(path)

    # MDB_NOTLS: read txns aren't tied to the OS thread that opened them
    # (Julia tasks may migrate). MDB_NORDAHEAD: lookups are random-access,
    # so OS read-ahead is wasted I/O.
    env = LMDB.Environment(path;
                           mapsize    = mapsize,
                           maxreaders = maxreaders,
                           flags      = LMDB.MDB_NOTLS | LMDB.MDB_NORDAHEAD,
                           mode       = 0o644)
    dbi, psize = LMDB.Transaction(env) do txn
        d = LMDB.Database(txn)
        (d, LMDB.stat(txn, d).psize)
    end
    return Cache(env, dbi, psize, path)
end

"""
    env_info(cache::Cache) -> NamedTuple

Snapshot of the env's *live* page count and configured map size. The
eviction policy uses it to decide whether to prune.

`MDB_envinfo`'s `me_last_pgno` is a monotonic high-water mark; it
doesn't drop when entries get deleted. Eviction relies on `MDB_stat`'s
live-page accounting (branch + leaf + overflow), which *does* shrink
after a delete commits. One short read txn, around 10 µs.
"""
function env_info(cache::Cache)
    mapsize = LMDB.info(cache.env).mapsize
    s = LMDB.Transaction(cache.env; flags = LMDB.MDB_RDONLY) do txn
        LMDB.stat(txn, cache.dbi)
    end
    live_pages = s.branch_pages + s.leaf_pages + s.overflow_pages
    used_bytes = live_pages * cache.psize
    return (; mapsize, used_bytes, entries = s.entries)
end

@inline utilization(cache::Cache) =
    let i = env_info(cache); i.used_bytes / i.mapsize end

# ===========================================================================
# get / put!
# ===========================================================================

# Eviction trigger / target as fractions of mapsize. Margins live here
# for easy tuning; the 90% high-water mark is what Howard Chu (LMDB
# author) recommends.
const HIGH_WATER = 0.90
const LOW_WATER  = 0.75

# Overflow-page allocation can still hit MDB_MAP_FULL after ordinary LRU
# pruning if free pages are too fragmented. On that rare path, prune much
# harder and retry once.
const MAP_FULL_RETRY_WATER = 0.25

# Keep the eviction batch small so freed COW pages become available
# within a couple of write txns (per LMDB authors' advice).
const EVICT_BATCH = 100

"""
    get(cache, key) -> Union{Vector{UInt8}, Nothing}

Look up `key` in the cache. Returns a freshly-allocated copy of the
value on hit, or `nothing` on miss. The copy is necessary because LMDB
hands back a pointer into the memory-mapped data file, which a future
writer is allowed to reuse.

On hit, the entry's atime is bumped (at most once per session per key,
to avoid write-amplification). This drives the LRU eviction order.
"""
function get(cache::Cache, key::AbstractVector{UInt8})
    LMDB.isopen(cache.env) || error("DiskCache.get on closed cache")

    blob = LMDB.Transaction(cache.env; flags = LMDB.MDB_RDONLY) do txn
        LMDB.get(txn, cache.dbi, key, AtimedBlob, nothing)
    end
    blob === nothing && return nothing

    # Throttled atime refresh. Done outside the read txn (LMDB doesn't
    # allow nesting). Errors here are non-fatal; we'd rather return the
    # blob than fail the launch.
    Base.@lock cache.state_lock begin
        if !(key in cache.refreshed)
            push!(cache.refreshed, copy(key))
            try
                put_framed!(cache, key, time_ns(), blob)
            catch err
                @debug "atime refresh failed" exception=(err, catch_backtrace())
            end
        end
    end

    return blob
end

"""
    put!(cache, key, value)

Insert `key => value` into the cache. Existing entries are not
overwritten: under content addressing, a key collision means the values
are identical (or, vanishingly, a `hash` collision). Either way, the
first writer wins.

If the env plus the incoming value would cross [`HIGH_WATER`](@ref),
[`evict_lru!`](@ref) is run first to drop down to [`LOW_WATER`](@ref)
with enough room for the value.
"""
function put!(cache::Cache, key::AbstractVector{UInt8},
              value::AbstractVector{UInt8})
    LMDB.isopen(cache.env) || error("DiskCache.put! on closed cache")

    # Double-checked: the cheap utilization probe gates the lock
    # acquisition in the common case, when we're well below high water.
    incoming = ATIME_PREFIX + length(value)
    if should_evict(cache, incoming)
        Base.@lock cache.state_lock begin
            if should_evict(cache, incoming)
                evict_lru!(cache, LOW_WATER; room_for=incoming)
            end
        end
    end

    try
        put_framed!(cache, key, time_ns(), value)
    catch err
        is_map_full(err) || rethrow()
        Base.@lock cache.state_lock begin
            evict_lru!(cache, MAP_FULL_RETRY_WATER; room_for=incoming)
            put_framed!(cache, key, time_ns(), value)
        end
    end

    # Mark this key as "atime is fresh" so a subsequent get in the same
    # session doesn't redundantly bump it.
    Base.@lock cache.state_lock push!(cache.refreshed, copy(key))
    return
end

# Write `atime ‖ payload` directly into LMDB-allocated mmap pages,
# instead of building an intermediate `Vector{UInt8}` (now that
# `MDB_RESERVE` is reachable from tier-2).
function put_framed!(cache::Cache, key::AbstractVector{UInt8}, atime::UInt64,
                     payload::AbstractVector{UInt8})
    sz = ATIME_PREFIX + length(payload)
    LMDB.Transaction(cache.env) do txn
        LMDB.put_reserved!(txn, cache.dbi, key, sz) do buf
            unsafe_store!(Ptr{UInt64}(pointer(buf)), htol(atime))
            copyto!(buf, ATIME_PREFIX + 1, payload, 1, length(payload))
        end
    end
    return
end

# ===========================================================================
# Eviction
# ===========================================================================

"""
    evict_lru!(cache, target_ratio = LOW_WATER) -> Int

Walk the cache, sort entries by atime ascending (oldest first), and
delete enough of them in batched write txns to bring used bytes below
`target_ratio * mapsize - room_for`. Returns the number of entries evicted.

Called automatically from [`put!`](@ref) when usage plus the incoming
entry would cross [`HIGH_WATER`](@ref); also exposed for manual invocation.
"""
function should_evict(cache::Cache, room_for::Integer = 0)
    info = env_info(cache)
    return info.used_bytes + room_for > HIGH_WATER * info.mapsize
end

is_map_full(err) = err isa LMDB.LMDBError && err.code == LMDB.MDB_MAP_FULL

function evict_lru!(cache::Cache, target_ratio::Real = LOW_WATER;
                    room_for::Integer = 0)
    info = env_info(cache)
    target_bytes = max(0, floor(Int, target_ratio * info.mapsize) - room_for)
    info.used_bytes <= target_bytes && return 0
    bytes_to_free = info.used_bytes - target_bytes

    entries = collect_entries(cache)
    sort!(entries, by = e -> e[2])  # atime ascending

    evicted = 0
    nominated = 0
    batch = Vector{Vector{UInt8}}()
    sizehint!(batch, EVICT_BATCH)

    for (key, _atime, raw_size) in entries
        push!(batch, key)
        # Approximate freed bytes by entry size; LMDB's actual freed-page
        # count is fuzzier (page granularity, COW), but this is the right
        # order of magnitude.
        nominated += cld(raw_size, cache.psize) * cache.psize
        if length(batch) >= EVICT_BATCH || nominated >= bytes_to_free
            evicted += delete_batch!(cache, batch)
            empty!(batch)
            env_info(cache).used_bytes <= target_bytes && break
            nominated = 0
        end
    end
    isempty(batch) || (evicted += delete_batch!(cache, batch))

    # Drop refreshed-set entries we just deleted (they no longer exist).
    # Cheaper than per-key removal: clear the set entirely. Worst case
    # we do a few redundant atime bumps after eviction.
    empty!(cache.refreshed)

    return evicted
end

# Collect (key_copy, atime, raw_value_size) for every entry in the cache.
# Typed walk: the key gets unpacked as `Vector{UInt8}` (default Julia-owned
# copy), the value as the AtimeMeta NamedTuple (atime + size, no payload
# copy). Pre-eviction-format / malformed entries surface with `atime = 0`
# so they evict first.
function collect_entries(cache::Cache)
    entries = Tuple{Vector{UInt8}, UInt64, Int}[]
    LMDB.Transaction(cache.env; flags = LMDB.MDB_RDONLY) do txn
        LMDB.Cursor(txn, cache.dbi) do cur
            LMDB.walk(cur, Vector{UInt8}, AtimeMeta) do key, meta
                push!(entries, (key, meta.atime, meta.size))
            end
        end
    end
    return entries
end

# Delete a batch of keys in a single write txn.
function delete_batch!(cache::Cache, keys::Vector{Vector{UInt8}})
    LMDB.Transaction(cache.env) do txn
        deleted = 0
        for key in keys
            LMDB.delete!(txn, cache.dbi, key) && (deleted += 1)
        end
        deleted
    end
end

# ===========================================================================
# Key derivation
# ===========================================================================

"""
    SCHEMA_VERSION

Cache schema version, mixed into every [`compute_key`](@ref). Bump this
when the on-disk value layout changes incompatibly (e.g. add/move bytes
in the framed value, change the hash algorithm, or alter what inputs
the key covers). After the bump, no existing entry matches on lookup,
and LRU eventually evicts them.
"""
const SCHEMA_VERSION = UInt32(2)

"""
    compute_key(bytecode, sm_arch, opt_level, toolkit_version) -> Vector{UInt8}

Derive a SHA-256 content-addressable cache key for a Tile IR compilation.
The key covers the bytecode plus every input that changes the resulting
CUBIN: target arch, opt level, and the `tileiras` toolkit version
(typically the full `--version` stdout; see `cuTile.toolkit_version()`).

Any change to those inputs produces a fresh key. Old-toolkit entries no
longer match on lookup, and eventually evict via LRU. The
[`SCHEMA_VERSION`](@ref) prefix lets us invalidate every entry at once
when the value layout changes.
"""
function compute_key(bytecode::Vector{UInt8}, sm_arch::VersionNumber,
                     opt_level::Integer, toolkit_version::AbstractString)
    data = UInt8[]
    append_uint32_be!(data, SCHEMA_VERSION)
    append_field!(data, codeunits(toolkit_version))
    append_field!(data, codeunits(string(sm_arch)))
    append_uint32_be!(data, opt_level)
    append_field!(data, bytecode)
    return sha256(data)
end

function append_field!(data::Vector{UInt8}, bytes)
    n = length(bytes)
    n <= typemax(UInt32) || throw(ArgumentError("cache key field too large: $n bytes"))
    append_uint32_be!(data, n)
    append!(data, bytes)
    return data
end

function append_uint32_be!(data::Vector{UInt8}, x::Integer)
    0 <= x <= typemax(UInt32) || throw(ArgumentError("cache key integer out of range: $x"))
    u = UInt32(x)
    push!(data,
          UInt8((u >> 24) & 0xff),
          UInt8((u >> 16) & 0xff),
          UInt8((u >> 8) & 0xff),
          UInt8(u & 0xff))
    return data
end

# ===========================================================================
# Process-wide singleton
# ===========================================================================

mutable struct GlobalCacheState
    initialized::Bool
    cache::Union{Cache, Nothing}
end

const global_cache_state = Base.Lockable(GlobalCacheState(false, nothing))

"""
    global_cache() -> Union{Cache, Nothing}

Return the lazily-initialized process-wide disk cache, or `nothing` if
initialization failed. The cache lives in cuTile's Scratch.jl-managed
scratchspace (so Pkg can clean it up when cuTile is uninstalled).
Failures are remembered; subsequent calls don't keep retrying.
"""
function global_cache()
    Base.@lock global_cache_state begin
        st = global_cache_state[]
        if !st.initialized
            st.cache = try_init()
            st.initialized = true
        end
        return st.cache
    end
end

function try_init()
    try
        root = configured_cache_dir()
        root === nothing && return nothing
        mapsize = configured_mapsize()
        try
            return open(root; mapsize)
        catch err
            @debug "cuTile disk cache failed to open; wiping" path=root exception=(err, catch_backtrace())
            wipe_lmdb_files(root)
            return open(root; mapsize)
        end
    catch err
        @debug "cuTile disk cache disabled" exception=(err, catch_backtrace())
        return nothing
    end
end

function configured_cache_dir()
    setting = Base.get(ENV, "JULIA_CUTILE_CACHE_DIR", nothing)
    if setting === nothing
        # @get_scratch! resolves to cuTile's package UUID via moduleroot,
        # so the path is $DEPOT/scratchspaces/<cuTile-UUID>/disk_cache/.
        return @get_scratch!("disk_cache")
    end
    cache_setting_disabled(setting) && return nothing
    return abspath(expanduser(setting))
end

function configured_mapsize()
    setting = Base.get(ENV, "JULIA_CUTILE_CACHE_SIZE", nothing)
    setting === nothing && return DEFAULT_MAPSIZE
    return parse_cache_size(setting)
end

function cache_setting_disabled(setting::AbstractString)
    lowercase(strip(setting)) in ("", "0", "off", "none")
end

function parse_cache_size(setting::AbstractString)
    value = tryparse(UInt64, strip(setting))
    if value === nothing || value == 0
        throw(ArgumentError("JULIA_CUTILE_CACHE_SIZE must be a positive integer byte count"))
    end
    return value
end

function wipe_lmdb_files(path::AbstractString)
    rm(joinpath(path, "data.mdb"); force=true)
    rm(joinpath(path, "lock.mdb"); force=true)
    return
end

end # module DiskCache
