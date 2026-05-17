"""
    DiskCache

Content-addressable disk cache for compiled Tile IR → CUBIN artifacts.

cuTile's in-memory cache (on `CodeInstance` via `CompilerCaching`) only
covers a single session. This submodule adds an LMDB-backed cache that
persists across sessions, so the second run of a kernel skips `tileiras`
entirely. Modeled on JuliaLang/julia#61527 (LLVM `objcache`) and
cuTile Python's SQLite cache (`cuda.tile._cache`).

The implementation talks to LMDB directly via `LMDB_jll`. The public
surface is intentionally narrow (`open`, `close`, `get`, `put!`,
`compute_key`, `evict_lru!`, plus the lazy `global_cache` accessor) so we
can swap the backend to `LMDB.jl` later without touching call sites.

# Layout
- A single LMDB env at `\$(scratchspace)/disk_cache/`. The directory
  contains `data.mdb` + `lock.mdb`; wiping the cache means `rm -rf` of
  that directory or `Scratch.delete_scratch!`.
- Keys: `hash(SCHEMA_VERSION ‖ toolkit_version ‖ sm_arch ‖ opt_level ‖ bytecode)` —
  any input change produces a fresh key, so old-toolkit entries simply
  never match on lookup. Bump [`SCHEMA_VERSION`](@ref) to invalidate
  every existing entry on the next access (e.g. after a value-framing
  change).
- Values: 8 bytes little-endian `atime_ns` followed by the CUBIN bytes.
  The atime drives the LRU eviction policy.

# Eviction
LMDB provides no built-in cache replacement; `mdb_put` returns
`MDB_MAP_FULL` when the map is exhausted. We prune *before* hitting that
point (deletes are also copy-on-write in LMDB and need free pages, so
draining a fully-saturated map is unreliable). On every `put!`:

1. Compute env utilization via `mdb_env_info` + the cached page size.
2. If above [`HIGH_WATER`](@ref) (90%), call [`evict_lru!`](@ref)
   targeting [`LOW_WATER`](@ref) (75%).
3. Then write the new entry.

`evict_lru!` cursor-walks all entries collecting `(key, atime, size)`,
sorts by atime ascending, and deletes the oldest in batches of ~100 per
write txn (so freed pages become available within a couple of generations
— LMDB's COW semantics need that headroom).

Atime refresh on a hit is throttled to at most once per session per key,
avoiding the write-amplification that an unconditional refresh would
cause on hot kernels.
"""
module DiskCache

using LMDB_jll: liblmdb
using Scratch: @get_scratch!

# ===========================================================================
# Minimal LMDB binding
# ===========================================================================

const MDB_RDONLY      = Cuint(0x00020000)
const MDB_NOTLS       = Cuint(0x00200000)
const MDB_NORDAHEAD   = Cuint(0x00800000)
const MDB_NOOVERWRITE = Cuint(0x00000010)

const MDB_SUCCESS     = Cint(0)
const MDB_KEYEXIST    = Cint(-30799)
const MDB_NOTFOUND    = Cint(-30798)
const MDB_MAP_FULL    = Cint(-30792)

# MDB_cursor_op values (from lmdb.h)
const MDB_FIRST = Cuint(0)
const MDB_NEXT  = Cuint(8)

struct MDB_val
    mv_size::Csize_t
    mv_data::Ptr{Cvoid}
end

struct MDB_envinfo
    me_mapaddr::Ptr{Cvoid}
    me_mapsize::Csize_t
    me_last_pgno::Csize_t
    me_last_txnid::Csize_t
    me_maxreaders::Cuint
    me_numreaders::Cuint
end

struct MDB_stat
    ms_psize::Cuint
    ms_depth::Cuint
    ms_branch_pages::Csize_t
    ms_leaf_pages::Csize_t
    ms_overflow_pages::Csize_t
    ms_entries::Csize_t
end

errstr(ret::Cint) =
    unsafe_string(ccall((:mdb_strerror, liblmdb), Cstring, (Cint,), ret))

@inline function check(ret::Cint, what)
    iszero(ret) && return nothing
    error("LMDB $what failed: $(errstr(ret))")
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
    env::Ptr{Cvoid}                  # MDB_env*
    dbi::Cuint                       # main DB handle
    psize::Int                       # LMDB page size in bytes (cached at open)
    path::String
    refreshed::Set{Vector{UInt8}}    # keys whose atime we already bumped this session
    state_lock::ReentrantLock        # guards `refreshed` + serializes evictions

    function Cache(env::Ptr{Cvoid}, dbi::Cuint, psize::Integer, path::AbstractString)
        c = new(env, dbi, Int(psize), String(path),
                Set{Vector{UInt8}}(), ReentrantLock())
        finalizer(close, c)
        return c
    end
end

isopen(cache::Cache) = cache.env != C_NULL

"""
    close(cache::Cache)

Release the underlying LMDB environment. Idempotent.
"""
function close(cache::Cache)
    cache.env == C_NULL && return
    ccall((:mdb_env_close, liblmdb), Cvoid, (Ptr{Cvoid},), cache.env)
    cache.env = C_NULL
    return
end

"""
    open(path; mapsize = 1<<30, maxreaders = 510) -> Cache

Open or create the disk cache at `path`. The directory is created if
missing. `mapsize` is the maximum on-disk size in bytes (LMDB grows the
map sparsely up to this limit). `maxreaders` caps concurrent reader
transactions.
"""
function open(path::AbstractString; mapsize::Integer = (Csize_t(1) << 30),
              maxreaders::Integer = 510)
    mkpath(path)

    env_ref = Ref{Ptr{Cvoid}}(C_NULL)
    check(ccall((:mdb_env_create, liblmdb), Cint, (Ref{Ptr{Cvoid}},), env_ref),
          "mdb_env_create")
    env = env_ref[]

    try
        check(ccall((:mdb_env_set_maxreaders, liblmdb), Cint,
                    (Ptr{Cvoid}, Cuint), env, Cuint(maxreaders)),
              "mdb_env_set_maxreaders")
        check(ccall((:mdb_env_set_mapsize, liblmdb), Cint,
                    (Ptr{Cvoid}, Csize_t), env, Csize_t(mapsize)),
              "mdb_env_set_mapsize")

        # MDB_NOTLS: read txns aren't tied to the OS thread that opened them
        # (Julia tasks may migrate). MDB_NORDAHEAD: lookups are random-access,
        # so OS read-ahead is wasted I/O.
        flags = MDB_NOTLS | MDB_NORDAHEAD
        check(ccall((:mdb_env_open, liblmdb), Cint,
                    (Ptr{Cvoid}, Cstring, Cuint, Cushort),
                    env, path, flags, Cushort(0o644)),
              "mdb_env_open($(repr(path)))")

        dbi, psize = open_main_db_and_stat!(env)
        return Cache(env, dbi, psize, path)
    catch
        ccall((:mdb_env_close, liblmdb), Cvoid, (Ptr{Cvoid},), env)
        rethrow()
    end
end

# Get a reusable handle to the env's main (unnamed) DB and read out the page
# size in the same dummy write txn. The dbi handle is only valid in
# subsequent transactions after the opening txn commits, so we always go
# through this dance.
function open_main_db_and_stat!(env::Ptr{Cvoid})
    txn_ref = Ref{Ptr{Cvoid}}(C_NULL)
    check(ccall((:mdb_txn_begin, liblmdb), Cint,
                (Ptr{Cvoid}, Ptr{Cvoid}, Cuint, Ref{Ptr{Cvoid}}),
                env, C_NULL, Cuint(0), txn_ref),
          "mdb_txn_begin (init)")
    txn = txn_ref[]

    dbi_ref = Ref{Cuint}(0)
    ret = ccall((:mdb_dbi_open, liblmdb), Cint,
                (Ptr{Cvoid}, Ptr{Cchar}, Cuint, Ref{Cuint}),
                txn, Ptr{Cchar}(C_NULL), Cuint(0), dbi_ref)
    if !iszero(ret)
        ccall((:mdb_txn_abort, liblmdb), Cvoid, (Ptr{Cvoid},), txn)
        check(ret, "mdb_dbi_open (main)")
    end
    dbi = dbi_ref[]

    stat_ref = Ref{MDB_stat}()
    ret = ccall((:mdb_stat, liblmdb), Cint,
                (Ptr{Cvoid}, Cuint, Ref{MDB_stat}), txn, dbi, stat_ref)
    if !iszero(ret)
        ccall((:mdb_txn_abort, liblmdb), Cvoid, (Ptr{Cvoid},), txn)
        check(ret, "mdb_stat")
    end
    psize = Int(stat_ref[].ms_psize)

    check(ccall((:mdb_txn_commit, liblmdb), Cint, (Ptr{Cvoid},), txn),
          "mdb_txn_commit (init)")

    return dbi, psize
end

"""
    env_info(cache::Cache) -> NamedTuple

Snapshot of the env's *live* page count and configured map size. Used
by the eviction policy to decide whether to prune.

Note that `mdb_env_info`'s `me_last_pgno` is a monotonic high-water
mark — it doesn't drop when entries get deleted. Eviction relies on
`mdb_stat`'s live-page accounting (branch + leaf + overflow), which
*does* shrink after a delete commits. Cost is one short read txn —
~10 µs.
"""
function env_info(cache::Cache)
    info_ref = Ref{MDB_envinfo}()
    check(ccall((:mdb_env_info, liblmdb), Cint,
                (Ptr{Cvoid}, Ref{MDB_envinfo}),
                cache.env, info_ref),
          "mdb_env_info")
    mapsize = Int(info_ref[].me_mapsize)

    txn_ref = Ref{Ptr{Cvoid}}(C_NULL)
    check(ccall((:mdb_txn_begin, liblmdb), Cint,
                (Ptr{Cvoid}, Ptr{Cvoid}, Cuint, Ref{Ptr{Cvoid}}),
                cache.env, C_NULL, MDB_RDONLY, txn_ref),
          "mdb_txn_begin (stat)")
    txn = txn_ref[]
    stat_ref = Ref{MDB_stat}()
    try
        check(ccall((:mdb_stat, liblmdb), Cint,
                    (Ptr{Cvoid}, Cuint, Ref{MDB_stat}),
                    txn, cache.dbi, stat_ref),
              "mdb_stat")
    finally
        ccall((:mdb_txn_abort, liblmdb), Cvoid, (Ptr{Cvoid},), txn)
    end
    s = stat_ref[]
    live_pages = Int(s.ms_branch_pages) + Int(s.ms_leaf_pages) + Int(s.ms_overflow_pages)
    used_bytes = live_pages * cache.psize
    return (; mapsize, used_bytes, entries = Int(s.ms_entries))
end

@inline utilization(cache::Cache) = let i = env_info(cache); i.used_bytes / i.mapsize end

# ===========================================================================
# Value framing (atime prefix)
# ===========================================================================

const _ATIME_PREFIX = 8  # bytes for a UInt64 little-endian timestamp

@inline function pack_value(atime::UInt64, value::Vector{UInt8})
    out = Vector{UInt8}(undef, _ATIME_PREFIX + length(value))
    GC.@preserve out begin
        unsafe_store!(Ptr{UInt64}(pointer(out)), htol(atime))
    end
    @inbounds copyto!(out, _ATIME_PREFIX + 1, value, 1, length(value))
    return out
end

@inline function read_atime(p::Ptr{UInt8})
    return ltoh(unsafe_load(Ptr{UInt64}(p)))
end

# ===========================================================================
# get / put!
# ===========================================================================

# Eviction trigger / target as fractions of mapsize. Margins live here for
# easy tuning; values are intentionally Howard-Chu-recommended-ish (he
# suggests pruning at ~90%).
const HIGH_WATER = 0.90
const LOW_WATER  = 0.75

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
function get(cache::Cache, key::Vector{UInt8})
    cache.env != C_NULL || error("DiskCache.get on closed cache")

    txn_ref = Ref{Ptr{Cvoid}}(C_NULL)
    check(ccall((:mdb_txn_begin, liblmdb), Cint,
                (Ptr{Cvoid}, Ptr{Cvoid}, Cuint, Ref{Ptr{Cvoid}}),
                cache.env, C_NULL, MDB_RDONLY, txn_ref),
          "mdb_txn_begin (read)")
    txn = txn_ref[]

    blob = try
        GC.@preserve key begin
            key_val  = Ref(MDB_val(Csize_t(length(key)), pointer(key)))
            data_val = Ref(MDB_val(Csize_t(0), C_NULL))

            ret = ccall(
                (:mdb_get, liblmdb), Cint,
                (Ptr{Cvoid}, Cuint, Ref{MDB_val}, Ref{MDB_val}),
                txn, cache.dbi, key_val, data_val)

            ret == MDB_NOTFOUND && return nothing
            check(ret, "mdb_get")
        end

        sz = Int(data_val[].mv_size)
        # Defensively reject malformed entries (shorter than the atime prefix).
        # We never write those, but a corrupted env shouldn't crash the caller.
        sz < _ATIME_PREFIX && return nothing
        out = Vector{UInt8}(undef, sz - _ATIME_PREFIX)
        unsafe_copyto!(pointer(out),
                       Ptr{UInt8}(data_val[].mv_data) + _ATIME_PREFIX,
                       sz - _ATIME_PREFIX)
        out
    finally
        ccall((:mdb_txn_abort, liblmdb), Cvoid, (Ptr{Cvoid},), txn)
    end

    # Throttled atime refresh. Done outside the read txn (LMDB doesn't
    # allow nesting). Errors here are non-fatal — we'd rather return the
    # blob than fail the launch.
    Base.@lock cache.state_lock begin
        if !(key in cache.refreshed)
            push!(cache.refreshed, copy(key))
            try
                put_raw!(cache, key, pack_value(time_ns(), blob))
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
overwritten — under content addressing, a key collision means the values
are identical (or, vanishingly, a `hash` collision); either way, the
first writer wins.

If the env is above [`HIGH_WATER`](@ref), [`evict_lru!`](@ref) is run
first to drop down to [`LOW_WATER`](@ref).
"""
function put!(cache::Cache, key::Vector{UInt8}, value::Vector{UInt8})
    cache.env != C_NULL || error("DiskCache.put! on closed cache")

    # Double-checked: the cheap utilization probe gates the lock acquisition
    # in the common case (well below high water, no contention).
    if utilization(cache) > HIGH_WATER
        Base.@lock cache.state_lock begin
            if utilization(cache) > HIGH_WATER
                evict_lru!(cache, LOW_WATER)
            end
        end
    end

    framed = pack_value(time_ns(), value)
    put_raw!(cache, key, framed)

    # Mark this key as "atime is fresh" so a subsequent get in the same
    # session doesn't redundantly bump it.
    Base.@lock cache.state_lock push!(cache.refreshed, copy(key))
    return
end

# Single mdb_put with already-framed (atime-prefixed) value bytes.
function put_raw!(cache::Cache, key::Vector{UInt8}, framed::Vector{UInt8})
    txn_ref = Ref{Ptr{Cvoid}}(C_NULL)
    check(ccall((:mdb_txn_begin, liblmdb), Cint,
                (Ptr{Cvoid}, Ptr{Cvoid}, Cuint, Ref{Ptr{Cvoid}}),
                cache.env, C_NULL, Cuint(0), txn_ref),
          "mdb_txn_begin (write)")
    txn = txn_ref[]

    handed_off = false
    try
        key_val = Ref(MDB_val(Csize_t(length(key)),    pointer(key)))
        val_val = Ref(MDB_val(Csize_t(length(framed)), pointer(framed)))

        ret = GC.@preserve key framed ccall(
            (:mdb_put, liblmdb), Cint,
            (Ptr{Cvoid}, Cuint, Ref{MDB_val}, Ref{MDB_val}, Cuint),
            txn, cache.dbi, key_val, val_val, Cuint(0))  # plain overwrite
        check(ret, "mdb_put")

        # mdb_txn_commit frees the txn handle on both success and failure
        # (per lmdb.h). Mark as handed off *before* check() can throw, so
        # the finally block doesn't abort an already-freed pointer.
        ret = ccall((:mdb_txn_commit, liblmdb), Cint, (Ptr{Cvoid},), txn)
        handed_off = true
        check(ret, "mdb_txn_commit (write)")
    finally
        handed_off || ccall((:mdb_txn_abort, liblmdb), Cvoid, (Ptr{Cvoid},), txn)
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
`target_ratio * mapsize`. Returns the number of entries evicted.

Called automatically from [`put!`](@ref) when utilization crosses
[`HIGH_WATER`](@ref); also exposed for manual invocation if needed.
"""
function evict_lru!(cache::Cache, target_ratio::Real = LOW_WATER)
    info = env_info(cache)
    target_bytes = floor(Int, target_ratio * info.mapsize)
    info.used_bytes <= target_bytes && return 0
    bytes_to_free = info.used_bytes - target_bytes

    entries = collect_entries(cache)
    sort!(entries, by = e -> e[2])  # atime ascending

    evicted = 0
    freed = 0
    batch = Vector{Vector{UInt8}}()
    sizehint!(batch, EVICT_BATCH)

    for (key, _atime, raw_size) in entries
        push!(batch, key)
        # Approximate freed bytes by entry size; LMDB's actual freed-page
        # count is fuzzier (page granularity, COW), but this is the right
        # order of magnitude.
        freed += raw_size
        if length(batch) >= EVICT_BATCH
            evicted += delete_batch!(cache, batch)
            empty!(batch)
        end
        # Stop as soon as we've nominated enough bytes for eviction. The
        # check has to sit outside the batch-flush branch — otherwise a
        # cache with fewer than EVICT_BATCH entries flushes the entire
        # contents in one shot.
        freed >= bytes_to_free && break
    end
    isempty(batch) || (evicted += delete_batch!(cache, batch))

    # Drop refreshed-set entries we just deleted — they no longer exist.
    # Cheaper than per-key removal: clear the set entirely. Worst case
    # we do a few redundant atime bumps after eviction.
    empty!(cache.refreshed)

    return evicted
end

# Collect (key_copy, atime, raw_value_size) for every entry in the cache.
function collect_entries(cache::Cache)
    entries = Tuple{Vector{UInt8}, UInt64, Int}[]

    txn_ref = Ref{Ptr{Cvoid}}(C_NULL)
    check(ccall((:mdb_txn_begin, liblmdb), Cint,
                (Ptr{Cvoid}, Ptr{Cvoid}, Cuint, Ref{Ptr{Cvoid}}),
                cache.env, C_NULL, MDB_RDONLY, txn_ref),
          "mdb_txn_begin (evict-scan)")
    txn = txn_ref[]

    cursor_ref = Ref{Ptr{Cvoid}}(C_NULL)
    ret = ccall((:mdb_cursor_open, liblmdb), Cint,
                (Ptr{Cvoid}, Cuint, Ref{Ptr{Cvoid}}),
                txn, cache.dbi, cursor_ref)
    if !iszero(ret)
        ccall((:mdb_txn_abort, liblmdb), Cvoid, (Ptr{Cvoid},), txn)
        check(ret, "mdb_cursor_open")
    end
    cursor = cursor_ref[]

    try
        op = MDB_FIRST
        key_val  = Ref(MDB_val(Csize_t(0), C_NULL))
        data_val = Ref(MDB_val(Csize_t(0), C_NULL))
        while true
            ret = ccall((:mdb_cursor_get, liblmdb), Cint,
                        (Ptr{Cvoid}, Ref{MDB_val}, Ref{MDB_val}, Cuint),
                        cursor, key_val, data_val, op)
            ret == MDB_NOTFOUND && break
            check(ret, "mdb_cursor_get")

            kv, dv = key_val[], data_val[]

            keysz = Int(kv.mv_size)
            datasz = Int(dv.mv_size)

            key_copy = Vector{UInt8}(undef, keysz)
            unsafe_copyto!(pointer(key_copy), Ptr{UInt8}(kv.mv_data), keysz)

            atime = if datasz >= _ATIME_PREFIX
                read_atime(Ptr{UInt8}(dv.mv_data))
            else
                # Pre-eviction-format entries (or anything malformed) get
                # priority eviction by virtue of atime = 0.
                UInt64(0)
            end

            push!(entries, (key_copy, atime, datasz))
            op = MDB_NEXT
        end
    finally
        ccall((:mdb_cursor_close, liblmdb), Cvoid, (Ptr{Cvoid},), cursor)
        ccall((:mdb_txn_abort, liblmdb), Cvoid, (Ptr{Cvoid},), txn)
    end

    return entries
end

# Delete a batch of keys in a single write txn.
function delete_batch!(cache::Cache, keys::Vector{Vector{UInt8}})
    txn_ref = Ref{Ptr{Cvoid}}(C_NULL)
    check(ccall((:mdb_txn_begin, liblmdb), Cint,
                (Ptr{Cvoid}, Ptr{Cvoid}, Cuint, Ref{Ptr{Cvoid}}),
                cache.env, C_NULL, Cuint(0), txn_ref),
          "mdb_txn_begin (evict-delete)")
    txn = txn_ref[]

    deleted = 0
    handed_off = false
    try
        for key in keys
            key_val = Ref(MDB_val(Csize_t(length(key)), pointer(key)))
            ret = GC.@preserve key ccall(
                (:mdb_del, liblmdb), Cint,
                (Ptr{Cvoid}, Cuint, Ref{MDB_val}, Ptr{MDB_val}),
                txn, cache.dbi, key_val, C_NULL)
            if ret == MDB_NOTFOUND
                # Already gone (race or repeat); skip.
            else
                check(ret, "mdb_del")
                deleted += 1
            end
        end

        # mdb_txn_commit frees the txn handle on both success and failure
        # (per lmdb.h). Mark as handed off *before* check() can throw, so
        # the finally block doesn't abort an already-freed pointer.
        ret = ccall((:mdb_txn_commit, liblmdb), Cint, (Ptr{Cvoid},), txn)
        handed_off = true
        check(ret, "mdb_txn_commit (evict)")
    finally
        handed_off || ccall((:mdb_txn_abort, liblmdb), Cvoid, (Ptr{Cvoid},), txn)
    end
    return deleted
end

# ===========================================================================
# Key derivation
# ===========================================================================

"""
    SCHEMA_VERSION

Cache schema version, mixed into every [`compute_key`](@ref). Bump this
when the on-disk value layout changes incompatibly (e.g. add/move bytes
in the framed value, change the hash algorithm, or alter what inputs
the key covers). The bump invalidates every existing entry on the next
access — old entries simply never match, and LRU eventually evicts
them.

Mirrors `_CACHE_VERSION` in cuTile Python (`_cache.py`).
"""
const SCHEMA_VERSION = UInt32(2)

"""
    compute_key(bytecode, sm_arch, opt_level, toolkit_version) -> Vector{UInt8}

Derive an 8-byte content-addressable cache key for a Tile IR compilation.
The key covers the bytecode plus every input that changes the resulting
CUBIN: target arch, opt level, and the `tileiras` toolkit version
(typically the full `--version` stdout — see `cuTile.toolkit_version()`).

Any change to those inputs produces a fresh key — old-toolkit entries
simply never match on lookup, and eventually evict via LRU. The
[`SCHEMA_VERSION`](@ref) prefix lets us invalidate every entry at once
when the value layout changes.
"""
function compute_key(bytecode::Vector{UInt8}, sm_arch::VersionNumber,
                     opt_level::Integer, toolkit_version::AbstractString)
    h = hash(SCHEMA_VERSION)
    h = hash(toolkit_version, h)
    h = hash(sm_arch, h)
    h = hash(opt_level, h)
    h = hash(bytecode, h)
    return reinterpret(UInt8, [h])
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
        # @get_scratch! resolves to cuTile's package UUID via moduleroot,
        # so the path is $DEPOT/scratchspaces/<cuTile-UUID>/disk_cache/.
        root = @get_scratch!("disk_cache")
        return open(root)
    catch err
        @debug "cuTile disk cache disabled" exception=(err, catch_backtrace())
        return nothing
    end
end

end # module DiskCache
