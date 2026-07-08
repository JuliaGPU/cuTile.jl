using SHA: sha256

const DC = cuTile.DiskCache

@testset "DiskCache" begin
    @testset "toolkit version parsing" begin
        @test cuTile.parse_toolkit_version("tileiras V13.2.78\nBuild local") == "13.2.78"
        @test cuTile.parse_toolkit_version("future version format") == "future version format"
        @test cuTile.parse_toolkit_version("") === nothing
    end

    @testset "configuration" begin
        @test DC.cache_setting_disabled("")
        @test DC.cache_setting_disabled("0")
        @test DC.cache_setting_disabled(" OFF ")
        @test DC.cache_setting_disabled("none")
        @test !DC.cache_setting_disabled("cache-dir")

        @test DC.parse_cache_size("4096") == 4096
        @test_throws ArgumentError DC.parse_cache_size("0")
        @test_throws ArgumentError DC.parse_cache_size("not-a-size")

        mktempdir() do dir
            withenv("JULIA_CUTILE_CACHE_DIR" => dir) do
                @test DC.configured_cache_dir() == abspath(dir)
            end
            withenv("JULIA_CUTILE_CACHE_DIR" => "off") do
                @test DC.configured_cache_dir() === nothing
            end
            withenv("JULIA_CUTILE_CACHE_SIZE" => "8192") do
                @test DC.configured_mapsize() == 8192
            end

            write(joinpath(dir, "data.mdb"), "stale")
            write(joinpath(dir, "lock.mdb"), "stale")
            write(joinpath(dir, "keep"), "user data")
            DC.wipe_lmdb_files(dir)
            @test !isfile(joinpath(dir, "data.mdb"))
            @test !isfile(joinpath(dir, "lock.mdb"))
            @test isfile(joinpath(dir, "keep"))
        end
    end

    @testset "compute_key" begin
        bc = collect(b"some bytecode bytes")
        k = DC.compute_key(bc, v"12.0", 3, "13.1")
        @test length(k) == 32
        @test bytes2hex(k) == "fd117ab2268ddfc8e0dccaad206b80557fa4239ea12f2d83293bcfa95f9d485e"
        @test DC.compute_key(bc, v"12.0", 3, "13.1") == k
        @test DC.compute_key(bc, v"12.0", 3, "13.2") != k
        @test DC.compute_key(bc, v"12.0", 2, "13.1") != k
        @test DC.compute_key(bc, v"12.1", 3, "13.1") != k
        @test DC.compute_key(vcat(bc, [0x00]), v"12.0", 3, "13.1") != k
    end

    @testset "open / close / roundtrip" begin
        mktempdir() do dir
            cache = DC.open(dir; mapsize = 4 * 1024 * 1024)
            try
                k = sha256(b"hello")
                @test DC.get(cache, k) === nothing
                v = collect(b"world payload")
                DC.put!(cache, k, v)
                @test DC.get(cache, k) == v
                # idempotent: same key+value, same observable state
                DC.put!(cache, k, v)
                @test DC.get(cache, k) == v
            finally
                DC.close(cache)
            end
        end
    end

    @testset "eviction prefers older writes (within one session)" begin
        # 4 MB map, ~50 KB blobs ⇒ ~75 entries fit before eviction kicks in.
        # Write 25 "old" keys followed by 80 "new" keys (5+ MB total) and
        # confirm the LRU sort drops the older writes first.
        mktempdir() do dir
            blob_size = 50_000
            mapsize   = 4 * 1024 * 1024
            cache = DC.open(dir; mapsize)
            try
                old_keys = [sha256(UInt8[1, i % 256, (i ÷ 256) % 256]) for i in 1:25]
                for k in old_keys
                    DC.put!(cache, k, rand(UInt8, blob_size))
                end
                new_keys = [sha256(UInt8[2, i % 256, (i ÷ 256) % 256]) for i in 1:80]
                for k in new_keys
                    DC.put!(cache, k, rand(UInt8, blob_size))
                end

                old_present = count(k -> DC.get(cache, k) !== nothing, old_keys)
                new_present = count(k -> DC.get(cache, k) !== nothing, new_keys)
                total_before = length(old_keys) + length(new_keys)
                total_after  = old_present + new_present

                # Eviction must have run (we wrote ~5 MB into a 4 MB map)
                @test total_after < total_before
                # LRU honored recency — newer writes survive better
                @test new_present > old_present
                # Most of the old set is gone
                @test old_present < length(old_keys) ÷ 2
                # And most of the new set is intact
                @test new_present >= 0.6 * length(new_keys)
            finally
                DC.close(cache)
            end
        end
    end

    @testset "put! reserves room for incoming value" begin
        mktempdir() do dir
            blob_size = 50_000
            mapsize   = 4 * 1024 * 1024
            cache = DC.open(dir; mapsize)
            try
                i = 0
                while DC.env_info(cache).used_bytes <= 0.82 * mapsize
                    i += 1
                    k = sha256(UInt8[0x33, i % 256, (i ÷ 256) % 256])
                    DC.put!(cache, k, rand(UInt8, blob_size))
                end
                @test DC.env_info(cache).used_bytes < 0.90 * mapsize

                large_key = sha256(b"large incoming blob")
                large_blob = rand(UInt8, 900_000)
                DC.put!(cache, large_key, large_blob)
                @test DC.get(cache, large_key) == large_blob
            finally
                DC.close(cache)
            end
        end
    end

    @testset "cross-session atime refresh keeps read-hot keys" begin
        # The atime-refresh-on-hit path is throttled to once per session
        # per key. To exercise it we need at least three opens of the
        # same env: write everything, read the "hot" ones in a fresh
        # session (bumping their atime), then overflow in a third.
        mktempdir() do dir
            blob_size = 50_000
            mapsize   = 4 * 1024 * 1024

            keys = [sha256(UInt8[1, i]) for i in 1:25]

            # Session 1 — populate
            cache = DC.open(dir; mapsize)
            try
                for k in keys
                    DC.put!(cache, k, rand(UInt8, blob_size))
                end
            finally
                DC.close(cache)
            end

            # Session 2 — touch the "hot" half on read; bumps their atime
            hot  = keys[1:12]
            cold = keys[13:end]
            cache = DC.open(dir; mapsize)
            try
                for k in hot
                    @test DC.get(cache, k) !== nothing
                end
            finally
                DC.close(cache)
            end

            # Session 3 — write enough overflow to force *some* eviction
            # but not so much that it exhausts both cold and hot. With
            # 25 baseline + 50 overflow = 75 entries × ~52 KB ≈ 3.9 MB,
            # we cross HIGH_WATER once and drop ~10 entries (the oldest:
            # all cold, plus possibly a few first-bumped hot).
            overflow = [sha256(UInt8[2, i]) for i in 1:50]
            cache = DC.open(dir; mapsize)
            try
                for k in overflow
                    DC.put!(cache, k, rand(UInt8, blob_size))
                end
                hot_present  = count(k -> DC.get(cache, k) !== nothing, hot)
                cold_present = count(k -> DC.get(cache, k) !== nothing, cold)
                ovl_present  = count(k -> DC.get(cache, k) !== nothing, overflow)

                # Eviction must have triggered.
                @test (hot_present + cold_present + ovl_present) <
                      (length(hot) + length(cold) + length(overflow))
                # LRU honored recency: more "hot" survives than "cold".
                @test hot_present > cold_present
                # Strict LRU: if any cold survived, all hot must have too
                # (cold's atime < hot's atime by construction).
                @test cold_present == 0 || hot_present == length(hot)
            finally
                DC.close(cache)
            end
        end
    end

    @testset "manual evict_lru!" begin
        mktempdir() do dir
            cache = DC.open(dir; mapsize = 4 * 1024 * 1024)
            try
                for i in 1:50
                    DC.put!(cache, sha256(UInt8[i % 256, 0xff]), rand(UInt8, 50_000))
                end
                before = DC.env_info(cache).used_bytes
                n = DC.evict_lru!(cache, 0.25)
                after = DC.env_info(cache).used_bytes
                @test n > 0
                @test after < before
            finally
                DC.close(cache)
            end
        end
    end
end
