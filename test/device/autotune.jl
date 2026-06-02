using CUDA

const Exp = ct.Experimental

@testset "Autotune" begin

    function vadd_kernel(a::ct.TileArray{Float32,1},
                         b::ct.TileArray{Float32,1},
                         c::ct.TileArray{Float32,1},
                         tile::Int)
        pid = ct.bid(1)
        ta = ct.load(a, pid, (tile,))
        tb = ct.load(b, pid, (tile,))
        ct.store(c, pid, ta + tb)
        return nothing
    end

    function inplace_add_kernel(x::ct.TileArray{Float32,1},
                                tile::Int)
        pid = ct.bid(1)
        tx = ct.load(x, pid, (tile,))
        ct.store(x, pid, tx .+ 1f0)
        return nothing
    end

    function cache_probe_kernel(a::ct.TileArray{Float32,1},
                                b::ct.TileArray{Float32,1},
                                c::ct.TileArray{Float32,1},
                                tile::Int)
        pid = ct.bid(1)
        ta = ct.load(a, pid, (tile,))
        tb = ct.load(b, pid, (tile,))
        ct.store(c, pid, ta + tb)
        return nothing
    end

    function normal_const_entry_count(f, args)
        converted = map(ct.cuTileconvert, args)
        tt = Tuple{map(Core.Typeof, converted)...}
        argtypes, _ = ct.unwrap_argtypes(f, tt)

        world = Base.get_world_counter()
        key = ct.TileCacheKey(ct.default_sm_arch(), ct.bytecode_version(),
                              3, nothing, nothing, nothing)
        cache = ct.CompilerCaching.CacheView{ct.CuTileResults}(key, world)
        mi = ct.CompilerCaching.method_instance(f, argtypes; world)
        ci = get(cache, mi, nothing)
        ci === nothing && return 0

        cached = ct.CC.traverse_analysis_results(ci) do result
            result isa ct.CompilerCaching.CachedResult{ct.CuTileResults} ?
                result : nothing
        end
        cached === nothing && return 0
        return length(cached.const_entries)
    end

    n = 512
    a = CUDA.fill(1f0, n)
    b = CUDA.fill(2f0, n)
    c = CUDA.zeros(Float32, n)

    # cfg 1's `occupancy=nothing, num_ctas=nothing` slots are present for
    # `FixedSpace` shape uniformity — cfgs 2 and 3 carry real hint values,
    # and `FixedSpace{names, NT<:NamedTuple{names}}` requires every element
    # to share the same `names` set.
    configs = [
        (; tile=16, occupancy=nothing, num_ctas=nothing),
        (; tile=32, occupancy=2,       num_ctas=nothing),
        (; tile=64, occupancy=4,       num_ctas=2),
    ]
    args_fn = cfg -> (a, b, c, ct.Constant(cfg.tile))
    grid_fn = cfg -> cld(n, cfg.tile)

    @testset "basic tuning" begin
        Exp.clear_autotune_cache()
        result = Exp.autotune_launch(
            vadd_kernel, configs, grid_fn, args_fn;
            key=(:basic, n),
            tuning=(preset=:fast, refine_topk=0),
        )
        @test !result.cache_hit
        @test result.tuned_config in configs
        @test !isempty(result.tuning_record)
        @test Array(c) ≈ fill(3f0, n)
    end

    @testset "cache hit" begin
        fill!(c, 0f0)
        result = Exp.autotune_launch(
            vadd_kernel, configs, grid_fn, args_fn;
            key=(:basic, n),
            tuning=(preset=:fast, refine_topk=0),
        )
        @test result.cache_hit
        @test Array(c) ≈ fill(3f0, n)
    end

    @testset "force retune" begin
        fill!(c, 0f0)
        result = Exp.autotune_launch(
            vadd_kernel, configs, grid_fn, args_fn;
            key=(:basic, n),
            tuning=(preset=:fast, refine_topk=0, force=true),
        )
        @test !result.cache_hit
        @test Array(c) ≈ fill(3f0, n)
    end

    @testset "CartesianSpace" begin
        Exp.clear_autotune_cache()
        fill!(c, 0f0)
        # Keep `occupancy=(nothing, 2)` — legitimately tunes between
        # "no hint" and 2. Single-value `nothing` axes are noise (see
        # other testsets); a 2-value axis with `nothing` as one option
        # is meaningful.
        space = Exp.CartesianSpace(;
            tile=(16, 32), occupancy=(nothing, 2))
        result = Exp.autotune_launch(
            vadd_kernel, space, grid_fn, args_fn;
            key=(:cartesian, n),
            tuning=(preset=:fast, refine_topk=0),
        )
        @test hasproperty(result.tuned_config, :tile)
        @test hasproperty(result.tuned_config, :occupancy)
        @test Array(c) ≈ fill(3f0, n)
    end

    @testset "CartesianSpace with constraint" begin
        Exp.clear_autotune_cache()
        fill!(c, 0f0)
        space = Exp.CartesianSpace(
            cfg -> cfg.tile == 16;
            tile=(16, 32, 64))
        result = Exp.autotune_launch(
            vadd_kernel, space, grid_fn, args_fn;
            key=(:constrained, n),
            tuning=(preset=:fast, refine_topk=0),
        )
        @test result.tuned_config.tile == 16
        @test Array(c) ≈ fill(3f0, n)
    end

    @testset "CartesianSpace range axis" begin
        @test collect(Exp.CartesianSpace(tile=16:16:32)) ==
              [(; tile=16), (; tile=32)]
    end

    @testset "NamedTuple convenience → CartesianSpace" begin
        Exp.clear_autotune_cache()
        fill!(c, 0f0)
        result = Exp.autotune_launch(
            vadd_kernel,
            (tile=(16, 32),),
            grid_fn, args_fn;
            key=(:nt_convenience, n),
            tuning=(preset=:fast, refine_topk=0),
        )
        @test result.tuned_config.tile in (16, 32)
        @test Array(c) ≈ fill(3f0, n)
    end

    @testset "launch_args (inplace kernel)" begin
        x = CUDA.zeros(Float32, n)
        original_x = Array(x)
        Exp.clear_autotune_cache()
        result = Exp.autotune_launch(
            inplace_add_kernel,
            [(; tile=16), (; tile=32)],
            grid_fn,
            cfg -> (copy(x), ct.Constant(cfg.tile));
            launch_args=cfg -> (x, ct.Constant(cfg.tile)),
            key=(:inplace, n),
            tuning=(preset=:fast, refine_topk=0),
        )
        @test !result.cache_hit
        @test Array(x) == original_x .+ 1f0
    end

    @testset "refinement" begin
        Exp.clear_autotune_cache()
        fill!(c, 0f0)
        result = Exp.autotune_launch(
            vadd_kernel, configs, grid_fn, args_fn;
            key=(:refine, n),
            tuning=(warmup=1, reps=2, refine_topk=2, refine_reps=4),
        )
        @test !result.cache_hit
        # Refinement record replaces initial — has at most refine_topk entries
        @test length(result.tuning_record) <= 2
        @test Array(c) ≈ fill(3f0, n)
    end

    @testset "verify" begin
        Exp.clear_autotune_cache()
        fill!(c, 0f0)
        verify_called = Ref(false)
        result = Exp.autotune_launch(
            vadd_kernel, configs, grid_fn, args_fn;
            key=(:verify, n),
            tuning=(preset=:fast, refine_topk=0),
            verify=() -> let
                ref = Array(a) .+ Array(b)
                verify_called[] = true
                () -> (CUDA.@allowscalar all(isapprox.(Array(c), ref, atol=1f-5)))
            end,
        )
        @test verify_called[]
        @test Array(c) ≈ fill(3f0, n)
    end

    @testset "clear cache per-kernel per-key" begin
        Exp.clear_autotune_cache()
        Exp.autotune_launch(
            vadd_kernel, configs, grid_fn, args_fn;
            key=(:k1, n), tuning=(preset=:fast, refine_topk=0))
        Exp.autotune_launch(
            vadd_kernel, configs, grid_fn, args_fn;
            key=(:k2, n), tuning=(preset=:fast, refine_topk=0))

        # Clear only one key
        Exp.clear_autotune_cache(kernel=vadd_kernel, key=(:k1, n))
        fill!(c, 0f0)
        r1 = Exp.autotune_launch(
            vadd_kernel, configs, grid_fn, args_fn;
            key=(:k1, n), tuning=(preset=:fast, refine_topk=0))
        @test !r1.cache_hit  # was cleared

        fill!(c, 0f0)
        r2 = Exp.autotune_launch(
            vadd_kernel, configs, grid_fn, args_fn;
            key=(:k2, n), tuning=(preset=:fast, refine_topk=0))
        @test r2.cache_hit  # still cached
    end

    @testset "shared key across shapes" begin
        Exp.clear_autotune_cache()
        n2 = 1024
        a2 = CUDA.fill(1f0, n2)
        b2 = CUDA.fill(2f0, n2)
        c2 = CUDA.zeros(Float32, n2)
        shared_key = (:shape_agnostic, eltype(a))

        Exp.autotune_launch(
            vadd_kernel, configs, grid_fn, args_fn;
            key=shared_key, tuning=(preset=:fast, refine_topk=0))

        fill!(c2, 0f0)
        result = Exp.autotune_launch(
            vadd_kernel, configs,
            cfg -> cld(n2, cfg.tile),
            cfg -> (a2, b2, c2, ct.Constant(cfg.tile));
            key=shared_key, tuning=(preset=:fast, refine_topk=0))
        @test result.cache_hit
        @test result.grid == cld(n2, result.tuned_config.tile)
        @test Array(c2) ≈ fill(3f0, n2)
    end

    @testset "literal grid/args (no closure)" begin
        Exp.clear_autotune_cache()
        fill!(c, 0f0)
        # Pass `grid` and `args` as values rather than `cfg -> …` closures.
        # cfg-independent grid: cld(n, tile) == cld(512, 16) for every cfg
        # so the literal `32` happens to be valid here.
        result = Exp.autotune_launch(
            vadd_kernel,
            [(; tile=16)],
            cld(n, 16),                                # literal grid
            (a, b, c, ct.Constant(16));                # literal args
            key=(:literal, n),
            tuning=(preset=:fast, refine_topk=0))
        @test !result.cache_hit
        @test result.grid == cld(n, 16)
        @test Array(c) ≈ fill(3f0, n)
    end

    @testset "static num_ctas / occupancy as kwargs" begin
        Exp.clear_autotune_cache()
        fill!(c, 0f0)
        # `space` has no num_ctas/occupancy axes — they're static kwargs.
        result = Exp.autotune_launch(
            vadd_kernel,
            [(; tile=16), (; tile=32)],
            cfg -> cld(n, cfg.tile),
            cfg -> (a, b, c, ct.Constant(cfg.tile));
            key=(:static_hints, n),
            occupancy=2,
            tuning=(preset=:fast, refine_topk=0))
        @test !result.cache_hit
        @test Array(c) ≈ fill(3f0, n)
    end

    @testset "conflict: static + space axis" begin
        Exp.clear_autotune_cache()
        # Run-time path (opaque space): autotune_launch should reject.
        @test_throws ArgumentError Exp.autotune_launch(
            vadd_kernel,
            [(; tile=16, occupancy=2)],
            cfg -> cld(n, cfg.tile),
            cfg -> (a, b, c, ct.Constant(cfg.tile));
            key=(:conflict, n),
            occupancy=4,
            tuning=(preset=:fast, refine_topk=0))
    end

    @testset "conflict scans every config" begin
        space = Exp.FixedSpace(Any[(; tile=16), (; tile=32, occupancy=2)])
        @test_throws ArgumentError Exp.autotune_launch(
            vadd_kernel,
            space,
            cfg -> cld(n, cfg.tile),
            cfg -> (a, b, c, ct.Constant(cfg.tile));
            key=(:conflict_late, n),
            occupancy=4,
            tuning=(preset=:fast, refine_topk=0))
    end

    @testset "tuning validation" begin
        @test_throws ArgumentError Exp.autotune_launch(
            vadd_kernel,
            [(; tile=16)],
            grid_fn, args_fn;
            key=(:bad_reps, n),
            tuning=(preset=:fast, reps=0))

        @test_throws ArgumentError Exp.autotune_launch(
            vadd_kernel,
            [(; tile=16)],
            grid_fn, args_fn;
            key=(:bad_key, n),
            tuning=(preset=:fast, typo=1))
    end

    @testset "cached config must belong to current space" begin
        Exp.clear_autotune_cache()
        fill!(c, 0f0)
        r1 = Exp.autotune_launch(
            vadd_kernel,
            [(; tile=16)],
            grid_fn, args_fn;
            key=(:space_sensitive, n),
            tuning=(preset=:fast, refine_topk=0))
        @test !r1.cache_hit
        @test r1.tuned_config.tile == 16

        fill!(c, 0f0)
        r2 = Exp.autotune_launch(
            vadd_kernel,
            [(; tile=32)],
            grid_fn, args_fn;
            key=(:space_sensitive, n),
            tuning=(preset=:fast, refine_topk=0))
        @test !r2.cache_hit
        @test r2.tuned_config.tile == 32
        @test Array(c) ≈ fill(3f0, n)
    end

    @testset "only winner enters normal compiler cache" begin
        Exp.clear_autotune_cache()
        probe_c = CUDA.zeros(Float32, n)
        probe_configs = [(; tile=16), (; tile=32), (; tile=64)]
        probe_args = cfg -> (a, b, probe_c, ct.Constant(cfg.tile))
        @test normal_const_entry_count(cache_probe_kernel, probe_args(probe_configs[1])) == 0

        result = Exp.autotune_launch(
            cache_probe_kernel,
            probe_configs,
            cfg -> cld(n, cfg.tile),
            probe_args;
            key=(:temporary_candidates, n),
            tuning=(preset=:fast, refine_topk=0))

        @test result.tuned_config in probe_configs
        @test normal_const_entry_count(cache_probe_kernel, probe_args(probe_configs[1])) == 1
        @test Array(probe_c) ≈ fill(3f0, n)
    end

    @testset "@autotune macro: NT space" begin
        Exp.clear_autotune_cache()
        fill!(c, 0f0)
        result = Exp.@autotune(
            key = (:macro_nt, n),
            space = (tile=(16, 32, 64),),
            blocks = cld(n, $tile),
            tuning = (preset=:fast, refine_topk=0),
            vadd_kernel(a, b, c, ct.Constant($tile)),
        )
        @test !result.cache_hit
        @test result.tuned_config.tile in (16, 32, 64)
        @test Array(c) ≈ fill(3f0, n)
    end

    @testset "@autotune macro: vector space" begin
        Exp.clear_autotune_cache()
        fill!(c, 0f0)
        result = Exp.@autotune(
            key = (:macro_vec, n),
            space = [(; tile=16), (; tile=32)],
            blocks = cld(n, $tile),
            tuning = (preset=:fast, refine_topk=0),
            vadd_kernel(a, b, c, ct.Constant($tile)),
        )
        @test !result.cache_hit
        @test result.tuned_config.tile in (16, 32)
        @test Array(c) ≈ fill(3f0, n)
    end

    @testset "@autotune macro: tuple blocks + 2D \$interp" begin
        Exp.clear_autotune_cache()
        fill!(c, 0f0)
        # Use a 1D kernel but pass a Tuple blocks=(N, 1) to exercise the
        # tuple-grid + $X-interp-in-blocks path.
        result = Exp.@autotune(
            key = (:macro_tuple, n),
            space = (tile=(16, 32),),
            blocks = (cld(n, $tile), 1),
            tuning = (preset=:fast, refine_topk=0),
            vadd_kernel(a, b, c, ct.Constant($tile)),
        )
        @test result.grid == (cld(n, result.tuned_config.tile), 1)
        @test Array(c) ≈ fill(3f0, n)
    end

    @testset "@autotune macro: static num_ctas as kwarg" begin
        Exp.clear_autotune_cache()
        fill!(c, 0f0)
        result = Exp.@autotune(
            key = (:macro_static, n),
            space = (tile=(16, 32),),
            blocks = cld(n, $tile),
            occupancy = 2,
            tuning = (preset=:fast, refine_topk=0),
            vadd_kernel(a, b, c, ct.Constant($tile)),
        )
        @test !result.cache_hit
        @test Array(c) ≈ fill(3f0, n)
    end

    @testset "@autotune macro: macro-time conflict error" begin
        # Should error at macro expansion (not run time).
        @test_throws LoadError @eval Exp.@autotune(
            space = (tile=(16,), num_ctas=(1, 2)),
            blocks = 1,
            num_ctas = 4,
            kernel(a),
        )
    end

    @testset "@autotune macro: required kwargs" begin
        @test_throws LoadError @eval Exp.@autotune(blocks = 1, kernel(a))
        @test_throws LoadError @eval Exp.@autotune(space = (tile=(16,),), kernel(a))
        @test_throws LoadError @eval Exp.@autotune(space = (tile=(16,),), blocks = 1)
    end
end
