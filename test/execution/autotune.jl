using CUDA

const Exp = ct.Experimental

@testset "Autotune" begin
    @testset "@withconfig" begin
        grid_fn = Exp.@withconfig (cld(1024, $tile), 17)
        @test grid_fn((; tile=64)) == (16, 17)

        args_fn = Exp.@withconfig (ct.Constant($tile), $occ, 42)
        args = args_fn((; tile=32, occ=2))
        @test args[1] isa ct.Constant
        @test args[1][] == 32
        @test args[2] == 2
        @test args[3] == 42
    end

    function vadd_kernel(a::ct.TileArray{Float32,1},
                         b::ct.TileArray{Float32,1},
                         c::ct.TileArray{Float32,1},
                         tile::Int)
        pid = ct.bid(1)
        ta = ct.load(a, pid, (tile[],))
        tb = ct.load(b, pid, (tile[],))
        ct.store(c, pid, ta + tb)
        return nothing
    end

    function inplace_add_kernel(x::ct.TileArray{Float32,1},
                                tile::Int)
        pid = ct.bid(1)
        tx = ct.load(x, pid, (tile[],))
        ct.store(x, pid, tx .+ 1f0)
        return nothing
    end

    n = 512
    a = CUDA.fill(1f0, n)
    b = CUDA.fill(2f0, n)
    c = CUDA.zeros(Float32, n)

    configs = [
        (; tile=16, occupancy=nothing, num_ctas=nothing),
        (; tile=32, occupancy=2, num_ctas=nothing),
        (; tile=64, occupancy=4, num_ctas=2),
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
        space = Exp.CartesianSpace(;
            tile=(16, 32), occupancy=(nothing, 2), num_ctas=(nothing,))
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
            tile=(16, 32, 64), occupancy=(nothing,), num_ctas=(nothing,))
        result = Exp.autotune_launch(
            vadd_kernel, space, grid_fn, args_fn;
            key=(:constrained, n),
            tuning=(preset=:fast, refine_topk=0),
        )
        @test result.tuned_config.tile == 16
        @test Array(c) ≈ fill(3f0, n)
    end

    @testset "NamedTuple convenience → CartesianSpace" begin
        Exp.clear_autotune_cache()
        fill!(c, 0f0)
        result = Exp.autotune_launch(
            vadd_kernel,
            (tile=(16, 32), occupancy=(nothing,), num_ctas=(nothing,)),
            grid_fn, args_fn;
            key=(:nt_convenience, n),
            tuning=(preset=:fast, refine_topk=0),
        )
        @test result.tuned_config.tile in (16, 32)
        @test Array(c) ≈ fill(3f0, n)
    end

    @testset "launch_args_fn (inplace kernel)" begin
        x = CUDA.zeros(Float32, n)
        original_x = Array(x)
        Exp.clear_autotune_cache()
        result = Exp.autotune_launch(
            inplace_add_kernel,
            [(; tile=16), (; tile=32)],
            grid_fn,
            cfg -> (copy(x), ct.Constant(cfg.tile));
            launch_args_fn=cfg -> (x, ct.Constant(cfg.tile)),
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

    @testset "key_fn" begin
        Exp.clear_autotune_cache()
        call_count = Ref(0)
        my_key_fn = () -> begin
            call_count[] += 1
            return (:dynamic, Float32)
        end

        fill!(c, 0f0)
        r1 = Exp.autotune_launch(
            vadd_kernel, configs, grid_fn, args_fn;
            key_fn=my_key_fn, tuning=(preset=:fast, refine_topk=0))
        r2 = Exp.autotune_launch(
            vadd_kernel, configs, grid_fn, args_fn;
            key_fn=my_key_fn, tuning=(preset=:fast, refine_topk=0))
        @test !r1.cache_hit
        @test r2.cache_hit
        @test call_count[] == 2
        @test Array(c) ≈ fill(3f0, n)
    end
end
