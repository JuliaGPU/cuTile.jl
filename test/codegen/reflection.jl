spec = ct.ArraySpec{1}(16, true)
TT3 = Tuple{ct.TileArray{Float32,1,Int32,spec}, ct.TileArray{Float32,1,Int32,spec}, ct.TileArray{Float32,1,Int32,spec}}

function reflect_vadd(a, b, c)
    pid = ct.bid(1)
    tile_a = ct.load(a; index=pid, shape=(16,))
    tile_b = ct.load(b; index=pid, shape=(16,))
    ct.store(c; index=pid, tile=tile_a + tile_b)
    return
end

function capture_stdout(f)
    mktemp() do _, io
        redirect_stdout(io) do
            f()
        end
        seekstart(io)
        read(io, String)
    end
end

@testset "code_typed" begin
    @test @filecheck begin
        @check "get_tile_block_id"
        @check "load_partition_view"
        @check "addf"
        @check "store_partition_view"
        ct.code_typed(reflect_vadd, TT3)
    end
end

@testset "code_structured" begin
    @testset "optimize=false" begin
        @test @filecheck begin
            @check "StructuredIRCode"
            @check "get_tile_block_id"
            # Core intrinsics survive without optimization
            @check "Base.add_int"
            @check "addf"
            ct.code_structured(reflect_vadd, TT3; optimize=false)
        end
    end

    @testset "optimize=true" begin
        @test @filecheck begin
            @check "StructuredIRCode"
            # Token ordering inserts make_token
            @check "MakeTokenNode"
            @check "get_tile_block_id"
            @check "addf"
            # Core intrinsics lowered by normalize pass
            @check_not "Base.add_int"
            ct.code_structured(reflect_vadd, TT3)
        end
    end
end

@testset "Debug info" begin
    @testset "code_tiled debuginfo=true" begin
        @test @filecheck begin
            # Debug info entries can appear in any order
            @check_dag "di_file"
            @check_dag "di_compile_unit"
            @check_dag "name = \"reflect_vadd\""
            @check_dag "di_loc"
            @check_dag "callsite"
            @check_dag "name = \"bid\""
            @check_dag "name = \"load\""
            @check_dag "name = \"store\""

            ct.code_tiled(reflect_vadd, TT3; debuginfo=true)
        end
    end

    @testset "code_tiled default has no debug info" begin
        @test @filecheck begin
            @check_not "di_loc"
            @check_not "di_subprogram"
            @check_not "callsite"
            ct.code_tiled(reflect_vadd, TT3)
        end
    end
end

if ct.tileiras_available()
    @testset "code_ptx" begin
        @test @filecheck begin
            @check ".visible .entry reflect_vadd"
            @check "add.rn.f32"
            @check "ret;"
            ct.code_ptx(reflect_vadd, TT3; sm_arch=v"10.0")
        end
    end

    @testset "code_sass" begin
        output = sprint(io -> ct.code_sass(io, reflect_vadd, TT3; sm_arch=v"10.0"))
        @test occursin(r"\.target\s+sm_100", output)
        @test occursin(".text.reflect_vadd", output)
        job = ct.tile_job(reflect_vadd, TT3; sm_arch=v"10.0")
        stdout_output = capture_stdout(() -> ct.code_sass(job))
        @test stdout_output == output
    end

    @testset "compile(job)" begin
        job = ct.tile_job(reflect_vadd, TT3; sm_arch=v"10.0")
        cubin = ct.compile(job)
        @test cubin isa Vector{UInt8} && cubin[1:4] == b"\x7fELF"
        ptx = ct.extract_ptx(cubin)
        @test startswith(ptx, ".version") && occursin(".entry reflect_vadd", ptx)
        @test ptx == ct.extract_ptx(IOBuffer(cubin)) == sprint(ct.code_ptx, job)
        stdout_ptx = capture_stdout(() -> ct.code_ptx(job))
        @test stdout_ptx == ptx
        @test_throws ct.ObjectFile.MagicMismatch ct.extract_ptx(UInt8[0x7f, 0x45, 0x4c, 0x46, 2, 1, 1, 0])
    end

    @testset "jobs" begin
        job = ct.tile_job(reflect_vadd, TT3; sm_arch=v"10.0")
        @test job isa ct.TileJob && job.config.name == "reflect_vadd"
        @test job === ct.tile_job(reflect_vadd, TT3; sm_arch=v"10.0")
        @test job !== ct.tile_job(reflect_vadd, TT3; sm_arch=v"10.0", opt_level=1)
        @test ct.job_signature(job) == (reflect_vadd, TT3)
        @test sprint(show, only(ct.code_typed(job))) == sprint(show, only(ct.code_typed(reflect_vadd, TT3)))
        stdout_tiled = capture_stdout(() -> ct.code_tiled(job))
        @test stdout_tiled == sprint(ct.code_tiled, job)
    end

    @testset "shared inference" begin
        # Inference is independent of target and hints, so every Tile job shares
        # one partition and one CodeInstance; codegen results are per job.
        job1 = ct.tile_job(reflect_vadd, TT3; sm_arch=v"10.0")
        job2 = ct.tile_job(reflect_vadd, TT3; sm_arch=v"8.9", opt_level=1)
        res1, res2 = ct.compile_or_lookup(job1), ct.compile_or_lookup(job2)
        ci1, ci2 = (get(ct.inference_cache(job), job.source, nothing) for job in (job1, job2))
        @test ci1 === ci2 !== nothing
        @test ci1.owner === ct.TILE_CACHE_OWNER
        @test res1 !== res2 && res1.cuda_bin != res2.cuda_bin
        # results are keyed by config, not job: they survive a world bump
        bump() = nothing
        job3 = ct.tile_job(reflect_vadd, TT3; sm_arch=v"10.0")
        @test job3.world > job1.world && job3.config === job1.config
        @test ct.compile_or_lookup(job3) === res1

        # A targetless cached job is normalized before lookup, so its CUBIN is
        # stored under the architecture it was assembled for.
        if CUDA.functional() && !isempty(CUDA.devices())
            targetless = ct.tile_job(reflect_vadd, TT3)
            targeted = ct.with_target(targetless, ct.default_sm_arch())
            @test ct.compile_or_lookup(targetless) === ct.compile_or_lookup(targeted)
        end
    end
end

if ct.tileiras_version() >= v"13.4"
    @testset "code_tiled remarks" begin
        output = sprint(io -> ct.code_tiled(io, reflect_vadd, TT3;
                                            sm_arch=v"10.0", remarks=true))
        @test occursin("// tileiras optimization remarks", output)
        @test occursin("// --- !Passed", output)
        @test occursin("// Name:", output)
    end
end

@testset "Constant args" begin
    const_spec = ct.ArraySpec{1}(128, true, (0,), (32,))
    ConstTT = Tuple{ct.TileArray{Float32,1,Int32,const_spec}, ct.TileArray{Float32,1,Int32,const_spec},
                    ct.TileArray{Float32,1,Int32,const_spec}, ct.Constant{Int64, 16}}

    function reflect_const_vadd(a, b, c, tile_size::Int)
        pid = ct.bid(1)
        tile_a = ct.load(a; index=pid, shape=(tile_size,))
        tile_b = ct.load(b; index=pid, shape=(tile_size,))
        ct.store(c; index=pid, tile=tile_a + tile_b)
        return
    end

    @testset "code_typed" begin
        @test @filecheck begin
            # Constant folded: shape=(16,) appears as literal tuple
            @check "make_partition_view"
            @check "(16,)"
            @check "Tuple{16}"
            ct.code_typed(reflect_const_vadd, ConstTT)
        end
    end

    @testset "code_structured" begin
        @test @filecheck begin
            @check "make_partition_view"
            @check "(16,)"
            @check "Tuple{16}"
            ct.code_structured(reflect_const_vadd, ConstTT; optimize=false)
        end
    end
end

@testset "Type args" begin
    const_spec = ct.ArraySpec{1}(128, true, (0,), (32,))

    @test ct.Constant(Int) isa ct.Constant{Type{Int}, Int}

    @testset "code_tiled with Type parameter" begin
        function reflect_type_param(a, b, c, tile_size::Int, ::Type{T}) where T
            pid = ct.bid(1)
            tile_a = ct.load(a; index=pid, shape=(tile_size,))
            tile_b = ct.load(b; index=pid, shape=(tile_size,))
            ct.store(c; index=pid, tile=tile_a + tile_b + zeros(T, (tile_size,)))
            return
        end

        ConstTypeTT = Tuple{ct.TileArray{Float32,1,Int32,const_spec}, ct.TileArray{Float32,1,Int32,const_spec},
                            ct.TileArray{Float32,1,Int32,const_spec}, ct.Constant{Int64, 16},
                            Type{Float32}}

        @test @filecheck begin
            @check "load_view_tko"
            @check "addf"
            @check "store_view_tko"
            ct.code_tiled(reflect_type_param, ConstTypeTT)
        end
    end
end
