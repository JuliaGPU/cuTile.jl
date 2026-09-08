spec = ct.ArraySpec{1}(16, true)
TT3 = Tuple{ct.TileArray{Float32,1,Int32,spec}, ct.TileArray{Float32,1,Int32,spec}, ct.TileArray{Float32,1,Int32,spec}}

function reflect_vadd(a, b, c)
    pid = ct.bid(1)
    tile_a = ct.load(a; index=pid, shape=(16,))
    tile_b = ct.load(b; index=pid, shape=(16,))
    ct.store(c; index=pid, tile=tile_a + tile_b)
    return
end

function reflect_vadd_n(a, b, c, n)
    pid = ct.bid(1)
    tile_a = ct.load(a; index=pid, shape=(n,))
    tile_b = ct.load(b; index=pid, shape=(n,))
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
        @test job.const_argtypes === nothing
        @test job === ct.tile_job(reflect_vadd, TT3; sm_arch=v"10.0")
        @test job !== ct.tile_job(reflect_vadd, TT3; sm_arch=v"10.0", opt_level=1)
        @test ct.job_signature(job) == (reflect_vadd, TT3)
        @test sprint(show, job) == "TileJob(reflect_vadd($(join(TT3.parameters, ", "))); " *
                                   "sm_arch=10.0.0, bytecode_version=v\"$(ct.bytecode_version())\")"
        @test sprint(show, only(ct.code_typed(job))) == sprint(show, only(ct.code_typed(reflect_vadd, TT3)))
        stdout_tiled = capture_stdout(() -> ct.code_tiled(job))
        @test stdout_tiled == sprint(ct.code_tiled, job)

        # Constant arguments seed inference; the job restores them in its signature.
        const_tt = Tuple{TT3.parameters..., ct.Constant{Int, 16}}
        const_job = ct.tile_job(reflect_vadd_n, const_tt; sm_arch=v"10.0")
        @test const_job.const_argtypes == (ct.CC.Const(reflect_vadd_n), TT3.parameters..., ct.CC.Const(16))
        @test ct.job_signature(const_job) == (reflect_vadd_n, const_tt)
        @test const_job === ct.tile_job(reflect_vadd_n, const_tt; sm_arch=v"10.0")
        @test occursin("Constant{Int64, 16}", sprint(show, const_job))
        @test ct.compile_or_lookup(const_job) === ct.compile_or_lookup(const_job)

        # A job created without a target can run every stage before tileiras.
        targetless = ct.TileJob(job.source, nothing, job.world,
                                ct.TileConfig(ct.TileCompilerTarget(nothing, job.config.target.bytecode_version),
                                              job.config.params, job.config.name))
        @test occursin("addf", sprint(ct.code_tiled, targetless))
        @test_throws ArgumentError ct.compile(targetless)
        @test_throws ArgumentError ct.code_tiled(devnull, targetless; remarks=true)

        # Incompatible targets are rejected when the job is built.
        @test_throws ArgumentError ct.tile_job(reflect_vadd, TT3; sm_arch=v"7.5")
    end

    @testset "shared inference" begin
        # Inference is independent of target and hints, so every Tile job shares
        # one partition and one CodeInstance; codegen results are per job.
        job1 = ct.tile_job(reflect_vadd, TT3; sm_arch=v"10.0")
        job2 = ct.tile_job(reflect_vadd, TT3; sm_arch=v"8.9", opt_level=1)
        res1, res2 = ct.compile_or_lookup(job1), ct.compile_or_lookup(job2)
        ci1, ci2 = ct.infer(job1), ct.infer(job2)
        @test ci1 === ci2
        @test ci1.owner === ct.TILE_CACHE_OWNER
        @test res1 !== res2 && res1.cubin != res2.cubin
        # results are keyed by config, not job: they survive a world bump
        bump() = nothing
        job3 = ct.tile_job(reflect_vadd, TT3; sm_arch=v"10.0")
        @test job3.world > job1.world && job3.config === job1.config
        @test ct.compile_or_lookup(job3) === res1

        # The default target is the active device's, so reflection matches a launch.
        if CUDA.functional() && !isempty(CUDA.devices())
            job = ct.tile_job(reflect_vadd, TT3)
            @test job.config.target.sm_arch == ct.default_sm_arch()
            @test job === ct.tile_job(reflect_vadd, TT3; sm_arch=ct.default_sm_arch())
        end
    end

    @testset "compile hook" begin
        job = ct.tile_job(reflect_vadd, TT3; sm_arch=v"10.0")
        ct.compile_or_lookup(job)
        # GPUCompiler's hook observes hits as well as misses, once per distinct job
        seen = ct.TileJob[]
        with(GPUCompiler.compile_hook => (job -> push!(seen, job))) do
            ct.compile_or_lookup(job)
            ct.compile_or_lookup(job)
        end
        @test seen == [job, job]
        @test GPUCompiler.compile_hook[] === nothing

        # GPUCompiler's macros accept Tile jobs through the shared protocol; cuTile
        # re-exports the Julia-level ones rather than defining its own
        @test ct.var"@device_code_typed" === GPUCompiler.var"@device_code_typed"
        typed = ct.@device_code_typed ct.compile_or_lookup(job)
        @test collect(keys(typed)) == [job]
        @test sprint(show, only(typed[job])) == sprint(show, only(ct.code_typed(job)))
        warntype = sprint() do io
            GPUCompiler.@device_code_warntype io=io ct.compile_or_lookup(job)
        end
        @test occursin("reflect_vadd", warntype) && occursin("Body::Nothing", warntype)
        @test_throws ArgumentError GPUCompiler.@device_code_llvm ct.compile_or_lookup(job)

        # Nested macros restore the outer hook; child tasks inherit it, and
        # repeated jobs are printed only once in each scope.
        outer, inner = IOBuffer(), IOBuffer()
        ct.@device_code_structured io=outer begin
            ct.@device_code_structured io=inner ct.compile_or_lookup(job)
            @sync for _ in 1:2
                @async ct.compile_or_lookup(job)
            end
        end
        @test String(take!(outer)) == String(take!(inner)) != ""
        @test GPUCompiler.compile_hook[] === nothing
        @test_throws ErrorException ct.@device_code_structured error("reflection failed")
        @test GPUCompiler.compile_hook[] === nothing
    end

    @testset "concurrent cache misses" begin
        job = ct.tile_job(reflect_vadd, TT3; sm_arch=v"10.0", name="concurrent_vadd")
        ct.cached_results(job)
        entered, resume = Channel{Nothing}(1), Channel{Nothing}(1)
        late = @async with(GPUCompiler.compile_hook => (_ -> (put!(entered, nothing); take!(resume)))) do
            ct.compile_or_lookup(job)
        end
        take!(entered)
        first_res = try
            # A different configuration can compile while the first is suspended.
            other = ct.tile_job(reflect_vadd, TT3; sm_arch=v"10.0", name="concurrent_other")
            @test !isempty(ct.compile_or_lookup(other).cubin)
            ct.compile_or_lookup(job)
        finally
            put!(resume, nothing)
        end
        first_cubin = first_res.cubin
        last_res = fetch(late)
        @test last_res === first_res
        @test last_res.cubin === first_cubin
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
