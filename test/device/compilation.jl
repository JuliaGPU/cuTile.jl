using CUDA

@testset "job reflection uses launch options" begin
    function hinted_kernel(a, n)
        ct.store(a, 1, fill(1.0f0, (n,)))
        return
    end
    a = CUDA.zeros(Float32, 16)
    converted = (ct.cuTileconvert(a), ct.Constant(16))
    tt = Tuple{map(Core.Typeof, converted)...}
    job = ct.tile_job(hinted_kernel, tt; occupancy=4, name="hinted_launch")

    # Populate the launch cache, then reflect the warm launch with the same options.
    ct.cufunction(hinted_kernel, tt; occupancy=4, name="hinted_launch")
    output = IOBuffer()
    ct.@device_code_tiled io=output @cuda backend=cuTile occupancy=4 name="hinted_launch" hinted_kernel(a, ct.Constant(16))
    @test occursin(sprint(ct.code_tiled, job), String(take!(output)))
    @test Array(a) == ones(Float32, 16)
end

@testset "launches target the device" begin
    function device_kernel(a)
        ct.store(a, 1, fill(1.0f0, (16,)))
        return
    end
    a = CUDA.zeros(Float32, 16)
    tt = Tuple{Core.Typeof(ct.cuTileconvert(a))}
    device_arch = ct.device_sm_arch()
    @test ct.cufunction(device_kernel, tt) === ct.cufunction(device_kernel, tt; sm_arch=device_arch)
    # tileiras generates architecture-specific code, so other targets cannot run here.
    other_arch = device_arch == v"10.0" ? v"12.0" : v"10.0"
    @test_throws "Cannot execute" ct.cufunction(device_kernel, tt; sm_arch=other_arch)
end

@testset "linked kernels are cached per context" begin
    context_kernel() = nothing
    ctx = CUDA.context()
    job = ct.tile_job(context_kernel, Tuple{})
    res = ct.compile_or_lookup(job)
    cubin = res.cubin

    # All tasks must converge on the same linked kernel for this context.
    tasks = [Threads.@spawn CUDA.context!(ctx) do
        ct.cufunction(context_kernel)
    end for _ in 1:8]
    kernels = fetch.(tasks)
    kernel = first(kernels)
    @test all(k -> k === kernel, kernels)
    @test ct.cufunction(context_kernel) === kernel
    kernel()
    CUDA.synchronize()

    exclusive = CUDA.attribute(CUDA.device(), CUDA.DEVICE_ATTRIBUTE_COMPUTE_MODE) ==
                CUDA.COMPUTEMODE_EXCLUSIVE_PROCESS
    if !exclusive
        CUDA.CuContext(CUDA.device()) do other_ctx
            CUDA.context!(other_ctx) do
                other_kernel = ct.cufunction(context_kernel)
                @test other_kernel !== kernel
                @test ct.cufunction(context_kernel) === other_kernel
                stream = CUDA.CuStream()
                try
                    other_kernel(; stream)
                    CUDA.synchronize(stream)
                finally
                    finalize(stream)
                    finalize(other_kernel.fun.mod)
                end
            end
        end
        @test ct.cufunction(context_kernel) === kernel
        @test ct.compile_or_lookup(job).cubin === cubin
    end
end

@testset "mixed-backend reflection" begin
    function tile_kernel(a)
        ct.store(a, 1, fill(1.0f0, (16,)))
        return
    end
    function cuda_kernel(a)
        a[1] = 2.0f0
        return
    end
    a = CUDA.zeros(Float32, 16)

    # Warm both launch caches before inspecting the same expression.
    @cuda cuda_kernel(a)
    @cuda backend=cuTile tile_kernel(a)
    typed = CUDA.@device_code_typed begin
        @cuda cuda_kernel(a)
        @cuda backend=cuTile tile_kernel(a)
    end
    @test length(typed) == 2
    @test count(job -> job isa ct.TileJob, keys(typed)) == 1
    @test count(job -> job isa GPUCompiler.CompilerJob, keys(typed)) == 1

    warntype = sprint() do io
        CUDA.@device_code_warntype io=io begin
            @cuda cuda_kernel(a)
            @cuda backend=cuTile tile_kernel(a)
        end
    end
    @test occursin("cuda_kernel", warntype)
    @test occursin("tile_kernel", warntype)

    ptx = sprint() do io
        CUDA.@device_code_ptx io=io begin
            @cuda cuda_kernel(a)
            @cuda backend=cuTile tile_kernel(a)
        end
    end
    @test occursin("cuda_kernel", ptx)
    @test occursin("tile_kernel", ptx)

    tiled = sprint() do io
        ct.@device_code_tiled io=io begin
            @cuda cuda_kernel(a)
            @cuda backend=cuTile tile_kernel(a)
        end
    end
    @test occursin("tile_kernel", tiled)
    @test !occursin("cuda_kernel", tiled)
    @test_throws "no kernels executed" ct.@device_code_tiled @cuda cuda_kernel(a)
    @test GPUCompiler.compile_hook[] === nothing
    CUDA.synchronize()
end
