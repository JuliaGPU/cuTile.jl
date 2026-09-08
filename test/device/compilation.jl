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
