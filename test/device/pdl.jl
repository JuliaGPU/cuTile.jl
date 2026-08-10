if ct.bytecode_version() >= v"13.4" && CUDA.capability(CUDA.device()) >= v"9.0"
    @testset "programmatic dependent launch" begin
        function producer(a::ct.TileArray{Float32,1}, out::ct.TileArray{Float32,1})
            ct.grid_dependency_control_launch_dependents()
            tile = ct.load(a, 1, (32,))
            for _ in 1:10_000
                tile = tile .+ 1f0
            end
            ct.store(out, 1, tile)
            return
        end

        function consumer(b::ct.TileArray{Float32,1}, producer_out::ct.TileArray{Float32,1},
                          out::ct.TileArray{Float32,1})
            tile = ct.load(b, 1, (32,))
            for _ in 1:10_000
                tile = tile .+ 1f0
            end
            ct.grid_dependency_control_wait()
            ct.store(out, 1, tile + ct.load(producer_out, 1, (32,)))
            return
        end

        a = CuArray(Float32.(0:31))
        b = copy(a)
        producer_out = CUDA.zeros(Float32, 32)
        consumer_out = CUDA.zeros(Float32, 32)
        stream = CUDA.stream()

        ct.launch(producer, 1, a, producer_out; stream)
        ct.launch(consumer, 1, b, producer_out, consumer_out; dependent=true, stream)
        CUDA.synchronize(stream)
        @test Array(consumer_out) == Array(a) + Array(b) .+ 20_000

        fill!(producer_out, 0)
        fill!(consumer_out, 0)
        @cuda backend=cuTile blocks=1 stream producer(a, producer_out)
        @cuda backend=cuTile blocks=1 dependent=true stream consumer(b, producer_out, consumer_out)
        CUDA.synchronize(stream)
        @test Array(consumer_out) == Array(a) + Array(b) .+ 20_000
    end
end
