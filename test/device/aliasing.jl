# Arrays that overlap in memory: each launch detects the overlap and compiles
# a kernel that keeps accesses through the overlapping arrays ordered.

using CUDA

# Stores through `dst`, then loads through `src`; when both are the same
# array, the load must observe the store.
function store_then_load(dst, src, out)
    ct.store(dst, 1, ct.load(out, 1, (16,)) .+ 1f0)
    ct.store(out, 1, ct.load(src, 1, (16,)))
    return
end

@testset "overlapping arguments stay ordered" begin
    x = CUDA.zeros(Float32, 16)
    out = CUDA.fill(5f0, 16)
    kernel = @cuda backend=cuTile store_then_load(x, x, out)
    @test Array(out) == fill(6f0, 16)
    @test kernel.opts.alias_groups == (((2, ()), (3, ())),)

    # disjoint arguments compile the kernel that assumes no overlap
    x, y = CUDA.zeros(Float32, 16), CUDA.zeros(Float32, 16)
    kernel = @cuda backend=cuTile store_then_load(x, y, out)
    @test kernel.opts.alias_groups == ()
end

@testset "overlapping arrays inside a tuple argument" begin
    function store_then_load_pair(arrays, out)
        store_then_load(arrays[1], arrays[2], out)
        return
    end
    x = CUDA.zeros(Float32, 16)
    out = CUDA.fill(5f0, 16)
    @cuda backend=cuTile store_then_load_pair((x, x), out)
    @test Array(out) == fill(6f0, 16)
end

@testset "a kernel compiled for disjoint arrays launches a variant" begin
    x, y = CUDA.zeros(Float32, 16), CUDA.zeros(Float32, 16)
    out = CUDA.fill(5f0, 16)
    kernel = @cuda backend=cuTile launch=false store_then_load(x, y, out)
    @test kernel.opts.alias_groups == ()
    conv = cuTile.cuTileconvert
    kernel(conv(x), conv(x), conv(out))
    @test Array(out) == fill(6f0, 16)

    out .= 5f0
    cuTile.launch(store_then_load, 1, x, x, out)
    @test Array(out) == fill(6f0, 16)
end
