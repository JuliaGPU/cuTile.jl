using CUDA
import cuTile as ct

function vadd_kernel(a::Ptr{Float32}, b::Ptr{Float32}, c::Ptr{Float32})
    pid = ct.bid(0)

    tile_a = ct.load(a, pid, (16,))
    tile_b = ct.load(b, pid, (16,))
    result = tile_a + tile_b
    ct.store(c, pid, result)

    return
end

function main(; vector_size = 2^12, tile_size = 16)
    # Compile
    argtypes = Tuple{Ptr{Float32}, Ptr{Float32}, Ptr{Float32}}
    cubin = ct.compile(vadd_kernel, argtypes; sm_arch="sm_120")
    cumod = CuModule(cubin)
    cufunc = CuFunction(cumod, "vadd_kernel")

    # Create test data
    grid_size = cld(vector_size, tile_size)
    a = CUDA.rand(Float32, vector_size)
    b = CUDA.rand(Float32, vector_size)
    c = CUDA.zeros(Float32, vector_size)

    cudacall(cufunc,
             Tuple{CuPtr{Float32}, CuPtr{Float32}, CuPtr{Float32}},
             a, b, c;
             blocks=grid_size)

    @assert Array(a) + Array(b) ≈ Array(c)
    println("✓ vadd example passed!")
end

isinteractive() || main()
