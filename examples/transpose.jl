using CUDA
import cuTile as ct

function transpose_kernel(x::Ptr{Float32}, y::Ptr{Float32})
    bidx = ct.bid(0)
    bidy = ct.bid(1)

    input_tile = ct.load(x, (bidx, bidy), (32, 32))
    transposed_tile = ct.transpose(input_tile)
    ct.store(y, (bidy, bidx), transposed_tile)

    return
end

function main()
    # Matrix dimensions
    m, n = 1024, 512
    tm, tn = 32, 32

    # compile
    argtypes = Tuple{Ptr{Float32}, Ptr{Float32}}
    cubin = ct.compile(transpose_kernel, argtypes; sm_arch="sm_120")
    cumod = CuModule(cubin)
    cufunc = CuFunction(cumod, "transpose_kernel")

    # Create test data
    x = CUDA.rand(Float32, m, n)
    y = CUDA.zeros(Float32, n, m)

    grid_x = cld(m, tm)
    grid_y = cld(n, tn)
    cudacall(cufunc,
             Tuple{CuPtr{Float32}, CuPtr{Float32}},
             x, y;
             blocks=(grid_x, grid_y))

    @assert Array(y) == transpose(Array(x))
    println("✓ transpose example passed!")
end

isinteractive() || main()
