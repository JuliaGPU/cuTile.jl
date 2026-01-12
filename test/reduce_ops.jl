using cuTile
import cuTile as ct
using CUDA
using Test

@testset "reduce operations" begin

#======================================================================#
# CPU reference implementations
# =====================================================================#

cpu_reduce_add(a::AbstractArray, dims::Integer) = sum(a, dims=dims)
cpu_reduce_mul(a::AbstractArray, dims::Integer) = prod(a, dims=dims)
cpu_reduce_max(a::AbstractArray, dims::Integer) = maximum(a, dims=dims)
cpu_reduce_min(a::AbstractArray, dims::Integer) = minimum(a, dims=dims)

cpu_reduce_and(a::AbstractArray{<:Unsigned}, dims::Integer) = reduce((x, y) -> x & y, a, init=typemax(eltype(a)), dims=dims)
cpu_reduce_and(a::AbstractArray{<:Signed}, dims::Integer) = reduce((x, y) -> x & y, a, init=Int64(-1), dims=dims)
cpu_reduce_or(a::AbstractArray{<:Integer}, dims::Integer) = reduce((x, y) -> x | y, a, init=zero(eltype(a)), dims=dims)
cpu_reduce_xor(a::AbstractArray{<:Integer}, dims::Integer) = reduce((x, y) -> x ⊻ y, a, init=zero(eltype(a)), dims=dims)

#======================================================================#
# Float32 operations
#======================================================================#

@testset "Float32 reduce_add" begin
    function reduce_add_kernel(a::ct.TileArray{Float32,2}, b::ct.TileArray{Float32,1})
        pid = ct.bid(1)
        tile = ct.load(a, (pid, 1), (1, 128))
        sums = ct.reduce_sum(tile, 2)
        ct.store(b, pid, sums)
        return
    end

    m, n = 64, 128
    a = CUDA.rand(Float32, m, n)
    b = CUDA.zeros(Float32, m)

    ct.launch(reduce_add_kernel, m, a, b)

    a_cpu = Array(a)
    b_cpu = Array(b)
    for i in 1:m
        @test b_cpu[i] ≈ cpu_reduce_add(a_cpu[i:i, :], 2)[1] rtol=1e-3
    end
end

@testset "Float32 reduce_mul" begin
    function reduce_mul_kernel(a::ct.TileArray{Float32,2}, b::ct.TileArray{Float32,1})
        pid = ct.bid(1)
        tile = ct.load(a, (pid, 1), (1, 32))
        products = ct.reduce_mul(tile, 2)
        ct.store(b, pid, products)
        return
    end

    m, n = 32, 64
    a = CUDA.rand(Float32, m, n) .+ 0.1f0
    b = CUDA.ones(Float32, m)

    ct.launch(reduce_mul_kernel, m, a, b)

    a_cpu = Array(a)
    b_cpu = Array(b)
    for i in 1:m
        @test b_cpu[i] ≈ cpu_reduce_mul(a_cpu[i:i, :], 2)[1] rtol=1e-2
    end
end

@testset "Float32 reduce_max" begin
    function reduce_max_kernel(a::ct.TileArray{Float32,2}, b::ct.TileArray{Float32,1})
        pid = ct.bid(1)
        tile = ct.load(a, (pid, 1), (1, 128))
        maxes = ct.reduce_max(tile, 2)
        ct.store(b, pid, maxes)
        return
    end

    m, n = 64, 128
    a = CUDA.rand(Float32, m, n)
    b = CUDA.zeros(Float32, m)

    ct.launch(reduce_max_kernel, m, a, b)

    a_cpu = Array(a)
    b_cpu = Array(b)
    for i in 1:m
        @test b_cpu[i] ≈ cpu_reduce_max(a_cpu[i:i, :], 2)[1] rtol=1e-5
    end
end

@testset "Float32 reduce_min" begin
    function reduce_min_kernel(a::ct.TileArray{Float32,2}, b::ct.TileArray{Float32,1})
        pid = ct.bid(1)
        tile = ct.load(a, (pid, 1), (1, 128))
        mins = ct.reduce_min(tile, 2)
        ct.store(b, pid, mins)
        return
    end

    m, n = 64, 128
    a = CUDA.rand(Float32, m, n)
    b = CUDA.zeros(Float32, m)

    ct.launch(reduce_min_kernel, m, a, b)

    a_cpu = Array(a)
    b_cpu = Array(b)
    for i in 1:m
        @test b_cpu[i] ≈ cpu_reduce_min(a_cpu[i:i, :], 2)[1] rtol=1e-5
    end
end

#======================================================================#
# Float64 operations
#======================================================================#

@testset "Float64 reduce_add" begin
    function reduce_add_f64_kernel(a::ct.TileArray{Float64,2}, b::ct.TileArray{Float64,1})
        pid = ct.bid(1)
        tile = ct.load(a, (pid, 1), (1, 64))
        sums = ct.reduce_sum(tile, 2)
        ct.store(b, pid, sums)
        return
    end

    m, n = 32, 64
    a = CUDA.rand(Float64, m, n)
    b = CUDA.zeros(Float64, m)

    ct.launch(reduce_add_f64_kernel, m, a, b)

    a_cpu = Array(a)
    b_cpu = Array(b)
    for i in 1:m
        @test b_cpu[i] ≈ cpu_reduce_add(a_cpu[i:i, :], 2)[1] rtol=1e-5
    end
end

@testset "Float64 reduce_max" begin
    function reduce_max_f64_kernel(a::ct.TileArray{Float64,2}, b::ct.TileArray{Float64,1})
        pid = ct.bid(1)
        tile = ct.load(a, (pid, 1), (1, 64))
        maxes = ct.reduce_max(tile, 2)
        ct.store(b, pid, maxes)
        return
    end

    m, n = 32, 64
    a = CUDA.rand(Float64, m, n)
    b = CUDA.zeros(Float64, m)

    ct.launch(reduce_max_f64_kernel, m, a, b)

    a_cpu = Array(a)
    b_cpu = Array(b)
    for i in 1:m
        @test b_cpu[i] ≈ cpu_reduce_max(a_cpu[i:i, :], 2)[1] rtol=1e-5
    end
end

@testset "Float64 reduce_min" begin
    function reduce_min_f64_kernel(a::ct.TileArray{Float64,2}, b::ct.TileArray{Float64,1})
        pid = ct.bid(1)
        tile = ct.load(a, (pid, 1), (1, 64))
        mins = ct.reduce_min(tile, 2)
        ct.store(b, pid, mins)
        return
    end

    m, n = 32, 64
    a = CUDA.rand(Float64, m, n)
    b = CUDA.zeros(Float64, m)

    ct.launch(reduce_min_f64_kernel, m, a, b)

    a_cpu = Array(a)
    b_cpu = Array(b)
    for i in 1:m
        @test b_cpu[i] ≈ cpu_reduce_min(a_cpu[i:i, :], 2)[1] rtol=1e-5
    end
end

@testset "Float64 reduce_mul" begin
    function reduce_mul_f64_kernel(a::ct.TileArray{Float64,2}, b::ct.TileArray{Float64,1})
        pid = ct.bid(1)
        tile = ct.load(a, (pid, 1), (1, 32))
        products = ct.reduce_mul(tile, 2)
        ct.store(b, pid, products)
        return
    end

    m, n = 16, 32
    a = CUDA.rand(Float64, m, n) .+ 0.1
    b = CUDA.ones(Float64, m)

    ct.launch(reduce_mul_f64_kernel, m, a, b)

    a_cpu = Array(a)
    b_cpu = Array(b)
    for i in 1:m
        @test b_cpu[i] ≈ cpu_reduce_mul(a_cpu[i:i, :], 2)[1] rtol=1e-2
    end
end

#======================================================================#
# Int32 operations
#======================================================================#

@testset "Int32 reduce_add" begin
    function reduce_add_i32_kernel(a::ct.TileArray{Int32,2}, b::ct.TileArray{Int32,1})
        pid = ct.bid(1)
        tile = ct.load(a, (pid, 1), (1, 64))
        sums = ct.reduce_sum(tile, 2)
        ct.store(b, pid, sums)
        return
    end

    m, n = 32, 64
    a = CUDA.rand(Int32, m, n) .+ 1
    b = CUDA.zeros(Int32, m)

    ct.launch(reduce_add_i32_kernel, m, a, b)

    a_cpu = Array(a)
    b_cpu = Array(b)
    for i in 1:m
        @test b_cpu[i] == cpu_reduce_add(a_cpu[i:i, :], 2)[1]
    end
end

@testset "Int32 reduce_mul" begin
    function reduce_mul_i32_kernel(a::ct.TileArray{Int32,2}, b::ct.TileArray{Int32,1})
        pid = ct.bid(1)
        tile = ct.load(a, (pid, 1), (1, 16))
        products = ct.reduce_mul(tile, 2)
        ct.store(b, pid, products)
        return
    end

    m, n = 8, 16
    a = CUDA.rand(Int32, m, n) .% 10 .+ 2
    b = CUDA.ones(Int32, m)

    ct.launch(reduce_mul_i32_kernel, m, a, b)

    a_cpu = Array(a)
    b_cpu = Array(b)
    for i in 1:m
        @test b_cpu[i] == cpu_reduce_mul(a_cpu[i:i, :], 2)[1]
    end
end

@testset "Int32 reduce_max" begin
    function reduce_max_i32_kernel(a::ct.TileArray{Int32,2}, b::ct.TileArray{Int32,1})
        pid = ct.bid(1)
        tile = ct.load(a, (pid, 1), (1, 64))
        maxes = ct.reduce_max(tile, 2)
        ct.store(b, pid, maxes)
        return
    end

    m, n = 32, 64
    a = CUDA.rand(Int32, m, n)
    b = CUDA.fill(typemin(Int32), m)

    ct.launch(reduce_max_i32_kernel, m, a, b)

    a_cpu = Array(a)
    b_cpu = Array(b)
    for i in 1:m
        @test b_cpu[i] == cpu_reduce_max(a_cpu[i:i, :], 2)[1]
    end
end

@testset "Int32 reduce_min" begin
    function reduce_min_i32_kernel(a::ct.TileArray{Int32,2}, b::ct.TileArray{Int32,1})
        pid = ct.bid(1)
        tile = ct.load(a, (pid, 1), (1, 64))
        mins = ct.reduce_min(tile, 2)
        ct.store(b, pid, mins)
        return
    end

    m, n = 32, 64
    a = CUDA.rand(Int32, m, n)
    b = CUDA.fill(typemax(Int32), m)

    ct.launch(reduce_min_i32_kernel, m, a, b)

    a_cpu = Array(a)
    b_cpu = Array(b)
    for i in 1:m
        @test b_cpu[i] == cpu_reduce_min(a_cpu[i:i, :], 2)[1]
    end
end

@testset "Int32 reduce_and" begin
    function reduce_and_i32_kernel(a::ct.TileArray{Int32,2}, b::ct.TileArray{Int32,1})
        pid = ct.bid(1)
        tile = ct.load(a, (pid, 1), (1, 32))
        result = ct.reduce_and(tile, 2)
        ct.store(b, pid, result)
        return
    end

    m, n = 16, 32
    a = CUDA.rand(Int32, m, n)
    b = CUDA.zeros(Int32, m)

    ct.launch(reduce_and_i32_kernel, m, a, b)

    a_cpu = Array(a)
    b_cpu = Array(b)
    for i in 1:m
        @test b_cpu[i] == cpu_reduce_and(a_cpu[i:i, :], 2)[1]
    end
end

@testset "Int32 reduce_or" begin
    function reduce_or_i32_kernel(a::ct.TileArray{Int32,2}, b::ct.TileArray{Int32,1})
        pid = ct.bid(1)
        tile = ct.load(a, (pid, 1), (1, 32))
        result = ct.reduce_or(tile, 2)
        ct.store(b, pid, result)
        return
    end

    m, n = 16, 32
    a = CUDA.rand(Int32, m, n)
    b = CUDA.zeros(Int32, m)

    ct.launch(reduce_or_i32_kernel, m, a, b)

    a_cpu = Array(a)
    b_cpu = Array(b)
    for i in 1:m
        @test b_cpu[i] == cpu_reduce_or(a_cpu[i:i, :], 2)[1]
    end
end

@testset "Int32 reduce_xor" begin
    function reduce_xor_i32_kernel(a::ct.TileArray{Int32,2}, b::ct.TileArray{Int32,1})
        pid = ct.bid(1)
        tile = ct.load(a, (pid, 1), (1, 32))
        result = ct.reduce_xor(tile, 2)
        ct.store(b, pid, result)
        return
    end

    m, n = 16, 32
    a = CUDA.rand(Int32, m, n)
    b = CUDA.zeros(Int32, m)

    ct.launch(reduce_xor_i32_kernel, m, a, b)

    a_cpu = Array(a)
    b_cpu = Array(b)
    for i in 1:m
        @test b_cpu[i] == cpu_reduce_xor(a_cpu[i:i, :], 2)[1]
    end
end

#======================================================================#
# UInt32 operations - tests AND identity encoding fix
#======================================================================#

@testset "UInt32 reduce_add" begin
    function reduce_add_u32_kernel(a::ct.TileArray{UInt32,2}, b::ct.TileArray{UInt32,1})
        pid = ct.bid(1)
        tile = ct.load(a, (pid, 1), (1, 64))
        sums = ct.reduce_sum(tile, 2)
        ct.store(b, pid, sums)
        return
    end

    m, n = 32, 64
    a = CUDA.rand(UInt32, m, n)
    b = CUDA.zeros(UInt32, m)

    ct.launch(reduce_add_u32_kernel, m, a, b)

    a_cpu = Array(a)
    b_cpu = Array(b)
    for i in 1:m
        @test b_cpu[i] == cpu_reduce_add(a_cpu[i:i, :], 2)[1]
    end
end

@testset "UInt32 reduce_mul" begin
    function reduce_mul_u32_kernel(a::ct.TileArray{UInt32,2}, b::ct.TileArray{UInt32,1})
        pid = ct.bid(1)
        tile = ct.load(a, (pid, 1), (1, 16))
        products = ct.reduce_mul(tile, 2)
        ct.store(b, pid, products)
        return
    end

    m, n = 8, 16
    a = CUDA.rand(UInt32, m, n) .% 10 .+ 2
    b = CUDA.ones(UInt32, m)

    ct.launch(reduce_mul_u32_kernel, m, a, b)

    a_cpu = Array(a)
    b_cpu = Array(b)
    for i in 1:m
        @test b_cpu[i] == cpu_reduce_mul(a_cpu[i:i, :], 2)[1]
    end
end

@testset "UInt32 reduce_max" begin
    function reduce_max_u32_kernel(a::ct.TileArray{UInt32,2}, b::ct.TileArray{UInt32,1})
        pid = ct.bid(1)
        tile = ct.load(a, (pid, 1), (1, 64))
        maxes = ct.reduce_max(tile, 2)
        ct.store(b, pid, maxes)
        return
    end

    m, n = 32, 64
    a = CUDA.rand(UInt32, m, n)
    b = CUDA.zeros(UInt32, m)

    ct.launch(reduce_max_u32_kernel, m, a, b)

    a_cpu = Array(a)
    b_cpu = Array(b)
    for i in 1:m
        @test b_cpu[i] == cpu_reduce_max(a_cpu[i:i, :], 2)[1]
    end
end

@testset "UInt32 reduce_min" begin
    function reduce_min_u32_kernel(a::ct.TileArray{UInt32,2}, b::ct.TileArray{UInt32,1})
        pid = ct.bid(1)
        tile = ct.load(a, (pid, 1), (1, 64))
        mins = ct.reduce_min(tile, 2)
        ct.store(b, pid, mins)
        return
    end

    m, n = 32, 64
    a = CUDA.rand(UInt32, m, n)
    b = CUDA.fill(typemax(UInt32), m)

    ct.launch(reduce_min_u32_kernel, m, a, b)

    a_cpu = Array(a)
    b_cpu = Array(b)
    for i in 1:m
        @test b_cpu[i] == cpu_reduce_min(a_cpu[i:i, :], 2)[1]
    end
end

@testset "UInt32 reduce_and" begin
    function reduce_and_u32_kernel(a::ct.TileArray{UInt32,2}, b::ct.TileArray{UInt32,1})
        pid = ct.bid(1)
        tile = ct.load(a, (pid, 1), (1, 32))
        result = ct.reduce_and(tile, 2)
        ct.store(b, pid, result)
        return
    end

    m, n = 16, 32
    a = CUDA.rand(UInt32, m, n)
    b = CUDA.zeros(UInt32, m)

    ct.launch(reduce_and_u32_kernel, m, a, b)

    a_cpu = Array(a)
    b_cpu = Array(b)
    for i in 1:m
        @test b_cpu[i] == cpu_reduce_and(a_cpu[i:i, :], 2)[1]
    end
end

@testset "UInt32 reduce_or" begin
    function reduce_or_u32_kernel(a::ct.TileArray{UInt32,2}, b::ct.TileArray{UInt32,1})
        pid = ct.bid(1)
        tile = ct.load(a, (pid, 1), (1, 32))
        result = ct.reduce_or(tile, 2)
        ct.store(b, pid, result)
        return
    end

    m, n = 16, 32
    a = CUDA.rand(UInt32, m, n)
    b = CUDA.zeros(UInt32, m)

    ct.launch(reduce_or_u32_kernel, m, a, b)

    a_cpu = Array(a)
    b_cpu = Array(b)
    for i in 1:m
        @test b_cpu[i] == cpu_reduce_or(a_cpu[i:i, :], 2)[1]
    end
end

@testset "UInt32 reduce_xor" begin
    function reduce_xor_u32_kernel(a::ct.TileArray{UInt32,2}, b::ct.TileArray{UInt32,1})
        pid = ct.bid(1)
        tile = ct.load(a, (pid, 1), (1, 32))
        result = ct.reduce_xor(tile, 2)
        ct.store(b, pid, result)
        return
    end

    m, n = 16, 32
    a = CUDA.rand(UInt32, m, n)
    b = CUDA.zeros(UInt32, m)

    ct.launch(reduce_xor_u32_kernel, m, a, b)

    a_cpu = Array(a)
    b_cpu = Array(b)
    for i in 1:m
        @test b_cpu[i] == cpu_reduce_xor(a_cpu[i:i, :], 2)[1]
    end
end

#======================================================================#
# Int8 operations - smaller integer type for encoding tests
#======================================================================#

@testset "Int8 reduce_add" begin
    function reduce_add_i8_kernel(a::ct.TileArray{Int8,2}, b::ct.TileArray{Int8,1})
        pid = ct.bid(1)
        tile = ct.load(a, (pid, 1), (1, 32))
        sums = ct.reduce_sum(tile, 2)
        ct.store(b, pid, sums)
        return
    end

    m, n = 16, 32
    a = CUDA.rand(Int8, m, n)
    b = CUDA.zeros(Int8, m)

    ct.launch(reduce_add_i8_kernel, m, a, b)

    a_cpu = Array(a)
    b_cpu = Array(b)
    for i in 1:m
        @test Int32(b_cpu[i]) == cpu_reduce_add(a_cpu[i:i, :], 2)[1]
    end
end

@testset "Int8 reduce_max" begin
    function reduce_max_i8_kernel(a::ct.TileArray{Int8,2}, b::ct.TileArray{Int8,1})
        pid = ct.bid(1)
        tile = ct.load(a, (pid, 1), (1, 32))
        maxes = ct.reduce_max(tile, 2)
        ct.store(b, pid, maxes)
        return
    end

    m, n = 16, 32
    a = CUDA.rand(Int8, m, n)
    b = CUDA.fill(typemin(Int8), m)

    ct.launch(reduce_max_i8_kernel, m, a, b)

    a_cpu = Array(a)
    b_cpu = Array(b)
    for i in 1:m
        @test b_cpu[i] == cpu_reduce_max(a_cpu[i:i, :], 2)[1]
    end
end

@testset "Int8 reduce_min" begin
    function reduce_min_i8_kernel(a::ct.TileArray{Int8,2}, b::ct.TileArray{Int8,1})
        pid = ct.bid(1)
        tile = ct.load(a, (pid, 1), (1, 32))
        mins = ct.reduce_min(tile, 2)
        ct.store(b, pid, mins)
        return
    end

    m, n = 16, 32
    a = CUDA.rand(Int8, m, n)
    b = CUDA.fill(typemax(Int8), m)

    ct.launch(reduce_min_i8_kernel, m, a, b)

    a_cpu = Array(a)
    b_cpu = Array(b)
    for i in 1:m
        @test b_cpu[i] == cpu_reduce_min(a_cpu[i:i, :], 2)[1]
    end
end

@testset "Int8 reduce_and" begin
    function reduce_and_i8_kernel(a::ct.TileArray{Int8,2}, b::ct.TileArray{Int8,1})
        pid = ct.bid(1)
        tile = ct.load(a, (pid, 1), (1, 16))
        result = ct.reduce_and(tile, 2)
        ct.store(b, pid, result)
        return
    end

    m, n = 8, 16
    a = CUDA.rand(Int8, m, n)
    b = CUDA.zeros(Int8, m)

    ct.launch(reduce_and_i8_kernel, m, a, b)

    a_cpu = Array(a)
    b_cpu = Array(b)
    for i in 1:m
        @test Int32(b_cpu[i]) == cpu_reduce_and(a_cpu[i:i, :], 2)[1]
    end
end

@testset "Int8 reduce_or" begin
    function reduce_or_i8_kernel(a::ct.TileArray{Int8,2}, b::ct.TileArray{Int8,1})
        pid = ct.bid(1)
        tile = ct.load(a, (pid, 1), (1, 16))
        result = ct.reduce_or(tile, 2)
        ct.store(b, pid, result)
        return
    end

    m, n = 8, 16
    a = CUDA.rand(Int8, m, n)
    b = CUDA.zeros(Int8, m)

    ct.launch(reduce_or_i8_kernel, m, a, b)

    a_cpu = Array(a)
    b_cpu = Array(b)
    for i in 1:m
        @test Int32(b_cpu[i]) == cpu_reduce_or(a_cpu[i:i, :], 2)[1]
    end
end

@testset "Int8 reduce_xor" begin
    function reduce_xor_i8_kernel(a::ct.TileArray{Int8,2}, b::ct.TileArray{Int8,1})
        pid = ct.bid(1)
        tile = ct.load(a, (pid, 1), (1, 16))
        result = ct.reduce_xor(tile, 2)
        ct.store(b, pid, result)
        return
    end

    m, n = 8, 16
    a = CUDA.rand(Int8, m, n)
    b = CUDA.zeros(Int8, m)

    ct.launch(reduce_xor_i8_kernel, m, a, b)

    a_cpu = Array(a)
    b_cpu = Array(b)
    for i in 1:m
        @test Int32(b_cpu[i]) == cpu_reduce_xor(a_cpu[i:i, :], 2)[1]
    end
end

#======================================================================#
# Axis 0 reductions - verify both axes work
#======================================================================#

@testset "axis 0 reduce_sum Float32" begin
    function reduce_sum_axis0_kernel(a::ct.TileArray{Float32,2}, b::ct.TileArray{Float32,1})
        pid = ct.bid(1)
        tile = ct.load(a, (1, pid), (64, 1))
        sums = ct.reduce_sum(tile, 1)
        ct.store(b, pid, sums)
        return
    end

    m, n = 64, 128
    a = CUDA.rand(Float32, m, n)
    b = CUDA.zeros(Float32, n)

    ct.launch(reduce_sum_axis0_kernel, n, a, b)

    a_cpu = Array(a)
    b_cpu = Array(b)
    for j in 1:n
        @test b_cpu[j] ≈ cpu_reduce_add(a_cpu[:, j:j], 1)[1] rtol=1e-3
    end
end

@testset "axis 0 reduce_min Int32" begin
    function reduce_min_axis0_i32_kernel(a::ct.TileArray{Int32,2}, b::ct.TileArray{Int32,1})
        pid = ct.bid(1)
        tile = ct.load(a, (1, pid), (32, 1))
        mins = ct.reduce_min(tile, 1)
        ct.store(b, pid, mins)
        return
    end

    m, n = 32, 64
    a = CUDA.rand(Int32, m, n)
    b = CUDA.fill(typemax(Int32), n)

    ct.launch(reduce_min_axis0_i32_kernel, n, a, b)

    a_cpu = Array(a)
    b_cpu = Array(b)
    for j in 1:n
        @test b_cpu[j] == cpu_reduce_min(a_cpu[:, j:j], 1)[1]
    end
end

@testset "axis 0 reduce_max UInt32" begin
    function reduce_max_axis0_u32_kernel(a::ct.TileArray{UInt32,2}, b::ct.TileArray{UInt32,1})
        pid = ct.bid(1)
        tile = ct.load(a, (1, pid), (32, 1))
        maxes = ct.reduce_max(tile, 1)
        ct.store(b, pid, maxes)
        return
    end

    m, n = 32, 64
    a = CUDA.rand(UInt32, m, n)
    b = CUDA.zeros(UInt32, n)

    ct.launch(reduce_max_axis0_u32_kernel, n, a, b)

    a_cpu = Array(a)
    b_cpu = Array(b)
    for j in 1:n
        @test b_cpu[j] == cpu_reduce_max(a_cpu[:, j:j], 1)[1]
    end
end

@testset "axis 0 reduce_and UInt32" begin
    function reduce_and_axis0_u32_kernel(a::ct.TileArray{UInt32,2}, b::ct.TileArray{UInt32,1})
        pid = ct.bid(1)
        tile = ct.load(a, (1, pid), (16, 1))
        result = ct.reduce_and(tile, 1)
        ct.store(b, pid, result)
        return
    end

    m, n = 16, 32
    a = CUDA.rand(UInt32, m, n)
    b = CUDA.fill(typemax(UInt32), n)

    ct.launch(reduce_and_axis0_u32_kernel, n, a, b)

    a_cpu = Array(a)
    b_cpu = Array(b)
    for j in 1:n
        @test b_cpu[j] == cpu_reduce_and(a_cpu[:, j:j], 1)[1]
    end
end

end  # @testset "reduce operations"
