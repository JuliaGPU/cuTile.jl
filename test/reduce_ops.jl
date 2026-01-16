
using cuTile
import cuTile as ct
using CUDA
using Test

# Kernel factory to properly capture element type and operation
function makeReduceKernel(::Type{T}, op::Symbol) where {T}
    reduceFunc = if op == :reduce_min
        ct.reduce_min
    elseif op == :reduce_max
        ct.reduce_max
    elseif op == :reduce_sum
        ct.reduce_sum
    elseif op == :reduce_xor
        ct.reduce_xor
    elseif op == :reduce_or
        ct.reduce_or
    elseif op == :reduce_and
        ct.reduce_and
    end

    @inline function kernel(a::ct.TileArray{T,1}, b::ct.TileArray{T,1}, tileSz::ct.Constant{Int})
        ct.store(b, ct.bid(1), reduceFunc(ct.load(a, ct.bid(1), (tileSz[],)), Val(1)))
        return nothing
    end
    return kernel
end

# Test with UInt types
@testset for elType in [UInt16, UInt32, UInt64]
    @testset for op in [:reduce_min, :reduce_max, :reduce_sum, :reduce_xor, :reduce_or, :reduce_and]
        sz = 32
        N = 2^15

        # Create kernel using factory
        reduceKernel = try
            makeReduceKernel(elType, op)
        catch e
            @test_broken false
            rethrow()
        end

        # Create data and run kernel
        a_gpu = CUDA.rand(elType, N)
        b_gpu = CUDA.zeros(elType, cld(N, sz))
        try
            CUDA.@sync ct.launch(reduceKernel, cld(length(a_gpu), sz), a_gpu, b_gpu, ct.Constant(sz))
        catch e
            @test_broken false
            rethrow()
        end
        res = Array(b_gpu)

        # CPU computation
        a_cpu = Array(a_gpu)
        a_reshaped = reshape(a_cpu, sz, :)

        if op == :reduce_min
            cpu_result = minimum(a_reshaped, dims=1)[:]
        elseif op == :reduce_max
            cpu_result = maximum(a_reshaped, dims=1)[:]
        elseif op == :reduce_sum
            raw_sum = sum(a_reshaped, dims=1)[:]
            cpu_result = raw_sum .& typemax(elType)
        elseif op == :reduce_xor
            cpu_result = mapslices(x -> reduce(⊻, x), a_reshaped, dims=1)[:]
        elseif op == :reduce_or
            cpu_result = mapslices(x -> reduce(|, x), a_reshaped, dims=1)[:]
        elseif op == :reduce_and
            cpu_result = mapslices(x -> reduce(&, x), a_reshaped, dims=1)[:]
        end

        @test cpu_result == res
    end
end

# Test with signed Int types
@testset for elType in [Int16, Int32, Int64]
    @testset for op in [:reduce_min, :reduce_max, :reduce_sum, :reduce_xor, :reduce_or, :reduce_and]
        sz = 32
        N = 2^15

        # Create kernel using factory
        reduceKernel = try
            makeReduceKernel(elType, op)
        catch e
            @test_broken false
            rethrow()
        end

        # Create data and run kernel - use range to get negative values too
        a_gpu = CuArray{elType}(rand(-1000:1000, N))
        b_gpu = CUDA.zeros(elType, cld(N, sz))
        try
            CUDA.@sync ct.launch(reduceKernel, cld(length(a_gpu), sz), a_gpu, b_gpu, ct.Constant(sz))
        catch e
            @test_broken false
            rethrow()
        end
        res = Array(b_gpu)

        # CPU computation
        a_cpu = Array(a_gpu)
        a_reshaped = reshape(a_cpu, sz, :)

        if op == :reduce_min
            cpu_result = minimum(a_reshaped, dims=1)[:]
        elseif op == :reduce_max
            cpu_result = maximum(a_reshaped, dims=1)[:]
        elseif op == :reduce_sum
            cpu_result = sum(a_reshaped, dims=1)[:]
        elseif op == :reduce_xor
            cpu_result = mapslices(x -> reduce(⊻, x), a_reshaped, dims=1)[:]
        elseif op == :reduce_or
            cpu_result = mapslices(x -> reduce(|, x), a_reshaped, dims=1)[:]
        elseif op == :reduce_and
            cpu_result = mapslices(x -> reduce(&, x), a_reshaped, dims=1)[:]
        end

        @test cpu_result == res
    end
end

# Test with Float types
@testset for elType in [Float16, Float32, Float64]
    @testset for op in [:reduce_min, :reduce_max, :reduce_sum]
        sz = 32
        N = 2^15

        # Create kernel using factory
        reduceKernel = makeReduceKernel(elType, op)

        # Create data and run kernel
        a_gpu = CUDA.rand(elType, N)
        b_gpu = CUDA.zeros(elType, cld(N, sz))
        CUDA.@sync ct.launch(reduceKernel, cld(length(a_gpu), sz), a_gpu, b_gpu, ct.Constant(sz))
        res = Array(b_gpu)

        # CPU computation
        a_cpu = Array(a_gpu)
        a_reshaped = reshape(a_cpu, sz, :)

        if op == :reduce_min
            cpu_result = minimum(a_reshaped, dims=1)[:]
        elseif op == :reduce_max
            cpu_result = maximum(a_reshaped, dims=1)[:]
        elseif op == :reduce_sum
            cpu_result = sum(a_reshaped, dims=1)[:]
        end

        @test isapprox(cpu_result, res)
    end
end
