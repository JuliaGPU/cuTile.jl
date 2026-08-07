struct WideDeviceVector{T, A<:CuArray{T, 1}} <: AbstractVector{T}
    parent::A
    len::Int64
end
Base.size(a::WideDeviceVector) = (a.len,)
Base.strides(::WideDeviceVector) = (Int64(1),)
Base.pointer(a::WideDeviceVector) = pointer(a.parent)
Base.getindex(::WideDeviceVector, ::Int) = error("not implemented")

@testset "Int64 indexing" begin
    function copy_wide_prefix(src::ct.TileArray{Float32, 1},
                              dst::ct.TileArray{Float32, 1})
        ct.store(dst, 1, ct.load(src, 1, (16,)))
        return
    end

    src = CUDA.rand(Float32, 16)
    wide = WideDeviceVector(src, Int64(typemax(Int32)) + 1)
    dst = CUDA.zeros(Float32, 16)

    @test ct.indextype(ct.TileArray(wide)) === Int64
    @cuda backend=cuTile blocks=1 copy_wide_prefix(wide, dst)
    @test Array(dst) == Array(src)

    fill!(dst, 0)
    src64 = ct.TileArray(src; index=Int64)
    dst64 = ct.TileArray(dst; index=Int64)
    @test ct.indextype(src64) === ct.indextype(dst64) === Int64
    @cuda backend=cuTile blocks=1 copy_wide_prefix(src64, dst64)
    @test Array(dst) == Array(src)
end
