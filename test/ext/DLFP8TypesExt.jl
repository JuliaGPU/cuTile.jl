using DLFP8Types: Float8_E4M3FN, Float8_E5M2

@testset "DLFP8Types extension" begin

spec1d = ct.ArraySpec{1}(16, true)

# Float32 -> Float8_E4M3FN
@test @filecheck begin
    @check_label "entry"
    code_tiled(Tuple{ct.TileArray{Float32,1,spec1d}, ct.TileArray{Float32,1,spec1d}}) do a, b
        pid = ct.bid(1)
        tile = ct.load(a, pid, (16,))
        @check "ftof"
        converted = convert(ct.Tile{Float8_E4M3FN}, tile)
        ct.store(b, pid, convert(ct.Tile{Float32}, converted))
        return
    end
end

# Float32 -> Float8_E5M2
@test @filecheck begin
    @check_label "entry"
    code_tiled(Tuple{ct.TileArray{Float32,1,spec1d}, ct.TileArray{Float32,1,spec1d}}) do a, b
        pid = ct.bid(1)
        tile = ct.load(a, pid, (16,))
        @check "ftof"
        converted = convert(ct.Tile{Float8_E5M2}, tile)
        ct.store(b, pid, convert(ct.Tile{Float32}, converted))
        return
    end
end

end
