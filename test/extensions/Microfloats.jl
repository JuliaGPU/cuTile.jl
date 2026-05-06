using Microfloats: Float8_E4M3FN, Float8_E5M2, Float8_E8M0FNU, Float4_E2M1FN

@testset "Microfloats extension" begin

spec1d = ct.ArraySpec{1}(16, true)

# Float32 -> Float8_E4M3FN (always available; 13.1+)
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

# Float32 -> Float8_E5M2 (always available; 13.1+)
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

# Float32 -> Float8_E8M0FNU works on bytecode 13.2+
@test @filecheck begin
    @check_label "entry"
    code_tiled(Tuple{ct.TileArray{Float32,1,spec1d}, ct.TileArray{Float32,1,spec1d}};
               bytecode_version=v"13.2") do a, b
        pid = ct.bid(1)
        tile = ct.load(a, pid, (16,))
        @check "ftof"
        converted = convert(ct.Tile{Float8_E8M0FNU}, tile)
        ct.store(b, pid, convert(ct.Tile{Float32}, converted))
        return
    end
end

# Float8_E8M0FNU rejected on bytecode 13.1 with a clear version error
let kernel = (a, b) -> begin
        pid = ct.bid(1)
        tile = ct.load(a, pid, (16,))
        converted = convert(ct.Tile{Float8_E8M0FNU}, tile)
        ct.store(b, pid, convert(ct.Tile{Float32}, converted))
        return
    end
    @test_throws "v13.2+" code_tiled(devnull, kernel,
        Tuple{ct.TileArray{Float32,1,spec1d}, ct.TileArray{Float32,1,spec1d}};
        bytecode_version=v"13.1")
end

# Float4_E2M1FN requires bytecode 13.3 — rejected at 13.2 with a clear error
let kernel = (a, b) -> begin
        pid = ct.bid(1)
        tile = ct.load(a, pid, (16,))
        converted = convert(ct.Tile{Float4_E2M1FN}, tile)
        ct.store(b, pid, convert(ct.Tile{Float32}, converted))
        return
    end
    @test_throws "v13.3+" code_tiled(devnull, kernel,
        Tuple{ct.TileArray{Float32,1,spec1d}, ct.TileArray{Float32,1,spec1d}};
        bytecode_version=v"13.2")
end

end
