using Microfloats: Float8_E4M3FN, Float8_E5M2, Float8_E8M0FNU, Float4_E2M1FN

spec1d = ct.ArraySpec{1}(16, true)
spec2d = ct.ArraySpec{2}(16, true)

@testset "Microfloats codegen" begin

@testset "ftof" begin
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

@testset "reinterpret" begin
    # Whole-tile `reinterpret` between UInt8 and Float4_E2M1FN packs/unpacks two
    # FP4 per byte: a `Tile{UInt8,(8,)}` unpacks to a `Tile{Float4_E2M1FN,(16,)}`,
    # lowering to `cuda_tile.unpack` (13.3+).
    @test @filecheck begin
        @check_label "entry"
        code_tiled(Tuple{ct.TileArray{UInt8,1,spec1d}, ct.TileArray{Float32,1,spec1d}};
                   bytecode_version=v"13.3") do a, b
            pid = ct.bid(1)
            bytes = ct.load(a, pid, (8,))            # Tile{UInt8,(8,)}
            @check "unpack"
            fp4 = reinterpret(Float4_E2M1FN, bytes)  # Tile{Float4_E2M1FN,(16,)}
            ct.store(b, pid, convert(ct.Tile{Float32}, fp4))
            return
        end
    end

    # And the reverse packs FP4 back into bytes via `cuda_tile.pack` (13.3+).
    @test @filecheck begin
        @check_label "entry"
        code_tiled(Tuple{ct.TileArray{Float32,1,spec1d}, ct.TileArray{UInt8,1,spec1d}};
                   bytecode_version=v"13.3") do a, b
            pid = ct.bid(1)
            vals = ct.load(a, pid, (16,))
            fp4 = convert(ct.Tile{Float4_E2M1FN}, vals)  # Tile{Float4_E2M1FN,(16,)}
            @check "pack"
            ct.store(b, pid, reinterpret(UInt8, fp4))    # Tile{UInt8,(8,)}
            return
        end
    end
end

@testset "mma" begin
    # f8e4m3fn operands with an f32 accumulator lower to `mmaf` (13.1+).
    @test @filecheck begin
        @check_label "entry"
        code_tiled(Tuple{ct.TileArray{Float8_E4M3FN,2,spec2d}, ct.TileArray{Float8_E4M3FN,2,spec2d},
                         ct.TileArray{Float32,2,spec2d}}) do a, b, c
            ta = ct.load(a, (1, 1), (16, 16))
            tb = ct.load(b, (1, 1), (16, 16))
            @check "mmaf"
            ct.store(c, (1, 1), muladd(ta, tb, zeros(Float32, (16, 16))))
            return
        end
    end

    # f8e5m2 operands with an f16 accumulator also lower to `mmaf`.
    @test @filecheck begin
        @check_label "entry"
        code_tiled(Tuple{ct.TileArray{Float8_E5M2,2,spec2d}, ct.TileArray{Float8_E5M2,2,spec2d},
                         ct.TileArray{Float16,2,spec2d}}) do a, b, c
            ta = ct.load(a, (1, 1), (16, 16))
            tb = ct.load(b, (1, 1), (16, 16))
            @check "mmaf"
            ct.store(c, (1, 1), muladd(ta, tb, zeros(Float16, (16, 16))))
            return
        end
    end

    # A disallowed accumulator dtype for f8 (only f16/f32 are valid) is rejected
    # with a clear error rather than producing an mmaf op tileiras would reject.
    @test_throws "tileiras requires acc" code_tiled(
        Tuple{ct.TileArray{Float8_E4M3FN,2,spec2d}, ct.TileArray{Float8_E4M3FN,2,spec2d},
              ct.TileArray{Float64,2,spec2d}}) do a, b, c
        ta = ct.load(a, (1, 1), (16, 16))
        tb = ct.load(b, (1, 1), (16, 16))
        acc = zeros(Float64, (16, 16))
        Base.donotdelete(ct.Intrinsics.mma(ta, tb, acc))
        return
    end
end

@testset "fast_acc" begin
    # `fast_acc=true` on f8 operands still lowers to `mmaf` (the hint rides on
    # the op as a flag); requires bytecode 13.3.
    @test @filecheck begin
        @check_label "entry"
        code_tiled(Tuple{ct.TileArray{Float8_E4M3FN,2,spec2d}, ct.TileArray{Float8_E4M3FN,2,spec2d},
                         ct.TileArray{Float32,2,spec2d}};
                   bytecode_version=v"13.3") do a, b, c
            ta = ct.load(a, (1, 1), (16, 16))
            tb = ct.load(b, (1, 1), (16, 16))
            @check "mmaf"
            ct.store(c, (1, 1), muladd(ta, tb, zeros(Float32, (16, 16)); fast_acc=true))
            return
        end
    end

    # `fast_acc` is an FP8-only hint: requesting it for f16 inputs is rejected.
    @test_throws "only supported for fp8" code_tiled(
        Tuple{ct.TileArray{Float16,2,spec2d}, ct.TileArray{Float16,2,spec2d},
              ct.TileArray{Float32,2,spec2d}}; bytecode_version=v"13.3") do a, b, c
        ta = ct.load(a, (1, 1), (16, 16))
        tb = ct.load(b, (1, 1), (16, 16))
        ct.store(c, (1, 1), muladd(ta, tb, zeros(Float32, (16, 16)); fast_acc=true))
        return
    end

    # `fast_acc` requires bytecode 13.3 — rejected at 13.2 with a clear error.
    let kernel = (a, b, c) -> begin
            ta = ct.load(a, (1, 1), (16, 16))
            tb = ct.load(b, (1, 1), (16, 16))
            ct.store(c, (1, 1), muladd(ta, tb, zeros(Float32, (16, 16)); fast_acc=true))
            return
        end
        @test_throws "13.3" code_tiled(devnull, kernel,
            Tuple{ct.TileArray{Float8_E4M3FN,2,spec2d}, ct.TileArray{Float8_E4M3FN,2,spec2d},
                  ct.TileArray{Float32,2,spec2d}}; bytecode_version=v"13.2")
    end
end

@testset "mma_scaled" begin
    # MXFP8: f8 operands (M,K)/(K,N) with f8e8m0fnu block scales (M,K_s)/(K_s,N)
    # accumulate into f32. Block size B = K ÷ K_s = 64 ÷ 2 = 32.
    @test @filecheck begin
        @check_label "entry"
        code_tiled(Tuple{ct.TileArray{Float8_E4M3FN,2,spec2d}, ct.TileArray{Float8_E8M0FNU,2,spec2d},
                         ct.TileArray{Float8_E4M3FN,2,spec2d}, ct.TileArray{Float8_E8M0FNU,2,spec2d},
                         ct.TileArray{Float32,2,spec2d}};
                   bytecode_version=v"13.3") do x, x_scale, y, y_scale, z
            xt  = ct.load(x,       (1, 1), (16, 64))
            xst = ct.load(x_scale, (1, 1), (16, 2))
            yt  = ct.load(y,       (1, 1), (64, 16))
            yst = ct.load(y_scale, (1, 1), (2, 16))
            @check "mmaf_scaled"
            result = ct.muladd_scaled(xt, xst, yt, yst, zeros(Float32, (16, 16)))
            ct.store(z, (1, 1), result)
            return
        end
    end

    # NVFP4: f4e2m1fn operands with f8e4m3fn scales, B = 16. Operands enter as
    # unpacked FP4 tiles; `mmaf_scaled` accepts them too.
    @test @filecheck begin
        @check_label "entry"
        code_tiled(Tuple{ct.TileArray{Float32,2,spec2d}, ct.TileArray{Float8_E4M3FN,2,spec2d},
                         ct.TileArray{Float32,2,spec2d}, ct.TileArray{Float8_E4M3FN,2,spec2d},
                         ct.TileArray{Float32,2,spec2d}};
                   bytecode_version=v"13.3") do x, x_scale, y, y_scale, z
            xt  = convert(ct.Tile{Float4_E2M1FN}, ct.load(x, (1, 1), (16, 64)))
            xst = ct.load(x_scale, (1, 1), (16, 4))
            yt  = convert(ct.Tile{Float4_E2M1FN}, ct.load(y, (1, 1), (64, 16)))
            yst = ct.load(y_scale, (1, 1), (4, 16))
            @check "mmaf_scaled"
            result = ct.muladd_scaled(xt, xst, yt, yst, zeros(Float32, (16, 16)))
            ct.store(z, (1, 1), result)
            return
        end
    end

    # mma_scaled requires bytecode 13.3 — rejected at 13.2 with a clear error.
    let kernel = (x, x_scale, y, y_scale, z) -> begin
            xt  = ct.load(x,       (1, 1), (16, 64))
            xst = ct.load(x_scale, (1, 1), (16, 2))
            yt  = ct.load(y,       (1, 1), (64, 16))
            yst = ct.load(y_scale, (1, 1), (2, 16))
            ct.store(z, (1, 1), ct.muladd_scaled(xt, xst, yt, yst, zeros(Float32, (16, 16))))
            return
        end
        @test_throws "13.3" code_tiled(devnull, kernel,
            Tuple{ct.TileArray{Float8_E4M3FN,2,spec2d}, ct.TileArray{Float8_E8M0FNU,2,spec2d},
                  ct.TileArray{Float8_E4M3FN,2,spec2d}, ct.TileArray{Float8_E8M0FNU,2,spec2d},
                  ct.TileArray{Float32,2,spec2d}}; bytecode_version=v"13.2")
    end
end

end
