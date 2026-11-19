@testset "modified cascade on diamond-shaped use chains" begin
    # An add-with-carry chain: every carry `t` feeds both the next step and the
    # low byte `b`, which also feeds the next step, so each step is a diamond
    # in the use graph. A driver that visits users before their operands'
    # defs and makes up for it by cascading changes up the use chain revisits
    # a node once per use path, taking ~50 s for 20 steps.
    spec = ct.ArraySpec{1}(16, true)
    carry_chain(D) = @eval function (x, y, o)
        a = ct.load(x, 1, (16,))
        b = ct.load(y, 1, (16,))
        t = UInt16.(a)
        $((:(t = UInt16.(a) .+ UInt16.(b) .+ (t .>> 0x0008); b = UInt8.(t)) for _ in 1:D)...)
        ct.store(o, 1, b)
        return
    end
    argtypes = Tuple{ct.TileArray{UInt8,1,Int32,spec}, ct.TileArray{UInt8,1,Int32,spec},
                     ct.TileArray{UInt8,1,Int32,spec}}
    ct.code_structured(carry_chain(1), argtypes)  # compile the pass pipeline
    f = carry_chain(20)
    ct.code_typed(f, argtypes)  # keep inference out of the timing
    @test @elapsed(ct.code_structured(f, argtypes)) < 10
end
