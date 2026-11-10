# Codegen tests for the pattern-rewrite driver (`rewrite_patterns!`).

@testset "rewrite — modified cascade is not exponential in diamond depth" begin
    # An add-with-carry chain: every carry `t` is used both by the next step
    # and by the low byte `b`, which also feeds the next step. Each step is a
    # diamond in the use graph, so a node `D` steps down is reachable along
    # 2^D paths. The operand-modified cascade used to re-propagate once per
    # path, making normalization exponential in `D` (D=18 took ~35 s).
    spec = ct.ArraySpec{1}(16, true)
    D = 24
    steps = [:(t = UInt16.(a) .+ UInt16.(b) .+ (t .>> 0x0008); b = UInt8.(t)) for _ in 1:D]
    carry_chain = @eval function (x, y, o)
        a = ct.load(x, 1, (16,))
        b = ct.load(y, 1, (16,))
        t = UInt16.(a)
        $(steps...)
        ct.store(o, 1, b)
        return
    end
    argtypes = Tuple{ct.TileArray{UInt8,1,Int32,spec}, ct.TileArray{UInt8,1,Int32,spec},
                     ct.TileArray{UInt8,1,Int32,spec}}
    ct.code_typed(carry_chain, argtypes)  # keep inference out of the timing
    @test @elapsed(ct.code_structured(carry_chain, argtypes)) < 30
end
