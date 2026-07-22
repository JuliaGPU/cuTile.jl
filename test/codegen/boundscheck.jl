using IRStructurizer: StructuredIRCode, Block, IfOp, YieldOp
using Core: SSAValue, ReturnNode

@testset "bounds-check resolution" begin
    then_region = Block()
    then_region.terminator = YieldOp(Any[])
    else_region = Block()
    else_region.terminator = YieldOp(Any[])

    entry = Block()
    push!(entry, 1, Expr(:boundscheck), Bool)
    push!(entry, 2, IfOp(SSAValue(1), then_region, else_region), Nothing)
    push!(entry, 3, Expr(:call, getfield, (1, 2), 1, SSAValue(1)), Int)
    false_then = Block()
    false_then.terminator = YieldOp(Any[])
    false_else = Block()
    false_else.terminator = YieldOp(Any[])
    push!(entry, 4, Expr(:boundscheck, false), Bool)
    push!(entry, 5, IfOp(SSAValue(4), false_then, false_else), Nothing)
    entry.terminator = ReturnNode(nothing)
    sci = StructuredIRCode(Any[Any], Any[], entry, 5)

    ct.resolve_boundscheck!(sci)

    expected = Base.JLOptions().check_bounds == 2 ? false : true
    @test !haskey(entry, 1)
    @test !haskey(entry, 4)
    @test entry[2][:stmt].condition === expected
    @test entry[3][:stmt].args[4] === expected
    @test entry[5][:stmt].condition === (Base.JLOptions().check_bounds == 1)
end
