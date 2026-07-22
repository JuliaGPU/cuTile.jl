using IRStructurizer: StructuredIRCode, Block, IfOp, YieldOp, BreakOp, Undef,
                      instructions, eachblock
using Core: SSAValue, Argument, ReturnNode

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

count_ifops(sci) = count(inst -> inst[:stmt] isa IfOp,
                         Iterators.flatten(instructions(block) for block in eachblock(sci)))

@testset "constant branch folding" begin
    @testset "single result and region splicing" begin
        then_region = Block()
        push!(then_region, 1, Expr(:call, Core.Intrinsics.add_int, 1, 2), Int)
        push!(then_region, 2, Expr(:call, Core.Intrinsics.add_int, SSAValue(1), 3), Int)
        then_region.terminator = YieldOp(Any[SSAValue(2)])
        else_region = Block()
        else_region.terminator = YieldOp(Any[0])

        entry = Block()
        push!(entry, 3, IfOp(true, then_region, else_region), Int)
        entry.terminator = ReturnNode(SSAValue(3))
        sci = StructuredIRCode(Any[Any], Any[], entry, 3)
        entry.parent = sci

        ct.fold_constant_branches!(sci)

        @test count_ifops(sci) == 0
        @test entry[2][:stmt].args[2] == SSAValue(1)
        @test entry.terminator.val == SSAValue(2)
    end

    @testset "tuple results reach a fixpoint" begin
        outer_then = Block()
        outer_then.terminator = YieldOp(Any[true, 11])
        outer_else = Block()
        outer_else.terminator = YieldOp(Any[false, Undef(Int)])
        inner_then = Block()
        inner_then.terminator = YieldOp(Any[SSAValue(3)])
        inner_else = Block()
        inner_else.terminator = YieldOp(Any[0])

        entry = Block()
        push!(entry, 1, IfOp(true, outer_then, outer_else), Tuple{Bool, Int})
        push!(entry, 2, Expr(:call, getfield, SSAValue(1), 1), Bool)
        push!(entry, 3, Expr(:call, getfield, SSAValue(1), 2), Int)
        push!(entry, 4, IfOp(SSAValue(2), inner_then, inner_else), Int)
        entry.terminator = ReturnNode(SSAValue(4))
        sci = StructuredIRCode(Any[Any], Any[], entry, 4)

        ct.fold_constant_branches!(sci)

        @test count_ifops(sci) == 0
        @test entry.terminator.val == 11
    end

    @testset "splat condition" begin
        then_region = Block()
        then_region.terminator = YieldOp(Any[])
        else_region = Block()
        else_region.terminator = YieldOp(Any[])
        entry = Block()
        push!(entry, 1, Expr(:call, ct.Intrinsics.constant, (), true, Bool),
              ct.Tile{Bool, Tuple{}})
        push!(entry, 2, IfOp(SSAValue(1), then_region, else_region), Nothing)
        entry.terminator = ReturnNode(nothing)
        sci = StructuredIRCode(Any[Any], Any[], entry, 2)

        ct.fold_constant_branches!(sci)
        @test count_ifops(sci) == 0
    end

    @testset "unsupported and runtime branches remain" begin
        then_region = Block()
        then_region.terminator = BreakOp(Any[])
        else_region = Block()
        else_region.terminator = YieldOp(Any[])
        entry = Block()
        push!(entry, 1, IfOp(true, then_region, else_region), Nothing)
        entry.terminator = ReturnNode(nothing)
        break_sci = StructuredIRCode(Any[Any], Any[], entry, 1)

        ct.fold_constant_branches!(break_sci)
        @test count_ifops(break_sci) == 1

        then_region = Block()
        then_region.terminator = YieldOp(Any[])
        else_region = Block()
        else_region.terminator = YieldOp(Any[])
        entry = Block()
        push!(entry, 1, IfOp(Argument(2), then_region, else_region), Nothing)
        entry.terminator = ReturnNode(nothing)
        runtime_sci = StructuredIRCode(Any[Any, Bool], Any[], entry, 1)

        ct.fold_constant_branches!(runtime_sci)
        @test count_ifops(runtime_sci) == 1
    end

    @testset "empty region" begin
        then_region = Block()
        then_region.terminator = YieldOp(Any[])
        else_region = Block()
        entry = Block()
        push!(entry, 1, IfOp(false, then_region, else_region), Nothing)
        entry.terminator = ReturnNode(nothing)
        sci = StructuredIRCode(Any[Any], Any[], entry, 1)

        ct.fold_constant_branches!(sci)
        @test count_ifops(sci) == 0
    end
end
