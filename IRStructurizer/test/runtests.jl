using Test

using IRStructurizer
using IRStructurizer: Block, IfOp, ForOp, WhileOp, LoopOp, YieldOp, ContinueOp, BreakOp,
                      ConditionOp, ControlFlowOp, Statement, Operation, LocalSSA, validate_scf

# Helper to check if block contains a control flow op of given type
function has_cfop(block::Block, T::Type)
    for op in block.ops
        if op.expr isa T
            return true
        end
    end
    return false
end

# Helper to get control flow ops of a given type from block
function get_cfops(block::Block, T::Type)
    cfops = T[]
    for op in block.ops
        if op.expr isa T
            push!(cfops, op.expr)
        end
    end
    return cfops
end

# Helper to count non-control-flow operations (statements)
function count_stmts(block::Block)
    count(op -> !(op.expr isa ControlFlowOp), block.ops)
end

@testset "IRStructurizer" verbose=true begin

#=============================================================================
 Interface Tests
=============================================================================#

@testset "interface" begin

@testset "low-level API" begin
    g(x) = x > 0 ? x + 1 : x - 1
    ci, _ = only(code_typed(g, (Int,)))

    # Create flat, then structurize
    sci = StructuredCodeInfo(ci)
    @test !has_cfop(sci.entry, IfOp)

    structurize!(sci)
    @test has_cfop(sci.entry, IfOp)

    # code_structured does both steps
    sci2 = code_structured(g, Tuple{Int})
    @test has_cfop(sci2.entry, IfOp)
end

@testset "validation: UnstructuredControlFlowError" begin
    # Create unstructured view and verify validation fails
    g(x) = x > 0 ? x + 1 : x - 1
    ci, _ = only(code_typed(g, (Int,)))

    # Flat view has GotoIfNot
    sci = StructuredCodeInfo(ci)
    gotoifnot_idx = findfirst(s -> s isa Core.GotoIfNot, ci.code)
    @test gotoifnot_idx !== nothing
    # Check that the GotoIfNot is in ops (before structurization)
    @test any(op -> op.expr isa Core.GotoIfNot, sci.entry.ops)

    # Validation should throw
    @test_throws UnstructuredControlFlowError validate_scf(sci)

    # After structurize!, validation passes
    structurize!(sci)
    # GotoIfNot should no longer be present (replaced by IfOp in ops)
    @test !any(op -> op.expr isa Core.GotoIfNot, sci.entry.ops)
    validate_scf(sci)  # Should not throw
end

@testset "loop_patterning kwarg" begin
    # Test that loop_patterning=false produces LoopOp instead of ForOp
    function count_loop(n::Int)
        i = 0
        while i < n
            i += 1
        end
        return i
    end

    # With patterning (default): ForOp
    sci_with = code_structured(count_loop, Tuple{Int}; loop_patterning=true)
    loop_ops_with = get_cfops(sci_with.entry, ForOp)
    @test !isempty(loop_ops_with)
    @test loop_ops_with[1] isa ForOp

    # Without patterning: LoopOp
    sci_without = code_structured(count_loop, Tuple{Int}; loop_patterning=false)
    loop_ops_without = get_cfops(sci_without.entry, LoopOp)
    @test !isempty(loop_ops_without)
    @test loop_ops_without[1] isa LoopOp
end

@testset "display output format" begin
    # Verify display shows proper structure
    branch_test(x::Bool) = x ? 1 : 2

    sci = code_structured(branch_test, Tuple{Bool})

    io = IOBuffer()
    show(io, MIME"text/plain"(), sci)
    output = String(take!(io))

    @test occursin("StructuredCodeInfo", output)
    @test occursin("if ", output)
    @test occursin("else", output)
    @test occursin("return", output)

    # Compact display
    io = IOBuffer()
    show(io, sci)
    output = String(take!(io))

    @test occursin("StructuredCodeInfo", output)
    @test occursin("ops", output)
end

end  # interface

#=============================================================================
 CFG Analysis Tests
 Tests that control flow regions are correctly identified.
 Uses loop_patterning=false to get LoopOp for all loops, focusing on
 the CFG structure rather than loop classification.
=============================================================================#

@testset "CFG analysis" begin

@testset "acyclic regions" begin

@testset "block sequence" begin
    # Simple function: single addition (no control flow)
    f(x) = x + 1

    sci = code_structured(f, Tuple{Int})
    @test sci isa StructuredCodeInfo

    # Entry block: one operation (the add), no control flow ops
    @test length(sci.entry.ops) == 1
    @test !(sci.entry.ops[1].expr isa ControlFlowOp)
    @test sci.entry.terminator isa Core.ReturnNode

    # Multiple operations: (x + y) * (x - y)
    g(x, y) = (x + y) * (x - y)

    sci = code_structured(g, Tuple{Int, Int})
    @test sci isa StructuredCodeInfo

    # Entry block: 3 operations (add, sub, mul), no control flow ops
    @test length(sci.entry.ops) == 3
    @test all(!(op.expr isa ControlFlowOp) for op in sci.entry.ops)
    @test sci.entry.terminator isa Core.ReturnNode
end

@testset "if-then-else: diamond pattern" begin
    # Both branches converge (diamond CFG pattern)
    compute_branch(x::Int) = x > 0 ? x + 1 : x - 1

    sci = code_structured(compute_branch, Tuple{Int})
    @test sci isa StructuredCodeInfo

    # Entry: comparison op, then IfOp
    @test length(sci.entry.ops) == 2
    @test !(sci.entry.ops[1].expr isa ControlFlowOp)
    @test sci.entry.ops[2].expr isa IfOp

    if_op = sci.entry.ops[2].expr

    # Then branch: one op (addition), then return
    @test length(if_op.then_block.ops) == 1
    @test !(if_op.then_block.ops[1].expr isa ControlFlowOp)
    @test if_op.then_block.terminator isa Core.ReturnNode

    # Else branch: one op (subtraction), then return
    @test length(if_op.else_block.ops) == 1
    @test !(if_op.else_block.ops[1].expr isa ControlFlowOp)
    @test if_op.else_block.terminator isa Core.ReturnNode
end

@testset "if-then-else: bool condition (no comparison)" begin
    # Bool condition directly, no comparison needed
    branch_test(x::Bool) = x ? 1 : 2

    sci = code_structured(branch_test, Tuple{Bool})
    @test sci isa StructuredCodeInfo

    # Entry: exactly one IfOp, no other operations
    @test length(sci.entry.ops) == 1
    @test sci.entry.ops[1].expr isa IfOp

    if_op = sci.entry.ops[1].expr

    # Condition is the first argument (the Bool)
    @test if_op.condition isa Core.Argument
    @test if_op.condition.n == 2  # arg 1 is #self#

    # Then branch: empty ops, returns constant 1
    @test isempty(if_op.then_block.ops)
    @test if_op.then_block.terminator isa Core.ReturnNode
    @test if_op.then_block.terminator.val == 1

    # Else branch: empty ops, returns constant 2
    @test isempty(if_op.else_block.ops)
    @test if_op.else_block.terminator isa Core.ReturnNode
    @test if_op.else_block.terminator.val == 2
end

@testset "if-then-else: with comparison" begin
    # Comparison before branch
    cmp_branch(x::Int) = x > 0 ? x : -x

    sci = code_structured(cmp_branch, Tuple{Int})
    @test sci isa StructuredCodeInfo

    # Entry: one op (comparison), then IfOp
    @test length(sci.entry.ops) == 2
    @test !(sci.entry.ops[1].expr isa ControlFlowOp)
    @test sci.entry.ops[2].expr isa IfOp

    if_op = sci.entry.ops[2].expr

    # Condition references the comparison result (now LocalSSA)
    @test if_op.condition isa LocalSSA
    @test if_op.condition.id == 1  # First operation in block

    # Both branches terminate with return
    @test if_op.then_block.terminator isa Core.ReturnNode
    @test if_op.else_block.terminator isa Core.ReturnNode
end

@testset "termination: early return pattern" begin
    # One branch returns early, other continues
    function early_return(x::Int, y::Int)
        if x > y
            return y * x
        end
        y - x
    end

    sci = code_structured(early_return, Tuple{Int, Int})
    @test sci isa StructuredCodeInfo

    # Entry: [comparison_op, IfOp]
    @test length(sci.entry.ops) == 2
    @test !(sci.entry.ops[1].expr isa ControlFlowOp)
    @test sci.entry.ops[2].expr isa IfOp

    if_op = sci.entry.ops[2].expr

    # Both branches terminate with return
    @test if_op.then_block.terminator isa Core.ReturnNode
    @test if_op.else_block.terminator isa Core.ReturnNode
end

end  # acyclic regions

@testset "cyclic regions" begin

@testset "simple loop structure" begin
    # Test that loops are detected (produces LoopOp with loop_patterning=false)
    function simple_loop(n::Int)
        i = 0
        while i < n
            i += 1
        end
        return i
    end

    sci = code_structured(simple_loop, Tuple{Int}; loop_patterning=false)
    @test sci isa StructuredCodeInfo

    # Entry should have a LoopOp
    loop_ops = get_cfops(sci.entry, LoopOp)
    @test length(loop_ops) == 1
end

@testset "loop with condition" begin
    # Loop with condition check at header
    function spinloop(flag::Int)
        while flag != 0
            # spin
        end
        return flag
    end

    sci = code_structured(spinloop, Tuple{Int}; loop_patterning=false)
    @test sci isa StructuredCodeInfo

    # Entry should have a LoopOp
    loop_ops = get_cfops(sci.entry, LoopOp)
    @test length(loop_ops) == 1

    loop_op = loop_ops[1]

    # LoopOp body should contain the conditional structure
    @test loop_op.body isa Block
end

@testset "loop with body statements" begin
    # Loop with actual work in body
    function countdown(n::Int)
        while n > 0
            n -= 1
        end
        return n
    end

    sci = code_structured(countdown, Tuple{Int}; loop_patterning=false)
    @test sci isa StructuredCodeInfo

    # Entry should have a LoopOp
    loop_ops = get_cfops(sci.entry, LoopOp)
    @test length(loop_ops) == 1
end

@testset "nested loops" begin
    # Two nested loops (both become LoopOp with loop_patterning=false)
    function nested(n::Int, m::Int)
        acc = 0
        i = 0
        while i < n
            j = 0
            while j < m
                acc += 1
                j += 1
            end
            i += 1
        end
        return acc
    end

    sci = code_structured(nested, Tuple{Int, Int}; loop_patterning=false)
    @test sci isa StructuredCodeInfo

    # Entry should have outer LoopOp
    outer_loops = get_cfops(sci.entry, LoopOp)
    @test length(outer_loops) == 1

    # Find inner loop in outer loop's body
    outer_loop = outer_loops[1]
    function find_nested_loops(block::Block)
        loops = LoopOp[]
        for op in block.ops
            if op.expr isa LoopOp
                push!(loops, op.expr)
            elseif op.expr isa IfOp
                append!(loops, find_nested_loops(op.expr.then_block))
                append!(loops, find_nested_loops(op.expr.else_block))
            end
        end
        return loops
    end
    inner_loops = find_nested_loops(outer_loop.body)
    @test length(inner_loops) == 1
end

end  # cyclic regions

end  # CFG analysis

#=============================================================================
 IR Patterning Tests
 Tests that loops are correctly classified into ForOp, WhileOp, or LoopOp.
 Uses loop_patterning=true (default) to test pattern detection.
=============================================================================#

@testset "loop patterning" begin

@testset "ForOp detection" begin

@testset "bounded counter" begin
    # Simple counting loop: i = 0; while i < n; i += 1
    function count_to_n(n::Int)
        i = 0
        while i < n
            i += 1
        end
        return i
    end

    sci = code_structured(count_to_n, Tuple{Int})
    @test sci isa StructuredCodeInfo

    # Should produce ForOp
    for_ops = get_cfops(sci.entry, ForOp)
    @test length(for_ops) == 1

    for_op = for_ops[1]

    # Bounds: 0 to n, step 1
    @test for_op.lower == 0
    @test for_op.upper isa Core.Argument
    @test for_op.step == 1

    # Body terminates with ContinueOp
    @test for_op.body.terminator isa ContinueOp
end

@testset "bounded counter with accumulator" begin
    # Counting loop with loop-carried accumulator
    function sum_to_n(n::Int)
        i = 0
        acc = 0
        while i < n
            acc += i
            i += 1
        end
        return acc
    end

    sci = code_structured(sum_to_n, Tuple{Int})
    @test sci isa StructuredCodeInfo

    # Should produce ForOp
    for_ops = get_cfops(sci.entry, ForOp)
    @test length(for_ops) == 1

    for_op = for_ops[1]

    # Body has block args: [accumulator] (IV is stored separately in for_op.iv_arg)
    @test length(for_op.body.args) == 1

    # Loop produces one result (the final accumulator value)
    @test for_op.result_type !== Nothing
end

@testset "nested for loops" begin
    # Two nested counting loops
    function nested_count(n::Int, m::Int)
        acc = 0
        i = 0
        while i < n
            j = 0
            while j < m
                acc += 1
                j += 1
            end
            i += 1
        end
        return acc
    end

    sci = code_structured(nested_count, Tuple{Int, Int})
    @test sci isa StructuredCodeInfo

    # Entry contains at least: [init_op, outer_ForOp, (extraction stmts for loop results)]
    @test length(sci.entry.ops) >= 2
    # Find the ForOp (could have extraction statements after it)
    outer_idx = findfirst(op -> op.expr isa ForOp, sci.entry.ops)
    @test outer_idx !== nothing
    outer_loop = sci.entry.ops[outer_idx].expr

    # Outer body contains at least: [init_op, inner_ForOp, (extraction stmts)]
    @test length(outer_loop.body.ops) >= 2
    inner_idx = findfirst(op -> op.expr isa ForOp, outer_loop.body.ops)
    @test inner_idx !== nothing
    inner_loop = outer_loop.body.ops[inner_idx].expr

    # Inner loop has its own structure
    @test inner_loop.body.terminator isa ContinueOp
end

end  # ForOp detection

@testset "WhileOp detection" begin

@testset "condition-only spinloop" begin
    # While loop that is NOT a for-loop (no increment pattern)
    function spinloop(flag::Int)
        while flag != 0
            # spin - no body operations, just condition check
        end
        return flag
    end

    sci = code_structured(spinloop, Tuple{Int})
    @test sci isa StructuredCodeInfo

    # Entry: [WhileOp] - no setup operations
    @test length(sci.entry.ops) == 1
    @test sci.entry.ops[1].expr isa WhileOp

    while_op = sci.entry.ops[1].expr

    # MLIR-style two-region structure: before (condition) and after (body)
    # Condition is in the ConditionOp terminator of the before region
    @test while_op.before.terminator isa ConditionOp
    @test while_op.before.terminator.condition isa LocalSSA

    # No loop-carried values (flag is just re-read each iteration)
    @test isempty(while_op.init_values)
    @test isempty(while_op.before.args)

    # Before region has the condition computation operations
    @test !isempty(while_op.before.ops)
    @test all(!(op.expr isa ControlFlowOp) for op in while_op.before.ops)

    # After region terminates with YieldOp
    @test while_op.after.terminator isa YieldOp
end

@testset "decrementing loop (non-ForOp pattern)" begin
    # Decrementing loop - may be WhileOp or ForOp depending on detection
    function countdown(n::Int)
        while n > 0
            n -= 1
        end
        return n
    end

    sci = code_structured(countdown, Tuple{Int})
    @test sci isa StructuredCodeInfo

    # Entry should contain a loop op (could have extraction statements after it)
    @test !isempty(sci.entry.ops)
    loop_idx = findfirst(op -> op.expr isa Union{ForOp, WhileOp, LoopOp}, sci.entry.ops)
    @test loop_idx !== nothing
end

end  # WhileOp detection

@testset "LoopOp fallback" begin

@testset "dynamic step" begin
    # Loop where step is modified inside loop body (not a valid ForOp)
    function dynamic_step(n::Int)
        i = 0
        step = 1
        while i < n
            i += step
            step += 1  # Step changes each iteration
        end
        return i
    end

    sci = code_structured(dynamic_step, Tuple{Int}; loop_patterning=false)
    @test sci isa StructuredCodeInfo

    # With loop_patterning=false, should be LoopOp
    loop_ops = get_cfops(sci.entry, LoopOp)
    @test length(loop_ops) == 1
end

end  # LoopOp fallback

end  # loop patterning

#=============================================================================
 Nested Control Flow Tests
=============================================================================#

@testset "nested control flow" begin

@testset "if inside loop" begin
    # Loop containing conditional
    function loop_with_if(n::Int)
        acc = 0
        i = 0
        while i < n
            if i % 2 == 0
                acc += i
            end
            i += 1
        end
        return acc
    end

    sci = code_structured(loop_with_if, Tuple{Int})
    @test sci isa StructuredCodeInfo

    # Should have a loop op in entry
    loop_ops = [op.expr for op in sci.entry.ops if op.expr isa Union{ForOp, WhileOp, LoopOp}]
    @test !isempty(loop_ops)

    # The loop body should contain an IfOp
    loop_op = loop_ops[1]
    @test has_cfop(loop_op.body, IfOp)
end

@testset "loop inside if" begin
    # Conditional containing loop
    function if_with_loop(x::Int, n::Int)
        if x > 0
            i = 0
            while i < n
                i += 1
            end
            return i
        else
            return 0
        end
    end

    sci = code_structured(if_with_loop, Tuple{Int, Int})
    @test sci isa StructuredCodeInfo

    # Should have IfOp in entry
    if_ops = get_cfops(sci.entry, IfOp)
    @test !isempty(if_ops)

    if_op = if_ops[1]

    # Then branch should contain a loop
    function has_loop_op(block::Block)
        for op in block.ops
            if op.expr isa Union{ForOp, WhileOp, LoopOp}
                return true
            end
        end
        return false
    end
    @test has_loop_op(if_op.then_block)
end

end  # nested control flow

#=============================================================================
 Regression Tests
=============================================================================#

@testset "regression" begin

@testset "no duplicated operations after loop" begin
    # Operations after loop should not be duplicated
    function loop_then_compute(x::Int)
        i = 0
        while i < x
            i += 1
        end
        # This should appear exactly once
        result = i * 2
        return result
    end

    sci = code_structured(loop_then_compute, Tuple{Int})
    @test sci isa StructuredCodeInfo

    # Check display output: mul_int should appear exactly once
    io = IOBuffer()
    show(io, MIME"text/plain"(), sci)
    output = String(take!(io))
    @test count("mul_int", output) == 1
end

@testset "type preservation" begin
    f(x::Float64) = x + 1.0

    sci = code_structured(f, Tuple{Float64})
    @test sci isa StructuredCodeInfo

    # Float64 type should be preserved in ssavaluetypes
    @test !isempty(sci.code.ssavaluetypes)
    @test any(t -> t isa Type && t <: AbstractFloat, sci.code.ssavaluetypes)
end

@testset "multiple arguments" begin
    # Different argument types
    h(x::Int, y::Float64) = x + y

    sci = code_structured(h, Tuple{Int, Float64})
    @test sci isa StructuredCodeInfo
    @test sci.entry.terminator isa Core.ReturnNode
end

end  # regression

end  # @testset "IRStructurizer"
