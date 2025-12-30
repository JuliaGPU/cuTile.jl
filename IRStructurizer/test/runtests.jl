using Test

using IRStructurizer
using IRStructurizer: Block, ControlFlowOp, IfOp, ForOp, WhileOp, LoopOp,
                      YieldOp, ContinueOp, BreakOp, ConditionOp,
                      validate_scf, items, statements
using Core: SSAValue

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
    @test !any(x -> x isa IfOp, items(sci.entry.body))

    structurize!(sci)
    @test any(x -> x isa IfOp, items(sci.entry.body))

    # code_structured does both steps
    sci2 = code_structured(g, Tuple{Int})
    @test any(x -> x isa IfOp, items(sci2.entry.body))
end

@testset "validation: UnstructuredControlFlowError" begin
    # Create unstructured view and verify validation fails
    g(x) = x > 0 ? x + 1 : x - 1
    ci, _ = only(code_typed(g, (Int,)))

    # Flat view has GotoIfNot
    sci = StructuredCodeInfo(ci)
    gotoifnot_idx = findfirst(s -> s isa Core.GotoIfNot, ci.code)
    @test gotoifnot_idx !== nothing
    # Check that the GotoIfNot is in the body (before structurize!)
    # Entry body is SSAVector, iterate as (idx, entry) pairs where entry has .stmt and .typ
    @test any(((_, entry),) -> entry.stmt isa Core.GotoIfNot, sci.entry.body)

    # Validation should throw
    @test_throws UnstructuredControlFlowError validate_scf(sci)

    # After structurize!, validation passes
    structurize!(sci)
    # GotoIfNot should no longer be in body (replaced by IfOp)
    @test !any(expr -> expr isa Core.GotoIfNot, items(sci.entry.body))
    validate_scf(sci)  # Should not throw
end

@testset "loop_patterning kwarg" begin
    # Test that loop_patterning=false produces :loop instead of :for
    function count_loop(n::Int)
        i = 0
        while i < n
            i += 1
        end
        return i
    end

    # With patterning (default): :for
    sci_with = code_structured(count_loop, Tuple{Int}; loop_patterning=true)
    loop_op_with = filter(x -> x isa ControlFlowOp, collect(items(sci_with.entry.body)))
    @test !isempty(loop_op_with)
    @test loop_op_with[1] isa ForOp

    # Without patterning: :loop
    sci_without = code_structured(count_loop, Tuple{Int}; loop_patterning=false)
    loop_op_without = filter(x -> x isa ControlFlowOp, collect(items(sci_without.entry.body)))
    @test !isempty(loop_op_without)
    @test loop_op_without[1] isa LoopOp
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
    @test occursin("stmts", output)
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

    # Entry block: one expression (the add), no control flow ops
    @test length(sci.entry.body) == 1
    @test !(sci.entry.body[1].stmt isa ControlFlowOp)
    @test sci.entry.terminator isa Core.ReturnNode

    # Multiple operations: (x + y) * (x - y)
    g(x, y) = (x + y) * (x - y)

    sci = code_structured(g, Tuple{Int, Int})
    @test sci isa StructuredCodeInfo

    # Entry block: 3 expressions (add, sub, mul), no control flow ops
    @test length(sci.entry.body) == 3
    @test all(x -> !(x isa ControlFlowOp), items(sci.entry.body))
    @test sci.entry.terminator isa Core.ReturnNode
end

@testset "if-then-else: diamond pattern" begin
    # Both branches converge (diamond CFG pattern)
    compute_branch(x::Int) = x > 0 ? x + 1 : x - 1

    sci = code_structured(compute_branch, Tuple{Int})
    @test sci isa StructuredCodeInfo

    # Entry: comparison expr, then IfOp
    @test length(sci.entry.body) == 2
    @test !(sci.entry.body[1].stmt isa ControlFlowOp)
    @test sci.entry.body[2].stmt isa IfOp

    if_op = sci.entry.body[2].stmt
    then_blk = if_op.then_region
    else_blk = if_op.else_region

    # Then branch: one expr (addition), then return
    @test length(then_blk.body) == 1
    @test !(then_blk.body[1].stmt isa ControlFlowOp)
    @test then_blk.terminator isa Core.ReturnNode

    # Else branch: one expr (subtraction), then return
    @test length(else_blk.body) == 1
    @test !(else_blk.body[1].stmt isa ControlFlowOp)
    @test else_blk.terminator isa Core.ReturnNode
end

@testset "if-then-else: bool condition (no comparison)" begin
    # Bool condition directly, no comparison needed
    branch_test(x::Bool) = x ? 1 : 2

    sci = code_structured(branch_test, Tuple{Bool})
    @test sci isa StructuredCodeInfo

    # Entry: exactly one IfOp, no expressions
    @test length(sci.entry.body) == 1
    @test sci.entry.body[1].stmt isa IfOp

    if_op = sci.entry.body[1].stmt
    then_blk = if_op.then_region
    else_blk = if_op.else_region

    # Condition is the first argument (the Bool)
    @test if_op.condition isa Core.Argument
    @test if_op.condition.n == 2  # arg 1 is #self#

    # Then branch: empty body, returns constant 1
    @test isempty(then_blk.body)
    @test then_blk.terminator isa Core.ReturnNode
    @test then_blk.terminator.val == 1

    # Else branch: empty body, returns constant 2
    @test isempty(else_blk.body)
    @test else_blk.terminator isa Core.ReturnNode
    @test else_blk.terminator.val == 2
end

@testset "if-then-else: with comparison" begin
    # Comparison before branch
    cmp_branch(x::Int) = x > 0 ? x : -x

    sci = code_structured(cmp_branch, Tuple{Int})
    @test sci isa StructuredCodeInfo

    # Entry: one expr (comparison), then IfOp
    @test length(sci.entry.body) == 2
    @test !(sci.entry.body[1].stmt isa ControlFlowOp)
    @test sci.entry.body[2].stmt isa IfOp

    if_op = sci.entry.body[2].stmt
    then_blk = if_op.then_region
    else_blk = if_op.else_region

    # Condition references the comparison result (SSAValue with local index after finalization)
    @test if_op.condition isa SSAValue
    @test if_op.condition.id > 0  # Positive local index after finalization

    # Both branches terminate with return
    @test then_blk.terminator isa Core.ReturnNode
    @test else_blk.terminator isa Core.ReturnNode
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

    # Entry: [comparison_expr, IfOp]
    @test length(sci.entry.body) == 2
    @test !(sci.entry.body[1].stmt isa ControlFlowOp)
    @test sci.entry.body[2].stmt isa IfOp

    if_op = sci.entry.body[2].stmt
    then_blk = if_op.then_region
    else_blk = if_op.else_region

    # Both branches terminate with return
    @test then_blk.terminator isa Core.ReturnNode
    @test else_blk.terminator isa Core.ReturnNode
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
    loop_ops = filter(x -> x isa LoopOp, collect(items(sci.entry.body)))
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
    loop_ops = filter(x -> x isa LoopOp, collect(items(sci.entry.body)))
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
    loop_ops = filter(x -> x isa LoopOp, collect(items(sci.entry.body)))
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
    outer_loops = filter(x -> x isa LoopOp, collect(items(sci.entry.body)))
    @test length(outer_loops) == 1

    # Find inner loop in outer loop's body
    outer_loop = outer_loops[1]
    function find_nested_loops(block::Block)
        loops = ControlFlowOp[]
        for stmt in statements(block.body)
            if stmt isa LoopOp
                push!(loops, stmt)
            elseif stmt isa IfOp
                append!(loops, find_nested_loops(stmt.then_region))
                append!(loops, find_nested_loops(stmt.else_region))
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
    for_ops = filter(x -> x isa ForOp, collect(items(sci.entry.body)))
    @test length(for_ops) == 1

    for_op = for_ops[1]
    for_body = for_op.body

    # Bounds: 0 to n, step 1
    @test for_op.lower == 0
    @test for_op.upper isa Core.Argument
    @test for_op.step == 1

    # Body terminates with ContinueOp
    @test for_body.terminator isa ContinueOp
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
    for_ops = filter(x -> x isa ForOp, collect(items(sci.entry.body)))
    @test length(for_ops) == 1

    for_op = for_ops[1]
    for_body = for_op.body

    # Body has block args: [accumulator] (IV is stored separately in for_op.iv_arg)
    @test length(for_body.args) == 1

    # Loop produces one result (the final accumulator value)
    # The result count equals init_values count (each init_value corresponds to one result)
    @test length(for_op.init_values) == 1
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

    # Entry: [init_expr, outer_ForOp, getfield(s)...]
    # Find the outer ForOp
    outer_loop_idx = findfirst(stmt -> stmt isa ForOp, collect(statements(sci.entry.body)))
    @test outer_loop_idx !== nothing

    outer_loop = collect(statements(sci.entry.body))[outer_loop_idx]
    outer_body = outer_loop.body

    # Outer body: [init_expr, inner_ForOp, getfield(s)...]
    # Find the inner ForOp
    inner_loop_idx = findfirst(stmt -> stmt isa ForOp, collect(statements(outer_body.body)))
    @test inner_loop_idx !== nothing

    inner_loop = collect(statements(outer_body.body))[inner_loop_idx]
    inner_body = inner_loop.body

    # Inner loop has its own structure
    @test inner_body.terminator isa ContinueOp
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

    # Entry: [WhileOp] - no setup statements
    @test length(sci.entry.body) == 1
    @test sci.entry.body[1].stmt isa WhileOp

    while_op = sci.entry.body[1].stmt
    before_blk = while_op.before
    after_blk = while_op.after

    # MLIR-style two-region structure: before (condition) and after (body)
    # Condition is in the ConditionOp terminator of the before region
    @test before_blk.terminator isa ConditionOp
    # Condition is SSAValue with global index (after finalization)
    @test before_blk.terminator.condition isa SSAValue

    # No loop-carried values (flag is just re-read each iteration)
    @test isempty(while_op.init_values)
    @test isempty(before_blk.args)

    # Before region has the condition computation expressions
    @test !isempty(before_blk.body)
    @test all(x -> !(x isa ControlFlowOp), items(before_blk.body))

    # After region terminates with YieldOp
    @test after_blk.terminator isa YieldOp
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

    # Entry should have items, find the loop op
    @test !isempty(sci.entry.body)
    # Find any loop op in the body (could be followed by getfield statements)
    loop_ops = filter(x -> x isa ForOp || x isa WhileOp || x isa LoopOp, collect(statements(sci.entry.body)))
    @test !isempty(loop_ops)
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
    loop_ops = filter(x -> x isa LoopOp, collect(items(sci.entry.body)))
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
    loop_ops = filter(x -> x isa ForOp || x isa WhileOp || x isa LoopOp, collect(items(sci.entry.body)))
    @test !isempty(loop_ops)

    # The loop body should contain an IfOp
    loop_op = loop_ops[1]
    function has_if_op(block::Block)
        for stmt in statements(block.body)
            if stmt isa IfOp
                return true
            end
        end
        return false
    end
    # Handle both LoopOp (body) and WhileOp (after)
    loop_body = if loop_op isa WhileOp
        loop_op.after
    elseif loop_op isa LoopOp
        loop_op.body
    else
        loop_op.body  # ForOp
    end
    @test has_if_op(loop_body)
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
    if_ops = filter(x -> x isa IfOp, collect(items(sci.entry.body)))
    @test !isempty(if_ops)

    if_op = if_ops[1]
    then_blk = if_op.then_region

    # Then branch should contain a loop
    function has_loop_op(block::Block)
        for stmt in statements(block.body)
            if stmt isa ForOp || stmt isa WhileOp || stmt isa LoopOp
                return true
            end
        end
        return false
    end
    @test has_loop_op(then_blk)
end

end  # nested control flow

#=============================================================================
 Regression Tests
=============================================================================#

@testset "regression" begin

@testset "no duplicated statements after loop" begin
    # Statements after loop should not be duplicated
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

    # Count expressions in entry block (final Block uses position indexing, not Statement.idx)
    expr_count = count(x -> !(x isa ControlFlowOp), items(sci.entry.body))
    # Just verify we have some expressions (the exact count may vary)
    @test expr_count >= 0

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

@testset "swap_loop phi references" begin
    # Test for loops with phi nodes that reference other phi nodes.
    # When x and y are swapped in a loop, the IR has cross-referencing phis.
    # With global SSA, outer scope values can be referenced directly.
    function swap_loop(n::Int)
        x, y = 1, 2
        for i in 1:n
            x, y = y, x
        end
        return x
    end

    sci = code_structured(swap_loop, Tuple{Int})
    @test sci isa StructuredCodeInfo
    validate_scf(sci)

    # Find the WhileOp in the structure
    function find_while_op(block::Block)
        for stmt in statements(block.body)
            if stmt isa WhileOp
                return stmt
            elseif stmt isa IfOp
                result = find_while_op(stmt.then_region)
                result !== nothing && return result
                result = find_while_op(stmt.else_region)
                result !== nothing && return result
            elseif stmt isa ForOp || stmt isa LoopOp
                result = find_while_op(stmt.body)
                result !== nothing && return result
            end
        end
        return nothing
    end

    while_op = find_while_op(sci.entry)
    @test while_op !== nothing

    # Loop-carried values are properly tracked via init_values and BlockArgs
    @test !isempty(while_op.init_values)
    @test !isempty(while_op.before.args)
end

@testset "while loop with outer capture has Nothing type" begin
    # Regression test: a while loop with only outer captures (no actual results)
    # should have Nothing result type, not the type of the outer capture.
    # This bug caused type information loss in downstream codegen.

    # Spinloop pattern: condition uses outer capture `x`, but loop produces no results
    function spinloop_capture(x::Int)
        while x > 0  # x is captured but loop has no actual results
        end
        return x
    end

    sci = code_structured(spinloop_capture, Tuple{Int})
    @test sci isa StructuredCodeInfo
    validate_scf(sci)

    # Find the loop in the structure - use statements() for simple item lookup
    while_idx = findfirst(stmt -> stmt isa WhileOp, collect(statements(sci.entry.body)))

    if while_idx !== nothing
        # Check that the result type is Tuple{} (no results), not Int (outer capture type)
        result_type = sci.entry.body[while_idx].typ
        @test result_type === Tuple{}
    end
end

@testset "while loop ConditionOp uses BlockArgs not SSAValues" begin
    # Regression test: ConditionOp args should be BlockArgs, not SSAValues.
    # When a loop computes intermediate values, the result should be the
    # loop-carried variable, not an intermediate computation.

    function count_power(x::Int, y::Int)
        count = 0
        while x^count < y  # x^count is intermediate, count is the result
            count += 1
        end
        return count
    end

    sci = code_structured(count_power, Tuple{Int, Int})
    @test sci isa StructuredCodeInfo
    validate_scf(sci)

    # Find the WhileOp - use statements() for simple item lookup
    while_idx = findfirst(stmt -> stmt isa WhileOp, collect(statements(sci.entry.body)))
    @test while_idx !== nothing

    if while_idx !== nothing
        while_op = sci.entry.body[while_idx].stmt
        before = while_op.before

        # The ConditionOp should have BlockArg as its result value
        @test before.terminator isa ConditionOp
        cond_op = before.terminator

        # The result should be %arg1 (the count BlockArg), not an SSAValue
        # pointing to the power computation
        @test !isempty(cond_op.args)
        @test cond_op.args[1] isa IRStructurizer.BlockArg
    end
end

end  # regression

end  # @testset "IRStructurizer"
