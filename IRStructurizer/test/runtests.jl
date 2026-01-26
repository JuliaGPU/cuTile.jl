using Test
using FileCheck

using IRStructurizer
using IRStructurizer: Block, ControlFlowOp, IfOp, ForOp, WhileOp, LoopOp,
                      YieldOp, ContinueOp, BreakOp, ConditionOp,
                      validate_scf, statements
using Core: SSAValue
using Base: code_ircode

@testset "IRStructurizer" verbose=true begin

#=============================================================================
 Interface Tests
=============================================================================#

@testset "interface" begin

@testset "low-level API" begin
    g(x) = x > 0 ? x + 1 : x - 1
    ir, _ = only(code_ircode(g, (Int,)))

    # Create flat view (no structurization)
    sci_flat = StructuredIRCode(ir; structurize=false, validate=false)
    @test !any(x -> x isa IfOp, statements(sci_flat.entry.body))

    # Create structured view
    sci = StructuredIRCode(ir)
    @test any(x -> x isa IfOp, statements(sci.entry.body))

    # code_structured is a convenience wrapper
    sci2, _ = code_structured(g, Tuple{Int}) |> only
    @test any(x -> x isa IfOp, statements(sci2.entry.body))
end

@testset "validation: UnstructuredControlFlowError" begin
    # Create unstructured view and verify validation fails
    g(x) = x > 0 ? x + 1 : x - 1
    ir, _ = only(code_ircode(g, (Int,)))

    # Flat view has GotoIfNot
    sci_flat = StructuredIRCode(ir; structurize=false, validate=false)
    gotoifnot_idx = findfirst(s -> s isa Core.GotoIfNot, ir.stmts.stmt)
    @test gotoifnot_idx !== nothing
    # Check that the GotoIfNot is in the body
    @test any(((_, entry),) -> entry.stmt isa Core.GotoIfNot, sci_flat.entry.body)

    # Validation should throw on unstructured IR
    @test_throws UnstructuredControlFlowError validate_scf(sci_flat)

    # Structured view passes validation
    sci = StructuredIRCode(ir)
    @test !any(expr -> expr isa Core.GotoIfNot, statements(sci.entry.body))
    validate_scf(sci)  # Should not throw
end

@testset "ForOp detection during CFG analysis" begin
    # Test that counting loops are detected as ForOp during CFG analysis
    sci, _ = code_structured(Tuple{Int}) do n::Int
        i = 0
        while i < n
            i += 1
        end
        return i
    end |> only

    # Counting loop should produce ForOp
    loop_ops = filter(x -> x isa ControlFlowOp, collect(statements(sci.entry.body)))
    @test !isempty(loop_ops)
    @test loop_ops[1] isa ForOp
end

@testset "display output format" begin
    # Verify display shows proper structure
    sci, _ = code_structured(Tuple{Bool}) do x::Bool
        x ? 1 : 2
    end |> only

    io = IOBuffer()
    show(io, MIME"text/plain"(), sci)
    output = String(take!(io))

    @test occursin("StructuredIRCode", output)
    @test occursin("if ", output)
    @test occursin("else", output)
    @test occursin("return", output)
end

end  # interface

#=============================================================================
 CFG Analysis Tests
 Tests that control flow regions are correctly identified.
=============================================================================#

@testset "CFG analysis" begin

@testset "acyclic regions" begin

@testset "block sequence" begin
    # Simple function: single addition (no control flow)
    @test @filecheck begin
        @check_not "if"
        code_structured(Tuple{Int}) do x
            @check "add_int"
            @check "return"
            x + 1
        end
    end

    # Multiple operations: (x + y) * (x - y)
    @test @filecheck begin
        @check_not "if"
        code_structured(Tuple{Int, Int}) do x, y
            @check "add_int"
            @check "sub_int"
            @check "mul_int"
            @check "return"
            (x + y) * (x - y)
        end
    end
end

@testset "if-then-else: diamond pattern" begin
    # Both branches converge (diamond CFG pattern)
    @test @filecheck begin
        code_structured(Tuple{Int}) do x::Int
            @check "slt_int"
            @check "if"
            @check "add_int"
            @check "else"
            @check "sub_int"
            @check "return"
            x > 0 ? x + 1 : x - 1
        end
    end
end

@testset "if-then-else: bool condition (no comparison)" begin
    # Bool condition directly, no comparison needed
    @test @filecheck begin
        code_structured(Tuple{Bool}) do x::Bool
            @check "if"
            @check "return 1"
            @check "else"
            @check "return 2"
            x ? 1 : 2
        end
    end
end

@testset "if-then-else: with comparison" begin
    @test @filecheck begin
        code_structured(Tuple{Int}) do x::Int
            @check "slt_int"
            @check "if"
            @check "return"
            @check "else"
            @check "neg_int"
            @check "return"
            x > 0 ? x : -x
        end
    end
end

@testset "termination: early return pattern" begin
    # One branch returns early, other continues
    @test @filecheck begin
        code_structured(Tuple{Int, Int}) do x::Int, y::Int
            @check "if"
            if x > y
                @check "mul_int"
                @check "return"
                return y * x
            end
            @check "else"
            @check "sub_int"
            @check "return"
            y - x
        end
    end
end

end  # acyclic regions

@testset "cyclic regions" begin

@testset "simple loop structure - ForOp" begin
    @test @filecheck begin
        code_structured(Tuple{Int}) do n::Int
            i = 0
            @check "for %{{.*}} ="
            while i < n
                i += 1
            end
            @check "continue"
            return i
        end
    end
end

@testset "loop with condition" begin
    # Loop with condition check at header (empty body - self-loop pattern)
    @test @filecheck begin
        code_structured(Tuple{Int}) do flag::Int
            @check "loop ->"
            while flag != 0
                @check "not_int"
                # spin
            end
            return flag
        end
    end
end

@testset "loop with body statements" begin
    @test @filecheck begin
        code_structured(Tuple{Int}) do n::Int
            @check "while"
            @check "slt_int"
            while n > 0
                @check "sub_int"
                n -= 1
            end
            return n
        end
    end
end

@testset "nested loops" begin
    @test @filecheck begin
        code_structured(Tuple{Int, Int}) do n::Int, m::Int
            acc = 0
            i = 0
            @check "for %{{.*}} ="
            while i < n
                j = 0
                @check "for %{{.*}} ="
                while j < m
                    acc += 1
                    j += 1
                end
                i += 1
            end
            return acc
        end
    end
end

end  # cyclic regions

end  # CFG analysis

#=============================================================================
 Loop Classification Tests
 Tests that loops are correctly classified into ForOp, WhileOp, or LoopOp.
 ForOp is detected directly during CFG analysis for counting patterns.
=============================================================================#

@testset "loop classification" begin

@testset "ForOp detection" begin

@testset "bounded counter" begin
    @test @filecheck begin
        code_structured(Tuple{Int}) do n::Int
            i = 0
            @check "for %{{.*}} ="
            while i < n
                i += 1
            end
            @check "continue"
            return i
        end
    end

    # Also verify ForOp bounds programmatically (FileCheck can't check these)
    sci, _ = code_structured(Tuple{Int}) do n::Int
        i = 0
        while i < n
            i += 1
        end
        return i
    end |> only
    for_ops = filter(x -> x isa ForOp, collect(statements(sci.entry.body)))
    @test length(for_ops) == 1

    for_op = for_ops[1]
    @test for_op.lower == 0
    @test for_op.upper isa Core.Argument
    @test for_op.step == 1
end

@testset "bounded counter with accumulator" begin
    @test @filecheck begin
        code_structured(Tuple{Int}) do n::Int
            i = 0
            acc = 0
            @check "for %{{.*}} ="
            while i < n
                @check "add_int"
                acc += i
                i += 1
            end
            @check "continue"
            return acc
        end
    end

    # Verify block args and init_values (FileCheck can't check these)
    sci, _ = code_structured(Tuple{Int}) do n::Int
        i = 0
        acc = 0
        while i < n
            acc += i
            i += 1
        end
        return acc
    end |> only
    for_ops = filter(x -> x isa ForOp, collect(statements(sci.entry.body)))
    @test length(for_ops) == 1

    for_op = for_ops[1]
    @test length(for_op.body.args) == 1
    @test length(for_op.init_values) == 1
end

@testset "Julia for-in-range (1:n) stays as LoopOp" begin
    # Native for-in-range has complex iterator protocol IR (multiple GotoIfNots)
    # so it stays as LoopOp, not ForOp. Use while-loops for ForOp.
    @test @filecheck begin
        code_structured(Tuple{Int}) do n::Int
            acc = 0
            @check "loop"
            for i in 1:n
                @check "add_int"
                acc += i
            end
            return acc
        end
    end

    sci, _ = code_structured(Tuple{Int}) do n::Int
        acc = 0
        for i in 1:n
            acc += i
        end
        return acc
    end |> only
    validate_scf(sci)

    # Verify IR is valid (LoopOp is nested inside IfOps from iterator protocol)
    @test sci isa StructuredIRCode
end

@testset "nested for loops" begin
    @test @filecheck begin
        code_structured(Tuple{Int, Int}) do n::Int, m::Int
            acc = 0
            i = 0
            @check "for"
            while i < n
                j = 0
                @check "for"
                while j < m
                    acc += 1
                    j += 1
                end
                i += 1
            end
            return acc
        end
    end
end

end  # ForOp detection

@testset "WhileOp detection" begin

@testset "condition-only spinloop" begin
    @test @filecheck begin
        code_structured(Tuple{Int}) do flag::Int
            @check "loop ->"
            while flag != 0
                @check "not_int"
            end
            return flag
        end
    end
end

@testset "decrementing loop (non-ForOp pattern)" begin
    @test @filecheck begin
        code_structured(Tuple{Int}) do n::Int
            @check "while"
            @check "slt_int"
            while n > 0
                @check "sub_int"
                n -= 1
            end
            return n
        end
    end
end

end  # WhileOp detection

@testset "WhileOp/LoopOp fallback" begin

@testset "dynamic step" begin
    # Loop where step is modified inside loop body (not a valid ForOp)
    sci, _ = code_structured(Tuple{Int}) do n::Int
        i = 0
        step = 1
        while i < n
            i += step
            step += 1
        end
        return i
    end |> only
    @test sci isa StructuredIRCode

    # Should have some loop op (not ForOp since step changes)
    loop_ops = filter(x -> x isa ForOp || x isa WhileOp || x isa LoopOp, collect(statements(sci.entry.body)))
    @test length(loop_ops) >= 1
end

end  # WhileOp/LoopOp fallback

end  # loop classification

#=============================================================================
 Nested Control Flow Tests
=============================================================================#

@testset "nested control flow" begin

@testset "if inside loop" begin
    @test @filecheck begin
        code_structured(Tuple{Int}) do n::Int
            acc = 0
            i = 0
            @check "for"
            while i < n
                @check "if"
                if i % 2 == 0
                    @check "add_int"
                    acc += i
                end
                i += 1
            end
            return acc
        end
    end
end

@testset "loop inside if" begin
    @test @filecheck begin
        code_structured(Tuple{Int, Int}) do x::Int, n::Int
            @check "if"
            if x > 0
                i = 0
                @check "for"
                while i < n
                    i += 1
                end
                return i
            @check "else"
            else
                @check "return 0"
                return 0
            end
        end
    end
end

end  # nested control flow

#=============================================================================
 Regression Tests
=============================================================================#

@testset "regression" begin

@testset "no duplicated statements after loop" begin
    @test @filecheck begin
        code_structured(Tuple{Int}) do x::Int
            i = 0
            @check "for"
            while i < x
                i += 1
            end
            # This should appear exactly once
            @check "mul_int"
            result = i * 2
            @check_not "mul_int"
            @check "return"
            return result
        end
    end
end

@testset "type preservation" begin
    sci, _ = code_structured(Tuple{Float64}) do x::Float64
        x + 1.0
    end |> only

    # Float64 type should be preserved in entry block types
    @test !isempty(sci.entry.body)
    @test any(((_, entry),) -> entry.typ isa Type && entry.typ <: AbstractFloat, sci.entry.body)
end

@testset "multiple arguments" begin
    sci, _ = code_structured(Tuple{Int, Float64}) do x::Int, y::Float64
        x + y
    end |> only
    @test sci.entry.terminator isa Core.ReturnNode
end

@testset "swap_loop phi references" begin
    # Native for-in-range produces LoopOp (iterator protocol is non-SESE)
    @test @filecheck begin
        code_structured(Tuple{Int}) do n::Int
            x, y = 1, 2
            @check "loop"
            for i in 1:n
                x, y = y, x
            end
            return x
        end
    end

    sci, _ = code_structured(Tuple{Int}) do n::Int
        x, y = 1, 2
        for i in 1:n
            x, y = y, x
        end
        return x
    end |> only
    validate_scf(sci)
end

@testset "while loop with outer capture has Nothing type" begin
    # Regression test: a while loop with only outer captures (no actual results)
    # should have Nothing result type, not the type of the outer capture.

    sci, _ = code_structured(Tuple{Int}) do x::Int
        while x > 0
        end
        return x
    end |> only
    validate_scf(sci)

    # Find the loop in the structure
    while_idx = findfirst(stmt -> stmt isa WhileOp, collect(statements(sci.entry.body)))
    if while_idx !== nothing
        # Check that the result type is Tuple{} (no results), not Int
        result_type = sci.entry.body[while_idx].typ
        @test result_type === Tuple{}
    end
end

@testset "while loop ConditionOp uses BlockArgs not SSAValues" begin
    # Regression test: ConditionOp args should be BlockArgs, not SSAValues.

    sci, _ = code_structured(Tuple{Int, Int}) do x::Int, y::Int
        count = 0
        while x^count < y
            count += 1
        end
        return count
    end |> only
    validate_scf(sci)

    while_idx = findfirst(stmt -> stmt isa WhileOp, collect(statements(sci.entry.body)))
    @test while_idx !== nothing

    if while_idx !== nothing
        while_op = sci.entry.body[while_idx].stmt
        before = while_op.before

        @test before.terminator isa ConditionOp
        cond_op = before.terminator

        # The result should be BlockArg, not SSAValue
        @test !isempty(cond_op.args)
        @test cond_op.args[1] isa IRStructurizer.BlockArg
    end
end

@testset "SESE while-loop becomes ForOp, non-SESE stays LoopOp" begin
    # Simple SESE while-loop → ForOp
    sci_while, _ = code_structured(Tuple{Int}) do n::Int
        i = 0
        acc = 0
        while i < n
            acc += i
            i += 1
        end
        return acc
    end |> only
    validate_scf(sci_while)
    for_ops = filter(x -> x isa ForOp, collect(statements(sci_while.entry.body)))
    @test length(for_ops) == 1

    # Native for-in-range (non-SESE due to iterator protocol) → LoopOp
    # LoopOp is nested inside IfOps from iterator protocol's branch structure
    sci_for, _ = code_structured(Tuple{Int}) do n::Int
        acc = 0
        for i in 1:n
            acc += i
        end
        return acc
    end |> only
    validate_scf(sci_for)
    # LoopOp will be nested inside IfOps, just verify the IR is valid
    @test sci_for isa StructuredIRCode
end

@testset "while-loop mimicking iterator protocol stays valid" begin
    # A while-loop that performs operations similar to the iterator protocol
    # (multiple branches, comparisons) should still produce valid structured IR.
    # This previously caused issues when non-SESE loops were incorrectly matched.
    sci, _ = code_structured(Tuple{Int}) do n::Int
        # Mimic iterator: check if done, extract value, update state
        state = 1
        upper = n
        acc = 0
        while true
            # "done" check - similar to iterator protocol
            done = state > upper
            done && break
            # "extract" value
            i = state
            # body
            acc += i
            # "next" state
            state += 1
        end
        return acc
    end |> only

    # Should produce valid structured IR (no unstructured control flow)
    validate_scf(sci)
end

# If-then (no else) must yield phi values, not return Nothing
@testset "if-then yields phi values" begin
    @test @filecheck begin
        code_structured(Tuple{Bool}) do flag::Bool
            x = 0
            @check "if"
            if flag
                x = 1
            end
            @check "yield"
            @check "else"
            @check "yield"
            @check "getfield"
            return x
        end
    end
end

@testset "if-then phi inside loop" begin
    @test @filecheck begin
        code_structured(Tuple{Int, Bool}) do n::Int, flag::Bool
            acc = 0
            j = 1
            @check "for"
            while j <= n
                x = 0
                @check "if"
                if flag && j >= 2
                    x = 1
                end
                @check "yield"
                @check "getfield"
                acc += x
                j += 1
            end
            return acc
        end
    end
end

@testset "if-then with multiple phis" begin
    @test @filecheck begin
        code_structured(Tuple{Bool}) do flag::Bool
            x, y = 0, 0
            @check "if"
            if flag
                x, y = 1, 2
            end
            @check "yield"
            @check "getfield"
            @check "getfield"
            return x + y
        end
    end
end

end  # regression

#=============================================================================
 Integration Tests: Julia for-in-range patterns
=============================================================================#

@testset "Julia for-in-range integration" begin


@testset "sum_to_n: accumulator pattern" begin
    # Native for-in-range stays as LoopOp (iterator protocol is non-SESE)
    @test @filecheck begin
        code_structured(Tuple{Int}) do n
            acc = 0
            @check "loop"
            for i in 1:n
                @check "add_int"
                acc += i
            end
            return acc
        end
    end

    sci, _ = code_structured(Tuple{Int}) do n
        acc = 0
        for i in 1:n
            acc += i
        end
        return acc
    end |> only
    validate_scf(sci)
end

@testset "product: multiply pattern" begin
    @test @filecheck begin
        code_structured(Tuple{Int}) do n
            acc = 1
            @check "loop"
            for i in 1:n
                @check "mul_int"
                acc *= i
            end
            return acc
        end
    end

    sci, _ = code_structured(Tuple{Int}) do n
        acc = 1
        for i in 1:n
            acc *= i
        end
        return acc
    end |> only
    validate_scf(sci)
end

@testset "count_evens: conditional accumulator" begin
    @test @filecheck begin
        code_structured(Tuple{Int}) do n
            count = 0
            for i in 1:n
                @check "if"
                @check "rem_int"
                if i % 2 == 0
                    count += 1
                end
            end
            return count
        end
    end

    sci, _ = code_structured(Tuple{Int}) do n
        count = 0
        for i in 1:n
            if i % 2 == 0
                count += 1
            end
        end
        return count
    end |> only
    validate_scf(sci)
end

@testset "multiple accumulators" begin
    @test @filecheck begin
        code_structured(Tuple{Int}) do n
            sum = 0
            count = 0
            @check "loop"
            for i in 1:n
                @check "add_int"
                sum += i
                @check "add_int"
                count += 1
            end
            return sum, count
        end
    end

    sci, _ = code_structured(Tuple{Int}) do n
        sum = 0
        count = 0
        for i in 1:n
            sum += i
            count += 1
        end
        return sum, count
    end |> only
    validate_scf(sci)
end

@testset "nested for-in-range loops" begin
    # Both native for-in-range loops produce LoopOp (iterator protocol is non-SESE)
    @test @filecheck begin
        code_structured(Tuple{Int, Int}) do n, m
            acc = 0
            @check "loop"
            for i in 1:n
                @check "loop"
                for j in 1:m
                    @check "mul_int"
                    acc += i * j
                end
            end
            return acc
        end
    end

    sci, _ = code_structured(Tuple{Int, Int}) do n, m
        acc = 0
        for i in 1:n
            for j in 1:m
                acc += i * j
            end
        end
        return acc
    end |> only
    validate_scf(sci)
end

@testset "for-in-range with tuple destructuring" begin
    @test @filecheck begin
        code_structured(Tuple{Int}) do n
            x, y = 1, 2
            @check "loop"
            for i in 1:n
                x, y = y, x
            end
            return x
        end
    end

    sci, _ = code_structured(Tuple{Int}) do n
        x, y = 1, 2
        for i in 1:n
            x, y = y, x
        end
        return x
    end |> only
    validate_scf(sci)
end

@testset "for-in-range produces valid LoopOp" begin
    # Native for-in-range stays as LoopOp (iterator protocol is non-SESE)
    @test @filecheck begin
        code_structured(Tuple{Int}) do n
            last = 0
            @check "loop"
            for i in 1:n
                last = i
            end
            return last
        end
    end

    sci, _ = code_structured(Tuple{Int}) do n
        last = 0
        for i in 1:n
            last = i
        end
        return last
    end |> only
    validate_scf(sci)
end

end  # Julia for-in-range integration

end  # @testset "IRStructurizer"
