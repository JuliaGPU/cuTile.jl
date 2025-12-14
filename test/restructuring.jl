@testset "restructuring" verbose=true begin

using cuTile: code_structured, StructuredCodeInfo, Block, IfOp, ForOp, LoopOp,
              YieldOp, ContinueOp, BreakOp, validate_structured_control_flow,
              UnstructuredControlFlowError, get_typed_ir

# Helper to check if a block contains a specific control flow op type
function has_nested_op(block::Block, ::Type{T}) where T
    any(op -> op isa T, block.nested)
end

# Helper to count nested ops of a type
function count_nested_ops(block::Block, ::Type{T}) where T
    count(op -> op isa T, block.nested)
end

# Recursive helper to find all ops of a type in a StructuredCodeInfo
function find_all_ops(sci::StructuredCodeInfo, ::Type{T}) where T
    ops = T[]
    find_ops_in_block!(ops, sci.entry, T)
    return ops
end

function find_ops_in_block!(ops::Vector{T}, block::Block, ::Type{T}) where T
    for op in block.nested
        if op isa T
            push!(ops, op)
        end
        find_ops_in_op!(ops, op, T)
    end
end

function find_ops_in_op!(ops::Vector{T}, op::IfOp, ::Type{T}) where T
    find_ops_in_block!(ops, op.then_block, T)
    find_ops_in_block!(ops, op.else_block, T)
end

function find_ops_in_op!(ops::Vector{T}, op::ForOp, ::Type{T}) where T
    find_ops_in_block!(ops, op.body, T)
end

function find_ops_in_op!(ops::Vector{T}, op::LoopOp, ::Type{T}) where T
    find_ops_in_block!(ops, op.body, T)
end

@testset "straight-line code" begin
    # Simple function with no control flow
    f(x) = x + 1

    sci = code_structured(f, Tuple{Int})
    @test sci isa StructuredCodeInfo
    @test !isempty(sci.entry.stmts)
    @test isempty(sci.entry.nested)  # No nested control flow
    @test sci.entry.terminator isa Core.ReturnNode

    # Multiple operations, still straight-line
    g(x, y) = (x + y) * (x - y)

    sci = code_structured(g, Tuple{Int, Int})
    @test sci isa StructuredCodeInfo
    @test isempty(sci.entry.nested)
    @test sci.entry.terminator isa Core.ReturnNode
end

# Note: Julia's optimized IR often has multiple returns instead of merging
# branches, which makes pattern matching difficult. These tests verify
# that code_structured handles control flow gracefully, even if not
# fully restructured into nested ops.
@testset "control flow handling" begin
    # Ternary operator - may fall back to flat representation
    # due to separate returns in each branch
    f(x) = x > 0 ? x + 1 : x - 1

    sci = code_structured(f, Tuple{Int})
    @test sci isa StructuredCodeInfo
    # Should have statements (either restructured or flat)
    @test !isempty(sci.entry.stmts) || !isempty(sci.entry.nested)

    # Multiple returns
    function multi_return(x)
        if x < 0
            return -1
        elseif x == 0
            return 0
        else
            return 1
        end
    end

    sci = code_structured(multi_return, Tuple{Int})
    @test sci isa StructuredCodeInfo

    # Boolean short-circuit
    f_and(x, y) = x > 0 && y > 0
    sci = code_structured(f_and, Tuple{Int, Int})
    @test sci isa StructuredCodeInfo

    f_or(x, y) = x > 0 || y > 0
    sci = code_structured(f_or, Tuple{Int, Int})
    @test sci isa StructuredCodeInfo
end

@testset "terminating if-then-else" begin
    # Simple ternary with Bool condition - both branches return
    f(x) = x ? 1 : 2
    sci = code_structured(f, Tuple{Bool})
    @test sci isa StructuredCodeInfo
    @test has_nested_op(sci.entry, IfOp)

    # Verify the IfOp structure
    if_ops = find_all_ops(sci, IfOp)
    @test length(if_ops) == 1
    if_op = if_ops[1]
    @test if_op.then_block.terminator isa Core.ReturnNode
    @test if_op.else_block.terminator isa Core.ReturnNode

    # Ternary with computed condition - stmt before if, both branches compute + return
    g(x) = x > 0 ? x + 1 : x - 1
    sci = code_structured(g, Tuple{Int})
    @test sci isa StructuredCodeInfo
    @test has_nested_op(sci.entry, IfOp)

    # Should have condition computation before the IfOp
    @test !isempty(sci.entry.stmts)

    # Verify IfOp with computations in branches
    if_ops = find_all_ops(sci, IfOp)
    @test length(if_ops) == 1
    if_op = if_ops[1]
    # Each branch should have a computation statement and a return
    @test !isempty(if_op.then_block.stmts)
    @test !isempty(if_op.else_block.stmts)
    @test if_op.then_block.terminator isa Core.ReturnNode
    @test if_op.else_block.terminator isa Core.ReturnNode

    # Display should show if structure
    io = IOBuffer()
    show(io, MIME"text/plain"(), sci)
    output = String(take!(io))
    @test occursin("if %", output)  # Julia-style if
    @test occursin("return", output)

    # If-else with early return - the foo(x, y) example
    # Tests multi-statement branches with computations before returns
    function foo(x, y)
        if x > y
            return y * x
        end
        y^2 - x
    end

    sci = code_structured(foo, Tuple{Int, Int})
    @test sci isa StructuredCodeInfo
    @test has_nested_op(sci.entry, IfOp)

    # Should have condition computation before the IfOp
    @test !isempty(sci.entry.stmts)

    # Verify IfOp structure
    if_ops = find_all_ops(sci, IfOp)
    @test length(if_ops) == 1
    if_op = if_ops[1]

    # Then branch: y * x, return
    @test !isempty(if_op.then_block.stmts)
    @test if_op.then_block.terminator isa Core.ReturnNode

    # Else branch: y^2 - x computation, return
    @test !isempty(if_op.else_block.stmts)
    @test if_op.else_block.terminator isa Core.ReturnNode

    # Display should show proper structure
    io = IOBuffer()
    show(io, MIME"text/plain"(), sci)
    output = String(take!(io))
    @test occursin("if %", output)
    @test occursin("mul_int", output)  # y * x
    @test occursin("sub_int", output)  # y^2 - x
    @test count("return", output) == 2  # Both branches return
end

@testset "display output" begin
    # Test straight-line display
    f(x) = x + 1
    sci = code_structured(f, Tuple{Int})

    io = IOBuffer()
    show(io, MIME"text/plain"(), sci)
    output = String(take!(io))

    @test occursin("StructuredCodeInfo", output)
    @test occursin("%", output)  # Has SSA values
    @test occursin("return", output)  # Has terminator

    # Test compact display
    io = IOBuffer()
    show(io, sci)
    output = String(take!(io))

    @test occursin("StructuredCodeInfo", output)
    @test occursin("stmts", output)
end

@testset "display with control flow" begin
    # Terminating if-then-else should be properly restructured
    f(x) = x > 0 ? x + 1 : x - 1
    sci = code_structured(f, Tuple{Int})

    io = IOBuffer()
    show(io, MIME"text/plain"(), sci)
    output = String(take!(io))

    @test occursin("StructuredCodeInfo", output)
    @test occursin("if %", output)  # Julia-style if
    @test occursin("return", output)  # Both branches have returns
end

@testset "loop handling" begin
    # Simple while loop with accumulator - the bar(x, y) example from PLAN
    function bar(x, y)
        acc = 0
        while acc < x
            acc += y
        end
        return acc
    end

    sci = code_structured(bar, Tuple{Int, Int})
    @test sci isa StructuredCodeInfo

    # Should have detected a LoopOp
    loop_ops = find_all_ops(sci, LoopOp)
    @test length(loop_ops) == 1

    loop_op = loop_ops[1]
    # Loop should have init values (the initial accumulator value)
    @test !isempty(loop_op.init_values)
    # Loop body should have block arguments (for carried values)
    @test !isempty(loop_op.body.args)
    # Loop body should have statements (the condition and increment)
    @test !isempty(loop_op.body.stmts)
    # Loop body should have nested IfOp for the condition
    @test has_nested_op(loop_op.body, IfOp)

    # Check the inner IfOp structure
    inner_if_ops = find_all_ops(sci, IfOp)
    @test length(inner_if_ops) >= 1
    inner_if = inner_if_ops[end]  # The one inside the loop
    # Then branch should continue
    @test inner_if.then_block.terminator isa ContinueOp
    # Else branch should break (exit loop)
    @test inner_if.else_block.terminator isa BreakOp

    # Display should show while structure
    io = IOBuffer()
    show(io, MIME"text/plain"(), sci)
    output = String(take!(io))
    @test occursin("while", output)
    @test occursin("continue", output)
    @test occursin("break", output)

    # Count-down while loop
    function count_down(n)
        while n > 0
            n -= 1
        end
        return n
    end

    sci = code_structured(count_down, Tuple{Int})
    @test sci isa StructuredCodeInfo
    loop_ops = find_all_ops(sci, LoopOp)
    @test length(loop_ops) == 1

    # Simple for loop (converts to while-like IR)
    function sum_to_n(n)
        total = 0
        for i in 1:n
            total += i
        end
        return total
    end

    sci = code_structured(sum_to_n, Tuple{Int})
    @test sci isa StructuredCodeInfo
    # Note: For loops may have more complex IR due to iterate() calls
end

@testset "type preservation" begin
    # Verify that types from original CodeInfo are preserved
    f(x::Float64) = x + 1.0

    sci = code_structured(f, Tuple{Float64})
    @test sci isa StructuredCodeInfo

    # The underlying CodeInfo should have proper types
    @test !isempty(sci.code.ssavaluetypes)
    # Float64 operations should appear
    @test any(t -> t isa Type && t <: AbstractFloat, sci.code.ssavaluetypes)
end

@testset "argument handling" begin
    # Single argument
    f(x) = x * 2
    sci = code_structured(f, Tuple{Int})
    @test sci isa StructuredCodeInfo

    # Multiple arguments
    g(x, y, z) = x + y + z
    sci = code_structured(g, Tuple{Int, Int, Int})
    @test sci isa StructuredCodeInfo

    # Different types
    h(x::Int, y::Float64) = x + y
    sci = code_structured(h, Tuple{Int, Float64})
    @test sci isa StructuredCodeInfo
end

@testset "validation" begin
    # Straight-line code should pass validation
    f(x) = x + 1
    sci = code_structured(f, Tuple{Int}; validate=true)
    @test sci isa StructuredCodeInfo
    @test validate_structured_control_flow(sci)

    # Simple ternary should pass validation
    g(x) = x > 0 ? x + 1 : x - 1
    sci = code_structured(g, Tuple{Int}; validate=true)
    @test sci isa StructuredCodeInfo

    # If-then-else terminating should pass
    function foo(x, y)
        if x > y
            return y * x
        end
        y^2 - x
    end
    sci = code_structured(foo, Tuple{Int, Int}; validate=true)
    @test sci isa StructuredCodeInfo

    # While loop should pass validation
    function bar(x, y)
        acc = 0
        while acc < x
            acc += y
        end
        return acc
    end
    sci = code_structured(bar, Tuple{Int, Int}; validate=true)
    @test sci isa StructuredCodeInfo

    # Test that validate=false doesn't throw (default behavior)
    h(x) = x > 0 ? x : -x
    sci = code_structured(h, Tuple{Int})  # validate=false by default
    @test sci isa StructuredCodeInfo

    # Test direct validation call
    sci = code_structured(f, Tuple{Int})
    @test validate_structured_control_flow(sci) === true

    # Test UnstructuredControlFlowError is actually thrown for unstructured control flow
    # Manually create a StructuredCodeInfo with a GotoIfNot in stmts
    g_with_cf(x) = x > 0 ? x + 1 : x - 1
    ci, _ = get_typed_ir(g_with_cf, Tuple{Int})
    # Find the GotoIfNot statement index
    gotoifnot_idx = findfirst(s -> s isa Core.GotoIfNot, ci.code)
    @test gotoifnot_idx !== nothing
    # Create a block that incorrectly includes the GotoIfNot
    bad_block = Block(1)
    push!(bad_block.stmts, gotoifnot_idx)
    bad_sci = StructuredCodeInfo(ci, bad_block)
    @test_throws UnstructuredControlFlowError validate_structured_control_flow(bad_sci)

    # Verify error contains the statement index
    try
        validate_structured_control_flow(bad_sci)
    catch e
        @test e isa UnstructuredControlFlowError
        @test gotoifnot_idx in e.stmt_indices
    end
end

end  # @testset "restructuring"
