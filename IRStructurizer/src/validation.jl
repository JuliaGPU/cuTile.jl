# structured IR validation

export UnstructuredControlFlowError, check_global_ssa_refs

"""
    UnstructuredControlFlowError <: Exception

Exception thrown when unstructured control flow is detected in structured IR.
"""
struct UnstructuredControlFlowError <: Exception
    stmt_indices::Vector{Int}
end

function Base.showerror(io::IO, e::UnstructuredControlFlowError)
    print(io, "UnstructuredControlFlowError: unstructured control flow at statement(s): ",
          join(e.stmt_indices, ", "))
end

"""
    validate_scf(sci::StructuredCodeInfo) -> Bool

Validate that all control flow in the original CodeInfo has been properly
converted to structured control flow operations (IfOp, LoopOp, ForOp).

Throws `UnstructuredControlFlowError` if unstructured control flow remains.
Returns `true` if all control flow is properly structured.

The invariant is simple: no Statement in any `block.body` should contain
a `GotoNode` or `GotoIfNot` expression - those should have been replaced by
structured ops that the visitor descends into.
"""
function validate_scf(sci::StructuredCodeInfo)
    unstructured = Int[]

    # Walk all blocks and check that no statement is unstructured control flow
    each_stmt(sci.entry) do stmt::Statement
        if stmt.expr isa GotoNode || stmt.expr isa GotoIfNot
            push!(unstructured, stmt.idx)
        end
    end

    if !isempty(unstructured)
        throw(UnstructuredControlFlowError(sort!(unstructured)))
    end

    return true
end

#=============================================================================
 Global SSA Reference Validation
 Detects SSAValue references that should have been converted to LocalSSAValue
 or BlockArg. Used during the migration to local SSA numbering.
=============================================================================#

"""
    check_global_ssa_refs(sci::StructuredCodeInfo; strict::Bool=false) -> Int
    check_global_ssa_refs(block::Block; strict::Bool=false) -> Int

Count SSAValue references remaining in the structured IR.
Returns the count. If strict=true, errors when count > 0.

During migration from global to local SSA numbering:
- **Default (warning)**: Code continues to work, we track progress
- **Strict mode**: Enable once conversion is complete to enforce the invariant

SSAValue references are only valid at the top-level entry block. Inside nested
control flow blocks (IfOp, ForOp, WhileOp, LoopOp), all references should be
either LocalSSAValue (for values defined in the same block) or BlockArg (for
values captured from parent scope).
"""
function check_global_ssa_refs(sci::StructuredCodeInfo; strict::Bool=false)
    check_global_ssa_refs(sci.entry; strict, is_entry=true)
end

function check_global_ssa_refs(block::Block; strict::Bool=false, is_entry::Bool=false)
    count = count_ssavalues_in_nested(block; is_entry)
    if count > 0
        msg = "Block has $count global SSAValue references in nested structures"
        if strict
            error(msg)
        else
            @warn msg
        end
    end
    return count
end

"""
    count_ssavalues_in_nested(block::Block; is_entry::Bool=false) -> Int

Count SSAValue references inside nested control flow structures.
SSAValues in the entry block are allowed (they reference the original CodeInfo),
but SSAValues inside nested blocks (IfOp, ForOp, etc.) should have been
converted to LocalSSAValue or BlockArg.
"""
function count_ssavalues_in_nested(block::Block; is_entry::Bool=false)
    count = 0

    for item in block.body
        if item isa Statement
            if !is_entry
                # Inside nested block, count SSAValues in the expression
                count += count_ssavalues_in_value(item.expr)
            end
        else
            # Control flow op - always check nested blocks
            count += count_ssavalues_in_op(item)
        end
    end

    # Check terminator (except in entry block)
    if !is_entry && block.terminator !== nothing
        count += count_ssavalues_in_terminator(block.terminator)
    end

    return count
end

function count_ssavalues_in_op(op::IfOp)
    count = count_ssavalues_in_value(op.condition)
    count += count_ssavalues_in_nested(op.then_block)
    count += count_ssavalues_in_nested(op.else_block)
    return count
end

function count_ssavalues_in_op(op::ForOp)
    count = count_ssavalues_in_value(op.lower)
    count += count_ssavalues_in_value(op.upper)
    count += count_ssavalues_in_value(op.step)
    for v in op.init_values
        count += count_ssavalues_in_value(v)
    end
    count += count_ssavalues_in_nested(op.body)
    return count
end

function count_ssavalues_in_op(op::LoopOp)
    count = 0
    for v in op.init_values
        count += count_ssavalues_in_value(v)
    end
    count += count_ssavalues_in_nested(op.body)
    return count
end

function count_ssavalues_in_op(op::WhileOp)
    count = 0
    for v in op.init_values
        count += count_ssavalues_in_value(v)
    end
    count += count_ssavalues_in_nested(op.before)
    count += count_ssavalues_in_nested(op.after)
    return count
end

function count_ssavalues_in_terminator(term::YieldOp)
    count = 0
    for v in term.values
        count += count_ssavalues_in_value(v)
    end
    return count
end

function count_ssavalues_in_terminator(term::ContinueOp)
    count = 0
    for v in term.values
        count += count_ssavalues_in_value(v)
    end
    return count
end

function count_ssavalues_in_terminator(term::BreakOp)
    count = 0
    for v in term.values
        count += count_ssavalues_in_value(v)
    end
    return count
end

function count_ssavalues_in_terminator(term::ConditionOp)
    count = count_ssavalues_in_value(term.condition)
    for v in term.args
        count += count_ssavalues_in_value(v)
    end
    return count
end

function count_ssavalues_in_terminator(::ReturnNode)
    return 0  # ReturnNode in nested blocks handled elsewhere
end

function count_ssavalues_in_terminator(::Nothing)
    return 0
end

"""
    count_ssavalues_in_value(value) -> Int

Count SSAValue occurrences in an IR value, recursively traversing Expr trees.
"""
count_ssavalues_in_value(::SSAValue) = 1
count_ssavalues_in_value(::BlockArg) = 0
count_ssavalues_in_value(::LocalSSAValue) = 0
count_ssavalues_in_value(::Argument) = 0
count_ssavalues_in_value(::SlotNumber) = 0
count_ssavalues_in_value(::GlobalRef) = 0
count_ssavalues_in_value(::QuoteNode) = 0
count_ssavalues_in_value(::Nothing) = 0
count_ssavalues_in_value(::Number) = 0
count_ssavalues_in_value(::Symbol) = 0
count_ssavalues_in_value(::Type) = 0
count_ssavalues_in_value(::PiNode) = 0  # PiNode handled separately if needed

function count_ssavalues_in_value(expr::Expr)
    count = 0
    for arg in expr.args
        count += count_ssavalues_in_value(arg)
    end
    return count
end

# Fallback for other types
count_ssavalues_in_value(_) = 0
