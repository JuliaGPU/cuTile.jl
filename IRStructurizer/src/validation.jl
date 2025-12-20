# structured IR validation

export UnstructuredControlFlowError

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

The invariant is simple: no statement/operation should contain
a `GotoNode` or `GotoIfNot` expression - those should have been replaced by
structured ops that the visitor descends into.
"""
function validate_scf(sci::StructuredCodeInfo)
    unstructured = Int[]

    # Check both body (pre-conversion) and ops (post-conversion)
    _validate_block!(unstructured, sci.entry)

    if !isempty(unstructured)
        throw(UnstructuredControlFlowError(sort!(unstructured)))
    end

    return true
end

function _validate_block!(unstructured::Vector{Int}, block::Block)
    # Check all ops for unstructured control flow
    for (i, op) in enumerate(block.ops)
        if op.expr isa GotoNode || op.expr isa GotoIfNot
            push!(unstructured, i)
        elseif op.expr isa ControlFlowOp
            _validate_cfop!(unstructured, op.expr)
        end
    end
end

function _validate_cfop!(unstructured::Vector{Int}, op::IfOp)
    _validate_block!(unstructured, op.then_block)
    _validate_block!(unstructured, op.else_block)
end

function _validate_cfop!(unstructured::Vector{Int}, op::ForOp)
    _validate_block!(unstructured, op.body)
end

function _validate_cfop!(unstructured::Vector{Int}, op::LoopOp)
    _validate_block!(unstructured, op.body)
end

function _validate_cfop!(unstructured::Vector{Int}, op::WhileOp)
    _validate_block!(unstructured, op.before)
    _validate_block!(unstructured, op.after)
end
