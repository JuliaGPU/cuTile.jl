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

The invariant is simple: no statement index in any `block.body` should point
to a `GotoNode` or `GotoIfNot` - those should have been replaced by structured
ops that the visitor descends into.
"""
function validate_scf(sci::StructuredCodeInfo)
    stmts = sci.code.code
    unstructured = Int[]

    # Walk all blocks and check that no statement is unstructured control flow
    each_stmt(sci.entry) do idx
        stmt = stmts[idx]
        if stmt isa GotoNode || stmt isa GotoIfNot
            push!(unstructured, idx)
        end
    end

    if !isempty(unstructured)
        throw(UnstructuredControlFlowError(sort!(unstructured)))
    end

    return true
end
