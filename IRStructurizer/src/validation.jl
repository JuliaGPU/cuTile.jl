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

Validate that all control flow has been converted to structured ops.
Throws `UnstructuredControlFlowError` if GotoNode/GotoIfNot remains.
"""
function validate_scf(sci::StructuredCodeInfo)
    unstructured = Int[]
    validate_block!(unstructured, sci.entry)
    isempty(unstructured) || throw(UnstructuredControlFlowError(sort!(unstructured)))
    return true
end

function validate_block!(unstructured::Vector{Int}, block::Block)
    for (idx, entry) in block.body
        stmt = entry.stmt
        if stmt isa GotoNode || stmt isa GotoIfNot
            push!(unstructured, idx)
        elseif stmt isa IfOp
            validate_block!(unstructured, stmt.then_region)
            validate_block!(unstructured, stmt.else_region)
        elseif stmt isa LoopOp
            validate_block!(unstructured, stmt.body)
        end
    end
end
