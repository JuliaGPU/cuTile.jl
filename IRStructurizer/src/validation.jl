# structured IR validation

export UnstructuredControlFlowError, UnsubstitutedPhiError

"""
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
Exception thrown when phi nodes remain after block arg substitution.
"""
struct UnsubstitutedPhiError <: Exception
    stmt_indices::Vector{Int}
end

function Base.showerror(io::IO, e::UnsubstitutedPhiError)
    print(io, "UnsubstitutedPhiError: phi nodes remain at statement(s): ",
          join(e.stmt_indices, ", "))
end

"""
    validate_scf(entry::Block) -> Bool

Validate that all control flow has been converted to structured ops.
Throws `UnstructuredControlFlowError` if GotoNode/GotoIfNot remains.
"""
function validate_scf(entry::Block)
    unstructured = Int[]
    validate_no_gotos!(unstructured, entry)
    isempty(unstructured) || throw(UnstructuredControlFlowError(sort!(unstructured)))
    return true
end

validate_scf(sci::StructuredIRCode) = validate_scf(sci.entry)

function validate_no_gotos!(bad::Vector{Int}, block::Block)
    for (idx, entry) in block.body
        stmt = entry.stmt
        if stmt isa GotoNode || stmt isa GotoIfNot
            push!(bad, idx)
        elseif stmt isa IfOp
            validate_no_gotos!(bad, stmt.then_region)
            validate_no_gotos!(bad, stmt.else_region)
        elseif stmt isa LoopOp
            validate_no_gotos!(bad, stmt.body)
        end
    end
end

"""
    validate_no_phis(entry::Block) -> Bool

Validate that all phi nodes have been converted to BlockArgs.
Throws `UnsubstitutedPhiError` if PhiNode expressions remain.
"""
function validate_no_phis(entry::Block)
    remaining = Int[]
    validate_no_phis!(remaining, entry)
    isempty(remaining) || throw(UnsubstitutedPhiError(sort!(remaining)))
    return true
end

validate_no_phis(sci::StructuredIRCode) = validate_no_phis(sci.entry)

function validate_no_phis!(bad::Vector{Int}, block::Block)
    for (idx, entry) in block.body
        stmt = entry.stmt
        if stmt isa PhiNode
            push!(bad, idx)
        elseif stmt isa IfOp
            validate_no_phis!(bad, stmt.then_region)
            validate_no_phis!(bad, stmt.else_region)
        elseif stmt isa LoopOp
            validate_no_phis!(bad, stmt.body)
        end
    end
end
