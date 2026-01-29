# structured IR validation

export UnstructuredControlFlowError, UnsubstitutedPhiError, InvalidTerminatorError

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

"""
Exception thrown when structured control flow ops have invalid terminators.
"""
struct InvalidTerminatorError <: Exception
    messages::Vector{String}
end

function Base.showerror(io::IO, e::InvalidTerminatorError)
    print(io, "InvalidTerminatorError: ")
    for (i, msg) in enumerate(e.messages)
        i > 1 && print(io, "; ")
        print(io, msg)
    end
end

"""
    validate_terminators(sci::StructuredIRCode) -> Bool

Validate that all structured control flow ops have correct terminators.
Throws `InvalidTerminatorError` if any terminator is missing or invalid.

Validation rules:
- IfOp: both regions must have explicit terminator (never `nothing`)
- ForOp body: must have ContinueOp
- WhileOp before: must have ConditionOp
- WhileOp after: must have YieldOp
- LoopOp body: recursively validate nested ops
"""
function validate_terminators(sci::StructuredIRCode)
    errors = String[]
    validate_terminators!(errors, sci, sci.entry)
    isempty(errors) || throw(InvalidTerminatorError(errors))
    return true
end

# Convenience method for testing: wrap block in minimal SCI
function validate_terminators(entry::Block)
    sci = StructuredIRCode(Any[], Any[], entry, 0)
    return validate_terminators(sci)
end

function validate_terminators!(errors::Vector{String}, sci::StructuredIRCode, block::Block)
    for (idx, entry) in block.body
        stmt = entry.stmt
        if stmt isa IfOp
            validate_if_terminators!(errors, sci, stmt, idx)
        elseif stmt isa ForOp
            validate_for_terminators!(errors, sci, stmt, idx)
        elseif stmt isa WhileOp
            validate_while_terminators!(errors, sci, stmt, idx)
        elseif stmt isa LoopOp
            validate_loop_terminators!(errors, sci, stmt, idx)
        end
    end
end

function validate_if_terminators!(errors::Vector{String}, sci::StructuredIRCode, op::IfOp, idx::Int)
    then_term = op.then_region.terminator
    else_term = op.else_region.terminator

    # Both regions must have explicit terminators
    # Having `nothing` as terminator is always invalid for IfOp regions
    # Valid terminators: YieldOp, ReturnNode, ContinueOp, BreakOp (for IfOps inside loops)
    if then_term === nothing
        push!(errors, "IfOp at %$idx: then region must have explicit terminator, got nothing")
    end
    if else_term === nothing
        push!(errors, "IfOp at %$idx: else region must have explicit terminator, got nothing")
    end

    # Validate yield arity and types: both branches must yield same number of values with matching types
    if then_term isa YieldOp && else_term isa YieldOp
        then_arity = length(then_term.values)
        else_arity = length(else_term.values)
        if then_arity != else_arity
            push!(errors, "IfOp at %$idx: yield arity mismatch (then yields $then_arity, else yields $else_arity)")
        end

        # Type validation for matching positions
        for i in 1:min(then_arity, else_arity)
            then_type = resolve_type(sci, then_term.values[i])
            else_type = resolve_type(sci, else_term.values[i])
            if then_type !== nothing && else_type !== nothing && then_type != else_type
                push!(errors, "IfOp at %$idx: yield type mismatch at position $i (then: $then_type, else: $else_type)")
            end
        end
    end

    # Recursively validate nested ops
    validate_terminators!(errors, sci, op.then_region)
    validate_terminators!(errors, sci, op.else_region)
end

function validate_for_terminators!(errors::Vector{String}, sci::StructuredIRCode, op::ForOp, idx::Int)
    term = op.body.terminator
    if !(term isa ContinueOp)
        push!(errors, "ForOp at %$idx: body must have ContinueOp, got $(typeof(term))")
    end

    # Recursively validate nested ops
    validate_terminators!(errors, sci, op.body)
end

function validate_while_terminators!(errors::Vector{String}, sci::StructuredIRCode, op::WhileOp, idx::Int)
    before_term = op.before.terminator
    after_term = op.after.terminator

    if !(before_term isa ConditionOp)
        push!(errors, "WhileOp at %$idx: before region must have ConditionOp, got $(typeof(before_term))")
    end
    if !(after_term isa YieldOp)
        push!(errors, "WhileOp at %$idx: after region must have YieldOp, got $(typeof(after_term))")
    end

    # Recursively validate nested ops
    validate_terminators!(errors, sci, op.before)
    validate_terminators!(errors, sci, op.after)
end

function validate_loop_terminators!(errors::Vector{String}, sci::StructuredIRCode, op::LoopOp, idx::Int)
    # LoopOp body can have various terminators (BreakOp, ContinueOp, etc.)
    # Just recursively validate nested ops
    validate_terminators!(errors, sci, op.body)
end
