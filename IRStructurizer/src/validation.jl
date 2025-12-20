# structured IR validation

export UnstructuredControlFlowError, SSAValueReferenceError

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

#=============================================================================
 SSAValue Reference Validation
=============================================================================#

"""
    SSAValueReferenceError <: Exception

Exception thrown when SSAValue references are found in structured IR.
Post-structurization, all value references should be LocalSSA, BlockArg,
Argument, or SlotNumber - never SSAValue (which references original Julia IR).
"""
struct SSAValueReferenceError <: Exception
    locations::Vector{String}
end

function Base.showerror(io::IO, e::SSAValueReferenceError)
    print(io, "SSAValueReferenceError: SSAValue references found at: ",
          join(e.locations, ", "))
end

"""
    validate_no_ssavalue(sci::StructuredCodeInfo) -> Bool

Validate that no SSAValue references remain in the structured IR.
All value references should be LocalSSA (same block), BlockArg (loop carried),
Argument, or SlotNumber - never SSAValue (original Julia IR).

Throws `SSAValueReferenceError` if SSAValue references are found.
Returns `true` if no SSAValue references exist.
"""
function validate_no_ssavalue(sci::StructuredCodeInfo)
    locations = String[]
    _check_block_for_ssavalue!(locations, sci.entry, "entry")

    if !isempty(locations)
        throw(SSAValueReferenceError(locations))
    end

    return true
end

function _check_block_for_ssavalue!(locations::Vector{String}, block::Block, path::String)
    # Check ops
    for (i, op) in enumerate(block.ops)
        op_path = "$path.ops[$i]"
        if op.expr isa ControlFlowOp
            _check_cfop_for_ssavalue!(locations, op.expr, op_path)
        else
            _check_value_for_ssavalue!(locations, op.expr, op_path)
        end
    end

    # Check terminator
    if block.terminator !== nothing
        _check_terminator_for_ssavalue!(locations, block.terminator, "$path.terminator")
    end
end

function _check_value_for_ssavalue!(locations::Vector{String}, val::SSAValue, path::String)
    push!(locations, "$path: SSAValue($(val.id))")
end

function _check_value_for_ssavalue!(locations::Vector{String}, val::Expr, path::String)
    for (i, arg) in enumerate(val.args)
        _check_value_for_ssavalue!(locations, arg, "$path.args[$i]")
    end
end

function _check_value_for_ssavalue!(locations::Vector{String}, val::PiNode, path::String)
    _check_value_for_ssavalue!(locations, val.val, "$path.val")
end

function _check_value_for_ssavalue!(locations::Vector{String}, val, path::String)
    # LocalSSA, BlockArg, Argument, SlotNumber, constants - all OK
end

function _check_cfop_for_ssavalue!(locations::Vector{String}, op::IfOp, path::String)
    _check_value_for_ssavalue!(locations, op.condition, "$path.condition")
    for (i, v) in enumerate(op.init_values)
        _check_value_for_ssavalue!(locations, v, "$path.init_values[$i]")
    end
    _check_block_for_ssavalue!(locations, op.then_block, "$path.then_block")
    _check_block_for_ssavalue!(locations, op.else_block, "$path.else_block")
end

function _check_cfop_for_ssavalue!(locations::Vector{String}, op::ForOp, path::String)
    _check_value_for_ssavalue!(locations, op.lower, "$path.lower")
    _check_value_for_ssavalue!(locations, op.upper, "$path.upper")
    _check_value_for_ssavalue!(locations, op.step, "$path.step")
    for (i, v) in enumerate(op.init_values)
        _check_value_for_ssavalue!(locations, v, "$path.init_values[$i]")
    end
    _check_block_for_ssavalue!(locations, op.body, "$path.body")
end

function _check_cfop_for_ssavalue!(locations::Vector{String}, op::LoopOp, path::String)
    for (i, v) in enumerate(op.init_values)
        _check_value_for_ssavalue!(locations, v, "$path.init_values[$i]")
    end
    _check_block_for_ssavalue!(locations, op.body, "$path.body")
end

function _check_cfop_for_ssavalue!(locations::Vector{String}, op::WhileOp, path::String)
    for (i, v) in enumerate(op.init_values)
        _check_value_for_ssavalue!(locations, v, "$path.init_values[$i]")
    end
    _check_block_for_ssavalue!(locations, op.before, "$path.before")
    _check_block_for_ssavalue!(locations, op.after, "$path.after")
end

function _check_terminator_for_ssavalue!(locations::Vector{String}, term::ContinueOp, path::String)
    for (i, v) in enumerate(term.values)
        _check_value_for_ssavalue!(locations, v, "$path.values[$i]")
    end
end

function _check_terminator_for_ssavalue!(locations::Vector{String}, term::BreakOp, path::String)
    for (i, v) in enumerate(term.values)
        _check_value_for_ssavalue!(locations, v, "$path.values[$i]")
    end
end

function _check_terminator_for_ssavalue!(locations::Vector{String}, term::YieldOp, path::String)
    for (i, v) in enumerate(term.values)
        _check_value_for_ssavalue!(locations, v, "$path.values[$i]")
    end
end

function _check_terminator_for_ssavalue!(locations::Vector{String}, term::ConditionOp, path::String)
    _check_value_for_ssavalue!(locations, term.condition, "$path.condition")
    for (i, v) in enumerate(term.args)
        _check_value_for_ssavalue!(locations, v, "$path.args[$i]")
    end
end

function _check_terminator_for_ssavalue!(locations::Vector{String}, term::ReturnNode, path::String)
    if isdefined(term, :val)
        _check_value_for_ssavalue!(locations, term.val, "$path.val")
    end
end

function _check_terminator_for_ssavalue!(locations::Vector{String}, ::Nothing, path::String)
    # No terminator - OK
end
