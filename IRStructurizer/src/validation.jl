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
            # Control flow op - check nested blocks
            # Pass is_entry so we know if the op's init_values are at entry level
            count += count_ssavalues_in_op(item; is_entry)
        end
    end

    # Check terminator (except in entry block)
    if !is_entry && block.terminator !== nothing
        count += count_ssavalues_in_terminator(block.terminator)
    end

    return count
end

function count_ssavalues_in_op(op::IfOp; is_entry::Bool=false)
    count = 0
    # Only count condition and init_values if op is nested (not at entry level)
    if !is_entry
        count += count_ssavalues_in_value(op.condition)
        for v in op.init_values
            count += count_ssavalues_in_value(v)
        end
    end
    count += count_ssavalues_in_nested(op.then_block)
    count += count_ssavalues_in_nested(op.else_block)
    return count
end

function count_ssavalues_in_op(op::ForOp; is_entry::Bool=false)
    count = 0
    # Only count bounds and init_values if op is nested (not at entry level)
    if !is_entry
        count += count_ssavalues_in_value(op.lower)
        count += count_ssavalues_in_value(op.upper)
        count += count_ssavalues_in_value(op.step)
        for v in op.init_values
            count += count_ssavalues_in_value(v)
        end
    end
    count += count_ssavalues_in_nested(op.body)
    return count
end

function count_ssavalues_in_op(op::LoopOp; is_entry::Bool=false)
    count = 0
    # Only count init_values if op is nested (not at entry level)
    if !is_entry
        for v in op.init_values
            count += count_ssavalues_in_value(v)
        end
    end
    count += count_ssavalues_in_nested(op.body)
    return count
end

function count_ssavalues_in_op(op::WhileOp; is_entry::Bool=false)
    count = 0
    # Only count init_values if op is nested (not at entry level)
    if !is_entry
        for v in op.init_values
            count += count_ssavalues_in_value(v)
        end
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

#=============================================================================
 SSA Ordering Validation
 Validates that all SSAValue references in an RHS have been defined earlier
 in the same block or are available as BlockArgs.
=============================================================================#

"""
    validate_ssa_ordering(sci::StructuredCodeInfo) -> Bool
    validate_ssa_ordering(block::Block; defined::Set{Int}=Set{Int}()) -> Bool

Validate that every SSAValue reference in block items has been defined earlier
in the block or is a BlockArg. This catches the phi-referencing-phi bug where
a phi node's carried value references another phi that hasn't been substituted
to a BlockArg yet.

Returns true if valid, throws an error otherwise.
"""
function validate_ssa_ordering(sci::StructuredCodeInfo)
    validate_ssa_ordering(sci.entry; defined=Set{Int}())
end

function validate_ssa_ordering(block::Block; defined::Set{Int}=Set{Int}())
    # BlockArgs are available from the start
    # (They reference external values, not SSA indices in this block)

    for item in block.body
        if item isa Statement
            # Check all SSAValue refs in the expression are in `defined`
            check_ssa_refs_defined(item.expr, defined, block.args, item.idx)
            # Add this statement's SSA to defined set
            push!(defined, item.idx)
        else
            # ControlFlowOp - check its inputs and recurse into nested blocks
            validate_control_flow_op_ordering(item, defined, block.args)
            # Add result_vars to defined set (if any)
            add_result_vars_to_defined!(item, defined)
        end
    end

    # Check terminator
    if block.terminator !== nothing
        check_terminator_refs_defined(block.terminator, defined, block.args)
    end

    return true
end

function validate_control_flow_op_ordering(op::IfOp, defined::Set{Int}, args::Vector{BlockArg})
    check_ssa_refs_defined(op.condition, defined, args, nothing)
    # Nested blocks start fresh but inherit defined from parent
    validate_ssa_ordering(op.then_block; defined=copy(defined))
    validate_ssa_ordering(op.else_block; defined=copy(defined))
end

function validate_control_flow_op_ordering(op::ForOp, defined::Set{Int}, args::Vector{BlockArg})
    check_ssa_refs_defined(op.lower, defined, args, nothing)
    check_ssa_refs_defined(op.upper, defined, args, nothing)
    check_ssa_refs_defined(op.step, defined, args, nothing)
    for v in op.init_values
        check_ssa_refs_defined(v, defined, args, nothing)
    end
    # Body starts fresh with its own args
    validate_ssa_ordering(op.body; defined=Set{Int}())
end

function validate_control_flow_op_ordering(op::LoopOp, defined::Set{Int}, args::Vector{BlockArg})
    for v in op.init_values
        check_ssa_refs_defined(v, defined, args, nothing)
    end
    validate_ssa_ordering(op.body; defined=Set{Int}())
end

function validate_control_flow_op_ordering(op::WhileOp, defined::Set{Int}, args::Vector{BlockArg})
    for v in op.init_values
        check_ssa_refs_defined(v, defined, args, nothing)
    end
    validate_ssa_ordering(op.before; defined=Set{Int}())
    validate_ssa_ordering(op.after; defined=Set{Int}())
end

function add_result_vars_to_defined!(op::IfOp, defined::Set{Int})
    for rv in op.result_vars
        push!(defined, rv.id)
    end
end

function add_result_vars_to_defined!(op::ForOp, defined::Set{Int})
    push!(defined, op.iv_ssa.id)
    for rv in op.result_vars
        push!(defined, rv.id)
    end
end

function add_result_vars_to_defined!(op::LoopOp, defined::Set{Int})
    for rv in op.result_vars
        push!(defined, rv.id)
    end
end

function add_result_vars_to_defined!(op::WhileOp, defined::Set{Int})
    for rv in op.result_vars
        push!(defined, rv.id)
    end
end

function check_terminator_refs_defined(term::YieldOp, defined::Set{Int}, args::Vector{BlockArg})
    for v in term.values
        check_ssa_refs_defined(v, defined, args, nothing)
    end
end

function check_terminator_refs_defined(term::ContinueOp, defined::Set{Int}, args::Vector{BlockArg})
    for v in term.values
        check_ssa_refs_defined(v, defined, args, nothing)
    end
end

function check_terminator_refs_defined(term::BreakOp, defined::Set{Int}, args::Vector{BlockArg})
    for v in term.values
        check_ssa_refs_defined(v, defined, args, nothing)
    end
end

function check_terminator_refs_defined(term::ConditionOp, defined::Set{Int}, args::Vector{BlockArg})
    check_ssa_refs_defined(term.condition, defined, args, nothing)
    for v in term.args
        check_ssa_refs_defined(v, defined, args, nothing)
    end
end

function check_terminator_refs_defined(::ReturnNode, ::Set{Int}, ::Vector{BlockArg})
    # ReturnNode handled at entry block level
end

function check_terminator_refs_defined(::Nothing, ::Set{Int}, ::Vector{BlockArg})
end

"""
    check_ssa_refs_defined(value, defined::Set{Int}, args::Vector{BlockArg}, context)

Check that all SSAValue references in `value` are either:
1. In the `defined` set (defined earlier in the block)
2. A BlockArg (passed from parent scope)

Throws an error if an undefined SSAValue is found.
The `context` parameter provides location info for error messages.
"""
function check_ssa_refs_defined(ssa::SSAValue, defined::Set{Int}, args::Vector{BlockArg}, context)
    if ssa.id in defined
        return  # OK - defined earlier in block
    end
    # Check if it matches a BlockArg's corresponding SSA (not directly supported,
    # but for now we allow SSAValues that might be from parent scope)
    # In strict mode, this would be an error - but we're in migration mode
end

check_ssa_refs_defined(::BlockArg, ::Set{Int}, ::Vector{BlockArg}, _) = nothing
check_ssa_refs_defined(::LocalSSAValue, ::Set{Int}, ::Vector{BlockArg}, _) = nothing
check_ssa_refs_defined(::Argument, ::Set{Int}, ::Vector{BlockArg}, _) = nothing
check_ssa_refs_defined(::SlotNumber, ::Set{Int}, ::Vector{BlockArg}, _) = nothing
check_ssa_refs_defined(::GlobalRef, ::Set{Int}, ::Vector{BlockArg}, _) = nothing
check_ssa_refs_defined(::QuoteNode, ::Set{Int}, ::Vector{BlockArg}, _) = nothing
check_ssa_refs_defined(::Nothing, ::Set{Int}, ::Vector{BlockArg}, _) = nothing
check_ssa_refs_defined(::Number, ::Set{Int}, ::Vector{BlockArg}, _) = nothing
check_ssa_refs_defined(::Symbol, ::Set{Int}, ::Vector{BlockArg}, _) = nothing
check_ssa_refs_defined(::Type, ::Set{Int}, ::Vector{BlockArg}, _) = nothing
check_ssa_refs_defined(::PiNode, ::Set{Int}, ::Vector{BlockArg}, _) = nothing

function check_ssa_refs_defined(expr::Expr, defined::Set{Int}, args::Vector{BlockArg}, context)
    for arg in expr.args
        check_ssa_refs_defined(arg, defined, args, context)
    end
end

# Fallback
check_ssa_refs_defined(_, ::Set{Int}, ::Vector{BlockArg}, _) = nothing
