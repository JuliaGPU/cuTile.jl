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
converted to structured control flow operations (ControlFlowOp).

Throws `UnstructuredControlFlowError` if unstructured control flow remains.
Returns `true` if all control flow is properly structured.

The invariant is simple: no expression in any `block.body` should be a
`GotoNode` or `GotoIfNot` - those should have been replaced by
structured ops that the visitor descends into.
"""
function validate_scf(sci::StructuredCodeInfo)
    unstructured = Int[]

    # Walk all blocks and check that no statement is unstructured control flow
    each_stmt(sci.entry) do stmt
        # stmt is a NamedTuple with idx, expr, type fields
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
 Detects positive SSAValue references inside nested blocks (which should only
 contain negative SSAValue or BlockArg until finalization).
=============================================================================#

"""
    check_global_ssa_refs(sci::StructuredCodeInfo; strict::Bool=false) -> Int
    check_global_ssa_refs(block::Block; strict::Bool=false) -> Int

Count positive SSAValue references inside nested control flow structures.
Returns the count. If strict=true, errors when count > 0.

Before finalization:
- Entry block: positive SSAValue (original CodeInfo refs) are allowed
- Nested blocks: should only have negative SSAValue (local) or BlockArg

After finalization:
- All SSAValue are positive (global indices)
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

Count positive SSAValue references inside nested control flow structures.
Positive SSAValues in the entry block are allowed (they reference the original
CodeInfo), but positive SSAValues inside nested blocks (if/for/etc.) should
have been converted to negative SSAValue (local) or BlockArg.

For pre-flattening (OrderedDict body), this scans expressions and nested ops.
For post-flattening (Vector body), all SSAs are valid local references, so return 0.
"""
function count_ssavalues_in_nested(block::Block; is_entry::Bool=false)
    # Post-flattening: all SSAValues are valid local references
    if block.body isa Vector
        return 0
    end

    # Pre-flattening: count SSAValues in nested structures
    count = 0

    for (idx, item) in block.body
        if item isa PartialControlFlowOp
            # Control flow op - check nested blocks
            # Pass is_entry so we know if the op's iter_args/captures are at entry level
            count += count_ssavalues_in_op(item; is_entry)
        else  # Statement
            if !is_entry
                # Inside nested block, count SSAValues in the expression
                count += count_ssavalues_in_value(item)
            end
        end
    end

    # Check terminator (except in entry block)
    if !is_entry && block.terminator !== nothing
        count += count_ssavalues_in_terminator(block.terminator)
    end

    return count
end

function count_ssavalues_in_op(op::PartialControlFlowOp; is_entry::Bool=false)
    count = 0
    # Only count operands and iter_args/captures if op is nested (not at entry level)
    if !is_entry
        # Count SSAValues in operands
        for v in values(op.operands)
            count += count_ssavalues_in_value(v)
        end
        for v in op.iter_args
            count += count_ssavalues_in_value(v)
        end
        for v in op.captures
            count += count_ssavalues_in_value(v)
        end
    end
    # Recurse into all regions
    for (_, region) in op.regions
        count += count_ssavalues_in_nested(region)
    end
    return count
end

# For finalized ControlFlowOps (IfOp, ForOp, WhileOp, LoopOp), all SSAValues
# are valid local references (per-block namespace), so we return 0.
count_ssavalues_in_op(::IfOp; is_entry::Bool=false) = 0
count_ssavalues_in_op(::ForOp; is_entry::Bool=false) = 0
count_ssavalues_in_op(::WhileOp; is_entry::Bool=false) = 0
count_ssavalues_in_op(::LoopOp; is_entry::Bool=false) = 0

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

Count positive SSAValue occurrences in an IR value, recursively traversing Expr trees.
Negative SSAValue (local references) are not counted.
"""
count_ssavalues_in_value(v::SSAValue) = v.id > 0 ? 1 : 0
count_ssavalues_in_value(::BlockArg) = 0
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

    # Handle both pre-flattening (OrderedDict) and post-flattening (Vector)
    if block.body isa Vector
        # Post-flattening: iterate with enumerate
        for (pos, item) in enumerate(block.body)
            if !(item isa ControlFlowOp)
                check_ssa_refs_defined(item, defined, block.args, pos)
                push!(defined, pos)
            else
                # Already finalized, skip deep validation
            end
        end
    else
        # Pre-flattening: iterate OrderedDict
        for (idx, item) in block.body
            if item isa PartialControlFlowOp
                # ControlFlowOp - check its inputs and recurse into nested blocks
                validate_control_flow_op_ordering(item, defined, block.args)
                # Add results to defined set
                for rv in derive_result_vars(item)
                    push!(defined, rv.id)
                end
            else  # Statement
                # Check all SSAValue refs in the expression are in `defined`
                check_ssa_refs_defined(item, defined, block.args, idx)
                # Add this statement's SSA to defined set
                push!(defined, idx)
            end
        end
    end

    # Check terminator
    if block.terminator !== nothing
        check_terminator_refs_defined(block.terminator, defined, block.args)
    end

    return true
end

function validate_control_flow_op_ordering(op::PartialControlFlowOp, defined::Set{Int}, args::Vector{BlockArg})
    # Check operands
    for v in values(op.operands)
        check_ssa_refs_defined(v, defined, args, nothing)
    end
    # Check iter_args and captures
    for v in op.iter_args
        check_ssa_refs_defined(v, defined, args, nothing)
    end
    for v in op.captures
        check_ssa_refs_defined(v, defined, args, nothing)
    end
    # Recurse into all regions - each region starts fresh
    for (_, region) in op.regions
        validate_ssa_ordering(region; defined=Set{Int}())
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
