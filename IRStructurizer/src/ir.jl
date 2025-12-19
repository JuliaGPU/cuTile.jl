# structured IR definitions

#=============================================================================
 Block Arguments (for loop carried values)
=============================================================================#

"""
    BlockArg

Represents a block argument (similar to MLIR block arguments).
Used for loop carried values and condition branch results.
"""
struct BlockArg
    id::Int           # Sequential ID within block (1, 2, 3...)
    type::Any         # Julia type
end

#=============================================================================
 IR Values - references to SSA values or block arguments
=============================================================================#

# IRValue: Values used in structured IR
# - SSAValue, Argument, SlotNumber: references to Julia IR values
# - BlockArg: block arguments for control flow
# - Raw values (Integer, Float, etc.): compile-time constants
const IRValue = Any

#=============================================================================
 Terminator Operations
=============================================================================#

"""
    YieldOp

Yields values from a structured control flow region (if/loop body).
The yielded values become the results of the containing IfOp/LoopOp.
"""
struct YieldOp
    values::Vector{IRValue}
end

YieldOp() = YieldOp(IRValue[])

"""
    ContinueOp

Continue to the next iteration of a loop with updated carried values.
"""
struct ContinueOp
    values::Vector{IRValue}
end

ContinueOp() = ContinueOp(IRValue[])

"""
    BreakOp

Break out of a loop, yielding values.
"""
struct BreakOp
    values::Vector{IRValue}
end

BreakOp() = BreakOp(IRValue[])

"""
    ConditionOp

Terminator for the 'before' region of a WhileOp (MLIR scf.condition).
If condition is true, args are passed to the 'after' region.
If condition is false, args become the final loop results.
"""
struct ConditionOp
    condition::IRValue           # Boolean condition
    args::Vector{IRValue}        # Values passed to after region or used as break results
end

ConditionOp(cond::IRValue) = ConditionOp(cond, IRValue[])

const Terminator = Union{ReturnNode, YieldOp, ContinueOp, BreakOp, ConditionOp, Nothing}

#=============================================================================
 SSA Substitution (phi refs → block args)
=============================================================================#

"""
    Substitutions

A mapping from SSA value indices to BlockArgs.
Used during IR construction to replace phi node references with block arguments.
"""
const Substitutions = Dict{Int, BlockArg}

"""
    substitute_ssa(value, subs::Substitutions)

Recursively substitute SSAValues with BlockArgs according to the substitution map.
Used to convert phi node references to block argument references inside loop bodies.
"""
function substitute_ssa(value, subs::Substitutions)
    if value isa SSAValue && haskey(subs, value.id)
        return subs[value.id]
    elseif value isa Expr
        new_args = Any[substitute_ssa(a, subs) for a in value.args]
        return Expr(value.head, new_args...)
    elseif value isa PiNode
        return PiNode(substitute_ssa(value.val, subs), value.typ)
    elseif value isa PhiNode
        # Phi nodes shouldn't appear in structured IR, but handle gracefully
        new_values = Vector{Any}(undef, length(value.values))
        for i in eachindex(value.values)
            if isassigned(value.values, i)
                new_values[i] = substitute_ssa(value.values[i], subs)
            end
        end
        return PhiNode(value.edges, new_values)
    else
        return value
    end
end

# Convenience for empty substitutions
substitute_ssa(value) = value

#=============================================================================
 Statement - self-contained statement with type
=============================================================================#

"""
    Statement

A statement in structured IR. Self-contained with expression and type.
SSA substitutions (phi refs → block args) are applied during construction.
"""
struct Statement
    idx::Int      # Original statement index (for source mapping/debugging)
    expr::Any     # The expression (with SSA refs substituted where needed)
    type::Any     # The SSA value type
end

function Base.show(io::IO, stmt::Statement)
    print(io, "Statement(", stmt.idx, ", ", stmt.expr, ")")
end

#=============================================================================
 Structured Control Flow Operations
=============================================================================#

# Forward declaration for Block (needed for mutual recursion)
abstract type ControlFlowOp end

"""
    BlockItem

Union type for items in a block's body.
Can be either a Statement or a structured control flow operation.
"""
const BlockItem = Union{Statement, ControlFlowOp}

"""
    Block

A basic block containing statements and potentially nested control flow.
Statements are self-contained Statement objects with expressions and types.
Body items are interleaved - Statements and control flow ops can appear in any order.
"""
mutable struct Block
    id::Int
    args::Vector{BlockArg}           # Block arguments (for loop carried values)
    body::Vector{BlockItem}          # Interleaved Statements and control flow ops
    terminator::Terminator           # ReturnNode, ContinueOp, YieldOp, BreakOp, or nothing
end

Block(id::Int) = Block(id, BlockArg[], BlockItem[], nothing)

function Base.show(io::IO, block::Block)
    print(io, "Block(id=", block.id)
    if !isempty(block.args)
        print(io, ", args=", length(block.args))
    end
    n_stmts = count(x -> x isa Statement, block.body)
    n_ops = count(x -> x isa ControlFlowOp, block.body)
    print(io, ", stmts=", n_stmts)
    if n_ops > 0
        print(io, ", ops=", n_ops)
    end
    print(io, ")")
end

# Iteration protocol for Block
Base.iterate(block::Block, state=1) = state > length(block.body) ? nothing : (block.body[state], state + 1)
Base.length(block::Block) = length(block.body)
Base.eltype(::Type{Block}) = BlockItem

"""
    IfOp <: ControlFlowOp

Structured if-then-else with nested blocks.
Both branches must yield values of the same types.

The op itself can be used as an IRValue. For multiple results,
it produces a tuple type.
"""
struct IfOp <: ControlFlowOp
    condition::IRValue               # SSAValue or BlockArg for the condition
    then_block::Block
    else_block::Block
    result_type::Type                # Nothing, T, or Tuple{T1, T2, ...}
end

function Base.show(io::IO, op::IfOp)
    print(io, "IfOp(cond=", op.condition,
          ", then=Block(", op.then_block.id, ")",
          ", else=Block(", op.else_block.id, ")",
          ", result=", op.result_type, ")")
end

"""
    ForOp <: ControlFlowOp

Structured for loop with known bounds.
Used when loop bounds can be determined (e.g., from range iteration).

The op itself can be used as an IRValue. Result type is computed from
the loop-carried values (excluding IV which is internal to the loop).
"""
struct ForOp <: ControlFlowOp
    lower::IRValue                   # Lower bound
    upper::IRValue                   # Upper bound (exclusive)
    step::IRValue                    # Step value
    iv_arg::BlockArg                 # Block arg for induction variable (for body/printing)
    init_values::Vector{IRValue}     # Initial values for non-IV carried variables
    body::Block                      # Block args for carried values (not IV)
    result_type::Type                # Nothing, T, or Tuple{T1, T2, ...}
end

function Base.show(io::IO, op::ForOp)
    print(io, "ForOp(lower=", op.lower, ", upper=", op.upper, ", step=", op.step,
          ", iv=", op.iv_arg,
          ", init=", length(op.init_values),
          ", body=Block(", op.body.id, ")",
          ", result=", op.result_type, ")")
end

"""
    LoopOp <: ControlFlowOp

General loop with dynamic exit condition.
Used for while loops and when bounds cannot be determined.

Also used as the initial loop representation before pattern matching
upgrades it to ForOp or WhileOp.

The op itself can be used as an IRValue. For multiple results,
it produces a tuple type.
"""
struct LoopOp <: ControlFlowOp
    init_values::Vector{IRValue}     # Initial values for loop-carried variables
    body::Block                      # Has carried vars as block args
    result_type::Type                # Nothing, T, or Tuple{T1, T2, ...}
end

function Base.show(io::IO, op::LoopOp)
    print(io, "LoopOp(init=", length(op.init_values),
          ", body=Block(", op.body.id, ")",
          ", result=", op.result_type, ")")
end

"""
    WhileOp <: ControlFlowOp

Structured while loop with MLIR-style two-region structure (scf.while).

- `before`: Computes the condition, ends with ConditionOp(cond, args)
- `after`: Loop body, ends with YieldOp to pass values back to before region

When condition is true, args are passed to the after region.
When condition is false, args become the final loop results.

The op itself can be used as an IRValue. For multiple results,
it produces a tuple type.
"""
struct WhileOp <: ControlFlowOp
    before::Block                    # Condition computation, ends with ConditionOp
    after::Block                     # Loop body, ends with YieldOp
    init_values::Vector{IRValue}     # Initial values for loop-carried variables
    result_type::Type                # Nothing, T, or Tuple{T1, T2, ...}
end

function Base.show(io::IO, op::WhileOp)
    print(io, "WhileOp(before=Block(", op.before.id, ")",
          ", after=Block(", op.after.id, ")",
          ", init=", length(op.init_values),
          ", result=", op.result_type, ")")
end

#=============================================================================
 StructuredCodeInfo - the structured IR for a function
=============================================================================#

"""
    StructuredCodeInfo

Represents a function's code with a structured view of control flow.
The CodeInfo is kept for metadata (slotnames, argtypes, method info).
The entry Block contains self-contained Statement objects with expressions and types.

Create with `StructuredCodeInfo(ci)` for a flat (unstructured) view,
then call `structurize!(sci)` to convert control flow to structured ops.
"""
mutable struct StructuredCodeInfo
    const code::CodeInfo             # For metadata (slotnames, argtypes, etc.)
    entry::Block                     # Self-contained structured IR
end

"""
    StructuredCodeInfo(code::CodeInfo)

Create a flat (unstructured) StructuredCodeInfo from Julia CodeInfo.
All statements are placed sequentially in a single block as Statement objects,
with control flow statements (GotoNode, GotoIfNot) included as-is.

Call `structurize!(sci)` to convert to structured control flow.
"""
function StructuredCodeInfo(code::CodeInfo)
    stmts = code.code
    types = code.ssavaluetypes
    n = length(stmts)

    entry = Block(1)

    for i in 1:n
        stmt = stmts[i]
        if stmt isa ReturnNode
            entry.terminator = stmt
        else
            # Include ALL statements as Statement objects (no substitutions at entry level)
            push!(entry.body, Statement(i, stmt, types[i]))
        end
    end

    return StructuredCodeInfo(code, entry)
end

#=============================================================================
 Block Substitution (apply SSA → BlockArg mappings)
=============================================================================#

"""
    substitute_block!(block::Block, subs::Substitutions)

Apply SSA substitutions to all statements in a block and nested control flow.
Modifies the block in-place by replacing Statement objects with substituted versions.
"""
function substitute_block!(block::Block, subs::Substitutions)
    isempty(subs) && return  # No substitutions to apply

    # Substitute statements and recurse into nested control flow
    for (i, item) in enumerate(block.body)
        if item isa Statement
            new_expr = substitute_ssa(item.expr, subs)
            if new_expr !== item.expr
                block.body[i] = Statement(item.idx, new_expr, item.type)
            end
        else
            substitute_control_flow!(item, subs)
        end
    end

    # Substitute terminator
    if block.terminator !== nothing
        block.terminator = substitute_terminator(block.terminator, subs)
    end
end

"""
    substitute_control_flow!(op::ControlFlowOp, subs::Substitutions)

Apply SSA substitutions to a control flow operation and its nested blocks.
"""
function substitute_control_flow!(op::IfOp, subs::Substitutions)
    substitute_block!(op.then_block, subs)
    substitute_block!(op.else_block, subs)
end

function substitute_control_flow!(op::ForOp, subs::Substitutions)
    substitute_block!(op.body, subs)
end

function substitute_control_flow!(op::LoopOp, subs::Substitutions)
    substitute_block!(op.body, subs)
end

function substitute_control_flow!(op::WhileOp, subs::Substitutions)
    substitute_block!(op.before, subs)
    substitute_block!(op.after, subs)
end

"""
    substitute_terminator(term, subs::Substitutions)

Apply SSA substitutions to a terminator's values.
"""
function substitute_terminator(term::ContinueOp, subs::Substitutions)
    new_values = [substitute_ssa(v, subs) for v in term.values]
    return ContinueOp(new_values)
end

function substitute_terminator(term::BreakOp, subs::Substitutions)
    new_values = [substitute_ssa(v, subs) for v in term.values]
    return BreakOp(new_values)
end

function substitute_terminator(term::ConditionOp, subs::Substitutions)
    new_cond = substitute_ssa(term.condition, subs)
    new_args = [substitute_ssa(v, subs) for v in term.args]
    return ConditionOp(new_cond, new_args)
end

function substitute_terminator(term::YieldOp, subs::Substitutions)
    new_values = [substitute_ssa(v, subs) for v in term.values]
    return YieldOp(new_values)
end

function substitute_terminator(term::ReturnNode, subs::Substitutions)
    if isdefined(term, :val)
        new_val = substitute_ssa(term.val, subs)
        if new_val !== term.val
            return ReturnNode(new_val)
        end
    end
    return term
end

function substitute_terminator(term::Nothing, subs::Substitutions)
    return nothing
end

#=============================================================================
 Iteration Utilities
=============================================================================#

"""
    each_block(f, block::Block)

Recursively iterate over all blocks, calling f on each.
"""
function each_block(f, block::Block)
    f(block)
    for item in block.body
        if item isa ControlFlowOp
            each_block_in_op(f, item)
        end
    end
end

function each_block_in_op(f, op::IfOp)
    each_block(f, op.then_block)
    each_block(f, op.else_block)
end

function each_block_in_op(f, op::ForOp)
    each_block(f, op.body)
end

function each_block_in_op(f, op::LoopOp)
    each_block(f, op.body)
end

function each_block_in_op(f, op::WhileOp)
    each_block(f, op.before)
    each_block(f, op.after)
end

"""
    each_stmt(f, block::Block)

Recursively iterate over all statements, calling f on each Statement.
"""
function each_stmt(f, block::Block)
    for item in block.body
        if item isa Statement
            f(item)
        else
            each_stmt_in_op(f, item)
        end
    end
end

function each_stmt_in_op(f, op::IfOp)
    each_stmt(f, op.then_block)
    each_stmt(f, op.else_block)
end

function each_stmt_in_op(f, op::ForOp)
    each_stmt(f, op.body)
end

function each_stmt_in_op(f, op::LoopOp)
    each_stmt(f, op.body)
end

function each_stmt_in_op(f, op::WhileOp)
    each_stmt(f, op.before)
    each_stmt(f, op.after)
end

#=============================================================================
 Block Queries
=============================================================================#

"""
    defines(block::Block, ssa::SSAValue) -> Bool

Check if a block defines an SSA value (i.e., contains a Statement that produces it).
Searches nested blocks recursively.
"""
function defines(block::Block, ssa::SSAValue)
    for item in block.body
        if item isa Statement && item.idx == ssa.id
            return true
        elseif item isa IfOp
            defines(item.then_block, ssa) && return true
            defines(item.else_block, ssa) && return true
        elseif item isa LoopOp
            defines(item.body, ssa) && return true
        elseif item isa ForOp
            defines(item.body, ssa) && return true
        elseif item isa WhileOp
            defines(item.before, ssa) && return true
            defines(item.after, ssa) && return true
        end
    end
    return false
end

#=============================================================================
 Pretty Printing (Julia CodeInfo-style with colors)
=============================================================================#

"""
    compute_used_ssas(block::Block) -> BitSet

Compute which SSA values are used anywhere in the structured IR.
Used for coloring types appropriately (used values get cyan, unused get gray).
"""
function compute_used_ssas(block::Block)
    used = BitSet()
    _scan_uses!(used, block)
    return used
end

function _scan_uses!(used::BitSet, block::Block)
    for item in block.body
        if item isa Statement
            _scan_expr_uses!(used, item.expr)
        else
            _scan_control_flow_uses!(used, item)
        end
    end
    if block.terminator !== nothing
        _scan_terminator_uses!(used, block.terminator)
    end
end

function _scan_expr_uses!(used::BitSet, v::SSAValue)
    push!(used, v.id)
end

function _scan_expr_uses!(used::BitSet, v::Expr)
    for arg in v.args
        _scan_expr_uses!(used, arg)
    end
end

function _scan_expr_uses!(used::BitSet, v::PhiNode)
    for val in v.values
        _scan_expr_uses!(used, val)
    end
end

function _scan_expr_uses!(used::BitSet, v::PiNode)
    _scan_expr_uses!(used, v.val)
end

function _scan_expr_uses!(used::BitSet, v::GotoIfNot)
    _scan_expr_uses!(used, v.cond)
end

function _scan_expr_uses!(used::BitSet, v)
    # Other values (constants, GlobalRefs, etc.) don't reference SSA values
end

function _scan_terminator_uses!(used::BitSet, term::ReturnNode)
    if isdefined(term, :val)
        _scan_expr_uses!(used, term.val)
    end
end

function _scan_terminator_uses!(used::BitSet, term::YieldOp)
    for v in term.values
        _scan_expr_uses!(used, v)
    end
end

function _scan_terminator_uses!(used::BitSet, term::ContinueOp)
    for v in term.values
        _scan_expr_uses!(used, v)
    end
end

function _scan_terminator_uses!(used::BitSet, term::BreakOp)
    for v in term.values
        _scan_expr_uses!(used, v)
    end
end

function _scan_terminator_uses!(used::BitSet, term::ConditionOp)
    _scan_expr_uses!(used, term.condition)
    for v in term.args
        _scan_expr_uses!(used, v)
    end
end

function _scan_terminator_uses!(used::BitSet, ::Nothing)
end

function _scan_control_flow_uses!(used::BitSet, op::IfOp)
    _scan_expr_uses!(used, op.condition)
    _scan_uses!(used, op.then_block)
    _scan_uses!(used, op.else_block)
end

function _scan_control_flow_uses!(used::BitSet, op::ForOp)
    _scan_expr_uses!(used, op.lower)
    _scan_expr_uses!(used, op.upper)
    _scan_expr_uses!(used, op.step)
    for v in op.init_values
        _scan_expr_uses!(used, v)
    end
    _scan_uses!(used, op.body)
end

function _scan_control_flow_uses!(used::BitSet, op::LoopOp)
    for v in op.init_values
        _scan_expr_uses!(used, v)
    end
    _scan_uses!(used, op.body)
end

function _scan_control_flow_uses!(used::BitSet, op::WhileOp)
    for v in op.init_values
        _scan_expr_uses!(used, v)
    end
    _scan_uses!(used, op.before)
    _scan_uses!(used, op.after)
end

"""
    IRPrinter

Context for printing structured IR with proper indentation and value formatting.
Uses Julia's CodeInfo style with box-drawing characters and colors.
"""
mutable struct IRPrinter
    io::IO
    code::CodeInfo
    indent::Int
    line_prefix::String    # Prefix for continuation lines (│, spaces)
    is_last_stmt::Bool     # Whether current stmt is last in block
    used::BitSet           # Which SSA values are used (for type coloring)
    color::Bool            # Whether to use colors
end

function IRPrinter(io::IO, code::CodeInfo, entry::Block)
    used = compute_used_ssas(entry)
    color = get(io, :color, false)::Bool
    IRPrinter(io, code, 0, "", false, used, color)
end

function indent(p::IRPrinter, n::Int=1)
    new_prefix = p.line_prefix * "    "  # 4 spaces per indent level
    return IRPrinter(p.io, p.code, p.indent + n, new_prefix, false, p.used, p.color)
end

function print_indent(p::IRPrinter)
    # Color the line prefix (box-drawing characters from parent blocks)
    print_colored(p, p.line_prefix, :light_black)
end

# Helper for colored output
function print_colored(p::IRPrinter, s, color::Symbol)
    if p.color
        printstyled(p.io, s; color=color)
    else
        print(p.io, s)
    end
end

# Print an IR value (no special coloring, like Julia's code_typed)
function print_value(p::IRPrinter, v::SSAValue)
    print(p.io, "%", v.id)
end

function print_value(p::IRPrinter, v::BlockArg)
    print(p.io, "%arg", v.id)
end

function print_value(p::IRPrinter, v::Argument)
    # Use slot names if available from CodeInfo
    if v.n <= length(p.code.slotnames)
        name = p.code.slotnames[v.n]
        print(p.io, name)
    else
        print(p.io, "_", v.n)
    end
end

function print_value(p::IRPrinter, v::SlotNumber)
    print(p.io, "slot#", v.id)
end

function print_value(p::IRPrinter, v::QuoteNode)
    print(p.io, repr(v.value))
end

function print_value(p::IRPrinter, v::GlobalRef)
    print(p.io, v.mod, ".", v.name)
end

function print_value(p::IRPrinter, v)
    print(p.io, repr(v))
end

# String versions for places that need strings (e.g., join)
function format_value(p::IRPrinter, v::SSAValue)
    string("%", v.id)
end
function format_value(p::IRPrinter, v::BlockArg)
    string("%arg", v.id)
end
function format_value(p::IRPrinter, v::Argument)
    if v.n <= length(p.code.slotnames)
        name = p.code.slotnames[v.n]
        return string(name)
    else
        return string("_", v.n)
    end
end
function format_value(p::IRPrinter, v::SlotNumber)
    string("slot#", v.id)
end
function format_value(p::IRPrinter, v::QuoteNode)
    repr(v.value)
end
function format_value(p::IRPrinter, v::GlobalRef)
    string(v.mod, ".", v.name)
end
function format_value(p::IRPrinter, v)
    repr(v)
end

# Format type for printing (compact form)
function format_type(t)
    if t isa Core.Const
        string("Const(", repr(t.val), ")")
    elseif t isa Type
        string(t)
    else
        string(t)
    end
end

# Print type annotation with color based on whether the value is used
# Like Julia's code_typed: both :: and type share the same color
function print_type_annotation(p::IRPrinter, idx::Int, t)
    is_used = idx in p.used
    color = is_used ? :cyan : :light_black
    print_colored(p, string("::", format_type(t)), color)
end

# Print result type annotation for control flow ops
function print_result_type(p::IRPrinter, result_type::Type)
    if result_type === Nothing
        return
    end
    print(p.io, " -> ")
    print_colored(p, format_type(result_type), :cyan)
end

# Print a statement
function print_stmt(p::IRPrinter, stmt::Statement; prefix::String="│  ")
    print_indent(p)
    print_colored(p, prefix, :light_black)

    # Only show %N = when the value is used (like Julia's code_typed)
    is_used = stmt.idx in p.used
    if is_used
        print(p.io, "%", stmt.idx, " = ")
    else
        print(p.io, "     ")  # Padding to align with used values
    end
    print_expr(p, stmt.expr)
    print_type_annotation(p, stmt.idx, stmt.type)
    println(p.io)
end

# Check if a function reference is an intrinsic
function is_intrinsic_call(func)
    if func isa GlobalRef
        try
            f = getfield(func.mod, func.name)
            return f isa Core.IntrinsicFunction
        catch
            return false
        end
    end
    return false
end

# Print an expression (RHS of a statement)
function print_expr(p::IRPrinter, expr::Expr)
    if expr.head == :call
        func = expr.args[1]
        args = expr.args[2:end]
        # Check if this is an intrinsic call
        if is_intrinsic_call(func)
            print_colored(p, "intrinsic ", :light_black)
        end
        print_value(p, func)
        print(p.io, "(")
        for (i, a) in enumerate(args)
            i > 1 && print(p.io, ", ")
            print_value(p, a)
        end
        print(p.io, ")")
    elseif expr.head == :invoke
        mi = expr.args[1]
        func = expr.args[2]
        args = expr.args[3:end]
        print_colored(p, "invoke ", :light_black)
        if mi isa Core.MethodInstance
            print(p.io, mi.def.name)
            # Get argument types from MethodInstance signature
            sig = mi.specTypes isa DataType ? mi.specTypes.parameters : ()
            print(p.io, "(")
            for (i, a) in enumerate(args)
                i > 1 && print(p.io, ", ")
                print_value(p, a)
                # Print type annotation if available (sig includes function type at position 1)
                if i + 1 <= length(sig)
                    print_colored(p, string("::", sig[i + 1]), :cyan)
                end
            end
            print(p.io, ")")
        else
            print_value(p, func)
            print(p.io, "(")
            for (i, a) in enumerate(args)
                i > 1 && print(p.io, ", ")
                print_value(p, a)
            end
            print(p.io, ")")
        end
    elseif expr.head == :new
        print(p.io, "new ", expr.args[1], "(")
        for (i, a) in enumerate(expr.args[2:end])
            i > 1 && print(p.io, ", ")
            print_value(p, a)
        end
        print(p.io, ")")
    elseif expr.head == :foreigncall
        print(p.io, "foreigncall ", repr(expr.args[1]))
    elseif expr.head == :boundscheck
        print(p.io, "boundscheck")
    else
        print(p.io, expr.head, " ")
        for (i, a) in enumerate(expr.args)
            i > 1 && print(p.io, ", ")
            print_value(p, a)
        end
    end
end

function print_expr(p::IRPrinter, node::PhiNode)
    print(p.io, "φ (")
    first = true
    for (edge, val) in zip(node.edges, node.values)
        first || print(p.io, ", ")
        first = false
        print(p.io, "#", edge, " => ")
        if isassigned(node.values, findfirst(==(val), node.values))
            print_value(p, val)
        else
            print_colored(p, "#undef", :red)
        end
    end
    print(p.io, ")")
end

function print_expr(p::IRPrinter, node::PiNode)
    print(p.io, "π (")
    print_value(p, node.val)
    print(p.io, ", ", node.typ, ")")
end

function print_expr(p::IRPrinter, node::GotoNode)
    print(p.io, "goto #", node.label)
end

function print_expr(p::IRPrinter, node::GotoIfNot)
    print(p.io, "goto #", node.dest, " if not ")
    print_value(p, node.cond)
end

function print_expr(p::IRPrinter, node::ReturnNode)
    print(p.io, "return")
    if isdefined(node, :val)
        print(p.io, " ")
        print_value(p, node.val)
    end
end

function print_expr(p::IRPrinter, v)
    print_value(p, v)
end

# Print block arguments (for loops and structured control flow)
function print_block_args(p::IRPrinter, args::Vector{BlockArg})
    if isempty(args)
        return
    end
    print(p.io, "(")
    for (i, a) in enumerate(args)
        i > 1 && print(p.io, ", ")
        print(p.io, "%arg", a.id)
        # Block args are always "used" within their scope
        print_colored(p, string("::", format_type(a.type)), :cyan)
    end
    print(p.io, ")")
end

# Print iteration arguments with initial values
function print_iter_args(p::IRPrinter, args::Vector{BlockArg}, init_values::Vector{IRValue})
    if isempty(args)
        return
    end
    print(p.io, " iter_args(")
    for (i, (arg, init)) in enumerate(zip(args, init_values))
        i > 1 && print(p.io, ", ")
        print(p.io, "%arg", arg.id, " = ")
        print_value(p, init)
        # Block args are always "used" within their scope
        print_colored(p, string("::", format_type(arg.type)), :cyan)
    end
    print(p.io, ")")
end

# Print a terminator
function print_terminator(p::IRPrinter, term::ReturnNode; prefix::String="└──")
    print_indent(p)
    print_colored(p, prefix, :light_black)
    print(p.io, "      return")  # Padding to align with %N =
    if isdefined(term, :val)
        print(p.io, " ")
        print_value(p, term.val)
    end
    println(p.io)
end

function print_terminator(p::IRPrinter, term::YieldOp; prefix::String="└──")
    print_indent(p)
    print_colored(p, prefix, :light_black)
    print(p.io, "      ")  # Padding to align with %N =
    print_colored(p, "yield", :yellow)  # Structured keyword
    if !isempty(term.values)
        print(p.io, " ")
        for (i, v) in enumerate(term.values)
            i > 1 && print(p.io, ", ")
            print_value(p, v)
        end
    end
    println(p.io)
end

function print_terminator(p::IRPrinter, term::ContinueOp; prefix::String="└──")
    print_indent(p)
    print_colored(p, prefix, :light_black)
    print(p.io, "      ")  # Padding to align with %N =
    print_colored(p, "continue", :yellow)  # Structured keyword
    if !isempty(term.values)
        print(p.io, " ")
        for (i, v) in enumerate(term.values)
            i > 1 && print(p.io, ", ")
            print_value(p, v)
        end
    end
    println(p.io)
end

function print_terminator(p::IRPrinter, term::BreakOp; prefix::String="└──")
    print_indent(p)
    print_colored(p, prefix, :light_black)
    print(p.io, "      ")  # Padding to align with %N =
    print_colored(p, "break", :yellow)  # Structured keyword
    if !isempty(term.values)
        print(p.io, " ")
        for (i, v) in enumerate(term.values)
            i > 1 && print(p.io, ", ")
            print_value(p, v)
        end
    end
    println(p.io)
end

function print_terminator(p::IRPrinter, term::ConditionOp; prefix::String="└──")
    print_indent(p)
    print_colored(p, prefix, :light_black)
    print(p.io, "      ")  # Padding to align with %N =
    print_colored(p, "condition", :yellow)
    print(p.io, "(")
    print_value(p, term.condition)
    print(p.io, ")")
    if !isempty(term.args)
        print(p.io, " ")
        for (i, v) in enumerate(term.args)
            i > 1 && print(p.io, ", ")
            print_value(p, v)
        end
    end
    println(p.io)
end

function print_terminator(p::IRPrinter, ::Nothing; prefix::String="└──")
    # No terminator
end

# Print a block's contents (statements, nested ops, terminator)
function print_block_body(p::IRPrinter, block::Block)
    # Collect all items to print to determine which is last
    items = []

    for item in block.body
        if item isa Statement
            push!(items, (:stmt, item))
        else
            push!(items, (:nested, item))
        end
    end
    if block.terminator !== nothing
        push!(items, (:term, block.terminator))
    end

    for (i, item) in enumerate(items)
        is_last = (i == length(items))
        if item[1] == :stmt
            prefix = is_last ? "└──" : "│  "
            print_stmt(p, item[2]; prefix=prefix)
        elseif item[1] == :nested
            print_control_flow(p, item[2]; is_last=is_last)
        else  # :term
            print_terminator(p, item[2]; prefix="└──")
        end
    end
end

# Print IfOp (Julia-style)
function print_control_flow(p::IRPrinter, op::IfOp; is_last::Bool=false)
    prefix = is_last ? "└──" : "├──"
    cont_prefix = is_last ? "    " : "│   "

    print_indent(p)
    print_colored(p, prefix, :light_black)
    print(p.io, " ")

    print(p.io, "if ")
    print_value(p, op.condition)
    print_result_type(p, op.result_type)
    println(p.io)

    # Then block body (indented with continuation line)
    then_p = IRPrinter(p.io, p.code, p.indent + 1, p.line_prefix * cont_prefix, false, p.used, p.color)
    print_block_body(then_p, op.then_block)

    # else - aligned with "if"
    print_indent(p)
    print_colored(p, cont_prefix, :light_black)
    println(p.io, "else")

    # Else block body
    print_block_body(then_p, op.else_block)

    # end - aligned with "if"
    print_indent(p)
    print_colored(p, cont_prefix, :light_black)
    println(p.io, "end")
end

# Print ForOp (Julia-style)
function print_control_flow(p::IRPrinter, op::ForOp; is_last::Bool=false)
    prefix = is_last ? "└──" : "├──"
    cont_prefix = is_last ? "    " : "│   "

    print_indent(p)
    print_colored(p, prefix, :light_black)
    print(p.io, " ")

    # for %iv = %lb:%step:%ub
    print_colored(p, "for", :yellow)  # Structured keyword
    print(p.io, " %arg", op.iv_arg.id, " = ")
    print_value(p, op.lower)
    print(p.io, ":")
    print_value(p, op.step)
    print(p.io, ":")
    print_value(p, op.upper)

    # Print iteration arguments (carried values only, IV is separate)
    if !isempty(op.body.args)
        print_iter_args(p, op.body.args, op.init_values)
    end

    print_result_type(p, op.result_type)
    println(p.io)

    # Body - substitutions already applied
    body_p = IRPrinter(p.io, p.code, p.indent + 1, p.line_prefix * cont_prefix, false, p.used, p.color)
    print_block_body(body_p, op.body)

    print_indent(p)
    print_colored(p, cont_prefix, :light_black)
    println(p.io, "end")
end

# Print LoopOp (general loop - distinct from structured WhileOp)
function print_control_flow(p::IRPrinter, op::LoopOp; is_last::Bool=false)
    prefix = is_last ? "└──" : "├──"
    cont_prefix = is_last ? "    " : "│   "

    print_indent(p)
    print_colored(p, prefix, :light_black)
    print(p.io, " ")

    print_colored(p, "loop", :yellow)  # Structured keyword
    print_iter_args(p, op.body.args, op.init_values)
    print_result_type(p, op.result_type)
    println(p.io)

    # Body - substitutions already applied during construction
    body_p = IRPrinter(p.io, p.code, p.indent + 1, p.line_prefix * cont_prefix, false, p.used, p.color)
    print_block_body(body_p, op.body)

    print_indent(p)
    print_colored(p, cont_prefix, :light_black)
    println(p.io, "end")
end

# Print WhileOp (two-region while with before/after regions)
function print_control_flow(p::IRPrinter, op::WhileOp; is_last::Bool=false)
    prefix = is_last ? "└──" : "├──"
    cont_prefix = is_last ? "    " : "│   "

    print_indent(p)
    print_colored(p, prefix, :light_black)
    print(p.io, " ")

    print_colored(p, "while", :yellow)
    print_iter_args(p, op.before.args, op.init_values)
    print_result_type(p, op.result_type)
    println(p.io, " {")

    # Before region (condition computation)
    before_p = IRPrinter(p.io, p.code, p.indent + 1, p.line_prefix * cont_prefix, false, p.used, p.color)
    print_block_body(before_p, op.before)

    # "} do {" separator
    print_indent(p)
    print_colored(p, cont_prefix, :light_black)
    println(p.io, "} do {")

    # After region (loop body)
    after_p = IRPrinter(p.io, p.code, p.indent + 1, p.line_prefix * cont_prefix, false, p.used, p.color)
    print_block_body(after_p, op.after)

    print_indent(p)
    print_colored(p, cont_prefix, :light_black)
    println(p.io, "}")
end

# Main entry point: show for StructuredCodeInfo
function Base.show(io::IO, ::MIME"text/plain", sci::StructuredCodeInfo)
    # Get return type from last stmt if it's a return
    ret_type = "Any"
    for stmt in reverse(sci.code.code)
        if stmt isa ReturnNode && isdefined(stmt, :val)
            val = stmt.val
            if val isa SSAValue
                ret_type = format_type(sci.code.ssavaluetypes[val.id])
            else
                ret_type = format_type(typeof(val))
            end
            break
        end
    end

    color = get(io, :color, false)::Bool

    # Print header
    println(io, "StructuredCodeInfo(")

    p = IRPrinter(io, sci.code, sci.entry)

    # Print entry block body
    print_block_body(p, sci.entry)

    print(io, ") => ")
    if color
        printstyled(io, ret_type; color=:cyan)
        println(io)
    else
        println(io, ret_type)
    end
end

# Keep the simple show method for compact display
function Base.show(io::IO, sci::StructuredCodeInfo)
    print(io, "StructuredCodeInfo(", length(sci.code.code), " stmts, entry=Block#", sci.entry.id, ")")
end
