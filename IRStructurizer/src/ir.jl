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
 Local SSA References (block-local SSA numbering)
=============================================================================#

"""
    LocalSSA

Reference to a value defined by an operation in the same block.
The id is the 1-indexed position in the containing block's ops array.
"""
struct LocalSSA
    id::Int  # 1-indexed position in block's ops array
end

function Base.show(io::IO, ssa::LocalSSA)
    print(io, "LocalSSA(", ssa.id, ")")
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

#=============================================================================
 Operation - unified representation for block-local SSA
=============================================================================#

"""
    Operation

A unified operation in structured IR. Position in the block's ops array
determines its local SSA id. Can contain either an expression (from original
IR) or a control flow operation.

Fields:
- `expr`: The expression or control flow op
- `type`: Result type (Nothing if no result)
"""
struct Operation
    expr::Any     # Expr, ControlFlowOp, or simple value
    type::Type    # Result type (Nothing if no result)
end

function Base.show(io::IO, op::Operation)
    if op.expr isa ControlFlowOp
        print(io, "Operation(", typeof(op.expr).name.name, ", ", op.type, ")")
    else
        print(io, "Operation(", op.expr, ", ", op.type, ")")
    end
end

"""
    Block

A basic block containing operations and potentially nested control flow.
Operations are stored in the ops array - position determines local SSA id.
"""
mutable struct Block
    id::Int
    args::Vector{BlockArg}           # Block arguments (for loop carried values)
    ops::Vector{Operation}           # Position = local SSA id (1-indexed)
    terminator::Terminator           # ReturnNode, ContinueOp, YieldOp, BreakOp, or nothing

    # SSA mapping: original SSA idx -> local position (for incremental construction)
    ssa_map::Dict{Int, Int}
end

Block(id::Int) = Block(id, BlockArg[], Operation[], nothing, Dict{Int, Int}())

function Base.show(io::IO, block::Block)
    print(io, "Block(id=", block.id)
    if !isempty(block.args)
        print(io, ", args=", length(block.args))
    end
    n_stmts = count(x -> !(x.expr isa ControlFlowOp), block.ops)
    n_cfops = count(x -> x.expr isa ControlFlowOp, block.ops)
    print(io, ", ops=", length(block.ops))
    print(io, ")")
end

# Iteration protocol for Block - iterate over ops
function Base.iterate(block::Block, state=1)
    state > length(block.ops) ? nothing : (block.ops[state], state + 1)
end
function Base.length(block::Block)
    length(block.ops)
end
Base.eltype(::Type{Block}) = Operation

"""
    push_op!(block::Block, ssa_idx::Int, expr, type) -> Int

Push an operation to a block with SSA index tracking.
Returns the local position (1-indexed) of the new operation.
The ssa_idx is recorded in ssa_map for later reference resolution.
"""
function push_op!(block::Block, ssa_idx::Int, expr, @nospecialize(type))
    # Convert any SSAValue references in expr to LocalSSA
    new_expr = convert_to_local_ssa(expr, block.ssa_map)
    result_type = type isa Type ? type : typeof(type)
    push!(block.ops, Operation(new_expr, result_type))
    local_pos = length(block.ops)
    block.ssa_map[ssa_idx] = local_pos
    return local_pos
end

"""
    push_extraction!(block::Block, ssa_idx::Int, cfop_pos::Int, element_idx::Int, type)

Push an extraction statement for a control flow op result.
Used when a PhiNode references the result of an IfOp/LoopOp.
"""
function push_extraction!(block::Block, ssa_idx::Int, cfop_pos::Int, element_idx::Int, @nospecialize(type))
    if element_idx == 0
        # Single result - no tuple extraction needed
        expr = LocalSSA(cfop_pos)
    else
        # Multiple results - extract from tuple
        expr = Expr(:call, GlobalRef(Base, :getfield), LocalSSA(cfop_pos), element_idx)
    end
    result_type = type isa Type ? type : typeof(type)
    push!(block.ops, Operation(expr, result_type))
    local_pos = length(block.ops)
    block.ssa_map[ssa_idx] = local_pos
    return local_pos
end

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
 Convert ControlFlowOp References
=============================================================================#

"""
    convert_cfop_refs(cfop::ControlFlowOp, ssa_map::Dict{Int,Int}) -> ControlFlowOp

Convert SSAValue references in a control flow operation to LocalSSA.
Does NOT recurse into nested blocks (they have their own ssa_map).
"""
function convert_cfop_refs(op::IfOp, ssa_map::Dict{Int,Int})
    new_condition = convert_to_local_ssa(op.condition, ssa_map)
    IfOp(new_condition, op.then_block, op.else_block, op.result_type)
end

function convert_cfop_refs(op::ForOp, ssa_map::Dict{Int,Int})
    new_lower = convert_to_local_ssa(op.lower, ssa_map)
    new_upper = convert_to_local_ssa(op.upper, ssa_map)
    new_step = convert_to_local_ssa(op.step, ssa_map)
    new_init = [convert_to_local_ssa(v, ssa_map) for v in op.init_values]
    ForOp(new_lower, new_upper, new_step, op.iv_arg, new_init, op.body, op.result_type)
end

function convert_cfop_refs(op::LoopOp, ssa_map::Dict{Int,Int})
    new_init = [convert_to_local_ssa(v, ssa_map) for v in op.init_values]
    LoopOp(new_init, op.body, op.result_type)
end

function convert_cfop_refs(op::WhileOp, ssa_map::Dict{Int,Int})
    new_init = [convert_to_local_ssa(v, ssa_map) for v in op.init_values]
    WhileOp(op.before, op.after, new_init, op.result_type)
end

"""
    push_cfop!(block::Block, cfop::ControlFlowOp) -> Int

Push a control flow operation to a block.
Returns the local position (1-indexed) of the new operation.
The cfop's condition/values are converted to LocalSSA using the block's ssa_map.
"""
function push_cfop!(block::Block, cfop::ControlFlowOp)
    # Convert the cfop's values using block's ssa_map
    new_cfop = convert_cfop_refs(cfop, block.ssa_map)
    push!(block.ops, Operation(new_cfop, get_result_type(new_cfop)))
    return length(block.ops)
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
            # Include ALL statements (no substitutions at entry level for trivial functions)
            push_op!(entry, i, stmt, types[i])
        end
    end

    return StructuredCodeInfo(code, entry)
end

#=============================================================================
 Block Substitution (apply SSA → BlockArg mappings)
=============================================================================#

"""
    substitute_block!(block::Block, subs::Substitutions)

Apply SSA substitutions to all operations in a block and nested control flow.
Modifies the block in-place by replacing Operation expressions with substituted versions.
"""
function substitute_block!(block::Block, subs::Substitutions)
    isempty(subs) && return  # No substitutions to apply

    # Substitute operations and recurse into nested control flow
    for (i, op) in enumerate(block.ops)
        if op.expr isa ControlFlowOp
            substitute_control_flow!(op.expr, subs)
        else
            new_expr = substitute_ssa(op.expr, subs)
            if new_expr !== op.expr
                block.ops[i] = Operation(new_expr, op.type)
            end
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
    # Substitute init_values
    for (i, v) in enumerate(op.init_values)
        op.init_values[i] = substitute_ssa(v, subs)
    end
    # Substitute lower/upper/step bounds
    # These are already part of the block's ops, so they should be substituted there
    substitute_block!(op.body, subs)
end

function substitute_control_flow!(op::LoopOp, subs::Substitutions)
    # Substitute init_values
    for (i, v) in enumerate(op.init_values)
        op.init_values[i] = substitute_ssa(v, subs)
    end
    substitute_block!(op.body, subs)
end

function substitute_control_flow!(op::WhileOp, subs::Substitutions)
    # Substitute init_values
    for (i, v) in enumerate(op.init_values)
        op.init_values[i] = substitute_ssa(v, subs)
    end
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
 Capture Outer Scope References
=============================================================================#

"""
    collect_ssa_refs(value) -> Set{Int}

Collect all SSAValue.id references in a value (recursively for expressions).
"""
function collect_ssa_refs(value::SSAValue)
    Set{Int}([value.id])
end

function collect_ssa_refs(value::Expr)
    refs = Set{Int}()
    for arg in value.args
        union!(refs, collect_ssa_refs(arg))
    end
    refs
end

function collect_ssa_refs(value::PiNode)
    collect_ssa_refs(value.val)
end

function collect_ssa_refs(value)
    Set{Int}()  # Constants, BlockArgs, Arguments, etc.
end

"""
    collect_block_ssa_refs(block::Block) -> Set{Int}

Collect all SSAValue.id references used in a block and its nested control flow.
"""
function collect_block_ssa_refs(block::Block)
    refs = Set{Int}()

    for op in block.ops
        if op.expr isa ControlFlowOp
            union!(refs, collect_cfop_ssa_refs(op.expr))
        else
            union!(refs, collect_ssa_refs(op.expr))
        end
    end

    if block.terminator !== nothing
        union!(refs, collect_terminator_ssa_refs(block.terminator))
    end

    refs
end

function collect_cfop_ssa_refs(op::IfOp)
    refs = collect_ssa_refs(op.condition)
    union!(refs, collect_block_ssa_refs(op.then_block))
    union!(refs, collect_block_ssa_refs(op.else_block))
    refs
end

function collect_cfop_ssa_refs(op::ForOp)
    refs = collect_ssa_refs(op.lower)
    union!(refs, collect_ssa_refs(op.upper))
    union!(refs, collect_ssa_refs(op.step))
    for v in op.init_values
        union!(refs, collect_ssa_refs(v))
    end
    union!(refs, collect_block_ssa_refs(op.body))
    refs
end

function collect_cfop_ssa_refs(op::LoopOp)
    refs = Set{Int}()
    for v in op.init_values
        union!(refs, collect_ssa_refs(v))
    end
    union!(refs, collect_block_ssa_refs(op.body))
    refs
end

function collect_cfop_ssa_refs(op::WhileOp)
    refs = Set{Int}()
    for v in op.init_values
        union!(refs, collect_ssa_refs(v))
    end
    union!(refs, collect_block_ssa_refs(op.before))
    union!(refs, collect_block_ssa_refs(op.after))
    refs
end

function collect_terminator_ssa_refs(term::ContinueOp)
    refs = Set{Int}()
    for v in term.values
        union!(refs, collect_ssa_refs(v))
    end
    refs
end

function collect_terminator_ssa_refs(term::BreakOp)
    refs = Set{Int}()
    for v in term.values
        union!(refs, collect_ssa_refs(v))
    end
    refs
end

function collect_terminator_ssa_refs(term::ConditionOp)
    refs = collect_ssa_refs(term.condition)
    for v in term.args
        union!(refs, collect_ssa_refs(v))
    end
    refs
end

function collect_terminator_ssa_refs(term::YieldOp)
    refs = Set{Int}()
    for v in term.values
        union!(refs, collect_ssa_refs(v))
    end
    refs
end

function collect_terminator_ssa_refs(term::ReturnNode)
    isdefined(term, :val) ? collect_ssa_refs(term.val) : Set{Int}()
end

function collect_terminator_ssa_refs(::Nothing)
    Set{Int}()
end

"""
    get_defined_ssas(block::Block) -> Set{Int}

Get all original SSA indices defined within a block and its nested control flow.
Uses the block's ssa_map for tracking which original indices are defined here.
"""
function get_defined_ssas(block::Block)
    defined = Set{Int}(keys(block.ssa_map))
    # Also check nested control flow ops
    for op in block.ops
        if op.expr isa ControlFlowOp
            union!(defined, get_cfop_defined_ssas(op.expr))
        end
    end
    defined
end

function get_cfop_defined_ssas(op::IfOp)
    union(get_defined_ssas(op.then_block), get_defined_ssas(op.else_block))
end

function get_cfop_defined_ssas(op::ForOp)
    get_defined_ssas(op.body)
end

function get_cfop_defined_ssas(op::LoopOp)
    get_defined_ssas(op.body)
end

function get_cfop_defined_ssas(op::WhileOp)
    union(get_defined_ssas(op.before), get_defined_ssas(op.after))
end

"""
    capture_outer_refs!(block::Block, code::Vector{Any}, types::Vector{Any}) -> (Vector{IRValue}, Substitutions)

Identify SSAValue references to outer scope values and create block arguments for them.
Returns the init_values to add and substitutions to apply.
"""
function capture_outer_refs!(block::Block, code::Vector{Any}, types::Vector{Any})
    # Collect all SSA refs used in the block
    used_refs = collect_block_ssa_refs(block)

    # Subtract SSAs defined within this block
    defined = get_defined_ssas(block)
    outer_refs = setdiff(used_refs, defined)

    isempty(outer_refs) && return (IRValue[], Substitutions())

    # Sort for deterministic ordering
    outer_refs_sorted = sort!(collect(outer_refs))

    # Create block args and substitutions for each outer ref
    init_values = IRValue[]
    subs = Substitutions()
    next_arg_idx = length(block.args) + 1

    for ssa_id in outer_refs_sorted
        # Get the type from the original code
        ref_type = ssa_id <= length(types) ? types[ssa_id] : Any
        if ref_type isa Core.Const
            ref_type = typeof(ref_type.val)
        elseif ref_type isa Core.PartialStruct
            ref_type = ref_type.typ
        end

        # Create block arg
        arg = BlockArg(next_arg_idx, ref_type)
        push!(block.args, arg)

        # Add to init values
        push!(init_values, SSAValue(ssa_id))

        # Create substitution
        subs[ssa_id] = arg

        next_arg_idx += 1
    end

    # Apply substitutions to the block
    substitute_block!(block, subs)

    return (init_values, subs)
end

#=============================================================================
 Body to Ops Conversion (Statement → Operation with LocalSSA)
=============================================================================#

"""
    SSAToLocalMap

Mapping from original CodeInfo SSA indices to local SSA positions within a block.
"""
const SSAToLocalMap = Dict{Int, Int}

"""
    convert_to_local_ssa(value, ssa_map::SSAToLocalMap)

Convert SSAValue references to LocalSSA references using the mapping.
BlockArgs and other values pass through unchanged.
"""
function convert_to_local_ssa(value::SSAValue, ssa_map::SSAToLocalMap)
    if haskey(ssa_map, value.id)
        return LocalSSA(ssa_map[value.id])
    else
        # SSAValue not in this block - might be from outer scope, keep as-is
        return value
    end
end

function convert_to_local_ssa(value::Expr, ssa_map::SSAToLocalMap)
    new_args = Any[convert_to_local_ssa(a, ssa_map) for a in value.args]
    return Expr(value.head, new_args...)
end

function convert_to_local_ssa(value::PiNode, ssa_map::SSAToLocalMap)
    return PiNode(convert_to_local_ssa(value.val, ssa_map), value.typ)
end

function convert_to_local_ssa(value, ssa_map::SSAToLocalMap)
    # BlockArg, Argument, LocalSSA, constants, etc. pass through unchanged
    return value
end

"""
    convert_terminator_to_local_ssa(term, ssa_map::SSAToLocalMap)

Convert SSAValue references in a terminator to LocalSSA.
"""
function convert_terminator_to_local_ssa(term::ContinueOp, ssa_map::SSAToLocalMap)
    new_values = [convert_to_local_ssa(v, ssa_map) for v in term.values]
    return ContinueOp(new_values)
end

function convert_terminator_to_local_ssa(term::BreakOp, ssa_map::SSAToLocalMap)
    new_values = [convert_to_local_ssa(v, ssa_map) for v in term.values]
    return BreakOp(new_values)
end

function convert_terminator_to_local_ssa(term::ConditionOp, ssa_map::SSAToLocalMap)
    new_cond = convert_to_local_ssa(term.condition, ssa_map)
    new_args = [convert_to_local_ssa(v, ssa_map) for v in term.args]
    return ConditionOp(new_cond, new_args)
end

function convert_terminator_to_local_ssa(term::YieldOp, ssa_map::SSAToLocalMap)
    new_values = [convert_to_local_ssa(v, ssa_map) for v in term.values]
    return YieldOp(new_values)
end

function convert_terminator_to_local_ssa(term::ReturnNode, ssa_map::SSAToLocalMap)
    if isdefined(term, :val)
        new_val = convert_to_local_ssa(term.val, ssa_map)
        if new_val !== term.val
            return ReturnNode(new_val)
        end
    end
    return term
end

function convert_terminator_to_local_ssa(term::Nothing, ssa_map::SSAToLocalMap)
    return nothing
end

"""
    get_result_type(op::ControlFlowOp) -> Type

Get the result type for a control flow operation.
"""
get_result_type(op::IfOp) = op.result_type
get_result_type(op::ForOp) = op.result_type
get_result_type(op::LoopOp) = op.result_type
get_result_type(op::WhileOp) = op.result_type

#=============================================================================
 Iteration Utilities
=============================================================================#

"""
    each_block(f, block::Block)

Recursively iterate over all blocks, calling f on each.
"""
function each_block(f, block::Block)
    f(block)
    for op in block.ops
        if op.expr isa ControlFlowOp
            each_block_in_op(f, op.expr)
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

Recursively iterate over all operations, calling f on each non-ControlFlowOp Operation.
"""
function each_stmt(f, block::Block)
    for op in block.ops
        if op.expr isa ControlFlowOp
            each_stmt_in_op(f, op.expr)
        else
            f(op)
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
    defines(block::Block, ssa::LocalSSA) -> Bool

Check if a block defines a local SSA value (i.e., the ops array has that index).
"""
function defines(block::Block, ssa::LocalSSA)
    return 1 <= ssa.id <= length(block.ops)
end

"""
    defines(block::Block, ssa::SSAValue) -> Bool

Check if a block defines an SSA value. For the new ops-based format,
this always returns false since we use LocalSSA instead.
Kept for compatibility during pattern matching.
"""
function defines(block::Block, ssa::SSAValue)
    # In the new format with ops, we don't use SSAValue indices
    # This is only called during pattern matching which still uses SSAValue
    # For now, check if any operation references this SSA
    for (i, op) in enumerate(block.ops)
        if !(op.expr isa ControlFlowOp)
            # Check if expr references this SSA (it shouldn't after conversion)
            if _expr_defines_ssa(op.expr, ssa.id)
                return true
            end
        else
            cf = op.expr
            if cf isa IfOp
                defines(cf.then_block, ssa) && return true
                defines(cf.else_block, ssa) && return true
            elseif cf isa LoopOp
                defines(cf.body, ssa) && return true
            elseif cf isa ForOp
                defines(cf.body, ssa) && return true
            elseif cf isa WhileOp
                defines(cf.before, ssa) && return true
                defines(cf.after, ssa) && return true
            end
        end
    end
    return false
end

# Helper to check if an expression could define an SSA value
# This is a heuristic for backward compatibility
function _expr_defines_ssa(expr, ssa_id::Int)
    return false  # In the new format, expressions don't have SSA indices
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
    for op in block.ops
        if op.expr isa ControlFlowOp
            _scan_control_flow_uses!(used, op.expr)
        else
            _scan_expr_uses!(used, op.expr)
        end
    end
    if block.terminator !== nothing
        _scan_terminator_uses!(used, block.terminator)
    end
end

function _scan_expr_uses!(used::BitSet, v::LocalSSA)
    push!(used, v.id)
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
function print_value(p::IRPrinter, v::LocalSSA)
    print(p.io, "%", v.id)
end

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
function format_value(p::IRPrinter, v::LocalSSA)
    string("%", v.id)
end
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

# Print a statement (deprecated, use print_op for new code)
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

# Print an operation with local SSA index
function print_op(p::IRPrinter, local_idx::Int, op::Operation; prefix::String="│  ")
    print_indent(p)
    print_colored(p, prefix, :light_black)

    # Only show %N = when the value is used (like Julia's code_typed)
    is_used = local_idx in p.used
    if is_used
        print(p.io, "%", local_idx, " = ")
    else
        print(p.io, "     ")  # Padding to align with used values
    end
    print_expr(p, op.expr)
    print_type_annotation(p, local_idx, op.type)
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

# Print a block's contents (operations, nested ops, terminator)
function print_block_body(p::IRPrinter, block::Block)
    # Collect all items to print to determine which is last
    items = []

    for (idx, op) in enumerate(block.ops)
        if op.expr isa ControlFlowOp
            push!(items, (:nested, op.expr))
        else
            # Create a pseudo-statement for printing with local SSA index
            push!(items, (:stmt, idx, op))
        end
    end
    if block.terminator !== nothing
        push!(items, (:term, block.terminator))
    end

    for (i, item) in enumerate(items)
        is_last = (i == length(items))
        if item[1] == :stmt
            prefix = is_last ? "└──" : "│  "
            print_op(p, item[2], item[3]; prefix=prefix)
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
    n_ops = length(sci.entry.ops)
    print(io, "StructuredCodeInfo(", n_ops, " ops, entry=Block#", sci.entry.id, ")")
end
