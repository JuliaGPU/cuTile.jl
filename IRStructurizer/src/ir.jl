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
    id::Int
    type::Any  # Julia type
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
 SSAVector - Vector of (ssa_idx, stmt, type) triples
=============================================================================#

"""
    SSAEntry

Named tuple for SSAVector entries: `(; stmt, typ)`.
"""
const SSAEntry = @NamedTuple{stmt::Any, typ::Any}

"""
    SSAVector

A vector of `(ssa_idx, statement, type)` triples, ordered by insertion.
Used to store block body contents with their original Julia SSA indices.

Iteration yields `(idx, entry)` pairs where `entry` is a `SSAEntry` named tuple.
Indexing by position returns `SSAEntry`. Use `idx in v` to test presence by SSA index.
"""
struct SSAVector <: AbstractVector{Tuple{Int, SSAEntry}}
    data::Vector{Tuple{Int, Any, Any}}
end

SSAVector() = SSAVector(Tuple{Int,Any,Any}[])

# Iteration yields (idx, (; stmt, typ)) pairs
function Base.iterate(v::SSAVector)
    isempty(v.data) && return nothing
    idx, stmt, typ = v.data[1]
    return (idx, SSAEntry((stmt, typ))), 2
end

function Base.iterate(v::SSAVector, state::Int)
    state > length(v.data) && return nothing
    idx, stmt, typ = v.data[state]
    return (idx, SSAEntry((stmt, typ))), state + 1
end

Base.length(v::SSAVector) = length(v.data)
Base.size(v::SSAVector) = (length(v.data),)

# Positional indexing returns SSAEntry
Base.getindex(v::SSAVector, i::Int) = let (_, stmt, typ) = v.data[i]; SSAEntry((stmt, typ)) end

# Push raw tuple
Base.push!(v::SSAVector, item::Tuple{Int,Any,Any}) = push!(v.data, item)

# Test if SSA index is present
Base.in(ssa_idx::Int, v::SSAVector) = any(idx == ssa_idx for (idx, _, _) in v.data)

# Lazy iterators
indices(v::SSAVector) = (idx for (idx, _, _) in v.data)
statements(v::SSAVector) = (stmt for (_, stmt, _) in v.data)
types(v::SSAVector) = (typ for (_, _, typ) in v.data)

# Alias for backwards compatibility
items(v::SSAVector) = statements(v)

"""
    find_by_ssa(v::SSAVector, ssa_idx::Int) -> Union{SSAEntry, Nothing}

Find a statement and its type by SSA index. Returns SSAEntry or nothing.
"""
function find_by_ssa(v::SSAVector, ssa_idx::Int)
    for (idx, stmt, typ) in v.data
        idx == ssa_idx && return SSAEntry((stmt, typ))
    end
    nothing
end

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
 Abstract Control Flow Type
=============================================================================#

"""
    ControlFlowOp

Abstract type for all structured control flow operations.
"""
abstract type ControlFlowOp end

#=============================================================================
 Block (defined before control flow ops so they can reference it)
=============================================================================#

"""
    Block

A block of statements with block arguments and a terminator.
Body is stored as SSAVector of (ssa_idx, stmt, type) triples.
"""
mutable struct Block
    args::Vector{BlockArg}
    body::SSAVector
    terminator::Terminator
end

Block() = Block(BlockArg[], SSAVector(), nothing)

"""
    push!(block::Block, idx::Int, stmt, typ)

Push a statement or control flow op to a block with its SSA index and type.
"""
Base.push!(block::Block, idx::Int, stmt, typ) = push!(block.body, (idx, stmt, typ))

function Base.show(io::IO, block::Block)
    print(io, "Block(")
    if !isempty(block.args)
        print(io, "args=", length(block.args), ", ")
    end
    n_ops = count(((_, item, _),) -> item isa ControlFlowOp, block.body)
    n_exprs = length(block.body) - n_ops
    print(io, n_exprs + n_ops, " items")
    print(io, ")")
end

# Iteration protocol for Block - yields (idx, stmt, typ) triples
Base.iterate(block::Block) = iterate(block.body)
Base.iterate(block::Block, state) = iterate(block.body, state)
Base.length(block::Block) = length(block.body)
Base.eltype(::Type{Block}) = Tuple{Int,Any,Any}

#=============================================================================
 Structurization Context
=============================================================================#

"""
    StructurizationContext

Context for IR construction. Holds metadata that shouldn't be part of the IR node types:
- ssavaluetypes: Julia types for each SSA value (from CodeInfo)
- next_ssa_idx: Next available SSA index for synthesized values (e.g., loop tuples)
"""
mutable struct StructurizationContext
    ssavaluetypes::Any  # Vector of types (from CodeInfo.ssavaluetypes)
    next_ssa_idx::Int   # Next available SSA index for synthesized values
end

#=============================================================================
 Control Flow Types
=============================================================================#

"""
    IfOp

Structured if-then-else operation.
"""
mutable struct IfOp <: ControlFlowOp
    condition::IRValue
    then_region::Block
    else_region::Block
end

function Base.show(io::IO, ::IfOp)
    print(io, "IfOp()")
end

"""
    ForOp

Counted for-loop with lower/upper/step bounds.
init_values = initial values for loop-carried variables.
"""
mutable struct ForOp <: ControlFlowOp
    lower::IRValue
    upper::IRValue
    step::IRValue
    iv_arg::BlockArg
    body::Block
    init_values::Vector{IRValue}
end

function Base.show(io::IO, op::ForOp)
    print(io, "ForOp(")
    if !isempty(op.init_values)
        print(io, "init_values=", length(op.init_values))
    end
    print(io, ")")
end

"""
    WhileOp

MLIR-style while loop with before (condition) and after (body) regions.
init_values = initial values for loop-carried variables.
"""
mutable struct WhileOp <: ControlFlowOp
    before::Block
    after::Block
    init_values::Vector{IRValue}
end

function Base.show(io::IO, op::WhileOp)
    print(io, "WhileOp(")
    if !isempty(op.init_values)
        print(io, "init_values=", length(op.init_values))
    end
    print(io, ")")
end

"""
    LoopOp

General loop with dynamic exit via BreakOp/ContinueOp.
init_values = initial values for loop-carried variables.
"""
mutable struct LoopOp <: ControlFlowOp
    body::Block
    init_values::Vector{IRValue}
end

function Base.show(io::IO, op::LoopOp)
    print(io, "LoopOp(")
    if !isempty(op.init_values)
        print(io, "init_values=", length(op.init_values))
    end
    print(io, ")")
end

#=============================================================================
 collect_results_ssavals - extract result info from terminators
=============================================================================#

"""
    collect_results_ssavals(op::LoopOp) -> Vector{SSAValue}

Derive result SSAValues from a LoopOp's terminator. Used to identify loop-carried values for
BlockArg creation.
"""
function collect_results_ssavals(op::LoopOp)
    body = op.body::Block
    break_vals = find_break_values(body)
    return SSAValue[v for v in break_vals if v isa SSAValue]
end

# Other control flow ops - results are determined by iter_args count, not SSA indices
collect_results_ssavals(::IfOp) = SSAValue[]
collect_results_ssavals(::ForOp) = SSAValue[]

"""
    collect_results_ssavals(op::WhileOp) -> Vector{SSAValue}

Derive result SSAValues from a WhileOp's ConditionOp terminator.
Used to identify loop-carried values for BlockArg creation.
"""
function collect_results_ssavals(op::WhileOp)
    before = op.before::Block
    term = before.terminator
    if term isa ConditionOp
        return SSAValue[v for v in term.args if v isa SSAValue]
    end
    return SSAValue[]
end

"""
    find_break_values(block::Block) -> Vector{IRValue}

Find BreakOp values in a block, searching inside nested IfOp.
"""
function find_break_values(block::Block)
    # Check block terminator
    if block.terminator isa BreakOp
        return block.terminator.values
    end
    # Search in nested IfOp (common loop structure)
    for stmt in statements(block.body)
        if stmt isa IfOp
            else_blk = stmt.else_region::Block
            if else_blk.terminator isa BreakOp
                return else_blk.terminator.values
            end
        end
    end
    return IRValue[]
end

#=============================================================================
 StructuredCodeInfo - the structured IR for a function
=============================================================================#

"""
    StructuredCodeInfo

Represents a function's code with a structured view of control flow.
The CodeInfo is kept for metadata (slotnames, argtypes, method info).

After structurize!(), the entry Block contains the final structured IR with
expressions and control flow ops.

Create with `StructuredCodeInfo(ci)` for a flat (unstructured) view,
then call `structurize!(sci)` to convert control flow to structured ops.
"""
mutable struct StructuredCodeInfo
    const code::CodeInfo                      # For metadata (slotnames, argtypes, etc.)
    entry::Block                              # Structured IR
    max_ssa_idx::Int                          # Max SSA index (may exceed length(code.code) after structurization)
end

"""
    StructuredCodeInfo(code::CodeInfo)

Create a flat (unstructured) StructuredCodeInfo from Julia CodeInfo.
All statements are placed sequentially in a single block,
with control flow statements (GotoNode, GotoIfNot) included as-is.

Call `structurize!(sci)` to convert to structured control flow.
"""
function StructuredCodeInfo(code::CodeInfo)
    stmts = code.code
    types = code.ssavaluetypes
    n = length(stmts)

    entry = Block()

    for i in 1:n
        stmt = stmts[i]
        if stmt isa ReturnNode
            entry.terminator = stmt
        else
            # Include ALL statements (no substitutions at entry level)
            push!(entry, i, stmt, types[i])
        end
    end

    return StructuredCodeInfo(code, entry, n)
end

#=============================================================================
 Block Substitution (apply SSA → BlockArg mappings)
=============================================================================#

"""
    apply_substitutions!(block::Block, subs::Substitutions)

Apply SSA substitutions within a block. Does not recurse into control flow regions.
Each control flow op's entry values (condition, init_values) are substituted,
but the nested regions are handled by process_block_args! dispatch.
"""
function apply_substitutions!(block::Block, subs::Substitutions)
    isempty(subs) && return

    new_body = SSAVector()
    for (idx, entry) in block.body
        if entry.stmt isa ControlFlowOp
            apply_substitutions!(entry.stmt, subs)
            push!(new_body, (idx, entry.stmt, entry.typ))
        else
            new_expr = substitute_ssa(entry.stmt, subs)
            push!(new_body, (idx, new_expr, entry.typ))
        end
    end
    block.body = new_body

    if block.terminator !== nothing
        block.terminator = substitute_terminator(block.terminator, subs)
    end
end

function apply_substitutions!(op::IfOp, subs::Substitutions)
    op.condition = substitute_ssa(op.condition, subs)
    apply_substitutions!(op.then_region, subs)
    apply_substitutions!(op.else_region, subs)
end

function apply_substitutions!(op::LoopOp, subs::Substitutions)
    for (j, v) in enumerate(op.init_values)
        op.init_values[j] = substitute_ssa(v, subs)
    end
    apply_substitutions!(op.body, subs)
end

function apply_substitutions!(op::WhileOp, subs::Substitutions)
    for (j, v) in enumerate(op.init_values)
        op.init_values[j] = substitute_ssa(v, subs)
    end
    apply_substitutions!(op.before, subs)
    apply_substitutions!(op.after, subs)
end

function apply_substitutions!(op::ForOp, subs::Substitutions)
    op.lower = substitute_ssa(op.lower, subs)
    op.upper = substitute_ssa(op.upper, subs)
    op.step = substitute_ssa(op.step, subs)
    for (j, v) in enumerate(op.init_values)
        op.init_values[j] = substitute_ssa(v, subs)
    end
    apply_substitutions!(op.body, subs)
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
