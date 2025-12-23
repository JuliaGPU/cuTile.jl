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

# Forward declarations needed for mutual recursion
# (The actual types are defined below)

#=============================================================================
 Structurization Context
=============================================================================#

"""
    StructurizationContext

Context for IR construction (Phases 1-4). Holds metadata that shouldn't be part
of the IR node types themselves:
- ssavaluetypes: Julia types for each SSA value (from CodeInfo)
- result_vars: Mapping from op identity to its result SSAValues

Using objectid(op) as key allows PartialControlFlowOp to remain simple while
still tracking which SSAValues each op produces.
"""
mutable struct StructurizationContext
    ssavaluetypes::Any  # Vector of types (from CodeInfo.ssavaluetypes)
    result_vars::Dict{UInt, Vector{SSAValue}}  # objectid(op) => result_vars
end

StructurizationContext(ssavaluetypes) =
    StructurizationContext(ssavaluetypes, Dict{UInt, Vector{SSAValue}}())

"""
    get_result_vars(ctx::StructurizationContext, op) -> Vector{SSAValue}

Get result_vars for an op, returning empty vector if not found.
"""
get_result_vars(ctx::StructurizationContext, op) =
    get(ctx.result_vars, objectid(op), SSAValue[])

"""
    set_result_vars!(ctx::StructurizationContext, op, vars::Vector{SSAValue})

Store result_vars for an op.
"""
set_result_vars!(ctx::StructurizationContext, op, vars::Vector{SSAValue}) =
    ctx.result_vars[objectid(op)] = vars

#=============================================================================
 Unified Control Flow (PartialControlFlowOp / PartialBlock)
=============================================================================#

# PartialControlFlowOp - defined first since PartialBlock references it
"""
    PartialControlFlowOp

Unified control flow operation with head::Symbol for type discrimination.
Used during IR construction (Phases 1-4).

Heads:
- :if - if-then-else, regions[:then] and regions[:else]
- :loop - general loop, regions[:body]
- :for - counted for loop, regions[:body], operands has lower/upper/step/iv_arg
- :while - MLIR-style while, regions[:before] and regions[:after]

Note: result_vars are stored in StructurizationContext, not on the op itself.
"""
mutable struct PartialControlFlowOp
    head::Symbol
    regions::Dict{Symbol, Any}  # Values are PartialBlock (forward reference)
    init_values::Vector{IRValue}
    operands::NamedTuple
end

# Convenience constructor
function PartialControlFlowOp(head::Symbol, regions::Dict{Symbol, <:Any};
                              init_values::Vector{IRValue}=IRValue[],
                              operands::NamedTuple=NamedTuple())
    PartialControlFlowOp(head, regions, init_values, operands)
end

function Base.show(io::IO, op::PartialControlFlowOp)
    print(io, "PartialControlFlowOp(:", op.head)
    print(io, ", regions=[", join(keys(op.regions), ", "), "]")
    if !isempty(op.init_values)
        print(io, ", init=", length(op.init_values))
    end
    print(io, ")")
end

# Accessors for PartialControlFlowOp
condition(op::PartialControlFlowOp) = op.head == :if ? op.operands.condition : error("no condition for :$(op.head)")
lower(op::PartialControlFlowOp) = op.head == :for ? op.operands.lower : error("no lower for :$(op.head)")
upper(op::PartialControlFlowOp) = op.head == :for ? op.operands.upper : error("no upper for :$(op.head)")
step(op::PartialControlFlowOp) = op.head == :for ? op.operands.step : error("no step for :$(op.head)")
iv_arg(op::PartialControlFlowOp) = op.head == :for ? op.operands.iv_arg : error("no iv_arg for :$(op.head)")

then_block(op::PartialControlFlowOp) = get(op.regions, :then, nothing)
else_block(op::PartialControlFlowOp) = get(op.regions, :else, nothing)
body_block(op::PartialControlFlowOp) = get(op.regions, :body, nothing)
before_block(op::PartialControlFlowOp) = get(op.regions, :before, nothing)
after_block(op::PartialControlFlowOp) = get(op.regions, :after, nothing)

const PartialBlockItem = Union{Statement, PartialControlFlowOp}

"""
    PartialBlock

A basic block with SSA metadata, used during IR construction (Phases 1-4).
Same as Block but with explicit name to distinguish from final flattened Block.
"""
mutable struct PartialBlock
    args::Vector{BlockArg}
    body::Vector{PartialBlockItem}
    terminator::Terminator
end

PartialBlock() = PartialBlock(BlockArg[], PartialBlockItem[], nothing)

function Base.show(io::IO, block::PartialBlock)
    print(io, "PartialBlock(")
    if !isempty(block.args)
        print(io, "args=", length(block.args), ", ")
    end
    n_stmts = count(x -> x isa Statement, block.body)
    n_ops = count(x -> x isa PartialControlFlowOp, block.body)
    print(io, "stmts=", n_stmts)
    if n_ops > 0
        print(io, ", ops=", n_ops)
    end
    print(io, ")")
end

# Iteration protocol for PartialBlock
Base.iterate(block::PartialBlock, state=1) = state > length(block.body) ? nothing : (block.body[state], state + 1)
Base.length(block::PartialBlock) = length(block.body)
Base.eltype(::Type{PartialBlock}) = PartialBlockItem

#=============================================================================
 Final Types (Block / ControlFlowOp) - Output of Phase 5
=============================================================================#

#=============================================================================
 Final Control Flow Types
 These are the immutable types output by finalize_ir!.
 Each has direct field access instead of head::Symbol + operands::NamedTuple.
=============================================================================#

"""
    IfOp

Structured if-then-else operation.
Regions are Block (forward reference, typed as Any).
"""
struct IfOp
    condition::IRValue
    then_region::Any    # Block (forward reference)
    else_region::Any    # Block
    init_values::Vector{IRValue}
end

function Base.show(io::IO, op::IfOp)
    print(io, "IfOp(")
    if !isempty(op.init_values)
        print(io, "init=", length(op.init_values))
    end
    print(io, ")")
end

"""
    ForOp

Counted for-loop with lower/upper/step bounds.
"""
struct ForOp
    lower::IRValue
    upper::IRValue
    step::IRValue
    iv_arg::BlockArg
    body::Any           # Block (forward reference)
    init_values::Vector{IRValue}
end

function Base.show(io::IO, op::ForOp)
    print(io, "ForOp(")
    if !isempty(op.init_values)
        print(io, "init=", length(op.init_values))
    end
    print(io, ")")
end

"""
    WhileOp

MLIR-style while loop with before (condition) and after (body) regions.
"""
struct WhileOp
    before::Any     # Block (forward reference), ends with ConditionOp
    after::Any      # Block
    init_values::Vector{IRValue}
end

function Base.show(io::IO, op::WhileOp)
    print(io, "WhileOp(")
    if !isempty(op.init_values)
        print(io, "init=", length(op.init_values))
    end
    print(io, ")")
end

"""
    LoopOp

General loop with dynamic exit via BreakOp/ContinueOp.
"""
struct LoopOp
    body::Any       # Block (forward reference)
    init_values::Vector{IRValue}
end

function Base.show(io::IO, op::LoopOp)
    print(io, "LoopOp(")
    if !isempty(op.init_values)
        print(io, "init=", length(op.init_values))
    end
    print(io, ")")
end

"""
    ControlFlowOp

Union of all control flow operation types.
"""
const ControlFlowOp = Union{IfOp, ForOp, WhileOp, LoopOp}

"""
    BlockItem

Union type for items in a block's body.
Can be either an expression (Any) or a ControlFlowOp.
"""
const BlockItem = Union{Any, ControlFlowOp}

"""
    Block

Final immutable block. Output of finalize_ir!.
Body contains expressions and ControlFlowOps interleaved.
Types vector is parallel to body.
"""
struct Block
    args::Vector{BlockArg}
    body::Vector{Any}           # Expressions and ControlFlowOps interleaved
    types::Vector{Any}          # Parallel to body (type for each item)
    terminator::Terminator
end

Block() = Block(BlockArg[], Any[], Any[], nothing)

function Base.show(io::IO, block::Block)
    print(io, "Block(")
    if !isempty(block.args)
        print(io, "args=", length(block.args), ", ")
    end
    n_exprs = count(x -> !(x isa ControlFlowOp), block.body)
    n_ops = count(x -> x isa ControlFlowOp, block.body)
    print(io, "exprs=", n_exprs)
    if n_ops > 0
        print(io, ", ops=", n_ops)
    end
    print(io, ")")
end

# Iteration protocol for Block
Base.iterate(block::Block, state=1) = state > length(block.body) ? nothing : (block.body[state], state + 1)
Base.length(block::Block) = length(block.body)
Base.eltype(::Type{Block}) = Any

#=============================================================================
 StructuredCodeInfo - the structured IR for a function
=============================================================================#

"""
    StructuredCodeInfo

Represents a function's code with a structured view of control flow.
The CodeInfo is kept for metadata (slotnames, argtypes, method info).

After structurize!(), the entry Block contains the final structured IR with
expressions and ControlFlowOps.

Create with `StructuredCodeInfo(ci)` for a flat (unstructured) view,
then call `structurize!(sci)` to convert control flow to structured ops.
"""
mutable struct StructuredCodeInfo
    const code::CodeInfo                      # For metadata (slotnames, argtypes, etc.)
    entry::Union{PartialBlock, Block}         # Structured IR (Block after finalization)
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

    entry = PartialBlock()

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
    substitute_block_shallow!(block::PartialBlock, subs::Substitutions)

Apply SSA substitutions within a block, recursing into :if ops (same scope) but NOT into :loop ops.
Used by apply_block_args! to let each :loop op handle its own substitution separately.

Substitutes:
- Statement expressions
- :loop init_values (outer scope references)
- Block terminators
- :if contents (since they're part of the same scope, not a new binding context)

Does NOT recurse into :loop ops - each handles its own substitution via apply_block_args!.
"""
function substitute_block_shallow!(block::PartialBlock, subs::Substitutions)
    isempty(subs) && return  # No substitutions to apply

    for (i, item) in enumerate(block.body)
        if item isa Statement
            new_expr = substitute_ssa(item.expr, subs)
            if new_expr !== item.expr
                block.body[i] = Statement(item.idx, new_expr, item.type)
            end
        elseif item isa PartialControlFlowOp
            if item.head == :loop
                # Only substitute init_values (which are in parent scope)
                # The body is handled by recursion in apply_block_args!
                for (j, v) in enumerate(item.init_values)
                    item.init_values[j] = substitute_ssa(v, subs)
                end
            elseif item.head == :if
                # IfOps are part of the same scope - recurse into them
                then_blk = item.regions[:then]::PartialBlock
                else_blk = item.regions[:else]::PartialBlock
                substitute_block_shallow!(then_blk, subs)
                substitute_block_shallow!(else_blk, subs)
            end
        end
    end

    # Substitute terminator
    if block.terminator !== nothing
        block.terminator = substitute_terminator(block.terminator, subs)
    end
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
    each_stmt(f, block::PartialBlock)

Recursively iterate over all statements in a PartialBlock.
"""
function each_stmt(f, block::PartialBlock)
    for item in block.body
        if item isa Statement
            f(item)
        elseif item isa PartialControlFlowOp
            for (_, region) in item.regions
                each_stmt(f, region)
            end
        end
    end
end

"""
    each_block(f, block::PartialBlock)

Recursively iterate over all blocks in a PartialBlock.
"""
function each_block(f, block::PartialBlock)
    f(block)
    for item in block.body
        if item isa PartialControlFlowOp
            for (_, region) in item.regions
                each_block(f, region)
            end
        end
    end
end

"""
    each_stmt(f, block::Block)

Recursively iterate over all expressions in a final Block.
Calls f with (index, expr, type) for each expression in the body.
"""
function each_stmt(f, block::Block)
    for (i, (expr, typ)) in enumerate(zip(block.body, block.types))
        if expr isa ControlFlowOp
            each_stmt_in_op(f, expr)
        else
            # For final Block, we call with a pseudo-Statement-like object
            f((idx=i, expr=expr, type=typ))
        end
    end
end

# Helper to iterate over regions in concrete ControlFlowOp types
each_stmt_in_op(f, op::IfOp) = (each_stmt(f, op.then_region); each_stmt(f, op.else_region))
each_stmt_in_op(f, op::ForOp) = each_stmt(f, op.body)
each_stmt_in_op(f, op::WhileOp) = (each_stmt(f, op.before); each_stmt(f, op.after))
each_stmt_in_op(f, op::LoopOp) = each_stmt(f, op.body)

"""
    each_block(f, block::Block)

Recursively iterate over all blocks in a final Block.
"""
function each_block(f, block::Block)
    f(block)
    for expr in block.body
        if expr isa ControlFlowOp
            each_block_in_op(f, expr)
        end
    end
end

# Helper to iterate over blocks in concrete ControlFlowOp types
each_block_in_op(f, op::IfOp) = (each_block(f, op.then_region); each_block(f, op.else_region))
each_block_in_op(f, op::ForOp) = each_block(f, op.body)
each_block_in_op(f, op::WhileOp) = (each_block(f, op.before); each_block(f, op.after))
each_block_in_op(f, op::LoopOp) = each_block(f, op.body)

#=============================================================================
 Block Queries
=============================================================================#

"""
    collect_outer_refs(block::PartialBlock, defined::Set{Int}; recursive::Bool=false) -> Vector{SSAValue}

Collect all SSAValue references in the block that are NOT in the defined set.
These are "outer" references that need to be captured as BlockArgs.

The defined set should contain SSA indices that are available in the current scope
(e.g., from the parent block's statements or control flow op results).

If recursive=false (default): Only collects from direct statements and the terminator.
Does NOT recurse into nested control flow ops - those will have their own capture pass.

If recursive=true: Also collects from nested control flow ops' blocks.
Use this when the outer op needs to know ALL outer refs in its entire subtree.
"""
function collect_outer_refs(block::PartialBlock, defined::Set{Int}; recursive::Bool=false)
    outer_refs = SSAValue[]
    seen = Set{Int}()

    # Collect from block body
    for item in block.body
        if item isa Statement
            collect_ssa_refs!(outer_refs, seen, item.expr, defined)
        elseif recursive && item isa PartialControlFlowOp
            # Recurse into nested control flow ops
            collect_control_flow_refs!(outer_refs, seen, item, defined)
        end
    end

    # Collect from terminator
    if block.terminator !== nothing
        collect_terminator_refs!(outer_refs, seen, block.terminator, defined)
    end

    return outer_refs
end

"""
    collect_control_flow_refs!(refs, seen, op, defined)

Collect SSAValue references from a control flow op's nested blocks.
"""
function collect_control_flow_refs!(refs::Vector{SSAValue}, seen::Set{Int}, op::PartialControlFlowOp, defined::Set{Int})
    # Collect from operands
    if op.head == :if
        collect_ssa_refs!(refs, seen, op.operands.condition, defined)
    elseif op.head == :for
        collect_ssa_refs!(refs, seen, op.operands.lower, defined)
        collect_ssa_refs!(refs, seen, op.operands.upper, defined)
        collect_ssa_refs!(refs, seen, op.operands.step, defined)
    end
    # Collect from init_values
    for v in op.init_values
        collect_ssa_refs!(refs, seen, v, defined)
    end
    # Recurse into all regions
    for (_, region) in op.regions
        collect_block_refs!(refs, seen, region, defined)
    end
end

"""
    collect_block_refs!(refs, seen, block, defined)

Recursively collect SSAValue references from a block.
"""
function collect_block_refs!(refs::Vector{SSAValue}, seen::Set{Int}, block::PartialBlock, defined::Set{Int})
    for item in block.body
        if item isa Statement
            collect_ssa_refs!(refs, seen, item.expr, defined)
        elseif item isa PartialControlFlowOp
            collect_control_flow_refs!(refs, seen, item, defined)
        end
    end
    if block.terminator !== nothing
        collect_terminator_refs!(refs, seen, block.terminator, defined)
    end
end

"""
    collect_ssa_refs!(refs, seen, value, defined)

Collect SSAValue references from a value that are not in the defined set.
Adds to refs and marks in seen to avoid duplicates.
"""
function collect_ssa_refs!(refs::Vector{SSAValue}, seen::Set{Int}, value::SSAValue, defined::Set{Int})
    if value.id ∉ defined && value.id ∉ seen
        push!(refs, value)
        push!(seen, value.id)
    end
end

function collect_ssa_refs!(refs::Vector{SSAValue}, seen::Set{Int}, value::Expr, defined::Set{Int})
    for arg in value.args
        collect_ssa_refs!(refs, seen, arg, defined)
    end
end

function collect_ssa_refs!(refs::Vector{SSAValue}, seen::Set{Int}, value::PiNode, defined::Set{Int})
    collect_ssa_refs!(refs, seen, value.val, defined)
end

function collect_ssa_refs!(refs::Vector{SSAValue}, seen::Set{Int}, value::PhiNode, defined::Set{Int})
    for i in eachindex(value.values)
        if isassigned(value.values, i)
            collect_ssa_refs!(refs, seen, value.values[i], defined)
        end
    end
end

# Base cases: other value types don't contain SSAValue references
collect_ssa_refs!(::Vector{SSAValue}, ::Set{Int}, ::BlockArg, ::Set{Int}) = nothing
collect_ssa_refs!(::Vector{SSAValue}, ::Set{Int}, ::Argument, ::Set{Int}) = nothing
collect_ssa_refs!(::Vector{SSAValue}, ::Set{Int}, ::SlotNumber, ::Set{Int}) = nothing
collect_ssa_refs!(::Vector{SSAValue}, ::Set{Int}, ::GlobalRef, ::Set{Int}) = nothing
collect_ssa_refs!(::Vector{SSAValue}, ::Set{Int}, ::QuoteNode, ::Set{Int}) = nothing
collect_ssa_refs!(::Vector{SSAValue}, ::Set{Int}, ::Nothing, ::Set{Int}) = nothing
collect_ssa_refs!(::Vector{SSAValue}, ::Set{Int}, ::Number, ::Set{Int}) = nothing
collect_ssa_refs!(::Vector{SSAValue}, ::Set{Int}, ::Symbol, ::Set{Int}) = nothing
collect_ssa_refs!(::Vector{SSAValue}, ::Set{Int}, ::Type, ::Set{Int}) = nothing
collect_ssa_refs!(::Vector{SSAValue}, ::Set{Int}, _, ::Set{Int}) = nothing  # Fallback

"""
    collect_terminator_refs!(refs, seen, term, defined)

Collect SSAValue references from a terminator that are not in the defined set.
"""
function collect_terminator_refs!(refs::Vector{SSAValue}, seen::Set{Int}, term::YieldOp, defined::Set{Int})
    for v in term.values
        collect_ssa_refs!(refs, seen, v, defined)
    end
end

function collect_terminator_refs!(refs::Vector{SSAValue}, seen::Set{Int}, term::ContinueOp, defined::Set{Int})
    for v in term.values
        collect_ssa_refs!(refs, seen, v, defined)
    end
end

function collect_terminator_refs!(refs::Vector{SSAValue}, seen::Set{Int}, term::BreakOp, defined::Set{Int})
    for v in term.values
        collect_ssa_refs!(refs, seen, v, defined)
    end
end

function collect_terminator_refs!(refs::Vector{SSAValue}, seen::Set{Int}, term::ConditionOp, defined::Set{Int})
    collect_ssa_refs!(refs, seen, term.condition, defined)
    for v in term.args
        collect_ssa_refs!(refs, seen, v, defined)
    end
end

function collect_terminator_refs!(refs::Vector{SSAValue}, seen::Set{Int}, term::ReturnNode, defined::Set{Int})
    if isdefined(term, :val)
        collect_ssa_refs!(refs, seen, term.val, defined)
    end
end

function collect_terminator_refs!(::Vector{SSAValue}, ::Set{Int}, ::Nothing, ::Set{Int})
end

#=============================================================================
 Local SSA Conversion (positive SSAValue → negative SSAValue within blocks)
=============================================================================#

# Type alias for SSA to local SSA mapping: original SSA id -> negative SSA id
const SSAToLocalMap = Dict{Int, Int}

"""
    convert_ssa_in_value(value, idx_to_local::SSAToLocalMap)

Convert SSAValue references in a value to negative SSAValue indices where applicable.
Returns the converted value (or original if no conversion needed).
Negative SSAValue indices represent block-local values and will be converted to
positive global indices during finalization.
"""
function convert_ssa_in_value(value::SSAValue, idx_to_local::SSAToLocalMap)
    if haskey(idx_to_local, value.id)
        return SSAValue(idx_to_local[value.id])
    end
    return value
end

function convert_ssa_in_value(value::Expr, idx_to_pos::SSAToLocalMap)
    new_args = Any[convert_ssa_in_value(a, idx_to_pos) for a in value.args]
    # Only create new Expr if something changed
    if new_args != value.args
        return Expr(value.head, new_args...)
    end
    return value
end

function convert_ssa_in_value(value::PiNode, idx_to_pos::SSAToLocalMap)
    new_val = convert_ssa_in_value(value.val, idx_to_pos)
    if new_val !== value.val
        return PiNode(new_val, value.typ)
    end
    return value
end

function convert_ssa_in_value(value::PhiNode, idx_to_pos::SSAToLocalMap)
    new_values = Vector{Any}(undef, length(value.values))
    changed = false
    for i in eachindex(value.values)
        if isassigned(value.values, i)
            new_val = convert_ssa_in_value(value.values[i], idx_to_pos)
            new_values[i] = new_val
            if new_val !== value.values[i]
                changed = true
            end
        end
    end
    if changed
        return PhiNode(value.edges, new_values)
    end
    return value
end

# Base cases: values that don't contain SSAValue references
convert_ssa_in_value(value::BlockArg, ::SSAToLocalMap) = value
convert_ssa_in_value(value::Argument, ::SSAToLocalMap) = value
convert_ssa_in_value(value::SlotNumber, ::SSAToLocalMap) = value
convert_ssa_in_value(value::GlobalRef, ::SSAToLocalMap) = value
convert_ssa_in_value(value::QuoteNode, ::SSAToLocalMap) = value
convert_ssa_in_value(value::Nothing, ::SSAToLocalMap) = value
convert_ssa_in_value(value::Number, ::SSAToLocalMap) = value
convert_ssa_in_value(value::Symbol, ::SSAToLocalMap) = value
convert_ssa_in_value(value::Type, ::SSAToLocalMap) = value
convert_ssa_in_value(value, ::SSAToLocalMap) = value  # Fallback

"""
    convert_ssa_in_terminator(term, idx_to_local)

Convert SSAValue references in a terminator to negative SSAValue indices.
"""
function convert_ssa_in_terminator(term::ContinueOp, idx_to_pos::SSAToLocalMap)
    new_values = [convert_ssa_in_value(v, idx_to_pos) for v in term.values]
    return ContinueOp(new_values)
end

function convert_ssa_in_terminator(term::BreakOp, idx_to_pos::SSAToLocalMap)
    new_values = [convert_ssa_in_value(v, idx_to_pos) for v in term.values]
    return BreakOp(new_values)
end

function convert_ssa_in_terminator(term::YieldOp, idx_to_pos::SSAToLocalMap)
    new_values = [convert_ssa_in_value(v, idx_to_pos) for v in term.values]
    return YieldOp(new_values)
end

function convert_ssa_in_terminator(term::ConditionOp, idx_to_pos::SSAToLocalMap)
    new_cond = convert_ssa_in_value(term.condition, idx_to_pos)
    new_args = [convert_ssa_in_value(v, idx_to_pos) for v in term.args]
    return ConditionOp(new_cond, new_args)
end

function convert_ssa_in_terminator(term::ReturnNode, idx_to_pos::SSAToLocalMap)
    if isdefined(term, :val)
        new_val = convert_ssa_in_value(term.val, idx_to_pos)
        if new_val !== term.val
            return ReturnNode(new_val)
        end
    end
    return term
end

function convert_ssa_in_terminator(term::Nothing, ::SSAToLocalMap)
    return nothing
end

"""
    convert_to_local_ssa!(block::PartialBlock, ctx::StructurizationContext)

Convert SSAValue references within a block to negative SSAValue indices.
Each statement and control flow op result gets a unique negative index.
These negative indices are block-local and will be converted to positive
global indices during finalization.
"""
function convert_to_local_ssa!(block::PartialBlock, ctx::StructurizationContext)
    # Build mapping: Statement.idx → unique negative index
    # Each result gets its own unique negative index (no result_idx needed)
    idx_to_local = SSAToLocalMap()
    counter = 0
    for item in block.body
        if item isa Statement
            counter -= 1
            idx_to_local[item.idx] = counter
        elseif item isa PartialControlFlowOp
            # Control flow ops produce values through result_vars (stored in context)
            # Each result_var gets its own unique negative index
            for rv in get_result_vars(ctx, item)
                counter -= 1
                idx_to_local[rv.id] = counter
            end
        end
    end

    # Convert SSAValues in statements
    for (i, item) in enumerate(block.body)
        if item isa Statement
            new_expr = convert_ssa_in_value(item.expr, idx_to_local)
            if new_expr !== item.expr
                block.body[i] = Statement(item.idx, new_expr, item.type)
            end
        elseif item isa PartialControlFlowOp
            convert_to_local_ssa_in_op!(item, idx_to_local, ctx)
        end
    end

    # Convert SSAValues in terminator
    if block.terminator !== nothing
        block.terminator = convert_ssa_in_terminator(block.terminator, idx_to_local)
    end
end

function convert_to_local_ssa_in_op!(op::PartialControlFlowOp, parent_idx_to_local::SSAToLocalMap,
                                     ctx::StructurizationContext)
    # Convert init_values using parent's mapping
    for (i, v) in enumerate(op.init_values)
        op.init_values[i] = convert_ssa_in_value(v, parent_idx_to_local)
    end

    # For :for ops, convert operands (lower, upper, step)
    if op.head == :for
        new_lower = convert_ssa_in_value(op.operands.lower, parent_idx_to_local)
        new_upper = convert_ssa_in_value(op.operands.upper, parent_idx_to_local)
        new_step = convert_ssa_in_value(op.operands.step, parent_idx_to_local)
        op.operands = (lower=new_lower, upper=new_upper, step=new_step, iv_arg=op.operands.iv_arg)
    elseif op.head == :if
        new_cond = convert_ssa_in_value(op.operands.condition, parent_idx_to_local)
        op.operands = (condition=new_cond,)
    end

    # Recursively convert nested blocks
    for (_, region) in op.regions
        convert_to_local_ssa!(region, ctx)
    end
end

#=============================================================================
 IR Finalization (PartialBlock/PartialControlFlowOp → Block/ControlFlowOp)
=============================================================================#

"""
    negate_negative_ssa(value)

Convert negative SSAValue indices to positive local indices.
SSAValue(-n) → SSAValue(n). Positive indices are left unchanged.
Each block has its own local SSA namespace.
"""
negate_negative_ssa(v::SSAValue) = v.id < 0 ? SSAValue(-v.id) : v

function negate_negative_ssa(v::Expr)
    new_args = Any[negate_negative_ssa(a) for a in v.args]
    new_args != v.args ? Expr(v.head, new_args...) : v
end

function negate_negative_ssa(v::PiNode)
    new_val = negate_negative_ssa(v.val)
    new_val !== v.val ? PiNode(new_val, v.typ) : v
end

negate_negative_ssa(v::Any) = v

function negate_terminator_ssa(term::YieldOp)
    new_values = [negate_negative_ssa(v) for v in term.values]
    new_values != term.values ? YieldOp(new_values) : term
end

function negate_terminator_ssa(term::ContinueOp)
    new_values = [negate_negative_ssa(v) for v in term.values]
    new_values != term.values ? ContinueOp(new_values) : term
end

function negate_terminator_ssa(term::BreakOp)
    new_values = [negate_negative_ssa(v) for v in term.values]
    new_values != term.values ? BreakOp(new_values) : term
end

function negate_terminator_ssa(term::ConditionOp)
    new_cond = negate_negative_ssa(term.condition)
    new_args = [negate_negative_ssa(v) for v in term.args]
    (new_cond !== term.condition || new_args != term.args) ?
        ConditionOp(new_cond, new_args) : term
end

function negate_terminator_ssa(term::ReturnNode)
    if isdefined(term, :val)
        new_val = negate_negative_ssa(term.val)
        new_val !== term.val ? ReturnNode(new_val) : term
    else
        term
    end
end

negate_terminator_ssa(term::Nothing) = term

"""
    finalize_ir(block::PartialBlock, ctx::StructurizationContext) -> Block

Convert a PartialBlock to a final Block by:
1. Converting negative SSAValue to positive local SSAValue (per-block)
2. Flattening Statement wrappers to (expr, type) pairs in body/types vectors
3. Converting PartialControlFlowOp to ControlFlowOp
4. Resolving control flow op types to Julia types via context
5. Recursively processing all nested regions

SSA indices are local to each block: body[i] defines SSAValue(i).
"""
function finalize_ir(block::PartialBlock, ctx::StructurizationContext)::Block
    finalize_block(block, ctx)
end

function finalize_block(block::PartialBlock, ctx::StructurizationContext)::Block
    body = Any[]
    types = Any[]

    for item in block.body
        if item isa Statement
            push!(body, negate_negative_ssa(item.expr))
            push!(types, item.type)
        elseif item isa PartialControlFlowOp
            push!(body, finalize_control_flow_op(item, ctx))
            # Store resolved Julia types for codegen
            # Resolve SSAValue references to actual Julia types
            result_vars = get_result_vars(ctx, item)
            if isempty(result_vars)
                push!(types, Nothing)
            elseif length(result_vars) == 1
                push!(types, ctx.ssavaluetypes[result_vars[1].id])
            else
                push!(types, Tuple{(ctx.ssavaluetypes[rv.id] for rv in result_vars)...})
            end
        end
    end

    terminator = block.terminator === nothing ? nothing :
        negate_terminator_ssa(block.terminator)

    return Block(copy(block.args), body, types, terminator)
end

"""
    finalize_control_flow_op(op::PartialControlFlowOp, ctx::StructurizationContext) -> ControlFlowOp

Convert a PartialControlFlowOp to a final ControlFlowOp.
Nested regions are finalized recursively with their own local SSA namespaces.
"""
function finalize_control_flow_op(op::PartialControlFlowOp, ctx::StructurizationContext)::ControlFlowOp
    # Convert all regions (each has its own local SSA namespace)
    regions = Dict{Symbol, Block}()
    for (name, region) in op.regions
        regions[name] = finalize_block(region::PartialBlock, ctx)
    end

    init_values = [negate_negative_ssa(v) for v in op.init_values]

    # Construct the appropriate concrete type with converted operands
    if op.head == :if
        return IfOp(
            negate_negative_ssa(op.operands.condition),
            regions[:then],
            regions[:else],
            init_values
        )
    elseif op.head == :for
        return ForOp(
            negate_negative_ssa(op.operands.lower),
            negate_negative_ssa(op.operands.upper),
            negate_negative_ssa(op.operands.step),
            op.operands.iv_arg,
            regions[:body],
            init_values
        )
    elseif op.head == :while
        return WhileOp(
            regions[:before],
            regions[:after],
            init_values
        )
    else  # :loop
        return LoopOp(
            regions[:body],
            init_values
        )
    end
end

#=============================================================================
 Pretty Printing (Julia CodeInfo-style with colors)
=============================================================================#

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

# Compute which SSA values are used (for type coloring)
function compute_used_ssas(block::PartialBlock)
    used = BitSet()
    _scan_uses!(used, block)
    return used
end

function _scan_uses!(used::BitSet, block::PartialBlock)
    for item in block.body
        if item isa Statement
            _scan_expr_uses!(used, item.expr)
        elseif item isa PartialControlFlowOp
            _scan_control_flow_uses!(used, item)
        end
    end
    if block.terminator !== nothing
        _scan_terminator_uses!(used, block.terminator)
    end
end

function _scan_control_flow_uses!(used::BitSet, op::PartialControlFlowOp)
    # Scan operands
    if op.head == :if
        _scan_expr_uses!(used, op.operands.condition)
    elseif op.head == :for
        _scan_expr_uses!(used, op.operands.lower)
        _scan_expr_uses!(used, op.operands.upper)
        _scan_expr_uses!(used, op.operands.step)
    end
    # Scan init_values
    for v in op.init_values
        _scan_expr_uses!(used, v)
    end
    # Recursively scan regions
    for (_, region) in op.regions
        _scan_uses!(used, region)
    end
end

# Final Block/ControlFlowOp usage scanning
function compute_used_ssas(block::Block)
    used = BitSet()
    _scan_uses!(used, block)
    return used
end

function _scan_uses!(used::BitSet, block::Block)
    for (expr, _) in zip(block.body, block.types)
        if expr isa ControlFlowOp
            _scan_control_flow_uses!(used, expr)
        else
            _scan_expr_uses!(used, expr)
        end
    end
    if block.terminator !== nothing
        _scan_terminator_uses!(used, block.terminator)
    end
end

function _scan_control_flow_uses!(used::BitSet, op::IfOp)
    _scan_expr_uses!(used, op.condition)
    for v in op.init_values
        _scan_expr_uses!(used, v)
    end
    _scan_uses!(used, op.then_region)
    _scan_uses!(used, op.else_region)
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

function _scan_control_flow_uses!(used::BitSet, op::WhileOp)
    for v in op.init_values
        _scan_expr_uses!(used, v)
    end
    _scan_uses!(used, op.before)
    _scan_uses!(used, op.after)
end

function _scan_control_flow_uses!(used::BitSet, op::LoopOp)
    for v in op.init_values
        _scan_expr_uses!(used, v)
    end
    _scan_uses!(used, op.body)
end

# Compute which SSAValue indices are used in a Block (for pretty printing)
function compute_block_used(block::Block)
    used = BitSet()
    for expr in block.body
        if expr isa IfOp
            _scan_expr_uses!(used, expr.condition)
        elseif expr isa ForOp
            _scan_expr_uses!(used, expr.lower)
            _scan_expr_uses!(used, expr.upper)
            _scan_expr_uses!(used, expr.step)
            for v in expr.init_values
                _scan_expr_uses!(used, v)
            end
        elseif expr isa LoopOp || expr isa WhileOp
            for v in expr.init_values
                _scan_expr_uses!(used, v)
            end
        else
            _scan_expr_uses!(used, expr)
        end
    end
    # Scan terminator
    if block.terminator !== nothing
        _scan_terminator_uses!(used, block.terminator)
    end
    return used
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
    block_used::BitSet     # Which SSA values are used in current block (for nested blocks)
    max_idx_width::Int     # Max width of "%N = " for alignment (like Julia's SSA printer)
    color::Bool            # Whether to use colors
end

function IRPrinter(io::IO, code::CodeInfo, entry::PartialBlock)
    used = compute_used_ssas(entry)
    color = get(io, :color, false)::Bool
    IRPrinter(io, code, 0, "", false, used, BitSet(), 0, color)
end

function IRPrinter(io::IO, code::CodeInfo, entry::Block)
    used = compute_used_ssas(entry)
    block_used = compute_block_used(entry)
    # Compute max index width for alignment: "%N = " where N is the max index
    max_idx_width = ndigits(length(entry.body)) + 4  # % + digits + space + = + space
    color = get(io, :color, false)::Bool
    IRPrinter(io, code, 0, "", false, used, block_used, max_idx_width, color)
end

function indent(p::IRPrinter, n::Int=1)
    new_prefix = p.line_prefix * "    "  # 4 spaces per indent level
    return IRPrinter(p.io, p.code, p.indent + n, new_prefix, false, p.used, p.block_used, p.max_idx_width, p.color)
end

# Create a child printer for a nested Block with fresh block_used and max_idx_width
function child_printer(p::IRPrinter, nested_block::Block, cont_prefix::String)
    nested_block_used = compute_block_used(nested_block)
    nested_max_idx_width = ndigits(length(nested_block.body)) + 4
    IRPrinter(p.io, p.code, p.indent + 1, p.line_prefix * cont_prefix, false, p.used, nested_block_used, nested_max_idx_width, p.color)
end

# Print region header: "├ label(%arg1::Type):" (regions always use ├, closing is on last line of content)
function print_region_header(p::IRPrinter, label::String, args::Vector{BlockArg})
    print_indent(p)
    print_colored(p, "├", :light_black)
    print(p.io, " ", label)
    if !isempty(args)
        print(p.io, "(")
        for (i, arg) in enumerate(args)
            i > 1 && print(p.io, ", ")
            print(p.io, "%arg", arg.id)
            print_colored(p, string("::", format_type(arg.type)), :cyan)
        end
        print(p.io, ")")
    end
    println(p.io, ":")
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

# Format result variables (string version for backwards compat)
function format_results(p::IRPrinter, results::Vector{SSAValue})
    if isempty(results)
        ""
    elseif length(results) == 1
        r = results[1]
        typ = p.code.ssavaluetypes[r.id]
        string(format_value(p, r), "::", format_type(typ))
    else
        parts = [string(format_value(p, r), "::", format_type(p.code.ssavaluetypes[r.id]))
                 for r in results]
        string("(", join(parts, ", "), ")")
    end
end

# Print result variables with type colors
function print_results(p::IRPrinter, results::Vector{SSAValue})
    if isempty(results)
        return
    elseif length(results) == 1
        r = results[1]
        print(p.io, "%", r.id)
        is_used = r.id in p.used
        color = is_used ? :cyan : :light_black
        print_colored(p, string("::", format_type(p.code.ssavaluetypes[r.id])), color)
    else
        print(p.io, "(")
        for (i, r) in enumerate(results)
            i > 1 && print(p.io, ", ")
            print(p.io, "%", r.id)
            is_used = r.id in p.used
            color = is_used ? :cyan : :light_black
            print_colored(p, string("::", format_type(p.code.ssavaluetypes[r.id])), color)
        end
        print(p.io, ")")
    end
end

# Print a statement
function print_stmt(p::IRPrinter, stmt::Statement; prefix::String="│  ")
    print_indent(p)
    print_colored(p, prefix, :light_black)
    print(p.io, " ")

    # Only show %N = when the value is used (like Julia's code_typed)
    is_used = stmt.idx in p.used
    if is_used
        print(p.io, "%", stmt.idx, " = ")
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
    if !isempty(prefix)
        print_colored(p, prefix, :light_black)
        print(p.io, " ")
    end
    print(p.io, "return")
    if isdefined(term, :val)
        print(p.io, " ")
        print_value(p, term.val)
    end
    println(p.io)
end

function print_terminator(p::IRPrinter, term::Union{YieldOp,ContinueOp,BreakOp,ConditionOp}; prefix::String="└──")
    print_indent(p)
    if !isempty(prefix)
        print_colored(p, prefix, :light_black)
        print(p.io, " ")
    end
    print_terminator_content(p, term)
    println(p.io)
end

function print_terminator(p::IRPrinter, ::Nothing; prefix::String="└──")
    # No terminator
end

# Print a PartialBlock's contents
function print_block_body(p::IRPrinter, block::PartialBlock)
    items = []
    for item in block.body
        if item isa Statement
            push!(items, (:stmt, item))
        elseif item isa PartialControlFlowOp
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

# Print PartialControlFlowOp (dispatches on head)
function print_control_flow(p::IRPrinter, op::PartialControlFlowOp; is_last::Bool=false)
    if op.head == :if
        print_if_op(p, op; is_last=is_last)
    elseif op.head == :for
        print_for_op(p, op; is_last=is_last)
    elseif op.head == :while
        print_while_op(p, op; is_last=is_last)
    elseif op.head == :loop
        print_loop_op(p, op; is_last=is_last)
    else
        error("Unknown op head: $(op.head)")
    end
end

function print_if_op(p::IRPrinter, op::PartialControlFlowOp; is_last::Bool=false)
    prefix = is_last ? "└──" : "├──"
    cont_prefix = is_last ? "    " : "│   "

    print_indent(p)
    print_colored(p, prefix, :light_black)
    print(p.io, " ")

    # Note: result_vars are stored in context, not on op, so we don't print them here

    print(p.io, "if ")
    print_value(p, op.operands.condition)
    println(p.io)

    then_blk = op.regions[:then]::PartialBlock
    else_blk = op.regions[:else]::PartialBlock

    then_p = IRPrinter(p.io, p.code, p.indent + 1, p.line_prefix * cont_prefix, false, p.used, p.color)
    print_block_body(then_p, then_blk)

    print_indent(p)
    print_colored(p, cont_prefix, :light_black)
    println(p.io, "else")

    print_block_body(then_p, else_blk)

    print_indent(p)
    print_colored(p, cont_prefix, :light_black)
    println(p.io, "end")
end

function print_for_op(p::IRPrinter, op::PartialControlFlowOp; is_last::Bool=false)
    prefix = is_last ? "└──" : "├──"
    cont_prefix = is_last ? "    " : "│   "

    print_indent(p)
    print_colored(p, prefix, :light_black)
    print(p.io, " ")

    # Note: result_vars are stored in context, not on op, so we don't print them here

    body = op.regions[:body]::PartialBlock

    print_colored(p, "for", :yellow)
    print(p.io, " %arg", op.operands.iv_arg.id, " = ")
    print_value(p, op.operands.lower)
    print(p.io, ":")
    print_value(p, op.operands.step)
    print(p.io, ":")
    print_value(p, op.operands.upper)

    if !isempty(body.args)
        print_iter_args(p, body.args, op.init_values)
    end

    println(p.io)

    body_p = IRPrinter(p.io, p.code, p.indent + 1, p.line_prefix * cont_prefix, false, p.used, p.color)
    print_block_body(body_p, body)

    print_indent(p)
    print_colored(p, cont_prefix, :light_black)
    println(p.io, "end")
end

function print_loop_op(p::IRPrinter, op::PartialControlFlowOp; is_last::Bool=false)
    prefix = is_last ? "└──" : "├──"
    cont_prefix = is_last ? "    " : "│   "

    print_indent(p)
    print_colored(p, prefix, :light_black)
    print(p.io, " ")

    # Note: result_vars are stored in context, not on op, so we don't print them here

    body = op.regions[:body]::PartialBlock

    print_colored(p, "loop", :yellow)
    print_iter_args(p, body.args, op.init_values)
    println(p.io)

    body_p = IRPrinter(p.io, p.code, p.indent + 1, p.line_prefix * cont_prefix, false, p.used, p.color)
    print_block_body(body_p, body)

    print_indent(p)
    print_colored(p, cont_prefix, :light_black)
    println(p.io, "end")
end

function print_while_op(p::IRPrinter, op::PartialControlFlowOp; is_last::Bool=false)
    prefix = is_last ? "└──" : "├──"
    cont_prefix = is_last ? "    " : "│   "

    print_indent(p)
    print_colored(p, prefix, :light_black)
    print(p.io, " ")

    # Note: result_vars are stored in context, not on op, so we don't print them here

    before = op.regions[:before]::PartialBlock
    after = op.regions[:after]::PartialBlock

    print_colored(p, "while", :yellow)
    print_iter_args(p, before.args, op.init_values)
    println(p.io, " {")

    before_p = IRPrinter(p.io, p.code, p.indent + 1, p.line_prefix * cont_prefix, false, p.used, p.color)
    print_block_body(before_p, before)

    print_indent(p)
    print_colored(p, cont_prefix, :light_black)
    println(p.io, "} do {")

    after_p = IRPrinter(p.io, p.code, p.indent + 1, p.line_prefix * cont_prefix, false, p.used, p.color)
    print_block_body(after_p, after)

    print_indent(p)
    print_colored(p, cont_prefix, :light_black)
    println(p.io, "}")
end

#=============================================================================
 Pretty Printing for Final Block/ControlFlowOp
=============================================================================#

# Print expression with type annotation (for final Block)
# prefix can be "│  " for continuation, "└──" for last, or "" for no prefix (inside regions)
function print_expr_with_type(p::IRPrinter, idx::Int, expr, typ; prefix::String="│  ")
    print_indent(p)
    if !isempty(prefix)
        print_colored(p, prefix, :light_black)
        print(p.io, " ")
    end

    # For final Block, we use position index instead of SSA index
    # Type color: cyan if used, grey if unused (matches Julia's SSA IR printer)
    is_used = idx in p.block_used
    if is_used
        idx_s = string(idx)
        pad = " "^(p.max_idx_width - length(idx_s) - 4)  # -4 for "% = "
        print(p.io, "%", idx_s, pad, " = ")
    else
        # Pad to align with "%N = " (like Julia's SSA printer)
        print(p.io, " "^p.max_idx_width)
    end
    print_expr(p, expr)

    # Print type annotation
    color = is_used ? :cyan : :light_black
    print_colored(p, string("::", format_type(typ)), color)
    println(p.io)
end

# Print a Block's contents (final type)
# If inside_region=true, don't show statement-level box drawing (just use line_prefix)
# If is_last_region=true, modify the last line to show outer block closing
function print_block_body(p::IRPrinter, block::Block; inside_region::Bool=false, is_last_region::Bool=false, cascade_depth::Int=0)
    items = []
    for (i, (expr, typ)) in enumerate(zip(block.body, block.types))
        if expr isa ControlFlowOp
            push!(items, (:nested, i, expr, typ))
        else
            push!(items, (:expr, i, expr, typ))
        end
    end
    if block.terminator !== nothing
        push!(items, (:term, block.terminator))
    end

    for (i, item) in enumerate(items)
        is_last = (i == length(items))
        if item[1] == :expr
            if inside_region
                prefix = ""
            else
                prefix = is_last ? "└──" : "│  "
            end
            if is_last && is_last_region
                # Replace trailing │ characters in line_prefix with └ for outer block closing
                print_expr_with_type_closing(p, item[2], item[3], item[4]; cascade_depth=cascade_depth)
            else
                print_expr_with_type(p, item[2], item[3], item[4]; prefix=prefix)
            end
        elseif item[1] == :nested
            if is_last && is_last_region
                # Cascade the closing - don't use └── here (is_last=false), and increment cascade_depth
                # so the nested op knows to close this level too
                print_control_flow(p, item[3], item[2]; is_last=false, cascade_depth=cascade_depth + 1)
            else
                print_control_flow(p, item[3], item[2]; is_last=is_last, cascade_depth=cascade_depth)
            end
        else  # :term
            if inside_region
                prefix = ""
            else
                prefix = "└──"
            end
            if is_last && is_last_region
                # Replace trailing │ characters in line_prefix with └ for outer block closing
                print_terminator_closing(p, item[2]; cascade_depth=cascade_depth)
            else
                print_terminator(p, item[2]; prefix=prefix)
            end
        end
    end
end

"""
    close_trailing_boxes(prefix::String, count::Int) -> String

Replace `count` trailing │ characters in prefix: the last one becomes └, the rest become spaces.
E.g., close_trailing_boxes("│   │   │   ", 2) → "│   └       " (closes 2 levels)
"""
function close_trailing_boxes(prefix::String, count::Int)
    count <= 0 && return prefix

    # Find all │ positions (working backwards)
    chars = collect(prefix)
    box_positions = Int[]
    for i in length(chars):-1:1
        if chars[i] == '│'
            push!(box_positions, i)
        end
    end

    # Replace up to `count` trailing │ characters
    n_to_replace = min(count, length(box_positions))
    for (i, pos) in enumerate(box_positions[1:n_to_replace])
        if i == 1
            # Last │ becomes └
            chars[pos] = '└'
        else
            # Other trailing │ become space
            chars[pos] = ' '
        end
    end

    return String(chars)
end

# Print expression for last line of last region (replace trailing │ with └)
function print_expr_with_type_closing(p::IRPrinter, idx::Int, expr, typ; cascade_depth::Int=0)
    # Replace (cascade_depth + 1) trailing │ characters: last one becomes └, rest become spaces
    closing_prefix = close_trailing_boxes(p.line_prefix, cascade_depth + 1)
    print_colored(p, closing_prefix, :light_black)

    # Type color: cyan if used, grey if unused (matches Julia's SSA IR printer)
    is_used = idx in p.block_used
    if is_used
        idx_s = string(idx)
        pad = " "^(p.max_idx_width - length(idx_s) - 4)  # -4 for "% = "
        print(p.io, "%", idx_s, pad, " = ")
    else
        # Pad to align with "%N = " (like Julia's SSA printer)
        print(p.io, " "^p.max_idx_width)
    end
    print_expr(p, expr)

    # Print type annotation
    color = is_used ? :cyan : :light_black
    print_colored(p, string("::", format_type(typ)), color)
    println(p.io)
end

# Print terminator for last line of last region (replace trailing │ with └)
function print_terminator_closing(p::IRPrinter, term; cascade_depth::Int=0)
    closing_prefix = close_trailing_boxes(p.line_prefix, cascade_depth + 1)
    print_colored(p, closing_prefix, :light_black)
    print_terminator_content(p, term)
    println(p.io)
end

# Print just the terminator content (keyword + values), no prefix or newline
function print_terminator_content(p::IRPrinter, term::YieldOp)
    print_colored(p, "yield", :yellow)
    print_terminator_values(p, term.values)
end
function print_terminator_content(p::IRPrinter, term::ContinueOp)
    print_colored(p, "continue", :yellow)
    print_terminator_values(p, term.values)
end
function print_terminator_content(p::IRPrinter, term::BreakOp)
    print_colored(p, "break", :yellow)
    print_terminator_values(p, term.values)
end
function print_terminator_content(p::IRPrinter, term::ConditionOp)
    print_colored(p, "condition", :yellow)
    print(p.io, "(")
    print_value(p, term.condition)
    print(p.io, ")")
    print_terminator_values(p, term.args)
end
function print_terminator_content(p::IRPrinter, term::ReturnNode)
    print(p.io, "return")
    if isdefined(term, :val)
        print(p.io, " ")
        print_value(p, term.val)
    end
end

function print_terminator_values(p::IRPrinter, values)
    if !isempty(values)
        print(p.io, " ")
        for (i, v) in enumerate(values)
            i > 1 && print(p.io, ", ")
            print_value(p, v)
        end
    end
end

# Print ControlFlowOp (final type, dispatches via multiple dispatch)
print_control_flow(p::IRPrinter, op::IfOp, pos::Int; is_last::Bool=false, cascade_depth::Int=0) = print_if_op_final(p, op, pos; is_last, cascade_depth)
print_control_flow(p::IRPrinter, op::ForOp, pos::Int; is_last::Bool=false, cascade_depth::Int=0) = print_for_op_final(p, op, pos; is_last, cascade_depth)
print_control_flow(p::IRPrinter, op::WhileOp, pos::Int; is_last::Bool=false, cascade_depth::Int=0) = print_while_op_final(p, op, pos; is_last, cascade_depth)
print_control_flow(p::IRPrinter, op::LoopOp, pos::Int; is_last::Bool=false, cascade_depth::Int=0) = print_loop_op_final(p, op, pos; is_last, cascade_depth)

function print_if_op_final(p::IRPrinter, op::IfOp, pos::Int; is_last::Bool=false, cascade_depth::Int=0)
    # When cascade_depth > 0, keep │ going (don't use └──) so inner content can close it
    use_closing = is_last && cascade_depth == 0
    prefix = use_closing ? "└──" : "├──"
    cont_prefix = use_closing ? "    " : "│   "

    print_indent(p)
    print_colored(p, prefix, :light_black)
    print(p.io, " ")

    # Show assignment if this position is used
    if pos in p.block_used
        print(p.io, "%", pos, " = ")
    end

    print(p.io, "if ")
    print_value(p, op.condition)
    println(p.io)

    # Print "then:" region header
    then_p = child_printer(p, op.then_region, cont_prefix)
    print_region_header(then_p, "then", op.then_region.args)
    # Print then region body (inside_region=true, no statement-level box drawing)
    then_body_p = child_printer(then_p, op.then_region, "│   ")
    print_block_body(then_body_p, op.then_region; inside_region=true)

    # Print "else:" region header
    else_p = child_printer(p, op.else_region, cont_prefix)
    print_region_header(else_p, "else", op.else_region.args)
    # Print else region body (last region, show closing on last line)
    # Pass cascade_depth so innermost item can close all cascaded levels
    else_body_p = child_printer(else_p, op.else_region, "│   ")
    print_block_body(else_body_p, op.else_region; inside_region=true, is_last_region=true, cascade_depth=cascade_depth)
end

function print_for_op_final(p::IRPrinter, op::ForOp, pos::Int; is_last::Bool=false, cascade_depth::Int=0)
    # When cascade_depth > 0, keep │ going so inner content can close it
    use_closing = is_last && cascade_depth == 0
    prefix = use_closing ? "└──" : "├──"
    cont_prefix = use_closing ? "    " : "│   "

    print_indent(p)
    print_colored(p, prefix, :light_black)
    print(p.io, " ")

    # Show assignment if this position is used
    if pos in p.block_used
        print(p.io, "%", pos, " = ")
    end

    print_colored(p, "for", :yellow)
    print(p.io, " %arg", op.iv_arg.id, " = ")
    print_value(p, op.lower)
    print(p.io, ":")
    print_value(p, op.step)
    print(p.io, ":")
    print_value(p, op.upper)

    if !isempty(op.body.args)
        print_iter_args(p, op.body.args, op.init_values)
    end

    println(p.io)

    # Body is inline (no region header for single-region ops)
    # Always pass is_last_region=true so the last item closes the for's box drawing
    # Pass cascade_depth so innermost item can close all cascaded levels
    body_p = child_printer(p, op.body, cont_prefix)
    print_block_body(body_p, op.body; is_last_region=true, cascade_depth=cascade_depth)
end

function print_loop_op_final(p::IRPrinter, op::LoopOp, pos::Int; is_last::Bool=false, cascade_depth::Int=0)
    # When cascade_depth > 0, keep │ going so inner content can close it
    use_closing = is_last && cascade_depth == 0
    prefix = use_closing ? "└──" : "├──"
    cont_prefix = use_closing ? "    " : "│   "

    print_indent(p)
    print_colored(p, prefix, :light_black)
    print(p.io, " ")

    # Show assignment if this position is used
    if pos in p.block_used
        print(p.io, "%", pos, " = ")
    end

    print_colored(p, "loop", :yellow)
    print_iter_args(p, op.body.args, op.init_values)
    println(p.io)

    # Body is inline (no region header for single-region ops)
    # Always pass is_last_region=true so the last item closes the loop's box drawing
    # Pass cascade_depth so innermost item can close all cascaded levels
    body_p = child_printer(p, op.body, cont_prefix)
    print_block_body(body_p, op.body; is_last_region=true, cascade_depth=cascade_depth)
end

function print_while_op_final(p::IRPrinter, op::WhileOp, pos::Int; is_last::Bool=false, cascade_depth::Int=0)
    # When cascade_depth > 0, keep │ going so inner content can close it
    use_closing = is_last && cascade_depth == 0
    prefix = use_closing ? "└──" : "├──"
    cont_prefix = use_closing ? "    " : "│   "

    print_indent(p)
    print_colored(p, prefix, :light_black)
    print(p.io, " ")

    # Show assignment if this position is used
    if pos in p.block_used
        print(p.io, "%", pos, " = ")
    end

    print_colored(p, "while", :yellow)
    print_iter_args(p, op.before.args, op.init_values)
    println(p.io)

    # Print "before" region header
    before_p = child_printer(p, op.before, cont_prefix)
    print_region_header(before_p, "before", op.before.args)
    # Print before region body (inside_region=true, no statement-level box drawing)
    before_body_p = child_printer(before_p, op.before, "│   ")
    print_block_body(before_body_p, op.before; inside_region=true)

    # Print "do" region header
    after_p = child_printer(p, op.after, cont_prefix)
    print_region_header(after_p, "do", op.after.args)
    # Print do region body (last region, show closing on last line)
    # Pass cascade_depth so innermost item can close all cascaded levels
    after_body_p = child_printer(after_p, op.after, "│   ")
    print_block_body(after_body_p, op.after; inside_region=true, is_last_region=true, cascade_depth=cascade_depth)
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
    n_body = length(sci.entry.body)
    print(io, "StructuredCodeInfo(", length(sci.code.code), " stmts, entry=Block(", n_body, " items))")
end
