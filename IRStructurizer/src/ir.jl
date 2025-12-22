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
 Local SSA Values (block-local references)
=============================================================================#

"""
    LocalSSAValue

Reference to a value defined by an operation in the same block.
The id is the 1-indexed position in the containing block's ops array.

This type will replace global SSAValue references inside structured blocks
once the local SSA refactoring is complete. For now, it coexists with
global SSAValue references during the migration.
"""
struct LocalSSAValue
    id::Int
end

Base.show(io::IO, v::LocalSSAValue) = print(io, "\$", v.id)

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
 Unified Control Flow (PartialControlFlowOp / PartialBlock)
=============================================================================#

# PartialControlFlowOp - defined first since PartialBlock references it
"""
    PartialControlFlowOp

Unified control flow operation with head::Symbol for type discrimination.
Used during IR construction (Phases 1-4). Contains result_vars for SSA tracking.

Heads:
- :if - if-then-else, regions[:then] and regions[:else]
- :loop - general loop, regions[:body]
- :for - counted for loop, regions[:body], operands has lower/upper/step/iv_arg
- :while - MLIR-style while, regions[:before] and regions[:after]
"""
mutable struct PartialControlFlowOp
    head::Symbol
    regions::Dict{Symbol, Any}  # Values are PartialBlock (forward reference)
    init_values::Vector{IRValue}
    operands::NamedTuple
    result_vars::Vector{SSAValue}
end

# Convenience constructor
function PartialControlFlowOp(head::Symbol, regions::Dict{Symbol, <:Any};
                              init_values::Vector{IRValue}=IRValue[],
                              operands::NamedTuple=NamedTuple(),
                              result_vars::Vector{SSAValue}=SSAValue[])
    PartialControlFlowOp(head, regions, init_values, operands, result_vars)
end

function Base.show(io::IO, op::PartialControlFlowOp)
    print(io, "PartialControlFlowOp(:", op.head)
    print(io, ", regions=[", join(keys(op.regions), ", "), "]")
    if !isempty(op.init_values)
        print(io, ", init=", length(op.init_values))
    end
    if !isempty(op.result_vars)
        print(io, ", results=", length(op.result_vars))
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
    id::Int
    args::Vector{BlockArg}
    body::Vector{PartialBlockItem}
    terminator::Terminator
end

PartialBlock(id::Int) = PartialBlock(id, BlockArg[], PartialBlockItem[], nothing)

function Base.show(io::IO, block::PartialBlock)
    print(io, "PartialBlock(id=", block.id)
    if !isempty(block.args)
        print(io, ", args=", length(block.args))
    end
    n_stmts = count(x -> x isa Statement, block.body)
    n_ops = count(x -> x isa PartialControlFlowOp, block.body)
    print(io, ", stmts=", n_stmts)
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

"""
    ControlFlowOp

Final immutable control flow operation. Output of finalize_ir!.
Uses head::Symbol for type discrimination (:if, :for, :while, :loop).

Unlike PartialControlFlowOp, this has no result_vars (not needed after finalization).
"""
struct ControlFlowOp
    head::Symbol
    regions::Dict{Symbol, Any}  # Values are Block (forward reference)
    init_values::Vector{IRValue}
    operands::NamedTuple
end

function Base.show(io::IO, op::ControlFlowOp)
    print(io, "ControlFlowOp(:", op.head)
    print(io, ", regions=[", join(keys(op.regions), ", "), "]")
    if !isempty(op.init_values)
        print(io, ", init=", length(op.init_values))
    end
    print(io, ")")
end

# Accessors for ControlFlowOp (same as PartialControlFlowOp)
condition(op::ControlFlowOp) = op.head == :if ? op.operands.condition : error("no condition for :$(op.head)")
lower(op::ControlFlowOp) = op.head == :for ? op.operands.lower : error("no lower for :$(op.head)")
upper(op::ControlFlowOp) = op.head == :for ? op.operands.upper : error("no upper for :$(op.head)")
step(op::ControlFlowOp) = op.head == :for ? op.operands.step : error("no step for :$(op.head)")
iv_arg(op::ControlFlowOp) = op.head == :for ? op.operands.iv_arg : error("no iv_arg for :$(op.head)")

then_block(op::ControlFlowOp) = get(op.regions, :then, nothing)
else_block(op::ControlFlowOp) = get(op.regions, :else, nothing)
body_block(op::ControlFlowOp) = get(op.regions, :body, nothing)
before_block(op::ControlFlowOp) = get(op.regions, :before, nothing)
after_block(op::ControlFlowOp) = get(op.regions, :after, nothing)

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

    entry = PartialBlock(1)

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
            for (_, region) in expr.regions
                each_stmt(f, region)
            end
        else
            # For final Block, we call with a pseudo-Statement-like object
            f((idx=i, expr=expr, type=typ))
        end
    end
end

"""
    each_block(f, block::Block)

Recursively iterate over all blocks in a final Block.
"""
function each_block(f, block::Block)
    f(block)
    for expr in block.body
        if expr isa ControlFlowOp
            for (_, region) in expr.regions
                each_block(f, region)
            end
        end
    end
end

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
collect_ssa_refs!(::Vector{SSAValue}, ::Set{Int}, ::LocalSSAValue, ::Set{Int}) = nothing
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
 Local SSA Conversion (SSAValue → LocalSSAValue within blocks)
=============================================================================#

"""
    convert_ssa_in_value(value, idx_to_pos::Dict{Int, Int})

Convert SSAValue references in a value to LocalSSAValue where applicable.
Returns the converted value (or original if no conversion needed).
"""
function convert_ssa_in_value(value::SSAValue, idx_to_pos::Dict{Int, Int})
    if haskey(idx_to_pos, value.id)
        return LocalSSAValue(idx_to_pos[value.id])
    end
    return value
end

function convert_ssa_in_value(value::Expr, idx_to_pos::Dict{Int, Int})
    new_args = Any[convert_ssa_in_value(a, idx_to_pos) for a in value.args]
    # Only create new Expr if something changed
    if new_args != value.args
        return Expr(value.head, new_args...)
    end
    return value
end

function convert_ssa_in_value(value::PiNode, idx_to_pos::Dict{Int, Int})
    new_val = convert_ssa_in_value(value.val, idx_to_pos)
    if new_val !== value.val
        return PiNode(new_val, value.typ)
    end
    return value
end

function convert_ssa_in_value(value::PhiNode, idx_to_pos::Dict{Int, Int})
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
convert_ssa_in_value(value::LocalSSAValue, ::Dict{Int, Int}) = value
convert_ssa_in_value(value::BlockArg, ::Dict{Int, Int}) = value
convert_ssa_in_value(value::Argument, ::Dict{Int, Int}) = value
convert_ssa_in_value(value::SlotNumber, ::Dict{Int, Int}) = value
convert_ssa_in_value(value::GlobalRef, ::Dict{Int, Int}) = value
convert_ssa_in_value(value::QuoteNode, ::Dict{Int, Int}) = value
convert_ssa_in_value(value::Nothing, ::Dict{Int, Int}) = value
convert_ssa_in_value(value::Number, ::Dict{Int, Int}) = value
convert_ssa_in_value(value::Symbol, ::Dict{Int, Int}) = value
convert_ssa_in_value(value::Type, ::Dict{Int, Int}) = value
convert_ssa_in_value(value, ::Dict{Int, Int}) = value  # Fallback

"""
    convert_ssa_in_terminator(term, idx_to_pos)

Convert SSAValue references in a terminator to LocalSSAValue where applicable.
"""
function convert_ssa_in_terminator(term::ContinueOp, idx_to_pos::Dict{Int, Int})
    new_values = [convert_ssa_in_value(v, idx_to_pos) for v in term.values]
    return ContinueOp(new_values)
end

function convert_ssa_in_terminator(term::BreakOp, idx_to_pos::Dict{Int, Int})
    new_values = [convert_ssa_in_value(v, idx_to_pos) for v in term.values]
    return BreakOp(new_values)
end

function convert_ssa_in_terminator(term::YieldOp, idx_to_pos::Dict{Int, Int})
    new_values = [convert_ssa_in_value(v, idx_to_pos) for v in term.values]
    return YieldOp(new_values)
end

function convert_ssa_in_terminator(term::ConditionOp, idx_to_pos::Dict{Int, Int})
    new_cond = convert_ssa_in_value(term.condition, idx_to_pos)
    new_args = [convert_ssa_in_value(v, idx_to_pos) for v in term.args]
    return ConditionOp(new_cond, new_args)
end

function convert_ssa_in_terminator(term::ReturnNode, idx_to_pos::Dict{Int, Int})
    if isdefined(term, :val)
        new_val = convert_ssa_in_value(term.val, idx_to_pos)
        if new_val !== term.val
            return ReturnNode(new_val)
        end
    end
    return term
end

function convert_ssa_in_terminator(term::Nothing, ::Dict{Int, Int})
    return nothing
end

"""
    convert_to_local_ssa!(block::PartialBlock)

Convert SSAValue references within a block to LocalSSAValue references.
"""
function convert_to_local_ssa!(block::PartialBlock)
    # Build mapping: Statement.idx → position in block.body
    # Also include PartialControlFlowOp result_vars (loop-produced values map to loop position)
    idx_to_pos = Dict{Int, Int}()
    for (pos, item) in enumerate(block.body)
        if item isa Statement
            idx_to_pos[item.idx] = pos
        elseif item isa PartialControlFlowOp
            # Control flow ops produce values through result_vars (e.g., loop-carried values)
            # All result_vars map to this position since the op produces them
            for rv in item.result_vars
                idx_to_pos[rv.id] = pos
            end
        end
    end

    # Convert SSAValues in statements
    for (i, item) in enumerate(block.body)
        if item isa Statement
            new_expr = convert_ssa_in_value(item.expr, idx_to_pos)
            if new_expr !== item.expr
                block.body[i] = Statement(item.idx, new_expr, item.type)
            end
        elseif item isa PartialControlFlowOp
            convert_to_local_ssa_in_op!(item, idx_to_pos)
        end
    end

    # Convert SSAValues in terminator
    if block.terminator !== nothing
        block.terminator = convert_ssa_in_terminator(block.terminator, idx_to_pos)
    end
end

function convert_to_local_ssa_in_op!(op::PartialControlFlowOp, parent_idx_to_pos::Dict{Int, Int})
    # Convert init_values using parent's mapping
    for (i, v) in enumerate(op.init_values)
        op.init_values[i] = convert_ssa_in_value(v, parent_idx_to_pos)
    end

    # For :for ops, convert operands (lower, upper, step)
    if op.head == :for
        new_lower = convert_ssa_in_value(op.operands.lower, parent_idx_to_pos)
        new_upper = convert_ssa_in_value(op.operands.upper, parent_idx_to_pos)
        new_step = convert_ssa_in_value(op.operands.step, parent_idx_to_pos)
        op.operands = (lower=new_lower, upper=new_upper, step=new_step, iv_arg=op.operands.iv_arg)
    elseif op.head == :if
        new_cond = convert_ssa_in_value(op.operands.condition, parent_idx_to_pos)
        op.operands = (condition=new_cond,)
    end

    # Recursively convert nested blocks
    for (_, region) in op.regions
        convert_to_local_ssa!(region)
    end
end

#=============================================================================
 IR Finalization (PartialBlock/PartialControlFlowOp → Block/ControlFlowOp)
=============================================================================#

"""
    finalize_ir(block::PartialBlock) -> Block

Convert a PartialBlock to a final Block by:
1. Flattening Statement wrappers to (expr, type) pairs in body/types vectors
2. Converting PartialControlFlowOp to ControlFlowOp (dropping result_vars)
3. Recursively processing all nested regions
"""
function finalize_ir(block::PartialBlock)::Block
    body = Any[]
    types = Any[]

    for item in block.body
        if item isa Statement
            push!(body, item.expr)
            push!(types, item.type)
        elseif item isa PartialControlFlowOp
            push!(body, finalize_control_flow_op(item))
            # For control flow ops, store the result types (Tuple if multiple)
            if isempty(item.result_vars)
                push!(types, Nothing)
            elseif length(item.result_vars) == 1
                push!(types, item.result_vars[1])  # Store SSAValue for reference
            else
                push!(types, item.result_vars)  # Store Vector{SSAValue}
            end
        end
    end

    return Block(copy(block.args), body, types, block.terminator)
end

"""
    finalize_control_flow_op(op::PartialControlFlowOp) -> ControlFlowOp

Convert a PartialControlFlowOp to a final ControlFlowOp by:
1. Converting all nested regions from PartialBlock to Block
2. Dropping result_vars (not needed after finalization)
"""
function finalize_control_flow_op(op::PartialControlFlowOp)::ControlFlowOp
    # Convert all regions
    final_regions = Dict{Symbol, Any}()
    for (name, region) in op.regions
        final_regions[name] = finalize_ir(region::PartialBlock)
    end

    # Create final ControlFlowOp without result_vars
    return ControlFlowOp(
        op.head,
        final_regions,
        copy(op.init_values),
        op.operands
    )
end

#=============================================================================
 Pretty Printing (Julia CodeInfo-style with colors)
=============================================================================#

function _scan_expr_uses!(used::BitSet, v::SSAValue)
    push!(used, v.id)
end

function _scan_expr_uses!(used::BitSet, v::LocalSSAValue)
    # LocalSSAValue references block positions, not original SSA indices
    # They can't be tracked in the same BitSet - just skip
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

function _scan_control_flow_uses!(used::BitSet, op::ControlFlowOp)
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

function IRPrinter(io::IO, code::CodeInfo, entry::PartialBlock)
    used = compute_used_ssas(entry)
    color = get(io, :color, false)::Bool
    IRPrinter(io, code, 0, "", false, used, color)
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

function print_value(p::IRPrinter, v::LocalSSAValue)
    print(p.io, "\$", v.id)
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
function format_value(p::IRPrinter, v::LocalSSAValue)
    string("\$", v.id)
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

    if !isempty(op.result_vars)
        print_results(p, op.result_vars)
        print(p.io, " = ")
    end

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

    if !isempty(op.result_vars)
        print_results(p, op.result_vars)
        print(p.io, " = ")
    end

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

    if !isempty(op.result_vars)
        print_results(p, op.result_vars)
        print(p.io, " = ")
    end

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

    if !isempty(op.result_vars)
        print_results(p, op.result_vars)
        print(p.io, " = ")
    end

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
function print_expr_with_type(p::IRPrinter, idx::Int, expr, typ; prefix::String="│  ")
    print_indent(p)
    print_colored(p, prefix, :light_black)

    # For final Block, we use position index instead of SSA index
    # Type color depends on whether the value is used
    is_used = idx in p.used
    if is_used
        print(p.io, "\$", idx, " = ")
    else
        print(p.io, "     ")  # Padding to align
    end
    print_expr(p, expr)

    # Print type annotation
    color = is_used ? :cyan : :light_black
    print_colored(p, string("::", format_type(typ)), color)
    println(p.io)
end

# Print a Block's contents (final type)
function print_block_body(p::IRPrinter, block::Block)
    items = []
    for (i, (expr, typ)) in enumerate(zip(block.body, block.types))
        if expr isa ControlFlowOp
            push!(items, (:nested, expr, typ))
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
            prefix = is_last ? "└──" : "│  "
            print_expr_with_type(p, item[2], item[3], item[4]; prefix=prefix)
        elseif item[1] == :nested
            print_control_flow(p, item[2]; is_last=is_last)
        else  # :term
            print_terminator(p, item[2]; prefix="└──")
        end
    end
end

# Print ControlFlowOp (final type, dispatches on head)
function print_control_flow(p::IRPrinter, op::ControlFlowOp; is_last::Bool=false)
    if op.head == :if
        print_if_op_final(p, op; is_last=is_last)
    elseif op.head == :for
        print_for_op_final(p, op; is_last=is_last)
    elseif op.head == :while
        print_while_op_final(p, op; is_last=is_last)
    elseif op.head == :loop
        print_loop_op_final(p, op; is_last=is_last)
    else
        error("Unknown op head: $(op.head)")
    end
end

function print_if_op_final(p::IRPrinter, op::ControlFlowOp; is_last::Bool=false)
    prefix = is_last ? "└──" : "├──"
    cont_prefix = is_last ? "    " : "│   "

    print_indent(p)
    print_colored(p, prefix, :light_black)
    print(p.io, " ")

    print(p.io, "if ")
    print_value(p, op.operands.condition)
    println(p.io)

    then_blk = op.regions[:then]::Block
    else_blk = op.regions[:else]::Block

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

function print_for_op_final(p::IRPrinter, op::ControlFlowOp; is_last::Bool=false)
    prefix = is_last ? "└──" : "├──"
    cont_prefix = is_last ? "    " : "│   "

    print_indent(p)
    print_colored(p, prefix, :light_black)
    print(p.io, " ")

    body = op.regions[:body]::Block

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

function print_loop_op_final(p::IRPrinter, op::ControlFlowOp; is_last::Bool=false)
    prefix = is_last ? "└──" : "├──"
    cont_prefix = is_last ? "    " : "│   "

    print_indent(p)
    print_colored(p, prefix, :light_black)
    print(p.io, " ")

    body = op.regions[:body]::Block

    print_colored(p, "loop", :yellow)
    print_iter_args(p, body.args, op.init_values)
    println(p.io)

    body_p = IRPrinter(p.io, p.code, p.indent + 1, p.line_prefix * cont_prefix, false, p.used, p.color)
    print_block_body(body_p, body)

    print_indent(p)
    print_colored(p, cont_prefix, :light_black)
    println(p.io, "end")
end

function print_while_op_final(p::IRPrinter, op::ControlFlowOp; is_last::Bool=false)
    prefix = is_last ? "└──" : "├──"
    cont_prefix = is_last ? "    " : "│   "

    print_indent(p)
    print_colored(p, prefix, :light_black)
    print(p.io, " ")

    before = op.regions[:before]::Block
    after = op.regions[:after]::Block

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
