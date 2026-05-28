# IR Rewriter, modeled on MLIR's RewriterBase/IRRewriter.
#
# All IR mutations during a rewrite session go through a `Rewriter`, which
# in return keeps an incremental use-def index up to date and fires
# `notify_*` hooks so higher-level drivers can react. Correspondences:
#
#   RewriterBase                  -> Rewriter
#   IROperand intrinsic use-list  -> (users, extra_uses) maps
#   RewriterBase::Listener        -> notify_inserted!/_modified!/_erased!

using Core: SSAValue

#=============================================================================
 Listener protocol
=============================================================================#

# Default no-op listener callbacks. Drivers override by adding a method on
# their listener type (e.g. `notify_inserted!(d::RewriteDriver, block, inst)`).
"""Fires after the Rewriter inserts an instruction."""
notify_inserted!(::Any, ::Block, ::Instruction) = nothing

"""Fires when the Rewriter replaces a stmt. Drivers use this to react to
func changes or mark users dirty."""
notify_modified!(::Any, ::Block, ::SSAValue, @nospecialize(old_stmt), @nospecialize(new_stmt)) = nothing

"""Fires once an instruction has been erased; the SSA value is no longer live."""
notify_erased!(::Any, ::Block, ::SSAValue, @nospecialize(old_stmt)) = nothing

#=============================================================================
 Rewriter
=============================================================================#

"""
    Rewriter(sci::StructuredIRCode; listener=nothing)

IR mutation handle for `sci`. Holds an incremental use-def index; query it
with `users(rewriter, val)` and `use_count(rewriter, val)`. Mutate the IR
only through `insert_before!`, `replace_stmt!`, `erase!`, or
`replace_uses!` on the rewriter, which keeps the index consistent and fires
the listener.

`listener` is dispatched on its type via the `notify_*` generic functions.
Defaults to `nothing` (no-op).
"""
mutable struct Rewriter
    sci::StructuredIRCode
    # `users`: SSA value -> SSAs of stmts that reference it. `extra_uses`
    # holds the count of terminator uses, which have no SSA owner.
    users::Dict{SSAValue, Vector{SSAValue}}
    extra_uses::Dict{SSAValue, Int}
    listener::Any
end

function Rewriter(sci::StructuredIRCode; listener=nothing)
    r = Rewriter(sci, Dict{SSAValue, Vector{SSAValue}}(),
                 Dict{SSAValue, Int}(), listener)
    build_use_index!(r)
    return r
end

#=============================================================================
 Queries
=============================================================================#

"""Users (by SSA) of `val`, or an empty list if `val` has none recorded."""
users(r::Rewriter, val::SSAValue) = get(r.users, val, SSAValue[])
users(::Rewriter, ::Any) = SSAValue[]

"""Total use count for `val`, including terminator uses."""
use_count(r::Rewriter, val::SSAValue) =
    length(users(r, val)) + get(r.extra_uses, val, 0)

#=============================================================================
 Mutation API
=============================================================================#

"""
    insert_before!(r::Rewriter, block::Block, ref, stmt, type; flag=UInt32(0))

Insert `stmt` at `ref` in `block`. Updates the use-index for the new stmt's
operands and fires `notify_inserted!`. Returns the new `Instruction`.
"""
function insert_before!(r::Rewriter, block::Block, ref, @nospecialize(stmt),
                        @nospecialize(type); flag::UInt32=UInt32(0))
    inst = IRStructurizer.insert_before!(block, ref, stmt, type; flag=flag)
    register_stmt_uses!(r, SSAValue(inst), stmt)
    notify_inserted!(r.listener, block, inst)
    return inst
end

"""
    replace_stmt!(r::Rewriter, block::Block, val::SSAValue, new_stmt; flag=nothing)

Replace the stmt at `val` with `new_stmt`. Updates the use-index for both
sets of operands and fires `notify_modified!`. Pass `flag` to also update
the IR_FLAG_* bitmask; otherwise only the stmt slot changes (type and flag
are preserved).
"""
function replace_stmt!(r::Rewriter, block::Block, val::SSAValue,
                       @nospecialize(new_stmt); flag::Union{UInt32, Nothing}=nothing)
    haskey(block, val.id) || return
    old_stmt = block[val.id][:stmt]
    unregister_stmt_uses!(r, val, old_stmt)
    if flag === nothing
        block[val.id] = (stmt=new_stmt,)
    else
        block[val.id] = (stmt=new_stmt, flag=flag)
    end
    register_stmt_uses!(r, val, new_stmt)
    notify_modified!(r.listener, block, val, old_stmt, new_stmt)
end

"""
    erase!(r::Rewriter, block::Block, val::SSAValue)

Erase the stmt at `val` from `block`. Drops the op's operand uses from the
use-index, then fires `notify_erased!`. Listeners can inspect `old_stmt` to
cascade operand-defs through their own worklist for dead-op elim.
"""
function erase!(r::Rewriter, block::Block, val::SSAValue)
    haskey(block, val.id) || return
    old_stmt = block[val.id][:stmt]
    unregister_stmt_uses!(r, val, old_stmt)
    delete!(block, val.id)
    delete!(r.users, val)
    delete!(r.extra_uses, val)
    notify_erased!(r.listener, block, val, old_stmt)
end

"""
    replace_uses!(r::Rewriter, val::SSAValue, new_val)

RAUW: replace every use of `val` in the SCI with `new_val`, and move the
use-index entries to `new_val`. The walk covers every operand kind (Expr
args, CF op fields, terminators). No listener event fires; only operand
slots change, not the user stmts themselves.
"""
function replace_uses!(r::Rewriter, val::SSAValue, @nospecialize(new_val))
    IRStructurizer.replace_uses!(r.sci.entry, val, new_val)
    transfer_uses!(r, val, new_val)
end

#=============================================================================
 Use-index maintenance (internal)
=============================================================================#

function for_expr_operands(f, expr::Expr)
    start = expr.head === :invoke ? 3 : 2
    for i in start:length(expr.args)
        f(expr.args[i])
    end
end

function add_use!(r::Rewriter, @nospecialize(operand), user::SSAValue)
    operand isa SSAValue || return
    push!(get!(Vector{SSAValue}, r.users, operand), user)
end

function remove_use!(r::Rewriter, @nospecialize(operand), user::SSAValue)
    operand isa SSAValue || return
    list = get(r.users, operand, nothing)
    list === nothing && return
    i = findfirst(==(user), list)
    i === nothing && return
    deleteat!(list, i)
    isempty(list) && delete!(r.users, operand)
end

# Incremental maintenance only needs the Expr path: drivers never produce
# CF ops or terminators. `register_owned_stmt_uses!` handles those once at
# initial build time.
function register_stmt_uses!(r::Rewriter, val::SSAValue, @nospecialize(stmt))
    stmt isa Expr || return
    for_expr_operands(stmt) do op
        add_use!(r, op, val)
    end
end

function unregister_stmt_uses!(r::Rewriter, val::SSAValue, @nospecialize(stmt))
    stmt isa Expr || return
    for_expr_operands(stmt) do op
        remove_use!(r, op, val)
    end
end

# Reassign every index entry from `old` to `new_val` so subsequent queries
# find the users under `new_val`. The IR-level mutation lives in
# `replace_uses!`; this only touches the bookkeeping.
function transfer_uses!(r::Rewriter, old::SSAValue, @nospecialize(new_val))
    old_users = get(r.users, old, nothing)
    if old_users !== nothing
        if new_val isa SSAValue
            dest = get!(Vector{SSAValue}, r.users, new_val)
            append!(dest, old_users)
        end
        delete!(r.users, old)
    end
    old_extra = get(r.extra_uses, old, 0)
    if old_extra > 0
        if new_val isa SSAValue
            r.extra_uses[new_val] = get(r.extra_uses, new_val, 0) + old_extra
        end
        delete!(r.extra_uses, old)
    end
end

# One-shot SCI walk that seeds the use-index. Dispatch mirrors
# `IRStructurizer.walk_uses!(::Block)` so every operand kind is covered:
# Expr args, CF op fields, ReturnNode/PiNode/alias stmt operands, and
# block terminators.
function build_use_index!(r::Rewriter)
    for block in eachblock(r.sci)
        for i in 1:length(block.body.ssa_idxes)
            owner = SSAValue(block.body.ssa_idxes[i])
            register_owned_stmt_uses!(r, owner, block.body.stmts[i])
        end
        register_terminator_uses!(r, block.terminator)
    end
end

function register_owned_stmt_uses!(r::Rewriter, owner::SSAValue, @nospecialize(stmt))
    if stmt isa Expr
        for_expr_operands(stmt) do op
            add_use!(r, op, owner)
        end
    elseif stmt isa Core.ReturnNode
        isdefined(stmt, :val) && add_use!(r, stmt.val, owner)
    elseif stmt isa Core.PiNode
        add_use!(r, stmt.val, owner)
    elseif stmt isa SSAValue
        # Alias/forwarding stmt: the stmt slot itself IS the operand.
        add_use!(r, stmt, owner)
    elseif stmt isa ControlFlowOp
        register_cf_op_uses!(r, owner, stmt)
    end
end

function register_cf_op_uses!(r::Rewriter, owner::SSAValue, op::IfOp)
    add_use!(r, op.condition, owner)
end

function register_cf_op_uses!(r::Rewriter, owner::SSAValue, op::ForOp)
    add_use!(r, op.lower, owner)
    add_use!(r, op.upper, owner)
    add_use!(r, op.step, owner)
    for iv in op.init_values
        add_use!(r, iv, owner)
    end
end

function register_cf_op_uses!(r::Rewriter, owner::SSAValue,
                              op::Union{WhileOp, LoopOp})
    for iv in op.init_values
        add_use!(r, iv, owner)
    end
end

# Terminator operands have no SSA owner, so they're recorded as anonymous
# counts. The dead-op-elim shortcut in `use_count` needs them.
function register_terminator_uses!(r::Rewriter, term)
    if term isa Core.ReturnNode
        isdefined(term, :val) && bump_extra_use!(r, term.val)
    elseif term isa ConditionOp
        bump_extra_use!(r, term.condition)
        for arg in term.args
            bump_extra_use!(r, arg)
        end
    elseif term isa Union{ContinueOp, BreakOp, YieldOp}
        for v in term.values
            bump_extra_use!(r, v)
        end
    end
end

function bump_extra_use!(r::Rewriter, @nospecialize(val))
    val isa SSAValue || return
    r.extra_uses[val] = get(r.extra_uses, val, 0) + 1
end
