# IR Rewriter — mirrors MLIR's RewriterBase/IRRewriter.
#
# `Rewriter` wraps a `StructuredIRCode` and is the single channel through
# which all IR mutations flow during a rewrite session. It maintains a
# use-def index incrementally as a side effect of the mutation API; clients
# don't touch the index. Each mutation also fires `notify_*` listener hooks,
# letting higher-level drivers (e.g. `RewriteDriver` for greedy pattern
# rewriting) maintain their own bookkeeping (defs, worklist, …) without
# having to reach into the IR themselves.
#
# Mirrors:
#   MLIR RewriterBase  →  Rewriter
#   MLIR IROperand intrinsic use-list  →  (users, extra_uses) maps
#   MLIR Listener / RewriterBase::Listener  →  notify_inserted!/_modified!/_erased!

using Core: SSAValue

#=============================================================================
 Listener protocol
=============================================================================#

# Default no-op listener callbacks. Drivers override via dispatch on their
# listener type — e.g. `notify_inserted!(d::RewriteDriver, block, inst)`.
"""Called after an instruction is inserted by the Rewriter."""
notify_inserted!(::Any, ::Block, ::Instruction) = nothing

"""Called after an instruction's stmt is replaced by the Rewriter.
`old_stmt` is the stmt that was in place before; `new_stmt` is the one now
written. Use this to react to func changes, mark users dirty, etc."""
notify_modified!(::Any, ::Block, ::SSAValue, @nospecialize(old_stmt), @nospecialize(new_stmt)) = nothing

"""Called after an instruction is erased by the Rewriter. `old_stmt` is the
stmt that was removed; the SSA value is no longer live."""
notify_erased!(::Any, ::Block, ::SSAValue, @nospecialize(old_stmt)) = nothing

#=============================================================================
 Rewriter
=============================================================================#

"""
    Rewriter(sci::StructuredIRCode; listener=nothing)

IR mutation handle for `sci`. Maintains an incremental use-def index — query
with `users(rewriter, val)` / `use_count(rewriter, val)`. All IR mutations
should go through `insert_before!` / `replace_stmt!` / `erase!` /
`replace_uses!` on the rewriter so the index stays consistent and the
listener gets fired.

`listener` is an opaque object whose type drives the `notify_*` callbacks
via Julia multi-dispatch. Defaults to `nothing` (no-op).
"""
mutable struct Rewriter
    sci::StructuredIRCode
    # SSA value → SSAs of stmts that reference it. Captures every operand
    # site reached by the build-time SCI walk (Expr args, CF op fields,
    # ReturnNode/PiNode/alias stmts). Terminator uses live in `extra_uses`
    # because terminators have no SSA owner.
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

Insert `stmt` at `ref` in `block`. Registers operand uses and fires
`notify_inserted!` on the listener. Returns the newly created `Instruction`.
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

Replace the stmt at `val` with `new_stmt`. Updates the use-index (deregister
old operands, register new) and fires `notify_modified!`. If `flag` is given,
the IR_FLAG_* bitmask is updated too; otherwise only the stmt slot changes
(type and flag preserved).
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

Erase the stmt at `val` from `block`. Deregisters this op's operand uses and
fires `notify_erased!`. The listener may inspect `old_stmt` to enqueue
operand-defs for dead-op elim cascading.
"""
function erase!(r::Rewriter, block::Block, val::SSAValue)
    haskey(block, val.id) || return
    old_stmt = block[val.id][:stmt]
    unregister_stmt_uses!(r, val, old_stmt)
    delete!(block, val.id)
    # Any leftover index entry for `val` would now be stale (nothing produces
    # val anymore); drop it so subsequent queries report empty.
    delete!(r.users, val)
    delete!(r.extra_uses, val)
    notify_erased!(r.listener, block, val, old_stmt)
end

"""
    replace_uses!(r::Rewriter, val::SSAValue, new_val)

Replace every use of `val` in the SCI with `new_val`. The IR walk handles
every operand kind (Expr args, CF op fields, terminators) and the use-index
entries are transferred to `new_val`. No listener event is fired — only the
users' operand slots changed; their stmts as a whole did not.
"""
function replace_uses!(r::Rewriter, val::SSAValue, @nospecialize(new_val))
    IRStructurizer.replace_uses!(r.sci.entry, val, new_val)
    transfer_uses!(r, val, new_val)
end

#=============================================================================
 Use-index maintenance (internal)
=============================================================================#

# Iterate the SSA-positional operands of an Expr (skipping head/callee slot).
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

# Register/unregister operand uses for a stmt owned by `val`. The mutation
# API only ever creates or replaces `Expr` stmts (CF ops and terminators are
# not produced by drivers), so we only need the Expr path here. Initial-build
# populates the other stmt kinds via `register_owned_stmt_uses!`.
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

# Move all index entries from `old` to `new_val` after a RAUW. The IR-level
# `replace_uses!` has already rewritten the user stmts; this just shifts the
# bookkeeping so subsequent queries find users under `new_val`.
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

# Build the initial use-index by walking the entire SCI once. Mirrors the
# dispatch in `IRStructurizer.walk_uses!(::Block)` so every operand kind that
# the IR could reference is captured: Expr args, CF op fields, ReturnNode /
# PiNode / alias stmt operands, and block terminators.
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

# Terminator operands have no SSA owner; record them as anonymous counts so
# `use_count` reflects them (needed for the dead-op elim shortcut).
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
