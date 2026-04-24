# Control-flow transform helpers
#
# Reusable mechanics for passes that thread additional SSA values through
# structured control flow (`ForOp`, `LoopOp`, `WhileOp`, `IfOp`). Two layers:
#
#   `extract_carry_results!` — append N trailing fields to a CF op's tuple
#   result and insert matching `getfield` extractions at the parent block.
#   The single low-level mechanic, used by passes that already manage
#   per-region recursion, terminator updates, and any per-arm logic
#   themselves (e.g. `token_order_pass!`, where mid-body
#   `BreakOp`/`ContinueOp` carry different per-terminator token states).
#
#   `thread_through_loop!` and `thread_through_branches!` — the high-level
#   wrappers that own the full pattern: push carries, recurse into body
#   region(s) via a callback, install the same yield values at every
#   reachable terminator, extract the parent-side results. Modelled on
#   MLIR's `LoopLikeOpInterface::replaceWithAdditionalYields`. Suitable for
#   state-threading passes whose per-region final state is a single value
#   (e.g. ambient counters threaded across loop iterations).
#
# IRStructurizer already provides the primary carry API — `carries(op)`,
# `push!(::LoopCarries, init_val, type)`, `body_arg`, `after_arg`,
# `term_value!`, `update_type!`, `reachable_terminators`. This file builds on
# it.

#=============================================================================
 Tuple-result extension
=============================================================================#

"""
    extract_carry_results!(parent_block::Block, inst::Instruction,
                           types::AbstractVector) -> Vector{SSAValue}

Append `types` as new trailing fields to `inst`'s tuple result type and insert
one `getfield` extraction per appended field into `parent_block`. Returns the
SSA values of the extracted fields, in the order they were appended.

`inst`'s current result type may be `Nothing`, a single type `T`, or a
`Tuple{...}`; the extension preserves any existing fields. Use this when an
external mechanism has populated the new tuple positions (e.g. `carries(op)`
for loops, or `append!(operands(yield), ...)` for `IfOp` arms) and now needs
to expose them at the parent block.
"""
function extract_carry_results!(parent_block::Block, inst::Instruction,
                                types::AbstractVector)
    # Read the live type from the block's storage rather than `inst.typ`,
    # which is a snapshot taken at instruction creation and may be stale
    # if a prior pass (or earlier call here) ran `update_type!`.
    old_type = value_type(parent_block, SSAValue(inst))
    user_types = if old_type === Nothing
        Type[]
    elseif old_type <: Tuple
        collect(Type, old_type.parameters)
    else
        Type[old_type]
    end
    merged = isempty(user_types) && isempty(types) ? Nothing :
             Tuple{user_types..., types...}
    update_type!(parent_block, inst, merged)
    n_user = length(user_types)
    result = SSAValue[]
    last_ref = SSAValue(inst)
    for (i, ty) in enumerate(types)
        gf_inst = insert_after!(parent_block, last_ref,
            Expr(:call, GlobalRef(Core, :getfield), SSAValue(inst), n_user + i), ty)
        push!(result, SSAValue(gf_inst))
        last_ref = SSAValue(gf_inst)
    end
    return result
end

#=============================================================================
 High-level region threading
=============================================================================#

"""
    thread_through_loop!(parent_block, inst, op, inits, type, body_fn) -> Vector{SSAValue}

Add `length(inits)` loop-carried values of `type` to `op`. Each carry is
initialized to the corresponding entry of `inits` (an `SSAValue`,
`BlockArgument`, or other operand). `body_fn(region, body_args)` is invoked
once per body region with the carries' block args (a `Vector{BlockArgument}`,
one per init) and must return a `Vector` of `length(inits)` values to install
at every reachable terminator of that region.

- `ForOp` / `LoopOp`: `body_fn` is called once on `op.body`.
- `WhileOp`: `body_fn` is called once on `op.before` (with `body_arg`s) and
  once on `op.after` (with `after_arg`s); the same yield values are
  installed at the matching region's terminators.

Returns the SSA values extracted at `parent_block`, one per init, in order.

Model: same yield per terminator. Consumers that need different values at
different terminators of the same region (e.g. mid-body `BreakOp` carrying
different state than the body's tail `ContinueOp`) should drop to
`carries(op)` + `extract_carry_results!` + `term_value!` directly.
"""
function thread_through_loop!(parent_block::Block, inst::Instruction,
                              op::Union{ForOp, LoopOp},
                              inits::AbstractVector,
                              @nospecialize(type::Type),
                              body_fn)
    new_carries = [push!(carries(op), init, type) for init in inits]
    install_region_yields!(op.body, new_carries, body_arg, body_fn)
    return extract_carry_results!(parent_block, inst, fill(type, length(inits)))
end

function thread_through_loop!(parent_block::Block, inst::Instruction,
                              op::WhileOp,
                              inits::AbstractVector,
                              @nospecialize(type::Type),
                              body_fn)
    new_carries = [push!(carries(op), init, type) for init in inits]
    install_region_yields!(op.before, new_carries, body_arg, body_fn)
    install_region_yields!(op.after,  new_carries, after_arg, body_fn)
    return extract_carry_results!(parent_block, inst, fill(type, length(inits)))
end

# Run `body_fn(region, args)` (where `args[i] = arg_for(carry_i)`) and install
# its returned values at every reachable terminator of `region`.
function install_region_yields!(region::Block, new_carries::Vector,
                                arg_for, body_fn)
    args = [arg_for(c) for c in new_carries]
    yields = body_fn(region, args)
    length(yields) == length(new_carries) ||
        throw(ArgumentError("body_fn must return $(length(new_carries)) values, got $(length(yields))"))
    for term in reachable_terminators(region)
        for (c, y) in zip(new_carries, yields)
            term_value!(c, term, y)
        end
    end
end

"""
    thread_through_branches!(parent_block, inst, op::IfOp, type, arm_fn) -> Vector{SSAValue}

For an `IfOp`, invoke `arm_fn(region)` on `op.then_region` and
`op.else_region`; the callback must return a `Vector` of values for that arm
to yield. Both vectors must have equal length `n`. Each arm's `YieldOp`
terminator is then extended with its yield values, and `op`'s tuple result is
extended by `n` fields of `type`. Returns the extracted SSA values, one per
yield, in order.

Both arms must end in a `YieldOp` — a `ContinueOp`/`BreakOp`/`ReturnNode` arm
(early-exit) cannot be merged via tuple extension.
"""
function thread_through_branches!(parent_block::Block, inst::Instruction,
                                  op::IfOp,
                                  @nospecialize(type::Type),
                                  arm_fn)
    then_yields = arm_fn(op.then_region)
    else_yields = arm_fn(op.else_region)
    length(then_yields) == length(else_yields) ||
        throw(ArgumentError("arm_fn returned mismatched yield counts: " *
                            "then=$(length(then_yields)), else=$(length(else_yields))"))
    append_yield_values!(op.then_region, then_yields)
    append_yield_values!(op.else_region, else_yields)
    return extract_carry_results!(parent_block, inst, fill(type, length(then_yields)))
end

function append_yield_values!(region::Block, values)
    term = region.terminator
    term isa YieldOp ||
        throw(IRError("thread_through_branches!: arm terminator is $(typeof(term)); " *
                      "expected YieldOp"))
    append!(operands(term), values)
end
