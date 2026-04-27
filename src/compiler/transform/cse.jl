# Common Subexpression Elimination
#
# Lightweight value-numbering on `StructuredIRCode`. Mirrors LLVM's
# `EarlyCSE.cpp` and MLIR's `Transforms/CSE.cpp`: a recursive walk over
# the structured-control-flow tree maintains a per-scope hash table
# mapping `(func, operands...)` to a canonical `SSAValue`. When an
# instruction's signature matches an already-defined SSA in the
# enclosing scope, all uses are redirected to the canonical SSA and the
# redundant instruction is erased.
#
# Dominance is implicit in the SCI shape: an instruction in block `B`
# dominates `B`'s remaining instructions and every instruction in `B`'s
# nested control-flow regions (recursively). The pass copies the
# parent's table on entering a sub-block — children see parent
# definitions, but additions inside one branch don't leak to siblings
# (e.g. `then` vs `else` of an `IfOp`).
#
# Side effects are gated through `classify_memory_op` (the same
# classifier `token_order_pass!` uses) plus a positive skip list for
# ops with hidden state that aren't memory in the token-chain sense
# (RNG, `assert`, `fpmode_*`, `format_string`, …). Anything else —
# arithmetic, casts, `make_tensor_view`, `make_partition_view`,
# `getfield`, `Core.tuple`, the `assume_*` annotations — is CSE-able.
#
# Single-pass: we don't iterate to fixpoint. Once a definition is
# canonicalised, every later use of an equivalent expression in
# program order resolves through it (via `replace_uses!`), so a single
# forward walk is sufficient.

"""
    cse_pass!(sci::StructuredIRCode)

Run common-subexpression elimination over `sci`. Replaces redundant
pure-op instructions with their canonical predecessor and erases the
redundancies. Does not iterate; assumes a single forward walk is
enough (cf. LLVM's `EarlyCSE`).
"""
function cse_pass!(sci::StructuredIRCode)
    cse_block!(sci.entry, Dict{Tuple, SSAValue}())
    return nothing
end

# Recursive walk. `parent_table` is the value-numbering table visible
# from the enclosing scope; this block extends it locally so additions
# don't leak to sibling branches.
function cse_block!(block::Block, parent_table::Dict{Tuple, SSAValue})
    table = copy(parent_table)
    snapshot = collect(instructions(block))
    for inst in snapshot
        s = stmt(inst)
        if s isa ControlFlowOp
            for sub in blocks(s)
                cse_block!(sub, table)
            end
            continue
        end
        cse_one!(block, inst, table)
    end
end

# Try to dedup a single instruction. On a hit, redirect all uses to
# the cached canonical SSA and delete this instruction. On a miss, add
# the signature to the table.
#
# Signature includes the SCI return-type annotation: two ops with the
# same operands but different result types are *not* equivalent. In
# this DSL, result type can vary independently of operands (scalar/
# tile boundary, integer-cast dispatch, broadcast widening), so dropping
# it from the key would silently merge `broadcast(1, (16,))::Tile{Int32}`
# with `broadcast(1, (16,))::Tile{Int64}`. Mirrors LLVM `EarlyCSE`'s
# `isIdenticalToWhenDefined` which compares both operands and result
# type implicitly through the value-level structural identity check.
function cse_one!(block::Block, inst::Instruction, table::Dict{Tuple, SSAValue})
    s = stmt(inst)
    s isa Expr || return
    call = resolve_call(block, inst)
    call === nothing && return
    func, ops = call
    is_pure_for_cse(func, inst.typ) || return
    sig = (func, inst.typ, Tuple(ops)...)
    canonical = get(table, sig, nothing)
    if canonical === nothing
        table[sig] = SSAValue(inst.ssa_idx)
    else
        replace_uses!(block_root(block), SSAValue(inst.ssa_idx), canonical)
        delete!(block, inst)
    end
    return
end

# Walk up to the SCI entry block to do a global use-replacement. `block`
# might be a nested CF body whose `replace_uses!` only sees its own
# region; uses in *enclosing* scopes can't reference an SSA defined
# inside the nested block, but uses in *sibling* nested blocks (and in
# the parent's instructions following the CF op) absolutely can.
function block_root(block::Block)
    p = block
    while p.parent isa Block
        p = p.parent
    end
    return p
end

#=============================================================================
 Purity classification
=============================================================================#

"""
    is_pure_for_cse(func, inst_type) -> Bool

Decide whether a call to `func` returning a value of type `inst_type`
is safe to CSE. An op is pure for CSE when it has no observable side
effect beyond producing its return value, and that return value is a
function of its operands alone.

Excludes:
- Memory and token-ordered ops (caught by `classify_memory_op`).
- Ops with no useful return SSA (`inst_type === Nothing`) — `assert`,
  `fpmode_begin`/`end`, `format_string`. Replacing identical calls
  doesn't help and can perturb effect ordering.
- RNG state intrinsics — `rng_advance`/`rng_set_seed` mutate per-stream
  counters; `rng_counter`/`rng_seed` *read* that mutable state, so the
  `lower_rng_state!` pass owns their threading. CSE-ing them before
  that pass runs would cross writes.
"""
function is_pure_for_cse(@nospecialize(func), @nospecialize(inst_type))
    classify_memory_op(func) == MEM_NONE || return false
    inst_type === Nothing && return false
    is_rng_func(func) && return false
    return true
end

# RNG state ops live until `lower_rng_state!` rewrites them. Treat them
# as impure for CSE so we don't fuse reads/writes of the per-stream
# counter before the dedicated pass owns the rewriting.
const RNG_FUNCS = (
    Intrinsics.rng_counter, Intrinsics.rng_advance,
    Intrinsics.rng_seed, Intrinsics.rng_set_seed,
    Intrinsics.rng_stream, Intrinsics.rng_default,
)
is_rng_func(@nospecialize(func)) = any(f -> f === func, RNG_FUNCS)
