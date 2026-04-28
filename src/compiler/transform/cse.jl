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
# Purity is decided by two checks:
#   1. The Julia-inferred `IR_FLAG_EFFECT_FREE` bit on the instruction
#      (carried through the SCI from `IRCode.stmts.flag`). This is the
#      authoritative source — it incorporates every `efunc` override
#      (RNG, atomics, asserts, fpmode_*, stores, print_tko all set
#      `effect_free=ALWAYS_FALSE`, so the bit is off).
#   2. `classify_memory_op == MEM_NONE`. Loads have no observable
#      side effect in Julia's sense (the load itself doesn't write),
#      so they get `IR_FLAG_EFFECT_FREE`, but their *value* depends on
#      memory state, not just operands — so CSE must still skip them.
#      The memory-op classifier provides exactly this distinction.
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
        s = inst[:stmt]
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
    s = inst[:stmt]
    s isa Expr || return
    call = resolve_call(block, inst)
    call === nothing && return
    func, ops = call
    is_pure_for_cse(inst, func) || return
    sig = (func, inst[:type], ops...)
    canonical = get(table, sig, nothing)
    if canonical === nothing
        table[sig] = SSAValue(inst)
    else
        # The redundant SSA is defined in `block`, so by SSA dominance its
        # uses are confined to `block` and the nested CF regions inside it.
        # `replace_uses!(block, …)` walks exactly that subtree (it recurses
        # into ControlFlowOps via `walk_uses!`).
        replace_uses!(block, SSAValue(inst), canonical)
        delete!(block, inst)
    end
    return
end

#=============================================================================
 Purity classification
=============================================================================#

"""
    is_pure_for_cse(inst::Instruction, func) -> Bool

Decide whether `inst` is safe to CSE. An op is pure for CSE when it has
no observable side effect beyond producing its return value, AND that
return value is a function of its operands alone.

Two checks:
- Julia inference's `IR_FLAG_EFFECT_FREE` bit on the instruction. This
  catches every `efunc(...) = effect_free=ALWAYS_FALSE` override (RNG
  state ops, atomics, asserts, fpmode_*, stores, print_tko, ...) — no
  per-pass skip lists needed.
- `classify_memory_op == MEM_NONE`. A load's `effect_free` bit is set
  (the load itself writes nothing), but its return value depends on
  the underlying memory state — so CSE must still treat loads as
  impure. The memory-op classifier provides that load-vs-pure split.
"""
function is_pure_for_cse(inst::Instruction, @nospecialize(func))
    CC.has_flag(inst[:flag], CC.IR_FLAG_EFFECT_FREE) || return false
    classify_memory_op(func) == MEM_NONE || return false
    return true
end
