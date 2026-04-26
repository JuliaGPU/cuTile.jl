# RNG passes
#
# Two passes lower the RNG placeholder intrinsics to concrete SSA before
# codegen. Both short-circuit to a no-op when no RNG intrinsic appears in
# the IR, so kernels that don't use random numbers pay zero cost.
#
#   1. `assign_rng_streams!`  — replaces every `rng_default()` call with
#      the literal `0` and every `rng_stream()` call with a unique
#      positive `Int` (1, 2, 3, ...). After this pass every remaining
#      RNG state intrinsic has an `Int` literal as its stream operand.
#      The `DeviceRNG` wrapper's `stream::Int` field collapses via SROA.
#
#   2. `lower_rng_state!`     — rewrites the four state placeholders:
#
#        rng_counter(s)      — uses become s's current counter SSA
#        rng_advance(s, n)   — emits addi(current_counter(s), n)
#        rng_seed(s)         — uses become s's current seed SSA
#        rng_set_seed(s, v)  — updates s's seed to `v` (no arithmetic)
#
#      into concrete SSA ops on per-stream `(counter, seed)` state slots.
#      Default both slots to `UInt32(0)`.
#
# Both slots per stream are threaded through structured CFG independently via
# the high-level `thread_through_loop!` / `thread_through_branches!` helpers
# in `transform/control_flow.jl`: only `(stream, slot)` pairs actually
# modified in a body become loop carries / IfOp yields.
#
# NO-OP: if no RNG intrinsic appears anywhere in the IR, the pass returns
# immediately — kernels that don't use random numbers pay zero cost.

#=============================================================================
 Call recognition
=============================================================================#

function matches_rng(stmt, target)
    stmt isa Expr || return false
    stmt.head === :call || stmt.head === :invoke || return false
    f_pos = stmt.head === :invoke ? 2 : 1
    f_pos <= length(stmt.args) || return false
    f = stmt.args[f_pos]
    f === target && return true
    f isa GlobalRef && isdefined(f.mod, f.name) && getfield(f.mod, f.name) === target && return true
    return false
end

is_rng_counter_call(s)  = matches_rng(s, Intrinsics.rng_counter)
is_rng_advance_call(s)  = matches_rng(s, Intrinsics.rng_advance)
is_rng_seed_call(s)     = matches_rng(s, Intrinsics.rng_seed)
is_rng_set_seed_call(s) = matches_rng(s, Intrinsics.rng_set_seed)
is_rng_stream_call(s)    = matches_rng(s, Intrinsics.rng_stream)
is_rng_default_call(s)  = matches_rng(s, Intrinsics.rng_default)

is_rng_read(s)   = is_rng_counter_call(s) || is_rng_seed_call(s)
is_rng_write(s)  = is_rng_advance_call(s) || is_rng_set_seed_call(s)
# Stream-creating intrinsics identify streams but don't read/write state.
# `assign_rng_streams!` rewrites them to Int literals before state threading.
is_rng_create_call(s) = is_rng_stream_call(s) || is_rng_default_call(s)
is_rng_call(s)   = is_rng_read(s) || is_rng_write(s) || is_rng_create_call(s)

# First non-callee argument of a :call/:invoke Expr.
function nth_operand(stmt::Expr, n::Int)
    start = stmt.head === :invoke ? 3 : 2
    idx = start + n - 1
    idx <= length(stmt.args) || throw(IRError("RNG intrinsic missing operand #$n"))
    return stmt.args[idx]
end

# Stream is always the 1st operand; the write payload (`n` or `s`) is the 2nd.
stream_operand(stmt::Expr) = nth_operand(stmt, 1)
payload_operand(stmt::Expr) = nth_operand(stmt, 2)

# Extract a compile-time integer from an operand. Accepts QuoteNode, literal,
# or SSA reference whose defining instruction is a visible integer literal.
function extract_const_int(operand, block::Block)
    if operand isa Integer
        return operand
    elseif operand isa QuoteNode && operand.value isa Integer
        return operand.value
    elseif operand isa SSAValue
        entry = get(block.body, operand.id, nothing)
        entry === nothing && return nothing
        s = entry.stmt
        if s isa Integer
            return s
        end
    end
    return nothing
end

#=============================================================================
 Stream keying
=============================================================================#

# Resolve a stream operand to a stable per-kernel key. After
# `assign_rng_streams!` runs, every stream operand is an `Int` literal
# (0 for the default stream, ≥1 for allocated streams). Julia's SROA
# collapses the `DeviceRNG(id).stream` wrapping so the operand is the
# literal directly; if SROA left a detour, we trace a simple chain.
function stream_key(operand, block::Block)
    if operand isa Integer
        return Int(operand)
    elseif operand isa QuoteNode && operand.value isa Integer
        return Int(operand.value)
    elseif operand isa SSAValue
        entry = get(block.body, operand.id, nothing)
        if entry !== nothing
            s = entry.stmt
            s isa Integer && return Int(s)
        end
    end
    throw(IRError("RNG stream operand must resolve to an `Int` literal (got $operand)"))
end

#=============================================================================
 Detection: does a block (or any nested block) modify counter/seed for a
 specific stream?
=============================================================================#

function block_modifies_slot(block::Block, pred, hkey::Int)
    for inst in instructions(block)
        s = stmt(inst)
        if pred(s) && stream_key(stream_operand(s::Expr), block) == hkey
            return true
        elseif s isa ControlFlowOp
            for b in blocks(s)
                block_modifies_slot(b, pred, hkey) && return true
            end
        end
    end
    return false
end

function sci_uses_rng(sci::StructuredIRCode)
    function walk(block::Block)
        for inst in instructions(block)
            s = stmt(inst)
            is_rng_call(s) && return true
            if s isa ControlFlowOp
                for b in blocks(s)
                    walk(b) && return true
                end
            end
        end
        return false
    end
    return walk(sci.entry)
end

# Collect every distinct stream ID that appears in the IR. Call once at
# pass entry so CFG handling can iterate over all `(stream, slot)` pairs.
# Only read/write intrinsics carry stream operands — stream-creating intrinsics
# are the *producers* and are skipped here.
function collect_streams(sci::StructuredIRCode)
    keys = Set{Int}()
    function walk(block::Block)
        for inst in instructions(block)
            s = stmt(inst)
            if is_rng_read(s) || is_rng_write(s)
                push!(keys, stream_key(stream_operand(s), block))
            elseif s isa ControlFlowOp
                for b in blocks(s)
                    walk(b)
                end
            end
        end
    end
    walk(sci.entry)
    return keys
end

#=============================================================================
 Per-stream state
=============================================================================#

# Slot values are arbitrary IR operands — `nothing` means "no value yet" (both
# slots default to UInt32(0) on first read), otherwise whatever flows from a
# write: literal UInt32, SSAValue, BlockArgument (CFG carry), kernel Argument,
# etc. Typed `Any` like other IR-handling code.
mutable struct RngState
    counter::Any
    seed::Any
end

RngState() = RngState(nothing, nothing)

const RngStateMap = Dict{Int, RngState}

# Ensure the map has an entry for `hkey`, returning it.
state_for(map::RngStateMap, hkey::Int) = get!(RngState, map, hkey)

current_counter(map::RngStateMap, hkey::Int) = begin
    v = state_for(map, hkey).counter
    v === nothing ? UInt32(0) : v
end
current_seed(map::RngStateMap, hkey::Int) = begin
    v = state_for(map, hkey).seed
    v === nothing ? UInt32(0) : v
end

# Copy the current map — used when forking scopes for CFG body processing
# so inner mutations don't leak until we explicitly merge them back via
# loop carries / IfOp yields.
copystate_for(map::RngStateMap) = RngStateMap(k => RngState(v.counter, v.seed) for (k, v) in map)

#=============================================================================
 Statement-level rewrites
=============================================================================#

function process_rng_counter!(block::Block, inst::Instruction, map::RngStateMap)
    hkey = stream_key(stream_operand(stmt(inst)), block)
    replace_uses!(block, SSAValue(inst), current_counter(map, hkey))
    delete!(block, inst)
end

function process_rng_seed!(block::Block, inst::Instruction, map::RngStateMap)
    hkey = stream_key(stream_operand(stmt(inst)), block)
    replace_uses!(block, SSAValue(inst), current_seed(map, hkey))
    delete!(block, inst)
end

function process_rng_advance!(block::Block, inst::Instruction, map::RngStateMap)
    s = stmt(inst)::Expr
    hkey = stream_key(stream_operand(s), block)
    n_operand = payload_operand(s)
    n = extract_const_int(n_operand, block)
    current = current_counter(map, hkey)
    rhs = n isa Integer ? UInt32(n) : n_operand
    new_stmt = Expr(:call, Intrinsics.addi, current, rhs)
    new_inst = insert_before!(block, inst, new_stmt, Tile{UInt32, Tuple{}})
    state_for(map, hkey).counter = SSAValue(new_inst.ssa_idx)
    delete!(block, inst)
end

function process_rng_set_seed!(block::Block, inst::Instruction, map::RngStateMap)
    s = stmt(inst)::Expr
    hkey = stream_key(stream_operand(s), block)
    operand = payload_operand(s)
    n = extract_const_int(operand, block)
    state_for(map, hkey).seed = n isa Integer ? UInt32(n) : operand
    delete!(block, inst)
end

#=============================================================================
 Block traversal
=============================================================================#

function transform_block!(block::Block, map::RngStateMap)
    snapshot = collect(instructions(block))
    for inst in snapshot
        s = stmt(inst)
        if s isa ControlFlowOp
            transform_cf!(block, inst, s, map)
        elseif is_rng_counter_call(s)
            process_rng_counter!(block, inst, map)
        elseif is_rng_advance_call(s)
            process_rng_advance!(block, inst, map)
        elseif is_rng_seed_call(s)
            process_rng_seed!(block, inst, map)
        elseif is_rng_set_seed_call(s)
            process_rng_set_seed!(block, inst, map)
        end
    end
end

#=============================================================================
 Control flow — per-stream × per-slot carry handling
=============================================================================#

struct SlotHandle
    hkey::Int
    which::Symbol                       # :counter or :seed
    pred::Function                      # which write-intrinsic mutates this slot
end

slot_get(map::RngStateMap, sh::SlotHandle) = getfield(state_for(map, sh.hkey), sh.which)
slot_set!(map::RngStateMap, sh::SlotHandle, val) =
    (setfield!(state_for(map, sh.hkey), sh.which, val); nothing)
slot_needs(sh::SlotHandle, block::Block) = block_modifies_slot(block, sh.pred, sh.hkey)

# Enumerate all slots across every stream seen at the top level.
function all_slots(streams)
    slots = SlotHandle[]
    for h in streams
        push!(slots, SlotHandle(h, :counter, is_rng_advance_call))
        push!(slots, SlotHandle(h, :seed,    is_rng_set_seed_call))
    end
    return slots
end

# Ensure a slot has a concrete SSA value available at `inst`'s insertion
# point, lifting `nothing`/literal to SSA via an identity `addi` if needed.
function materialize_slot!(parent_block::Block, inst::Instruction,
                            map::RngStateMap, sh::SlotHandle)
    cur = slot_get(map, sh)
    cur isa SSAValue && return cur
    cur isa IRStructurizer.BlockArgument && return cur
    cur === nothing && (cur = UInt32(0))
    lit = cur::UInt32
    lift_stmt = Expr(:call, Intrinsics.addi, lit, UInt32(0))
    new_inst = insert_before!(parent_block, inst, lift_stmt, Tile{UInt32, Tuple{}})
    v = SSAValue(new_inst.ssa_idx)
    slot_set!(map, sh, v)
    return v
end

const SLOT_TYPE = Tile{UInt32, Tuple{}}

function transform_cf!(parent_block::Block, inst::Instruction,
                       op::Union{ForOp, LoopOp, WhileOp}, map::RngStateMap)
    regions = op isa WhileOp ? (op.before, op.after) : (op.body,)
    active = filter(sh -> any(r -> slot_needs(sh, r), regions),
                    all_slots(keys(map)))
    if isempty(active)
        for r in regions
            transform_block!(r, copystate_for(map))
        end
        return
    end
    inits = [materialize_slot!(parent_block, inst, map, sh) for sh in active]
    extracted = thread_through_loop!(parent_block, inst, op, inits, SLOT_TYPE,
        (region, region_args) -> begin
            region_map = copystate_for(map)
            for (sh, ra) in zip(active, region_args)
                slot_set!(region_map, sh, ra)
            end
            transform_block!(region, region_map)
            return [slot_get(region_map, sh) for sh in active]
        end)
    for (sh, val) in zip(active, extracted)
        slot_set!(map, sh, val)
    end
end

function transform_cf!(parent_block::Block, inst::Instruction,
                       op::IfOp, map::RngStateMap)
    active = filter(sh -> slot_needs(sh, op.then_region) || slot_needs(sh, op.else_region),
                    all_slots(keys(map)))
    if isempty(active)
        transform_block!(op.then_region, copystate_for(map))
        transform_block!(op.else_region, copystate_for(map))
        return
    end
    # Each arm starts from the parent's current state and yields its final
    # per-slot value. `nothing` slots (never written in this arm) lift to
    # `UInt32(0)` to keep yields type-stable.
    extracted = thread_through_branches!(parent_block, inst, op, SLOT_TYPE,
        region -> begin
            region_map = copystate_for(map)
            transform_block!(region, region_map)
            return [(v = slot_get(region_map, sh); v === nothing ? UInt32(0) : v)
                    for sh in active]
        end)
    for (sh, val) in zip(active, extracted)
        slot_set!(map, sh, val)
    end
end

function transform_cf!(parent_block::Block, inst::Instruction,
                       op::ControlFlowOp, map::RngStateMap)
    for b in blocks(op)
        transform_block!(b, copystate_for(map))
    end
end

#=============================================================================
 Main entry point
=============================================================================#

"""
    assign_rng_streams!(sci::StructuredIRCode)

Walk the IR and rewrite every `Intrinsics.rng_default()` call to the
literal `0` and every `Intrinsics.rng_stream()` call to a unique positive
`Int` (1, 2, 3, ...). After this pass, every RNG state intrinsic
(`rng_counter`/`rng_advance`/`rng_seed`/`rng_set_seed`) has an `Int`
literal as its stream operand — which `lower_rng_state!` reads directly.

Runs before `lower_rng_state!`. No-op when no RNG intrinsics appear.
"""
function assign_rng_streams!(sci::StructuredIRCode)
    sci_uses_rng(sci) || return nothing
    next_id = Ref(1)
    function walk(block::Block)
        snapshot = collect(instructions(block))
        for inst in snapshot
            s = stmt(inst)
            if s isa ControlFlowOp
                for b in blocks(s)
                    walk(b)
                end
            elseif is_rng_stream_call(s)
                id = next_id[]
                next_id[] += 1
                replace_uses!(block, SSAValue(inst), id)
                delete!(block, inst)
            elseif is_rng_default_call(s)
                replace_uses!(block, SSAValue(inst), 0)
                delete!(block, inst)
            end
        end
    end
    walk(sci.entry)
    return nothing
end

"""
    lower_rng_state!(sci::StructuredIRCode)

Rewrite every RNG placeholder intrinsic (`rng_counter`, `rng_advance`,
`rng_seed`, `rng_set_seed`) into concrete SSA operations on per-stream
`(counter, seed)` state slots. Handles are integer IDs assigned by
`assign_rng_streams!` — every stream operand is an `Int` literal at
this point, so the pass reads it directly.

Zero-cost no-op when the IR contains no RNG intrinsic calls.
"""
function lower_rng_state!(sci::StructuredIRCode)
    sci_uses_rng(sci) || return nothing
    # Seed the state map so CFG handling iterates over every stream that
    # appears anywhere in the IR, not only those that were written before
    # the first enclosing loop/if was visited.
    map = RngStateMap()
    for hkey in collect_streams(sci)
        map[hkey] = RngState()
    end
    # Seed every stream's initial seed slot from `KernelState.seed` so that two
    # launches of the same kernel produce different output even for non-default
    # `DeviceRNG()` streams. The stream-ID mix in `rng_key` (language/random.jl)
    # provides cross-stream divergence within a single launch; the host seed
    # provides cross-launch divergence across all streams. In-kernel
    # `Random.seed!` still overwrites a stream's seed slot. Counter stays at
    # the default `UInt32(0)`.
    if !isempty(map)
        entry = sci.entry
        anchor = first(instructions(entry))
        T = Tile{UInt32, Tuple{}}
        ks_inst = insert_before!(entry, anchor,
                                 Expr(:call, Intrinsics.kernel_state), KernelState)
        seed_inst = insert_before!(entry, anchor,
            Expr(:call, Core.getfield, SSAValue(ks_inst.ssa_idx), QuoteNode(:seed)), T)
        seed_ssa = SSAValue(seed_inst.ssa_idx)
        for st in values(map)
            st.seed = seed_ssa
        end
    end
    transform_block!(sci.entry, map)
    return nothing
end
