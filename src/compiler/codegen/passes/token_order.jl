# Token Ordering Pass
#
# Transforms a StructuredIRCode by inserting explicit token operations
# (MakeTokenNode, JoinTokensNode, TokenResultNode) and adding token carries
# to loop/branch control flow. After this pass, codegen simply emits what
# the IR says — no manual token threading in control_flow.jl or intrinsics.
#
# WHY: Tile IR uses a token-based memory ordering model (similar to LLVM's
# token type). Every memory operation (load, store, atomic) consumes an input
# token and produces an output token. The chain of tokens defines the
# happens-before ordering between memory accesses.
#
# HOW: The pass maintains a `token_map: Dict{TokenKey, Any}` mapping each
# (alias_set, role) pair to its current token SSA value. Two roles exist per
# alias set:
#   - LAST_OP:    token from the most recent load or store (RAW/WAR tracking)
#   - LAST_STORE: token from the most recent store only (WAW tracking)
# Plus a global ACQUIRE token for acquire-ordered atomics.
#
# For loads, the input token comes from LAST_STORE of the same alias set
# (read-after-write dependency). For stores, the input token joins all
# LAST_OP tokens of overlapping alias sets (write-after-read + write-after-write).
# Release-ordered atomics additionally join ALL LAST_OP tokens across all alias
# sets (memory fence semantics). Acquire-ordered atomics update the global
# ACQUIRE token.
#
# The pass adds token carries to loops (init_values + block args + terminator
# operands) and token results to IfOp types, then inserts getfield extractions
# after control flow ops to update the parent scope's token_map.
#
# Mirrors cuTile Python's `token_order_pass`.

using Core: SSAValue, Argument, SlotNumber

#=============================================================================
 Memory effect classification
=============================================================================#

@enum MemoryEffect MEM_NONE MEM_LOAD MEM_STORE

"""
    MemoryEffects

Per-block summary of which alias sets are read/written.
"""
struct MemoryEffects
    effects::Dict{AliasSet, MemoryEffect}
    has_acquire::Bool
end

MemoryEffects() = MemoryEffects(Dict{AliasSet, MemoryEffect}(), false)

function Base.union(a::MemoryEffects, b::MemoryEffects)
    result = Dict{AliasSet, MemoryEffect}()
    for (k, v) in a.effects; result[k] = v; end
    for (k, v) in b.effects
        result[k] = max(get(result, k, MEM_NONE), v)
    end
    return MemoryEffects(result, a.has_acquire | b.has_acquire)
end

const EMPTY_MEMORY_EFFECTS = MemoryEffects()

#=============================================================================
 Resolve and classify IR expressions
=============================================================================#

function classify_memory_op(resolved_func)
    if resolved_func === Intrinsics.load_partition_view ||
       resolved_func === Intrinsics.load_ptr_tko
        return MEM_LOAD
    elseif resolved_func === Intrinsics.store_partition_view ||
           resolved_func === Intrinsics.store_ptr_tko
        return MEM_STORE
    elseif is_atomic_intrinsic(resolved_func)
        return MEM_STORE
    else
        return MEM_NONE
    end
end

function is_atomic_intrinsic(func)
    isdefined(Intrinsics, :atomic_cas) && func === Intrinsics.atomic_cas && return true
    for op in (:atomic_xchg, :atomic_add, :atomic_max, :atomic_min,
               :atomic_or, :atomic_and, :atomic_xor)
        isdefined(Intrinsics, op) && func === getfield(Intrinsics, op) && return true
    end
    return false
end

function get_alias_set_for_operand(alias_result::Dict{Any, AliasSet}, operand)
    if operand isa SSAValue || operand isa Argument || operand isa SlotNumber
        return get(alias_result, operand, ALIAS_UNIVERSE)
    end
    return ALIAS_UNIVERSE
end

#=============================================================================
 Compute per-block memory effects
=============================================================================#

function compute_block_memory_effects!(block::Block, alias_result::Dict{Any, AliasSet},
                                       cache::Dict{UInt64, MemoryEffects})
    block_id = objectid(block)
    haskey(cache, block_id) && return cache[block_id]

    effects = MemoryEffects()
    for inst in instructions(block)
        s = stmt(inst)
        if s isa ControlFlowOp
            for b in blocks(s)
                effects = union(effects, compute_block_memory_effects!(b, alias_result, cache))
            end
        else
            call = resolve_call(s)
            call === nothing && continue
            resolved_func, operands = call
            mem_effect = classify_memory_op(resolved_func)
            mem_effect == MEM_NONE && continue
            alias_set = get_alias_set_for_operand(alias_result, first(operands))
            effects.effects[alias_set] = max(get(effects.effects, alias_set, MEM_NONE), mem_effect)
            if is_atomic_intrinsic(resolved_func)
                mo = extract_memory_order(resolved_func, operands)
                if has_acquire_order(mo)
                    effects = MemoryEffects(effects.effects, true)
                end
            end
        end
    end
    cache[block_id] = effects
    return effects
end

#=============================================================================
 Token map (IR-level, SSAValue/BlockArg)
=============================================================================#

function collect_join_tokens_ir(token_key::TokenKey, token_map::Dict{TokenKey, Any},
                                memory_order=nothing)
    tokens_to_join = Any[token_map[token_key]]
    for (other_key, other_tok) in token_map
        should_join = false
        if other_key isa AcquireTokenKey
            should_join = true
        elseif other_key isa AliasTokenKey && token_key isa AliasTokenKey
            if memory_order !== nothing && has_release_order(memory_order)
                should_join = other_key.role == LAST_OP
            end
            if other_key.role == token_key.role
                alias_overlap = (other_key.alias_set isa AliasUniverse) ||
                    (token_key.alias_set isa AliasUniverse) ||
                    !isempty(intersect(other_key.alias_set, token_key.alias_set))
                should_join = should_join || alias_overlap
            end
        end
        if should_join && !any(t -> t === other_tok, tokens_to_join)
            push!(tokens_to_join, other_tok)
        end
    end
    return tokens_to_join
end

function get_input_token_ir!(block::Block, ref::SSAValue,
                              token_key::TokenKey, token_map::Dict{TokenKey, Any},
                              memory_order=nothing)
    haskey(token_map, token_key) || return token_map[ACQUIRE_TOKEN_KEY]
    tokens = collect_join_tokens_ir(token_key, token_map, memory_order)
    length(tokens) == 1 && return tokens[1]
    join_inst = insert_before!(block, ref, JoinTokensNode(tokens), TOKEN_TYPE)
    return SSAValue(join_inst)
end

function has_release_order(memory_order)
    memory_order === nothing && return false
    return memory_order === MemoryOrder.Release || memory_order === MemoryOrder.AcqRel
end

function has_acquire_order(memory_order)
    memory_order === nothing && return false
    return memory_order === MemoryOrder.Acquire || memory_order === MemoryOrder.AcqRel
end

"""
    extract_memory_order(resolved_func, operands) -> Union{MemoryOrder.T, Nothing}

Extract the compile-time memory_order from an atomic intrinsic's operands.
"""
function extract_memory_order(resolved_func, operands)
    is_atomic_intrinsic(resolved_func) || return nothing
    # CAS: (ptr, expected, desired, mask, memory_order, memory_scope)
    # RMW: (ptr, val, mask, memory_order, memory_scope)
    mo_idx = resolved_func === Intrinsics.atomic_cas ? 5 : 4
    mo_idx > length(operands) && return nothing
    mo_arg = operands[mo_idx]
    # The memory_order is typically a compile-time constant (QuoteNode or literal)
    if mo_arg isa QuoteNode
        return mo_arg.value
    elseif mo_arg isa MemoryOrder.T
        return mo_arg
    end
    return nothing
end

#=============================================================================
 Control flow exit tokens (matching Python's _get_cf_exit_tokens)
=============================================================================#

"""
    get_cf_exit_tokens(effects, token_map) -> Vector{Any}

Collect current tokens for each alias set with memory effects.
These are appended to ContinueOp/BreakOp/YieldOp when leaving a CF region.
"""
function get_cf_exit_tokens(effects::MemoryEffects, token_map::Dict{TokenKey, Any})
    tokens = Any[]
    for (alias_set, effect) in effects.effects
        effect == MEM_NONE && continue
        if effect == MEM_LOAD
            push!(tokens, token_map[last_op_key(alias_set)])
        elseif effect == MEM_STORE
            push!(tokens, token_map[last_op_key(alias_set)])
            push!(tokens, token_map[last_store_key(alias_set)])
        end
    end
    if effects.has_acquire
        push!(tokens, token_map[ACQUIRE_TOKEN_KEY])
    end
    return tokens
end

#=============================================================================
 The main pass
=============================================================================#

function token_order_pass!(sci::StructuredIRCode, alias_result::Dict{Any, AliasSet})
    effects_cache = Dict{UInt64, MemoryEffects}()
    compute_block_memory_effects!(sci.entry, alias_result, effects_cache)

    # Insert root MakeTokenNode at entry
    root_inst = pushfirst!(sci.entry, MakeTokenNode(), TOKEN_TYPE)
    root_token = SSAValue(root_inst)

    # Initialize: all alias sets start at root token
    token_map = Dict{TokenKey, Any}()
    for alias_set in Set(values(alias_result))
        token_map[last_op_key(alias_set)] = root_token
        token_map[last_store_key(alias_set)] = root_token
    end
    token_map[ACQUIRE_TOKEN_KEY] = root_token

    transform_block!(sci.entry, alias_result, token_map, effects_cache, nothing, nothing, nothing)
    return nothing
end

#=============================================================================
 Block transformation
=============================================================================#

function transform_block!(block::Block,
                           alias_result::Dict{Any, AliasSet},
                           token_map::Dict{TokenKey, Any},
                           effects_cache::Dict{UInt64, MemoryEffects},
                           loop_effects::Union{MemoryEffects, Nothing},
                           ifelse_effects::Union{MemoryEffects, Nothing},
                           token_carries)
    # Snapshot to avoid invalidation from insertions
    snapshot = collect(instructions(block))

    for inst in snapshot
        s = stmt(inst)
        if s isa ControlFlowOp
            transform_control_flow!(block, inst, s,
                                     alias_result, token_map, effects_cache, loop_effects, token_carries)
        else
            transform_statement!(block, inst, alias_result, token_map)
        end
    end

    # Append exit tokens to the block's terminator (for loops and branches)
    transform_terminator!(block, token_map, loop_effects, ifelse_effects, token_carries)
end

function transform_statement!(block::Block, inst::Inst,
                                alias_result::Dict{Any, AliasSet},
                                token_map::Dict{TokenKey, Any})
    s = stmt(inst)
    call = resolve_call(s)
    call === nothing && return
    resolved_func, operands = call
    mem_effect = classify_memory_op(resolved_func)
    mem_effect == MEM_NONE && return

    alias_set = get_alias_set_for_operand(alias_result, first(operands))

    if mem_effect == MEM_LOAD
        input_token = get_input_token_ir!(block, SSAValue(inst),
                                           last_store_key(alias_set), token_map)
        push!(s.args, input_token)

        result_inst = insert_after!(block, SSAValue(inst), TokenResultNode(inst.ssa_idx), TOKEN_TYPE)
        result_token = SSAValue(result_inst)

        # Eagerly join with last_op token (Python line 176-179)
        lop_key = last_op_key(alias_set)
        last_op_tok = token_map[lop_key]
        join_inst = insert_after!(block, SSAValue(result_inst),
                       JoinTokensNode([last_op_tok, result_token]), TOKEN_TYPE)
        token_map[lop_key] = SSAValue(join_inst)

    elseif mem_effect == MEM_STORE
        # For release-ordered atomics, join with ALL LAST_OP tokens (memory fence)
        memory_order = extract_memory_order(resolved_func, operands)
        input_token = get_input_token_ir!(block, SSAValue(inst),
                                           last_op_key(alias_set), token_map,
                                           memory_order)
        push!(s.args, input_token)

        result_inst = insert_after!(block, SSAValue(inst), TokenResultNode(inst.ssa_idx), TOKEN_TYPE)
        result_token = SSAValue(result_inst)

        token_map[last_op_key(alias_set)] = result_token
        token_map[last_store_key(alias_set)] = result_token

        # Only acquire/acq_rel atomics update the ACQUIRE token
        if is_atomic_intrinsic(resolved_func) && has_acquire_order(memory_order)
            token_map[ACQUIRE_TOKEN_KEY] = result_token
        end
    end
end

function transform_terminator!(block::Block, token_map::Dict{TokenKey, Any},
                                 loop_effects::Union{MemoryEffects, Nothing},
                                 ifelse_effects::Union{MemoryEffects, Nothing},
                                 token_carries=nothing)
    term = terminator(block)
    term === nothing && return

    # ConditionOp (WhileOp before-block): extend args with exit tokens so that
    # the codegen-generated BreakOp carries them.
    if term isa ConditionOp && loop_effects !== nothing
        exit_tokens = get_cf_exit_tokens(loop_effects, token_map)
        if token_carries !== nothing
            for (cr, tok) in zip(token_carries, exit_tokens)
                term_value!(cr, term, tok)
            end
        else
            append!(operands(term), exit_tokens)
        end
        return
    end

    effects = if (term isa ContinueOp || term isa BreakOp) && loop_effects !== nothing
        loop_effects
    elseif term isa YieldOp && ifelse_effects !== nothing
        ifelse_effects
    elseif term isa YieldOp && loop_effects !== nothing
        loop_effects
    else
        nothing
    end
    effects === nothing && return

    exit_tokens = get_cf_exit_tokens(effects, token_map)

    # ContinueOp/BreakOp are always direct loop exits → in carries → use term_value!
    # YieldOp with ifelse_effects → IfOp branch → NOT in carries → append
    # YieldOp with only loop_effects → WhileOp after block → in carries → term_value!
    if token_carries !== nothing && !(term isa YieldOp && ifelse_effects !== nothing)
        for (cr, tok) in zip(token_carries, exit_tokens)
            term_value!(cr, term, tok)
        end
    else
        append!(operands(term), exit_tokens)
    end
end

#=============================================================================
 Control flow transformation
=============================================================================#

# --- Loops (ForOp, LoopOp) ---
# Matching Python's Loop handling (token_order.py:228-280)

function transform_control_flow!(parent_block::Block, inst::Inst,
                                  op::Union{ForOp, LoopOp},
                                  alias_result, token_map, effects_cache,
                                  parent_loop_effects=nothing, parent_token_carries=nothing)
    transform_loop!(parent_block, inst, op, alias_result,
                     token_map, effects_cache)
end

"""
    extract_token_getfields!(parent_block, inst, start_idx, effects, token_map)

Insert getfield extractions after a control flow op for each per-alias token result.
Updates `token_map` with SSAValues pointing to the extracted tokens.
`start_idx` is the 0-based index of the first token result (i.e., number of user results).
"""
function extract_token_getfields!(parent_block::Block, inst::Inst, start_idx::Int,
                                    effects::MemoryEffects, token_map::Dict{TokenKey, Any})
    last_ref = SSAValue(inst)
    idx = start_idx
    for (alias_set, effect) in effects.effects
        effect == MEM_NONE && continue
        if effect >= MEM_LOAD
            idx += 1
            gf_inst = insert_after!(parent_block, last_ref,
                           Expr(:call, GlobalRef(Core, :getfield), SSAValue(inst), idx), TOKEN_TYPE)
            token_map[last_op_key(alias_set)] = SSAValue(gf_inst)
            last_ref = SSAValue(gf_inst)
        end
        if effect == MEM_STORE
            idx += 1
            gf_inst = insert_after!(parent_block, last_ref,
                           Expr(:call, GlobalRef(Core, :getfield), SSAValue(inst), idx), TOKEN_TYPE)
            token_map[last_store_key(alias_set)] = SSAValue(gf_inst)
            last_ref = SSAValue(gf_inst)
        end
    end
    if effects.has_acquire
        idx += 1
        gf_inst = insert_after!(parent_block, last_ref,
                       Expr(:call, GlobalRef(Core, :getfield), SSAValue(inst), idx), TOKEN_TYPE)
        token_map[ACQUIRE_TOKEN_KEY] = SSAValue(gf_inst)
    end
end

"""
    insert_token_result_getfields!(parent_block, inst, block_args, n_user, effects, token_map)

Insert getfield extractions after a loop for each per-alias token result.
Updates the result type to include TokenType parameters and calls
`extract_token_getfields!` for the actual extraction.
"""
function insert_token_result_getfields!(parent_block::Block, inst::Inst,
                                         block_args, n_user::Int,
                                         effects::MemoryEffects, token_map::Dict{TokenKey, Any})
    length(block_args) > n_user || return
    all_types = Type[is_token_type(arg.type) ? TokenType : arg.type for arg in block_args]
    update_type!(parent_block, inst, isempty(all_types) ? Nothing : Tuple{all_types...})
    extract_token_getfields!(parent_block, inst, n_user, effects, token_map)
end

"""
    add_token_carries!(loop_carries, body_token_map, token_map, effects) -> token_carry_pairs

Add per-alias-set token carries to a loop via the `carries()` API.
Returns `(key => CarryRef)` pairs: keys for token_map updates, CarryRef handles
for `term_value!`, `body_arg`, and `after_arg`.
"""
function add_token_carries!(loop_carries, body_token_map::Dict{TokenKey, Any},
                             token_map::Dict{TokenKey, Any}, effects::MemoryEffects)
    token_carry_pairs = Pair[]
    for (alias_set, effect) in effects.effects
        effect == MEM_NONE && continue
        if effect >= MEM_LOAD
            key = last_op_key(alias_set)
            cr = push!(loop_carries, token_map[key], TOKEN_TYPE)
            body_token_map[key] = body_arg(cr)
            push!(token_carry_pairs, key => cr)
        end
        if effect == MEM_STORE
            key = last_store_key(alias_set)
            cr = push!(loop_carries, token_map[key], TOKEN_TYPE)
            body_token_map[key] = body_arg(cr)
            push!(token_carry_pairs, key => cr)
        end
    end
    if effects.has_acquire
        cr = push!(loop_carries, token_map[ACQUIRE_TOKEN_KEY], TOKEN_TYPE)
        body_token_map[ACQUIRE_TOKEN_KEY] = body_arg(cr)
        push!(token_carry_pairs, ACQUIRE_TOKEN_KEY => cr)
    end
    return token_carry_pairs
end

"""
    transform_loop!(...)

Add per-alias-set token carries to a ForOp/LoopOp.
"""
function transform_loop!(parent_block::Block, inst::Inst,
                           op::Union{ForOp, LoopOp},
                           alias_result::Dict{Any, AliasSet},
                           token_map::Dict{TokenKey, Any},
                           effects_cache::Dict{UInt64, MemoryEffects})
    body = op.body
    body_effects = get(effects_cache, objectid(body), EMPTY_MEMORY_EFFECTS)

    body_token_map = copy(token_map)
    result_token_map = copy(token_map)
    n_user_carries = length(op.init_values)

    token_carry_pairs = add_token_carries!(carries(op), body_token_map, token_map, body_effects)
    token_carry_refs = last.(token_carry_pairs)

    # Recurse — pass token_carry_refs so transform_terminator! can overwrite
    # per-terminator carry values
    transform_block!(body, alias_result, body_token_map, effects_cache,
                      body_effects, nothing, token_carry_refs)

    insert_token_result_getfields!(parent_block, inst, body.args,
                                    n_user_carries, body_effects, result_token_map)
    merge!(token_map, result_token_map)
end

# --- WhileOp ---
# WhileOp has before/after regions. We treat it similarly to a loop but need to
# handle both regions.

function transform_control_flow!(parent_block::Block, inst::Inst,
                                  op::WhileOp,
                                  alias_result, token_map, effects_cache,
                                  parent_loop_effects=nothing, parent_token_carries=nothing)
    before_effects = get(effects_cache, objectid(op.before), EMPTY_MEMORY_EFFECTS)
    after_effects = get(effects_cache, objectid(op.after), EMPTY_MEMORY_EFFECTS)
    loop_effects = union(before_effects, after_effects)

    body_token_map = copy(token_map)
    result_token_map = copy(token_map)
    n_user_carries = length(op.init_values)

    token_carry_pairs = add_token_carries!(carries(op), body_token_map, token_map, loop_effects)
    token_carry_refs = last.(token_carry_pairs)

    # Build after_token_map from after block's args via after_arg()
    after_token_map = copy(token_map)
    for (key, cr) in token_carry_pairs
        after_token_map[key] = after_arg(cr)
    end

    # Transform before region — pass token_carry_refs for ConditionOp overwrite
    transform_block!(op.before, alias_result, body_token_map, effects_cache,
                      loop_effects, nothing, token_carry_refs)

    # Propagate before's final token state to after_token_map.
    # The after block receives values from before's ConditionOp, so it should
    # see the token state AFTER the before block's transformations (e.g., CAS result).
    for (key, val) in body_token_map
        after_token_map[key] = val
    end

    # Transform after region — also pass token_carry_refs for terminator overwrite
    transform_block!(op.after, alias_result, after_token_map, effects_cache,
                      loop_effects, nothing, token_carry_refs)

    insert_token_result_getfields!(parent_block, inst, op.before.args,
                                    n_user_carries, loop_effects, result_token_map)
    merge!(token_map, result_token_map)
end

# --- IfOp ---
# Matching Python's IfElse handling (token_order.py:294-334)

function transform_control_flow!(parent_block::Block, inst::Inst,
                                  op::IfOp,
                                  alias_result, token_map, effects_cache,
                                  parent_loop_effects=nothing, parent_token_carries=nothing)
    then_effects = get(effects_cache, objectid(op.then_region), EMPTY_MEMORY_EFFECTS)
    else_effects = get(effects_cache, objectid(op.else_region), EMPTY_MEMORY_EFFECTS)
    merged_effects = union(then_effects, else_effects)

    # Transform both branches. Pass parent_loop_effects + parent_token_carries so that
    # ContinueOp/BreakOp inside branches (common for LoopOp→IfOp patterns) get
    # token exit values overwritten correctly.
    then_map = copy(token_map)
    transform_block!(op.then_region, alias_result, then_map, effects_cache,
                      parent_loop_effects, merged_effects, parent_token_carries)
    else_map = copy(token_map)
    transform_block!(op.else_region, alias_result, else_map, effects_cache,
                      parent_loop_effects, merged_effects, parent_token_carries)

    # Count token results for type update
    n_token_results = 0
    for (_, effect) in merged_effects.effects
        effect == MEM_NONE && continue
        n_token_results += (effect == MEM_LOAD) ? 1 : 2
    end
    n_token_results += merged_effects.has_acquire ? 1 : 0

    if n_token_results > 0
        old_type = value_type(inst)
        user_types = if old_type === Nothing
            Type[]
        elseif old_type <: Tuple
            collect(Type, old_type.parameters)
        else
            Type[old_type]
        end
        new_type = Tuple{user_types..., fill(TokenType, n_token_results)...}
        update_type!(parent_block, inst, new_type)
        extract_token_getfields!(parent_block, inst, length(user_types), merged_effects, token_map)
    end
end

# Fallback
function transform_control_flow!(parent_block::Block, inst::Inst,
                                  op::ControlFlowOp,
                                  alias_result, token_map, effects_cache,
                                  parent_loop_effects=nothing, parent_token_carries=nothing)
end
