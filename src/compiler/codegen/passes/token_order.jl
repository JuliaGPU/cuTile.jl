# Token ordering pass
#
# Transforms a StructuredIRCode by inserting token operations (MakeToken, JoinTokens,
# TokenResult) and threading per-alias-set tokens through control flow.
# After this pass, codegen emits what's in the IR — no manual token threading.
#
# Mirrors cuTile Python's `token_order_pass` (res/cutile-python/src/cuda/tile/_passes/token_order.py).

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

function resolve_call(stmt)
    stmt isa Expr || return nothing
    if stmt.head === :call
        func_ref = stmt.args[1]
        operands = @view stmt.args[2:end]
    elseif stmt.head === :invoke
        func_ref = stmt.args[2]
        operands = @view stmt.args[3:end]
    else
        return nothing
    end
    resolved = if func_ref isa GlobalRef
        try; getfield(func_ref.mod, func_ref.name); catch; nothing; end
    else
        func_ref
    end
    resolved === nothing && return nothing
    return (resolved, operands)
end

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
    for (_, entry) in block.body
        if entry.stmt isa ControlFlowOp
            nested = compute_cf_memory_effects!(entry.stmt, alias_result, cache)
            effects = union(effects, nested)
        else
            call = resolve_call(entry.stmt)
            call === nothing && continue
            resolved_func, operands = call
            mem_effect = classify_memory_op(resolved_func)
            mem_effect == MEM_NONE && continue
            alias_set = get_alias_set_for_operand(alias_result, first(operands))
            effects.effects[alias_set] = max(get(effects.effects, alias_set, MEM_NONE), mem_effect)
            # Track acquire ordering for atomics
            if is_atomic_intrinsic(resolved_func)
                effects = MemoryEffects(effects.effects, true)
            end
        end
    end
    cache[block_id] = effects
    return effects
end

compute_cf_memory_effects!(op::IfOp, ar, c) =
    union(compute_block_memory_effects!(op.then_region, ar, c),
          compute_block_memory_effects!(op.else_region, ar, c))
compute_cf_memory_effects!(op::ForOp, ar, c) = compute_block_memory_effects!(op.body, ar, c)
compute_cf_memory_effects!(op::LoopOp, ar, c) = compute_block_memory_effects!(op.body, ar, c)
compute_cf_memory_effects!(op::WhileOp, ar, c) =
    union(compute_block_memory_effects!(op.before, ar, c),
          compute_block_memory_effects!(op.after, ar, c))
compute_cf_memory_effects!(::ControlFlowOp, _, _) = EMPTY_MEMORY_EFFECTS

#=============================================================================
 Token map (IR-level, SSAValue/BlockArg)
=============================================================================#

const IRToken = Any

function collect_join_tokens_ir(token_key::TokenKey, token_map::Dict{TokenKey, IRToken},
                                memory_order=nothing)
    tokens_to_join = IRToken[token_map[token_key]]
    for (other_key, other_tok) in token_map
        should_join = false
        if other_key isa AcquireTokenKey
            should_join = true
        elseif other_key isa AliasTokenKey && token_key isa AliasTokenKey
            if memory_order !== nothing && has_release_order(memory_order)
                should_join = other_key.role == LAST_OP
            end
            if other_key.role == token_key.role
                alias_overlap = !(other_key.alias_set isa AliasUniverse) &&
                    !(token_key.alias_set isa AliasUniverse) &&
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

function get_input_token_ir!(sci::StructuredIRCode, block::Block, before_ssa::Int,
                              token_key::TokenKey, token_map::Dict{TokenKey, IRToken},
                              memory_order=nothing)
    haskey(token_map, token_key) || return token_map[ACQUIRE_TOKEN_KEY]
    tokens = collect_join_tokens_ir(token_key, token_map, memory_order)
    length(tokens) == 1 && return tokens[1]
    join_ssa = new_ssa_idx!(sci)
    insert_before!(block.body, before_ssa, join_ssa, JoinTokensNode(tokens), TOKEN_TYPE)
    return SSAValue(join_ssa)
end

has_release_order(memory_order) = false

#=============================================================================
 Control flow exit tokens (matching Python's _get_cf_exit_tokens)
=============================================================================#

"""
    get_cf_exit_tokens(effects, token_map) -> Vector{IRToken}

Collect current tokens for each alias set with memory effects.
These are appended to ContinueOp/BreakOp/YieldOp when leaving a CF region.
"""
function get_cf_exit_tokens(effects::MemoryEffects, token_map::Dict{TokenKey, IRToken})
    tokens = IRToken[]
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
    root_ssa = new_ssa_idx!(sci)
    pushfirst!(sci.entry.body, (root_ssa, MakeTokenNode(), TOKEN_TYPE))
    root_token = SSAValue(root_ssa)

    # Initialize: all alias sets start at root token
    token_map = Dict{TokenKey, IRToken}()
    for alias_set in Set(values(alias_result))
        token_map[last_op_key(alias_set)] = root_token
        token_map[last_store_key(alias_set)] = root_token
    end
    token_map[ACQUIRE_TOKEN_KEY] = root_token

    transform_block!(sci, sci.entry, alias_result, token_map, effects_cache, nothing, nothing)
    return nothing
end

#=============================================================================
 Block transformation
=============================================================================#

function transform_block!(sci::StructuredIRCode, block::Block,
                           alias_result::Dict{Any, AliasSet},
                           token_map::Dict{TokenKey, IRToken},
                           effects_cache::Dict{UInt64, MemoryEffects},
                           loop_effects::Union{MemoryEffects, Nothing},
                           ifelse_effects::Union{MemoryEffects, Nothing})
    # Snapshot indices to avoid invalidation from insertions
    ssa_indices = collect(Int, block.body.ssa_idxes)

    for ssa_idx in ssa_indices
        entry = get(block.body, ssa_idx, nothing)
        entry === nothing && continue
        if entry.stmt isa ControlFlowOp
            transform_control_flow!(sci, block, ssa_idx, entry.stmt, entry.typ,
                                     alias_result, token_map, effects_cache, loop_effects)
        else
            transform_statement!(sci, block, ssa_idx, entry.stmt,
                                  alias_result, token_map)
        end
    end

    # Append exit tokens to the block's terminator (for loops and branches)
    transform_terminator!(block, token_map, loop_effects, ifelse_effects)
end

function transform_statement!(sci::StructuredIRCode, block::Block, ssa_idx::Int, stmt,
                                alias_result::Dict{Any, AliasSet},
                                token_map::Dict{TokenKey, IRToken})
    call = resolve_call(stmt)
    call === nothing && return
    resolved_func, operands = call
    mem_effect = classify_memory_op(resolved_func)
    mem_effect == MEM_NONE && return

    alias_set = get_alias_set_for_operand(alias_result, first(operands))

    if mem_effect == MEM_LOAD
        input_token = get_input_token_ir!(sci, block, ssa_idx,
                                           last_store_key(alias_set), token_map)
        push!(stmt.args, input_token)

        result_ssa = new_ssa_idx!(sci)
        insert_after!(block.body, ssa_idx, result_ssa, TokenResultNode(ssa_idx), TOKEN_TYPE)
        result_token = SSAValue(result_ssa)

        # Eagerly join with last_op token (Python line 176-179)
        lop_key = last_op_key(alias_set)
        last_op_tok = token_map[lop_key]
        join_ssa = new_ssa_idx!(sci)
        insert_after!(block.body, result_ssa, join_ssa,
                       JoinTokensNode([last_op_tok, result_token]), TOKEN_TYPE)
        token_map[lop_key] = SSAValue(join_ssa)

    elseif mem_effect == MEM_STORE
        input_token = get_input_token_ir!(sci, block, ssa_idx,
                                           last_op_key(alias_set), token_map)
        push!(stmt.args, input_token)

        result_ssa = new_ssa_idx!(sci)
        insert_after!(block.body, ssa_idx, result_ssa, TokenResultNode(ssa_idx), TOKEN_TYPE)
        result_token = SSAValue(result_ssa)

        token_map[last_op_key(alias_set)] = result_token
        token_map[last_store_key(alias_set)] = result_token

        # Atomics with acquire semantics update the ACQUIRE token
        if is_atomic_intrinsic(resolved_func)
            token_map[ACQUIRE_TOKEN_KEY] = result_token
        end
    end
end

function transform_terminator!(block::Block, token_map::Dict{TokenKey, IRToken},
                                 loop_effects::Union{MemoryEffects, Nothing},
                                 ifelse_effects::Union{MemoryEffects, Nothing})
    term = block.terminator
    term === nothing && return

    # ConditionOp (WhileOp before-block): extend args with exit tokens so that
    # the codegen-generated BreakOp carries them.
    if term isa ConditionOp && loop_effects !== nothing
        append!(term.args, get_cf_exit_tokens(loop_effects, token_map))
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
    append!(term.values, get_cf_exit_tokens(effects, token_map))
end

#=============================================================================
 Control flow transformation
=============================================================================#

# --- Loops (ForOp, LoopOp) ---
# Matching Python's Loop handling (token_order.py:228-280)

function transform_control_flow!(sci::StructuredIRCode, parent_block::Block,
                                  ssa_idx::Int, op::ForOp, @nospecialize(result_type),
                                  alias_result, token_map, effects_cache, parent_loop_effects=nothing)
    transform_loop!(sci, parent_block, ssa_idx, op, op.body, alias_result,
                     token_map, effects_cache)
end

function transform_control_flow!(sci::StructuredIRCode, parent_block::Block,
                                  ssa_idx::Int, op::LoopOp, @nospecialize(result_type),
                                  alias_result, token_map, effects_cache, parent_loop_effects=nothing)
    transform_loop!(sci, parent_block, ssa_idx, op, op.body, alias_result,
                     token_map, effects_cache)
end

"""
    transform_loop!(sci, parent_block, ssa_idx, op, body, alias_result, token_map, effects_cache)

Add per-alias-set token carries to a loop. For each alias set with memory effects
in the body, creates init_values, BlockArgs, and terminator exit tokens.
Then recurses into the body with the body-scoped token_map.
After the loop, inserts getfield extractions for the token results.
"""
function transform_loop!(sci::StructuredIRCode, parent_block::Block,
                           ssa_idx::Int, op::Union{ForOp, LoopOp}, body::Block,
                           alias_result::Dict{Any, AliasSet},
                           token_map::Dict{TokenKey, IRToken},
                           effects_cache::Dict{UInt64, MemoryEffects})
    body_effects = get(effects_cache, objectid(body), EMPTY_MEMORY_EFFECTS)

    body_token_map = copy(token_map)
    result_token_map = copy(token_map)

    # Track the number of user carries (before we add tokens)
    n_user_carries = length(op.init_values)

    # Add per-alias token carries (matching Python lines 245-264)
    carry_idx = n_user_carries  # 0-based index into results after user carries
    for (alias_set, effect) in body_effects.effects
        effect == MEM_NONE && continue
        if effect >= MEM_LOAD
            carry_idx += 1
            push!(op.init_values, token_map[last_op_key(alias_set)])
            body_arg = new_block_arg!(body, sci, TOKEN_TYPE)
            body_token_map[last_op_key(alias_set)] = body_arg
            # result_token_map will be updated below with getfield SSAs
        end
        if effect == MEM_STORE
            carry_idx += 1
            push!(op.init_values, token_map[last_store_key(alias_set)])
            body_arg = new_block_arg!(body, sci, TOKEN_TYPE)
            body_token_map[last_store_key(alias_set)] = body_arg
        end
    end
    if body_effects.has_acquire
        carry_idx += 1
        push!(op.init_values, token_map[ACQUIRE_TOKEN_KEY])
        body_arg = new_block_arg!(body, sci, TOKEN_TYPE)
        body_token_map[ACQUIRE_TOKEN_KEY] = body_arg
    end

    n_total_carries = length(op.init_values)

    # Recurse into body with body-scoped token map
    transform_block!(sci, body, alias_result, body_token_map, effects_cache,
                      body_effects, nothing)

    # After the loop: insert getfield extractions for token results.
    # Update the loop's type to include ALL carries (user + token) so that
    # codegen's getfield extraction works correctly.
    if n_total_carries > n_user_carries
        # Build result type from block args (authoritative source of all carries)
        all_types = Type[is_token_type(arg.type) ? TokenType : arg.type
                         for arg in body.args]
        new_type = isempty(all_types) ? Nothing : Tuple{all_types...}
        update_type!(parent_block.body, ssa_idx, new_type)


        # Insert getfield SSAs after the loop for each token result
        last_inserted = ssa_idx
        carry_idx = n_user_carries
        for (alias_set, effect) in body_effects.effects
            effect == MEM_NONE && continue
            if effect >= MEM_LOAD
                carry_idx += 1
                gf_ssa = new_ssa_idx!(sci)
                gf_expr = Expr(:call, GlobalRef(Core, :getfield), SSAValue(ssa_idx), carry_idx)
                insert_after!(parent_block.body, last_inserted, gf_ssa, gf_expr, TOKEN_TYPE)
                result_token_map[last_op_key(alias_set)] = SSAValue(gf_ssa)
                last_inserted = gf_ssa
            end
            if effect == MEM_STORE
                carry_idx += 1
                gf_ssa = new_ssa_idx!(sci)
                gf_expr = Expr(:call, GlobalRef(Core, :getfield), SSAValue(ssa_idx), carry_idx)
                insert_after!(parent_block.body, last_inserted, gf_ssa, gf_expr, TOKEN_TYPE)
                result_token_map[last_store_key(alias_set)] = SSAValue(gf_ssa)
                last_inserted = gf_ssa
            end
        end
        if body_effects.has_acquire
            carry_idx += 1
            gf_ssa = new_ssa_idx!(sci)
            gf_expr = Expr(:call, GlobalRef(Core, :getfield), SSAValue(ssa_idx), carry_idx)
            insert_after!(parent_block.body, last_inserted, gf_ssa, gf_expr, TOKEN_TYPE)
            result_token_map[ACQUIRE_TOKEN_KEY] = SSAValue(gf_ssa)
        end
    end

    # Update parent's token_map with loop result tokens
    merge!(token_map, result_token_map)
end

# --- WhileOp ---
# WhileOp has before/after regions. We treat it similarly to a loop but need to
# handle both regions.

function transform_control_flow!(sci::StructuredIRCode, parent_block::Block,
                                  ssa_idx::Int, op::WhileOp, @nospecialize(result_type),
                                  alias_result, token_map, effects_cache, parent_loop_effects=nothing)
    before_effects = get(effects_cache, objectid(op.before), EMPTY_MEMORY_EFFECTS)
    after_effects = get(effects_cache, objectid(op.after), EMPTY_MEMORY_EFFECTS)
    loop_effects = union(before_effects, after_effects)

    body_token_map = copy(token_map)
    result_token_map = copy(token_map)
    n_user_carries = length(op.init_values)

    # Add per-alias token carries to before/after blocks
    for (alias_set, effect) in loop_effects.effects
        effect == MEM_NONE && continue
        if effect >= MEM_LOAD
            push!(op.init_values, token_map[last_op_key(alias_set)])
            before_arg = new_block_arg!(op.before, sci, TOKEN_TYPE)
            after_arg = new_block_arg!(op.after, sci, TOKEN_TYPE)
            body_token_map[last_op_key(alias_set)] = before_arg
        end
        if effect == MEM_STORE
            push!(op.init_values, token_map[last_store_key(alias_set)])
            before_arg = new_block_arg!(op.before, sci, TOKEN_TYPE)
            after_arg = new_block_arg!(op.after, sci, TOKEN_TYPE)
            body_token_map[last_store_key(alias_set)] = before_arg
        end
    end
    if loop_effects.has_acquire
        push!(op.init_values, token_map[ACQUIRE_TOKEN_KEY])
        before_arg = new_block_arg!(op.before, sci, TOKEN_TYPE)
        after_arg = new_block_arg!(op.after, sci, TOKEN_TYPE)
        body_token_map[ACQUIRE_TOKEN_KEY] = before_arg
    end

    n_total_carries = length(op.init_values)

    # Build after_token_map from after block's args (not before's)
    after_token_map = copy(token_map)
    after_arg_idx = n_user_carries
    for (alias_set, effect) in loop_effects.effects
        effect == MEM_NONE && continue
        if effect >= MEM_LOAD
            after_arg_idx += 1
            after_token_map[last_op_key(alias_set)] = op.after.args[after_arg_idx]
        end
        if effect == MEM_STORE
            after_arg_idx += 1
            after_token_map[last_store_key(alias_set)] = op.after.args[after_arg_idx]
        end
    end
    if loop_effects.has_acquire
        after_arg_idx += 1
        after_token_map[ACQUIRE_TOKEN_KEY] = op.after.args[after_arg_idx]
    end

    # Transform before region (may update body_token_map, e.g., CAS in condition)
    transform_block!(sci, op.before, alias_result, body_token_map, effects_cache,
                      loop_effects, nothing)

    # Propagate before's final token state to after_token_map.
    # The after block receives values from before's ConditionOp, so it should
    # see the token state AFTER the before block's transformations (e.g., CAS result).
    for (key, val) in body_token_map
        after_token_map[key] = val
    end

    transform_block!(sci, op.after, alias_result, after_token_map, effects_cache,
                      loop_effects, nothing)

    # Insert getfield extractions for token results
    if n_total_carries > n_user_carries
        # Build result type from before block args (authoritative source of all carries)
        all_types = Type[is_token_type(arg.type) ? TokenType : arg.type
                         for arg in op.before.args]
        new_type = isempty(all_types) ? Nothing : Tuple{all_types...}
        update_type!(parent_block.body, ssa_idx, new_type)


        last_inserted = ssa_idx
        carry_idx = n_user_carries
        for (alias_set, effect) in loop_effects.effects
            effect == MEM_NONE && continue
            if effect >= MEM_LOAD
                carry_idx += 1
                gf_ssa = new_ssa_idx!(sci)
                gf_expr = Expr(:call, GlobalRef(Core, :getfield), SSAValue(ssa_idx), carry_idx)
                insert_after!(parent_block.body, last_inserted, gf_ssa, gf_expr, TOKEN_TYPE)
                result_token_map[last_op_key(alias_set)] = SSAValue(gf_ssa)
                last_inserted = gf_ssa
            end
            if effect == MEM_STORE
                carry_idx += 1
                gf_ssa = new_ssa_idx!(sci)
                gf_expr = Expr(:call, GlobalRef(Core, :getfield), SSAValue(ssa_idx), carry_idx)
                insert_after!(parent_block.body, last_inserted, gf_ssa, gf_expr, TOKEN_TYPE)
                result_token_map[last_store_key(alias_set)] = SSAValue(gf_ssa)
                last_inserted = gf_ssa
            end
        end
        if loop_effects.has_acquire
            carry_idx += 1
            gf_ssa = new_ssa_idx!(sci)
            gf_expr = Expr(:call, GlobalRef(Core, :getfield), SSAValue(ssa_idx), carry_idx)
            insert_after!(parent_block.body, last_inserted, gf_ssa, gf_expr, TOKEN_TYPE)
            result_token_map[ACQUIRE_TOKEN_KEY] = SSAValue(gf_ssa)
        end
    end

    merge!(token_map, result_token_map)
end

# --- IfOp ---
# Matching Python's IfElse handling (token_order.py:294-334)

function transform_control_flow!(sci::StructuredIRCode, parent_block::Block,
                                  ssa_idx::Int, op::IfOp, @nospecialize(result_type),
                                  alias_result, token_map, effects_cache, parent_loop_effects=nothing)
    then_effects = get(effects_cache, objectid(op.then_region), EMPTY_MEMORY_EFFECTS)
    else_effects = get(effects_cache, objectid(op.else_region), EMPTY_MEMORY_EFFECTS)
    merged_effects = union(then_effects, else_effects)

    # Transform both branches. Pass parent_loop_effects so that ContinueOp/BreakOp
    # inside branches (common for LoopOp→IfOp patterns) get token exit values.
    then_map = copy(token_map)
    transform_block!(sci, op.then_region, alias_result, then_map, effects_cache,
                      parent_loop_effects, merged_effects)
    else_map = copy(token_map)
    transform_block!(sci, op.else_region, alias_result, else_map, effects_cache,
                      parent_loop_effects, merged_effects)

    # Count token results and insert getfield extractions
    n_token_results = 0
    for (_, effect) in merged_effects.effects
        effect == MEM_NONE && continue
        n_token_results += (effect == MEM_LOAD) ? 1 : 2
    end
    n_token_results += merged_effects.has_acquire ? 1 : 0

    if n_token_results > 0
        # Update IfOp type to include token results
        old_type = get(parent_block.body, ssa_idx, nothing)
        old_type = get(parent_block.body, ssa_idx, nothing)
        if old_type !== nothing
            user_types = if old_type.typ === Nothing
                Type[]
            elseif old_type.typ <: Tuple
                collect(Type, old_type.typ.parameters)
            else
                Type[old_type.typ]
            end
            token_types = fill(TokenType, n_token_results)
            new_type = Tuple{user_types..., token_types...}
            update_type!(parent_block.body, ssa_idx, new_type)
        else
            user_types = Type[]
        end

        # Insert getfield extractions for token results
        last_inserted = ssa_idx
        result_idx = length(user_types)
        for (alias_set, effect) in merged_effects.effects
            effect == MEM_NONE && continue
            if effect >= MEM_LOAD
                result_idx += 1
                gf_ssa = new_ssa_idx!(sci)
                gf_expr = Expr(:call, GlobalRef(Core, :getfield), SSAValue(ssa_idx), result_idx)
                insert_after!(parent_block.body, last_inserted, gf_ssa, gf_expr, TOKEN_TYPE)
                token_map[last_op_key(alias_set)] = SSAValue(gf_ssa)
                last_inserted = gf_ssa
            end
            if effect == MEM_STORE
                result_idx += 1
                gf_ssa = new_ssa_idx!(sci)
                gf_expr = Expr(:call, GlobalRef(Core, :getfield), SSAValue(ssa_idx), result_idx)
                insert_after!(parent_block.body, last_inserted, gf_ssa, gf_expr, TOKEN_TYPE)
                token_map[last_store_key(alias_set)] = SSAValue(gf_ssa)
                last_inserted = gf_ssa
            end
        end
        if merged_effects.has_acquire
            result_idx += 1
            gf_ssa = new_ssa_idx!(sci)
            gf_expr = Expr(:call, GlobalRef(Core, :getfield), SSAValue(ssa_idx), result_idx)
            insert_after!(parent_block.body, last_inserted, gf_ssa, gf_expr, TOKEN_TYPE)
            token_map[ACQUIRE_TOKEN_KEY] = SSAValue(gf_ssa)
        end
    end
end

# Fallback
function transform_control_flow!(sci::StructuredIRCode, parent_block::Block,
                                  ssa_idx::Int, op::ControlFlowOp, @nospecialize(result_type),
                                  alias_result, token_map, effects_cache, parent_loop_effects=nothing)
end
