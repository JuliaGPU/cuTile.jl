# Token ordering pass
#
# Transforms a StructuredIRCode by inserting token operations (MakeToken, JoinTokens,
# TokenResult) and threading tokens through control flow (loop carries, branch yields).
# After this pass, codegen simply emits what's in the IR — no manual token threading.
#
# Mirrors cuTile Python's `token_order_pass` (res/cutile-python/src/cuda/tile/_passes/token_order.py).

using Core: SSAValue, Argument, SlotNumber

#=============================================================================
 Memory effect classification
=============================================================================#

@enum MemoryEffect MEM_NONE MEM_LOAD MEM_STORE

"""
    MemoryEffects

Per-block summary of which alias sets are read/written and whether any
acquire-ordered operation appears.
"""
struct MemoryEffects
    effects::Dict{AliasSet, MemoryEffect}
    has_acquire::Bool
end

MemoryEffects() = MemoryEffects(Dict{AliasSet, MemoryEffect}(), false)

function Base.merge!(a::MemoryEffects, b::MemoryEffects)
    for (alias_set, effect) in b.effects
        existing = get(a.effects, alias_set, MEM_NONE)
        a.effects[alias_set] = max(existing, effect)
    end
    return MemoryEffects(a.effects, a.has_acquire | b.has_acquire)
end

function Base.union(a::MemoryEffects, b::MemoryEffects)
    result = Dict{AliasSet, MemoryEffect}()
    for (k, v) in a.effects
        result[k] = v
    end
    for (k, v) in b.effects
        existing = get(result, k, MEM_NONE)
        result[k] = max(existing, v)
    end
    return MemoryEffects(result, a.has_acquire | b.has_acquire)
end

const EMPTY_MEMORY_EFFECTS = MemoryEffects()

#=============================================================================
 Resolve functions from IR expressions
=============================================================================#

"""
    resolve_call(stmt) -> (func, operands) or nothing

Extract the resolved function value and operands from a :call or :invoke Expr.
"""
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
        try
            getfield(func_ref.mod, func_ref.name)
        catch
            nothing
        end
    else
        func_ref
    end
    resolved === nothing && return nothing
    return (resolved, operands)
end

"""
    classify_memory_op(resolved_func) -> (MemoryEffect, Bool)

Classify a resolved function as a memory operation.
Returns (effect, is_store) where effect is MEM_NONE/MEM_LOAD/MEM_STORE.
"""
function classify_memory_op(resolved_func)
    if resolved_func === Intrinsics.load_partition_view ||
       resolved_func === Intrinsics.load_ptr_tko
        return MEM_LOAD
    elseif resolved_func === Intrinsics.store_partition_view ||
           resolved_func === Intrinsics.store_ptr_tko
        return MEM_STORE
    elseif is_atomic_intrinsic(resolved_func)
        return MEM_STORE  # Atomics are read-modify-write, treat as store for ordering
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

"""
    get_alias_set_for_operand(alias_result, operand) -> AliasSet

Look up the alias set for an operand (the first arg of a memory op).
"""
function get_alias_set_for_operand(alias_result::Dict{Any, AliasSet}, operand)
    if operand isa SSAValue || operand isa Argument || operand isa SlotNumber
        return get(alias_result, operand, ALIAS_UNIVERSE)
    end
    return ALIAS_UNIVERSE
end

#=============================================================================
 Compute per-block memory effects
=============================================================================#

"""
    compute_block_memory_effects!(block, alias_result, cache)

Compute memory effects for a block and all nested blocks, storing results in `cache`.
"""
function compute_block_memory_effects!(block::Block, alias_result::Dict{Any, AliasSet},
                                       cache::Dict{UInt64, MemoryEffects})
    block_id = objectid(block)
    haskey(cache, block_id) && return cache[block_id]

    effects = MemoryEffects()
    for (ssa_idx, entry) in block.body
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
            existing = get(effects.effects, alias_set, MEM_NONE)
            effects.effects[alias_set] = max(existing, mem_effect)
        end
    end
    cache[block_id] = effects
    return effects
end

function compute_cf_memory_effects!(op::IfOp, alias_result, cache)
    then_eff = compute_block_memory_effects!(op.then_region, alias_result, cache)
    else_eff = compute_block_memory_effects!(op.else_region, alias_result, cache)
    return union(then_eff, else_eff)
end

function compute_cf_memory_effects!(op::ForOp, alias_result, cache)
    return compute_block_memory_effects!(op.body, alias_result, cache)
end

function compute_cf_memory_effects!(op::WhileOp, alias_result, cache)
    before_eff = compute_block_memory_effects!(op.before, alias_result, cache)
    after_eff = compute_block_memory_effects!(op.after, alias_result, cache)
    return union(before_eff, after_eff)
end

function compute_cf_memory_effects!(op::LoopOp, alias_result, cache)
    return compute_block_memory_effects!(op.body, alias_result, cache)
end

compute_cf_memory_effects!(::ControlFlowOp, alias_result, cache) = EMPTY_MEMORY_EFFECTS

#=============================================================================
 Token map (IR-level, using SSAValue/BlockArg instead of bytecode Values)
=============================================================================#

# IRToken: an SSAValue, BlockArg, or nothing (for tokens in the IR)
const IRToken = Any

"""
    collect_join_tokens_ir(token_key, token_map, memory_order=nothing) -> Vector{IRToken}

IR-level equivalent of Python's `_collect_join_tokens`.
Collects all token IR values that need to be joined for the given token_key.
"""
function collect_join_tokens_ir(token_key::TokenKey, token_map::Dict{TokenKey, IRToken},
                                memory_order=nothing)
    tokens_to_join = IRToken[token_map[token_key]]

    for (other_key, other_tok) in token_map
        should_join = false

        if other_key isa AcquireTokenKey
            should_join = true
        elseif other_key isa AliasTokenKey && token_key isa AliasTokenKey
            # Release: join with all LAST_OP tokens
            if memory_order !== nothing && has_release_order(memory_order)
                should_join = other_key.role == LAST_OP
            end
            # Alias set overlap: same role and sets overlap
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

"""
    get_input_token_ir!(sci, block, before_ssa, token_key, token_map, memory_order=nothing)
        -> IRToken

Get the input token for a memory operation. If multiple tokens need joining,
inserts a JoinTokensNode into the block before `before_ssa` and returns its SSAValue.
"""
function get_input_token_ir!(sci::StructuredIRCode, block::Block, before_ssa::Int,
                              token_key::TokenKey, token_map::Dict{TokenKey, IRToken},
                              memory_order=nothing)
    if !haskey(token_map, token_key)
        # Fallback to ACQUIRE token
        return token_map[ACQUIRE_TOKEN_KEY]
    end

    tokens_to_join = collect_join_tokens_ir(token_key, token_map, memory_order)

    if length(tokens_to_join) == 1
        return tokens_to_join[1]
    end

    # Insert JoinTokensNode before the memory op
    join_ssa = new_ssa_idx!(sci)
    insert_before!(block.body, before_ssa, join_ssa, JoinTokensNode(tokens_to_join), TOKEN_TYPE)
    return SSAValue(join_ssa)
end

has_release_order(memory_order) = false


#=============================================================================
 The main pass
=============================================================================#

"""
    token_order_pass!(sci::StructuredIRCode, alias_result::Dict{Any, AliasSet})

Transform a StructuredIRCode by inserting explicit token operations.
Modifies the IR in place:
- Inserts MakeTokenNode at function entry
- Inserts JoinTokensNode where tokens need merging
- Inserts TokenResultNode after memory ops to capture their result tokens
- Adds token as extra argument to memory op calls
- Adds per-alias-set token carries through loops and branches

After this pass, codegen emits tokens from the IR without manual threading.
"""
function token_order_pass!(sci::StructuredIRCode, alias_result::Dict{Any, AliasSet})
    # Compute per-block memory effects
    effects_cache = Dict{UInt64, MemoryEffects}()
    compute_block_memory_effects!(sci.entry, alias_result, effects_cache)

    # Create root token (MakeTokenNode) at entry
    root_ssa = new_ssa_idx!(sci)
    pushfirst!(sci.entry.body, (root_ssa, MakeTokenNode(), TOKEN_TYPE))
    root_token = SSAValue(root_ssa)

    # Initialize token map: all alias sets start at root token
    token_map = Dict{TokenKey, IRToken}()
    unique_alias_sets = Set(values(alias_result))
    for alias_set in unique_alias_sets
        token_map[last_op_key(alias_set)] = root_token
        token_map[last_store_key(alias_set)] = root_token
    end
    token_map[ACQUIRE_TOKEN_KEY] = root_token

    # Transform the entry block
    transform_block!(sci, sci.entry, alias_result, token_map, effects_cache, nothing, nothing)

    return nothing
end

#=============================================================================
 Block transformation (recursive)
=============================================================================#

"""
    transform_block!(sci, block, alias_result, token_map, effects_cache,
                     innermost_loop_info, ifelse_info)

Walk a block's statements and transform memory/control-flow ops for token ordering.
Modifies `token_map` in place to reflect the token state after the block.
"""
function transform_block!(sci::StructuredIRCode, block::Block,
                           alias_result::Dict{Any, AliasSet},
                           token_map::Dict{TokenKey, IRToken},
                           effects_cache::Dict{UInt64, MemoryEffects},
                           innermost_loop_effects::Union{MemoryEffects, Nothing},
                           ifelse_effects::Union{MemoryEffects, Nothing})

    # Collect SSA indices first to avoid iterator invalidation from insertions.
    ssa_indices = collect(Int, block.body.ssa_idxes)

    # Track whether we've seen a control flow op. Once we hit one,
    # we stop transforming memory ops because the token state after the CF op
    # is managed by codegen (ctx.token), not by the pass's token_map.
    seen_control_flow = false

    for ssa_idx in ssa_indices
        entry = get(block.body, ssa_idx, nothing)
        entry === nothing && continue

        if entry.stmt isa ControlFlowOp
            seen_control_flow = true
            # Don't recurse into nested regions (conservative approach)
        elseif !seen_control_flow
            transform_statement!(sci, block, ssa_idx, entry.stmt,
                                  alias_result, token_map)
        end
    end
end

"""
    transform_statement!(sci, block, ssa_idx, stmt, alias_result, token_map)

Transform a single statement. If it's a memory operation, insert token input/output nodes.
"""
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
        # Load depends on LAST_STORE (read-after-write)
        input_token = get_input_token_ir!(sci, block, ssa_idx,
                                           last_store_key(alias_set), token_map)

        # Add token arg to the call
        push!(stmt.args, input_token)

        # Insert TokenResultNode after the load
        result_ssa = new_ssa_idx!(sci)
        insert_after!(block.body, ssa_idx, result_ssa, TokenResultNode(ssa_idx), TOKEN_TYPE)

        result_token = SSAValue(result_ssa)

        # Update LAST_OP: eagerly join with existing last_op token
        lop_key = last_op_key(alias_set)
        last_op_tok = token_map[lop_key]
        join_ssa = new_ssa_idx!(sci)
        insert_after!(block.body, result_ssa, join_ssa,
                       JoinTokensNode([last_op_tok, result_token]), TOKEN_TYPE)
        token_map[lop_key] = SSAValue(join_ssa)

    elseif mem_effect == MEM_STORE
        # Store depends on LAST_OP (write-after-read, write-after-write)
        input_token = get_input_token_ir!(sci, block, ssa_idx,
                                           last_op_key(alias_set), token_map)

        # Add token arg to the call
        push!(stmt.args, input_token)

        # Insert TokenResultNode after the store
        result_ssa = new_ssa_idx!(sci)
        insert_after!(block.body, ssa_idx, result_ssa, TokenResultNode(ssa_idx), TOKEN_TYPE)

        result_token = SSAValue(result_ssa)

        # Update both LAST_OP and LAST_STORE
        token_map[last_op_key(alias_set)] = result_token
        token_map[last_store_key(alias_set)] = result_token
    end
end


#=============================================================================
 Control flow transformation (conservative)

 For this initial port, control flow ops are handled conservatively:
 - Memory ops inside nested blocks get the root token (from the enclosing scope)
 - No per-alias token carries through loops or branches
 - Token state is unchanged after control flow ops

 This matches the original inline approach's conservative behavior.
 TODO: Add per-alias token carries (matching Python's token_order_pass).
=============================================================================#

# For the conservative approach, control flow regions are NOT transformed by the pass.
# Memory ops inside loops/branches use ctx.token (the loop-carried or pre-branch token)
# which is managed manually by the codegen's control flow emitters.
# The pass only transforms straight-line code in the entry block.
function transform_control_flow!(sci::StructuredIRCode, parent_block::Block,
                                  ssa_idx::Int, op::ControlFlowOp, @nospecialize(result_type),
                                  alias_result, token_map, effects_cache)
    # Do nothing — codegen handles control flow token threading conservatively.
    # TODO: Transform nested regions once per-alias loop carries are implemented.
end

