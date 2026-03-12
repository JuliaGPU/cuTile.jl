"""
    get_input_var(args) -> Any

Extract the pointer/array variable from memory operation arguments.
"""
function get_input_var(args)
    return args[1]
end

"""
    get_alias_set(ctx::CGCtx, var) -> AliasSet

Get the alias set for a variable from analysis results.
"""
function get_alias_set(ctx::CGCtx, var)
    # Trace to source
    source = trace_to_source(ctx, var)

    # Lookup in alias results
    return get(ctx.alias_result, source, ALIAS_UNIVERSE)
end

"""
    collect_join_tokens(ctx::CGCtx, token_key::TokenKey, memory_order=nothing) -> Vector{Value}

Collect all tokens that need to be joined for synchronization.
Based on Python's `_collect_join_tokens`.
"""
function collect_join_tokens(ctx::CGCtx, token_key::TokenKey, memory_order = nothing)
    tokens_to_join = [ctx.token_map[token_key]]

    for (other_key, other_token) in ctx.token_map
        should_join = false

        # Join with ACQUIRE token
        if other_key isa AcquireTokenKey
            should_join = true

            # Join if alias sets overlap
        elseif other_key isa AliasTokenKey && token_key isa AliasTokenKey
            # Release memory order: join with all LAST_OP tokens
            if memory_order !== nothing && has_release_order(memory_order)
                should_join = other_key.role == LAST_OP
            end

            # Alias set overlap: join if same role and sets overlap
            if other_key.role == token_key.role
                alias_overlap = !(other_key.alias_set isa AliasUniverse) &&
                    !(token_key.alias_set isa AliasUniverse) &&
                    !isempty(intersect(other_key.alias_set, token_key.alias_set))
                should_join = should_join || alias_overlap
            end
        end

        if should_join && !(other_token in tokens_to_join)
            push!(tokens_to_join, other_token)
        end
    end

    return tokens_to_join
end

"""
    get_input_token!(ctx::CGCtx, token_key::TokenKey, memory_order=nothing) 
        -> (Value, Union{Nothing, JoinOp})

Get the input token for an operation, potentially creating a join operation.
"""
function get_input_token!(ctx::CGCtx, token_key::TokenKey, memory_order = nothing)
    
    if !haskey(ctx.token_map, token_key)
        @warn "Token key not found in token_map!" token_key available_keys=keys(ctx.token_map)
        # Fallback to root token
        return (ctx.token_map[ACQUIRE_TOKEN_KEY], nothing)
    end
    tokens_to_join = collect_join_tokens(ctx, token_key, memory_order)

    if length(tokens_to_join) == 1
        return (tokens_to_join[1], nothing)
    end

    # Join multiple tokens
    result_token = encode_JoinTokensOp!(ctx.cb, ctx.token_type, tokens_to_join)

    return (result_token, nothing)  # Return nothing for join_op since its already been emitted
end

"""
    trace_to_source(ctx::CGCtx, var) -> Any

Trace a value back to its original source (Argument, SSAValue).
"""
function trace_to_source(ctx::CGCtx, var)
    # Returns if its an Argument or SSAValue
    if var isa Argument || var isa SSAValue
        return var
    end

    # Resolve for SlothNumber
    if var isa SlotNumber
        tv = get(ctx.slots, var.id, nothing)
        if tv !== nothing && is_arg_ref(tv)
            arg_idx, _ = tv.arg_ref
            return Argument(arg_idx)
        end
    end

    # Generic emit_value resolution
    tv = emit_value!(ctx, var)
    if tv !== nothing && is_arg_ref(tv)
        arg_idx, _ = tv.arg_ref
        return Argument(arg_idx)
    end

    # Return as is for unknown
    return var
end

"""
    has_release_order(memory_order) -> Bool

Check if memory order has release semantics.
For now, returns false (no memory order support yet).
"""
function has_release_order(memory_order)
    # TODO: Implement proper memory order checking when needed
    return false
end
