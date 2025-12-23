# Codegen: Julia IR -> Tile IR bytecode

#=============================================================================
 Public API
=============================================================================#

export code_tiled

"""
    emit_tileir(f, argtypes; name=nothing) -> Vector{UInt8}

Compile a Julia function to Tile IR bytecode.
"""
function emit_tileir(@nospecialize(f), @nospecialize(argtypes);
                     name::Union{String, Nothing} = nothing)
    target = TileTarget(f, argtypes)
    kernel_name = name === nothing ? string(target.mi.def.name) : name

    buf = write_bytecode!(1) do writer, func_buf
        emit_kernel!(writer, func_buf, target; name=kernel_name)
    end

    return buf
end

const cuda_tile_translate = @load_preference("cuda_tile_translate_path", "cuda-tile-translate")

function disassemble_tileir(bytecode::Vector{UInt8})::String
    mktempdir() do dir
        input_path = joinpath(dir, "kernel.tile")
        output_path = joinpath(dir, "kernel.disasm")
        write(input_path, bytecode)
        read(`$cuda_tile_translate --cudatilebc-to-mlir $input_path`, String)
    end
end

"""
    code_tiled(f, argtypes; name=nothing) -> String

Return the CUDA Tile IR for a Julia function as a textual MLIR representation.
Analogous to `code_typed` or `code_structured`.
"""
function code_tiled(@nospecialize(f), @nospecialize(argtypes);
                   name::Union{String, Nothing} = nothing)
    bytecode = emit_tileir(f, argtypes; name)
    disassemble_tileir(bytecode)
end


#=============================================================================
 Kernel Entry Point
=============================================================================#

"""
    emit_kernel!(writer, func_buf, target; name, is_entry=true)

Compile a TileTarget to Tile IR bytecode.
"""
function emit_kernel!(writer::BytecodeWriter, func_buf::Vector{UInt8},
                      target::TileTarget;
                      name::String = string(target.mi.def.name),
                      is_entry::Bool = true)
    ctx = CodegenContext(writer, target)
    tt = ctx.tt

    # Validate argument types are concrete
    for (i, argtype) in enumerate(target.argtypes)
        require_concrete_type(argtype, "kernel argument $i")
    end

    # Build parameter list, handling ghost types and struct destructuring
    param_types = TypeId[]
    param_mapping = Tuple{Int, Union{Nothing, Symbol}}[]

    for (i, argtype) in enumerate(target.argtypes)
        argtype_unwrapped = unwrap_type(argtype)
        if is_ghost_type(argtype_unwrapped)
            continue
        elseif should_destructure(argtype_unwrapped)
            # Destructure TileArray into flat parameters
            params = argtype_unwrapped.parameters
            ndims = params[2]::Integer
            for fi in 1:fieldcount(argtype_unwrapped)
                fname = fieldname(argtype_unwrapped, fi)
                ftype = fieldtype(argtype_unwrapped, fi)
                if fname === :sizes || fname === :strides
                    fcount = ndims
                    elem_type = Int32
                else
                    fcount = flat_field_count(ftype)
                    elem_type = ftype <: Ptr ? Ptr{params[1]} : (ftype <: Tuple ? eltype(ftype) : ftype)
                end
                for _ in 1:fcount
                    push!(param_types, tile_type_for_julia!(ctx, elem_type))
                    push!(param_mapping, (i, fname))
                end
            end
            ctx.arg_types[i] = argtype_unwrapped
        else
            push!(param_types, tile_type_for_julia!(ctx, argtype_unwrapped))
            push!(param_mapping, (i, nothing))
        end
    end

    # Return types
    result_types = TypeId[]
    if target.rettype !== Nothing && target.rettype !== Union{}
        push!(result_types, tile_type_for_julia!(ctx, target.rettype))
    end

    # Create function
    cb = add_function!(writer, func_buf, name, param_types, result_types; is_entry)
    ctx.cb = cb

    # Set up argument values
    arg_values = make_block_args!(cb, length(param_types))

    # Build arg_flat_values map
    field_values = Dict{Tuple{Int, Union{Nothing, Symbol}}, Vector{Value}}()
    for (param_idx, val) in enumerate(arg_values)
        key = param_mapping[param_idx]
        if !haskey(field_values, key)
            field_values[key] = Value[]
        end
        push!(field_values[key], val)
    end

    # Store in context and set up slot/argument mappings
    for (key, values) in field_values
        arg_idx, field = key
        ctx.arg_flat_values[key] = values

        if field === nothing
            # Regular argument - create concrete CGVal
            @assert length(values) == 1
            val = values[1]
            type_id = tile_type_for_julia!(ctx, target.argtypes[arg_idx])
            tv = CGVal(val, type_id, target.argtypes[arg_idx])
            ctx[SlotNumber(arg_idx + 1)] = tv
            ctx[Argument(arg_idx + 1)] = tv
        end
    end

    # For destructured args, create lazy CGVals that track the argument index
    for (arg_idx, argtype) in ctx.arg_types
        tv = arg_ref_value(arg_idx, Union{Symbol, Int}[], argtype)
        ctx[SlotNumber(arg_idx + 1)] = tv
        ctx[Argument(arg_idx + 1)] = tv
    end

    # Create memory ordering token
    token_type = Token(tt)
    ctx.token_type = token_type
    ctx.token = encode_MakeTokenOp!(cb, token_type)

    # Lower to structured IR
    sci = StructuredCodeInfo(target.ci)
    structurize!(sci)

    # Emit the structured IR (uses original Julia SSA indices everywhere)
    emit_block!(ctx, sci.entry)

    finalize_function!(func_buf, cb, writer.debug_info)
end

#=============================================================================
 Structured IR Emission
=============================================================================#

"""
    result_count(T) -> Int

Compute the number of results from a Block.types entry.
Block.types contains Julia types:
- For Statement: Julia type → 1 result
- For ControlFlowOp with 0 results: Nothing → 0 results
- For ControlFlowOp with 1 result: Julia type → 1 result
- For ControlFlowOp with N results: Tuple{T1, T2, ...} → N results
"""
function result_count(@nospecialize(T))
    T === Nothing && return 0
    T <: Tuple && return length(T.parameters)
    return 1
end

"""
    emit_block!(ctx, block::Block)

Emit bytecode for a structured IR block.
All SSA values use original Julia SSA indices (no local renumbering).
Values are stored in ctx.values by their original index.
"""
function emit_block!(ctx::CodegenContext, block::Block; skip_terminator::Bool=false)
    # Emit body items (interleaved expressions and control flow ops)
    for (i, (expr, result_type)) in enumerate(zip(block.body, block.types))
        # Get original Julia SSA index for this item
        ssa_idx = !isempty(block.original_indices) ? block.original_indices[i] : i
        if expr isa ControlFlowOp
            n_results = result_count(result_type)
            emit_control_flow_op!(ctx, expr, result_type, n_results, ssa_idx)
        else
            emit_statement!(ctx, expr, ssa_idx, result_type)
        end
    end

    # Emit terminator (unless skipped)
    if !skip_terminator && block.terminator !== nothing
        emit_terminator!(ctx, block.terminator)
    end
end

"""
    emit_control_flow_op!(ctx, op::ControlFlowOp, result_type, n_results, original_idx)

Emit bytecode for a structured control flow operation.
Uses multiple dispatch on the concrete ControlFlowOp type.
Results are stored at indices assigned AFTER nested regions (DFS order).
original_idx is the original Julia SSA index for cross-block references.
"""
emit_control_flow_op!(ctx::CodegenContext, op::IfOp, @nospecialize(result_type), n_results::Int, original_idx::Int) =
    emit_if_op!(ctx, op, result_type, n_results, original_idx)
emit_control_flow_op!(ctx::CodegenContext, op::ForOp, @nospecialize(result_type), n_results::Int, original_idx::Int) =
    emit_for_op!(ctx, op, result_type, n_results, original_idx)
emit_control_flow_op!(ctx::CodegenContext, op::WhileOp, @nospecialize(result_type), n_results::Int, original_idx::Int) =
    emit_while_op!(ctx, op, result_type, n_results, original_idx)
emit_control_flow_op!(ctx::CodegenContext, op::LoopOp, @nospecialize(result_type), n_results::Int, original_idx::Int) =
    emit_loop_op!(ctx, op, result_type, n_results, original_idx)

function emit_if_op!(ctx::CodegenContext, op::IfOp, @nospecialize(parent_result_type), n_results::Int, ssa_idx::Int)
    cb = ctx.cb
    then_blk = op.then_region
    else_blk = op.else_region

    # Get condition value
    cond_tv = emit_value!(ctx, op.condition)
    cond_tv === nothing && error("Cannot resolve condition for IfOp")

    # Determine result types from parent_result_type
    result_types = TypeId[]
    julia_result_types = Type[]
    if parent_result_type === Nothing
        # No results
    elseif parent_result_type <: Tuple
        for T in parent_result_type.parameters
            push!(result_types, tile_type_for_julia!(ctx, T))
            push!(julia_result_types, T)
        end
    else
        push!(result_types, tile_type_for_julia!(ctx, parent_result_type))
        push!(julia_result_types, parent_result_type)
    end
    n_user_results = length(result_types)
    # Add token type as additional result (for memory ordering)
    push!(result_types, ctx.token_type)

    # Save token before branches
    token_before = ctx.token

    # Emit IfOp with callback-based region building
    then_body = function(_)
        saved_block_args = copy(ctx.block_args)
        ctx.token = token_before  # Reset to pre-branch token
        emit_block!(ctx, then_blk)
        empty!(ctx.block_args)
        merge!(ctx.block_args, saved_block_args)
    end
    else_body = function(_)
        saved_block_args = copy(ctx.block_args)
        ctx.token = token_before  # Reset to pre-branch token
        emit_block!(ctx, else_blk)
        empty!(ctx.block_args)
        merge!(ctx.block_args, saved_block_args)
    end
    results = encode_IfOp!(then_body, else_body, cb, result_types, cond_tv.v)

    # Last result is the merged token from both branches
    ctx.token = results[end]

    # Store results by original Julia SSA index
    # For IfOp: results are stored at ssa_idx, ssa_idx+1, etc. (the merge phi positions)
    for i in 1:n_user_results
        tv = CGVal(results[i], result_types[i], julia_result_types[i])
        ctx.values[ssa_idx + i - 1] = tv
    end
end

function emit_for_op!(ctx::CodegenContext, op::ForOp, @nospecialize(parent_result_type), n_results::Int, ssa_idx::Int)
    cb = ctx.cb
    tt = ctx.tt
    body_blk = op.body

    # Get bounds values
    lower_tv = emit_value!(ctx, op.lower)
    upper_tv = emit_value!(ctx, op.upper)
    step_tv = emit_value!(ctx, op.step)
    iv_arg = op.iv_arg

    (lower_tv === nothing || upper_tv === nothing || step_tv === nothing) &&
        error("Cannot resolve ForOp bounds")

    # Get init values (iter_args are loop-carried values)
    init_values = Value[]
    for init_val in op.iter_args
        tv = emit_value!(ctx, init_val)
        (tv === nothing || tv.v === nothing) && error("Cannot resolve ForOp init value")
        push!(init_values, tv.v)
    end
    # Add token as additional init value (for memory ordering)
    push!(init_values, ctx.token)

    # Number of carries (iter_args) - these are the loop results
    n_carries = length(op.iter_args)

    # Determine result types from carries (body.args)
    result_types = TypeId[]
    for i in 1:n_carries
        body_arg = body_blk.args[i]
        type_id = tile_type_for_julia!(ctx, body_arg.type)
        push!(result_types, type_id)
    end
    # Add token type as additional result (for memory ordering)
    push!(result_types, ctx.token_type)

    # Number of user result types (excluding token)
    n_user_results = n_carries

    # Emit ForOp with callback-based region building
    body_builder = function(block_args)
        saved_block_args = copy(ctx.block_args)

        # Tile IR block args layout: [iv, carries..., token]
        # Julia IR body.args layout: [carries...]

        # Map the induction variable
        iv_type = tile_type!(tt, I32(tt), Int[])
        iv_tv = CGVal(block_args[1], iv_type, Int32)
        ctx[iv_arg] = iv_tv

        # Map carried values (body.args)
        for i in 1:n_carries
            body_arg = body_blk.args[i]
            shape = extract_tile_shape(body_arg.type)
            tv = CGVal(block_args[i + 1], result_types[i], body_arg.type, shape)
            ctx[body_arg] = tv
        end

        # Set token from last block arg
        ctx.token = block_args[end]

        emit_block!(ctx, body_blk)

        empty!(ctx.block_args)
        merge!(ctx.block_args, saved_block_args)
    end
    results = encode_ForOp!(body_builder, cb, result_types, lower_tv.v, upper_tv.v, step_tv.v, init_values)

    # Last result is the token
    ctx.token = results[end]

    # Store results by original Julia SSA index
    # For ForOp: ssa_idx is where the IV phi was, iter_arg results are at ssa_idx+1, ssa_idx+2, ...
    for i in 1:n_user_results
        type_id = tile_type_for_julia!(ctx, body_blk.args[i].type)
        shape = extract_tile_shape(body_blk.args[i].type)
        tv = CGVal(results[i], type_id, body_blk.args[i].type, shape)
        ctx.values[ssa_idx + i] = tv
    end
end

function emit_loop_op!(ctx::CodegenContext, op::LoopOp, @nospecialize(parent_result_type), n_results::Int, ssa_idx::Int)
    cb = ctx.cb
    body_blk = op.body

    # Get init values (iter_args are loop-carried values)
    init_values = Value[]
    for init_val in op.iter_args
        tv = emit_value!(ctx, init_val)
        (tv === nothing || tv.v === nothing) && error("Cannot resolve LoopOp init value")
        push!(init_values, tv.v)
    end
    # Add token as additional init value (for memory ordering)
    push!(init_values, ctx.token)

    # Number of carries (iter_args) - these are the loop results
    n_carries = length(op.iter_args)

    # Determine result types from carries (body.args)
    result_types = TypeId[]
    for i in 1:n_carries
        body_arg = body_blk.args[i]
        type_id = tile_type_for_julia!(ctx, body_arg.type)
        push!(result_types, type_id)
    end
    # Add token type as additional result (for memory ordering)
    push!(result_types, ctx.token_type)

    # Number of user result types (excluding token)
    n_user_results = n_carries

    # Emit LoopOp with callback-based region building
    body_builder = function(block_args)
        saved_block_args = copy(ctx.block_args)

        # Tile IR block args layout: [carries..., token]
        # Julia IR body.args layout: [carries...]

        # Map carried values (body.args)
        for i in 1:n_carries
            body_arg = body_blk.args[i]
            shape = extract_tile_shape(body_arg.type)
            ctx[body_arg] = CGVal(block_args[i], result_types[i], body_arg.type, shape)
        end

        # Set token from last block arg
        ctx.token = block_args[end]

        emit_block!(ctx, body_blk)

        # In Tile IR, if the loop body ends with an IfOp (even one with continue/break
        # in all branches), the if is NOT a terminator. We need an explicit terminator
        # after the if. Add an unreachable ContinueOp as fallback terminator.
        if body_blk.terminator === nothing
            fallback_operands = copy(block_args)
            fallback_operands[end] = ctx.token
            encode_ContinueOp!(ctx.cb, fallback_operands)
        end

        empty!(ctx.block_args)
        merge!(ctx.block_args, saved_block_args)
    end
    results = encode_LoopOp!(body_builder, cb, result_types, init_values)

    # Last result is the token
    ctx.token = results[end]

    # Store results by original Julia SSA index
    for i in 1:n_user_results
        type_id = tile_type_for_julia!(ctx, body_blk.args[i].type)
        shape = extract_tile_shape(body_blk.args[i].type)
        tv = CGVal(results[i], type_id, body_blk.args[i].type, shape)
        ctx.values[ssa_idx + i - 1] = tv
    end
end

function emit_while_op!(ctx::CodegenContext, op::WhileOp, @nospecialize(parent_result_type), n_results::Int, ssa_idx::Int)
    cb = ctx.cb
    before_blk = op.before
    after_blk = op.after

    # Get init values (iter_args are loop-carried values)
    init_values = Value[]
    for init_val in op.iter_args
        tv = emit_value!(ctx, init_val)
        (tv === nothing || tv.v === nothing) && error("Cannot resolve WhileOp init value: $init_val")
        push!(init_values, tv.v)
    end
    # Add token as additional init value (for memory ordering)
    push!(init_values, ctx.token)

    # Number of carries (iter_args) - these are the loop results
    n_carries = length(op.iter_args)

    # Determine result types from carries (before.args)
    result_types = TypeId[]
    for i in 1:n_carries
        before_arg = before_blk.args[i]
        type_id = tile_type_for_julia!(ctx, before_arg.type)
        push!(result_types, type_id)
    end
    # Add token type as additional result (for memory ordering)
    push!(result_types, ctx.token_type)

    # Number of user result types (excluding token)
    n_user_results = n_carries

    # Emit WhileOp as cuda_tile.loop with if-continue-break pattern
    # MLIR structure: before { stmts; condition(cond) args } do { stmts; yield vals }
    # Emitted as: loop { before_stmts; if(cond) { after_stmts; continue } else { break } }
    body_builder = function(block_args)
        saved_block_args = copy(ctx.block_args)

        # Tile IR block args layout: [carries..., token]
        # Julia IR before.args layout: [carries...]

        # Map carried values (before.args)
        for i in 1:n_carries
            before_arg = before_blk.args[i]
            shape = extract_tile_shape(before_arg.type)
            ctx[before_arg] = CGVal(block_args[i], result_types[i], before_arg.type, shape)
        end

        # Set token from last block arg
        ctx.token = block_args[end]

        # Emit "before" region
        emit_block!(ctx, before_blk)

        # Get condition from ConditionOp terminator
        cond_op = before_blk.terminator
        cond_op isa ConditionOp || error("WhileOp before region must end with ConditionOp")

        cond_tv = emit_value!(ctx, cond_op.condition)
        (cond_tv === nothing || cond_tv.v === nothing) && error("Cannot resolve WhileOp condition: $(cond_op.condition)")

        # Create if-then-else: if condition { after_region; continue } else { break }
        then_body = function(_)
            # Map "after" region block args - carries from ConditionOp.args
            for i in 1:n_carries
                after_arg = after_blk.args[i]
                if i <= length(cond_op.args)
                    tv = emit_value!(ctx, cond_op.args[i])
                    if tv !== nothing
                        ctx[after_arg] = tv
                    else
                        shape = extract_tile_shape(after_arg.type)
                        ctx[after_arg] = CGVal(block_args[i], result_types[i], after_arg.type, shape)
                    end
                end
            end

            # Emit "after" region body (skip terminator - we emit ContinueOp instead)
            emit_block!(ctx, after_blk; skip_terminator=true)

            # Emit ContinueOp with yield values from after region's YieldOp
            continue_operands = Value[]
            if after_blk.terminator isa YieldOp
                for val in after_blk.terminator.values
                    tv = emit_value!(ctx, val)
                    tv !== nothing && tv.v !== nothing && push!(continue_operands, tv.v)
                end
            end
            push!(continue_operands, ctx.token)
            encode_ContinueOp!(ctx.cb, continue_operands)
        end

        else_body = function(_)
            # Break with ConditionOp args (become loop results)
            break_operands = Value[]
            for arg in cond_op.args
                tv = emit_value!(ctx, arg)
                tv !== nothing && tv.v !== nothing && push!(break_operands, tv.v)
            end
            if isempty(break_operands)
                for i in 1:n_carries
                    push!(break_operands, block_args[i])
                end
            end
            push!(break_operands, ctx.token)
            encode_BreakOp!(ctx.cb, break_operands)
        end

        # Emit IfOp with NO results - continue/break are terminators to the enclosing loop
        # They don't yield values back to the IfOp
        if_result_types = TypeId[]
        encode_IfOp!(then_body, else_body, cb, if_result_types, cond_tv.v)

        # Add fallback ContinueOp as block terminator (required even though IfOp covers all paths)
        # The IfOp is not considered a terminator in CUDA Tile bytecode format
        fallback_operands = copy(block_args)
        # Token was updated by continue/break inside the branches, use the last known token
        encode_ContinueOp!(ctx.cb, fallback_operands)

        empty!(ctx.block_args)
        merge!(ctx.block_args, saved_block_args)
    end
    results = encode_LoopOp!(body_builder, cb, result_types, init_values)

    # Last result is the token
    ctx.token = results[end]

    # Store results by original Julia SSA index
    for i in 1:n_user_results
        type_id = tile_type_for_julia!(ctx, before_blk.args[i].type)
        shape = extract_tile_shape(before_blk.args[i].type)
        tv = CGVal(results[i], type_id, before_blk.args[i].type, shape)
        ctx.values[ssa_idx + i - 1] = tv
    end
end

"""
    emit_terminator!(ctx, terminator)

Emit bytecode for a block terminator.
"""
function emit_terminator!(ctx::CodegenContext, node::ReturnNode)
    emit_return!(ctx, node)
end

function emit_terminator!(ctx::CodegenContext, op::YieldOp)
    # Collect yield operands
    operands = Value[]
    for val in op.values
        tv = emit_value!(ctx, val)
        @debug "YieldOp value $val => $(tv !== nothing ? tv : "nothing")"
        tv !== nothing && tv.v !== nothing && push!(operands, tv.v)
    end
    # Append current token for memory ordering
    push!(operands, ctx.token)
    @debug "YieldOp operands: $operands ($(length(operands)) values)"
    encode_YieldOp!(ctx.cb, operands)
end

function emit_terminator!(ctx::CodegenContext, op::ContinueOp)
    # Collect continue operands (updated carried values)
    operands = Value[]
    for val in op.values
        tv = emit_value!(ctx, val)
        tv !== nothing && tv.v !== nothing && push!(operands, tv.v)
    end
    # Append current token for memory ordering
    push!(operands, ctx.token)
    encode_ContinueOp!(ctx.cb, operands)
end

function emit_terminator!(ctx::CodegenContext, op::BreakOp)
    # Collect break operands (final values)
    operands = Value[]
    for val in op.values
        tv = emit_value!(ctx, val)
        tv !== nothing && tv.v !== nothing && push!(operands, tv.v)
    end
    # Append current token for memory ordering
    push!(operands, ctx.token)
    encode_BreakOp!(ctx.cb, operands)
end

function emit_terminator!(ctx::CodegenContext, ::Nothing)
    # No terminator, nothing to emit
end

function emit_terminator!(ctx::CodegenContext, ::ConditionOp)
    # ConditionOp is handled specially by emit_while_op!, not emitted as a terminator
end


#=============================================================================
 Statement Emission
=============================================================================#

"""
    emit_statement!(ctx, stmt, ssa_idx, result_type)

Emit bytecode for a single SSA statement.
The ssa_idx is the original Julia SSA index to store the result at.
"""
function emit_statement!(ctx::CodegenContext, @nospecialize(stmt), ssa_idx::Int, @nospecialize(result_type))
    tv = nothing
    if stmt isa ReturnNode
        emit_return!(ctx, stmt)
    elseif stmt isa Expr
        tv = emit_expr!(ctx, stmt, result_type)
        if tv === nothing
            # If emit_expr! returns nothing, try emit_value! for ghost values like tuples
            tv = emit_value!(ctx, stmt)
        end
    elseif stmt isa GlobalRef
        tv = emit_value!(ctx, stmt)
    elseif stmt isa QuoteNode
        tv = emit_constant!(ctx, stmt.value, result_type)
    elseif stmt isa SlotNumber
        tv = ctx[stmt]
    elseif stmt isa PiNode
        # PiNode is a type narrowing assertion - store the resolved value
        tv = emit_value!(ctx, stmt)
    elseif stmt === nothing
        # No-op
    else
        @warn "Unhandled statement type" typeof(stmt) stmt
    end

    # Store result by original Julia SSA index
    if tv !== nothing
        ctx.values[ssa_idx] = tv
    end
end

"""
    emit_return!(ctx, node::ReturnNode)

Emit a return operation.
"""
function emit_return!(ctx::CodegenContext, node::ReturnNode)
    if !isdefined(node, :val) || node.val === nothing || (node.val isa GlobalRef && node.val.name === :nothing)
        encode_ReturnOp!(ctx.cb, Value[])
    else
        tv = emit_value!(ctx, node.val)
        if tv !== nothing && !is_ghost(tv)
            encode_ReturnOp!(ctx.cb, [tv.v])
        else
            encode_ReturnOp!(ctx.cb, Value[])
        end
    end
end

#=============================================================================
 Value Emission
=============================================================================#

"""
    emit_value!(ctx, ref) -> Union{CGVal, Nothing}

Emit/resolve a value reference to a CGVal using multiple dispatch.
"""
function emit_value!(ctx::CodegenContext, ssa::SSAValue)
    # First try to get from context (already processed)
    tv = ctx[ssa]
    tv !== nothing && return tv
    # For block-local SSAs (id > code length), they must be in ctx.values_stack
    code_stmts = code(ctx.target)
    ssa.id <= length(code_stmts) || error("Block-local SSAValue %$(ssa.id) not found in context")
    # Follow the SSA chain to the defining statement in CodeInfo
    stmt = code_stmts[ssa.id]
    emit_value!(ctx, stmt)
end
emit_value!(ctx::CodegenContext, arg::Argument) = ctx[arg]
emit_value!(ctx::CodegenContext, slot::SlotNumber) = ctx[slot]
emit_value!(ctx::CodegenContext, block_arg::BlockArg) = ctx[block_arg]

function emit_value!(ctx::CodegenContext, val::Integer)
    type_id = tile_type_for_julia!(ctx, Int32)
    bytes = reinterpret(UInt8, [Int32(val)])
    v = encode_ConstantOp!(ctx.cb, type_id, collect(bytes))
    CGVal(v, type_id, Int32, Int[], nothing, val)  # Preserve original type in constant
end

function emit_value!(ctx::CodegenContext, val::AbstractFloat)
    jltype = typeof(val)
    type_id = tile_type_for_julia!(ctx, jltype)
    bytes = reinterpret(UInt8, [jltype(val)])
    v = encode_ConstantOp!(ctx.cb, type_id, collect(bytes))
    CGVal(v, type_id, jltype, Int[], nothing, val)
end

function emit_value!(ctx::CodegenContext, node::QuoteNode)
    emit_value!(ctx, node.value)
end

function emit_value!(ctx::CodegenContext, expr::Expr)
    # Try to extract constant from Expr (e.g., call to tuple or type assert)
    if expr.head === :call
        callee = expr.args[1]
        # Handle Core.tuple - extract elements as constant tuple
        if callee isa GlobalRef && callee.mod === Core && callee.name === :tuple
            elements = Any[]
            for arg in expr.args[2:end]
                tv = emit_value!(ctx, arg)
                tv === nothing && return nothing
                tv.constant === nothing && return nothing
                push!(elements, tv.constant)
            end
            return ghost_value(typeof(Tuple(elements)), Tuple(elements))
        end
    end
    nothing
end

function emit_value!(ctx::CodegenContext, ref::GlobalRef)
    val = getfield(ref.mod, ref.name)
    ghost_value(typeof(val), val)
end

function emit_value!(ctx::CodegenContext, node::PiNode)
    # PiNode narrows the type. If the narrowed type is Type{T}, extract T
    if node.typ isa Type{<:Type}
        return ghost_value(node.typ, node.typ.parameters[1])
    end
    # Otherwise emit the underlying value
    emit_value!(ctx, node.val)
end

emit_value!(ctx::CodegenContext, ::Nothing) = nothing

"""
    get_constant(ctx, ref) -> Union{Any, Nothing}

Get the compile-time constant from an IR reference, or nothing if not available.
"""
function get_constant(ctx::CodegenContext, @nospecialize(ref))
    tv = emit_value!(ctx, ref)
    tv === nothing ? nothing : tv.constant
end

# Symbols are compile-time only values
emit_value!(ctx::CodegenContext, val::Symbol) = ghost_value(Symbol, val)

# Tuples are compile-time only values
emit_value!(ctx::CodegenContext, val::Tuple) = ghost_value(typeof(val), val)

# Types are compile-time only values
emit_value!(ctx::CodegenContext, @nospecialize(val::Type)) = ghost_value(Type{val}, val)

# Fallback for other types (constants embedded in IR)
function emit_value!(ctx::CodegenContext, @nospecialize(val))
    T = typeof(val)
    # Handle Val{V} instances
    if T <: Val && length(T.parameters) == 1
        return ghost_value(T, T.parameters[1])
    end
    # Handle Constant{T, V} instances
    if T <: Constant && length(T.parameters) >= 2
        return ghost_value(T, T.parameters[2])
    end
    @warn "Unhandled value type in emit_value!" typeof(val)
    nothing
end

#=============================================================================
 Expression Emission
=============================================================================#

"""
    emit_expr!(ctx, expr::Expr, result_type) -> Union{CGVal, Nothing}

Emit bytecode for an expression.
"""
function emit_expr!(ctx::CodegenContext, expr::Expr, @nospecialize(result_type))
    if expr.head === :call
        return emit_call!(ctx, expr, result_type)
    elseif expr.head === :invoke
        return emit_invoke!(ctx, expr, result_type)
    elseif expr.head === :(=)
        return emit_assignment!(ctx, expr, result_type)
    elseif expr.head === :new
        return nothing  # Struct construction handled elsewhere
    elseif expr.head === :foreigncall
        error("Foreign calls not supported in Tile IR")
    elseif expr.head === :boundscheck
        return nothing
    else
        @warn "Unhandled expression head" expr.head expr
        return nothing
    end
end

function emit_assignment!(ctx::CodegenContext, expr::Expr, @nospecialize(result_type))
    lhs = expr.args[1]
    rhs = expr.args[2]

    tv = emit_rhs!(ctx, rhs, result_type)

    if lhs isa SlotNumber && tv !== nothing
        ctx[lhs] = tv
    end

    return tv
end

function emit_rhs!(ctx::CodegenContext, @nospecialize(rhs), @nospecialize(result_type))
    if rhs isa Expr
        return emit_expr!(ctx, rhs, result_type)
    elseif rhs isa SSAValue || rhs isa SlotNumber || rhs isa Argument
        return emit_value!(ctx, rhs)
    elseif rhs isa QuoteNode
        return emit_constant!(ctx, rhs.value, result_type)
    elseif rhs isa GlobalRef
        return nothing
    else
        return emit_constant!(ctx, rhs, result_type)
    end
end

#=============================================================================
 Call Emission
=============================================================================#

"""
    emit_call!(ctx, expr::Expr, result_type) -> Union{CGVal, Nothing}

Emit bytecode for a function call.
"""
function emit_call!(ctx::CodegenContext, expr::Expr, @nospecialize(result_type))
    args = expr.args
    func = resolve_function(ctx, args[1])

    call_args = args[2:end]
    result = emit_intrinsic!(ctx, func, call_args, result_type)
    result === missing && error("Unknown function call: $func")
    return result
end

"""
    emit_invoke!(ctx, expr::Expr, result_type) -> Union{CGVal, Nothing}

Emit bytecode for a method invocation.
"""
function emit_invoke!(ctx::CodegenContext, expr::Expr, @nospecialize(result_type))
    # invoke has: (MethodInstance, func, args...)
    func = resolve_function(ctx, expr.args[2])
    call_args = expr.args[3:end]

    result = emit_intrinsic!(ctx, func, call_args, result_type)
    result === missing && error("Unknown function call: $func")
    return result
end

"""
    resolve_function(ctx, ref) -> Any

Resolve a function reference to its actual value.
"""
function resolve_function(ctx::CodegenContext, @nospecialize(ref))
    if ref isa GlobalRef
        val = getfield(ref.mod, ref.name)
        # If it's a module, return it for chained lookups
        return val
    elseif ref isa QuoteNode
        return ref.value
    elseif ref isa SSAValue
        stmt = code(ctx.target)[ref.id]
        if stmt isa GlobalRef
            return getfield(stmt.mod, stmt.name)
        elseif stmt isa QuoteNode
            return stmt.value
        end
    end
    return ref
end


