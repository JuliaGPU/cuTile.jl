# Codegen: Julia IR -> Tile IR bytecode

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
            # Regular argument - create concrete TileValue
            @assert length(values) == 1
            val = values[1]
            type_id = tile_type_for_julia!(ctx, target.argtypes[arg_idx])
            tv = TileValue(val, type_id, target.argtypes[arg_idx])
            ctx[SlotNumber(arg_idx + 1)] = tv
            ctx[Argument(arg_idx + 1)] = tv
        end
    end

    # For destructured args, create lazy TileValues that track the argument index
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
    structured_ir = lower_to_structured_ir(target)

    # Emit the structured IR
    emit_block!(ctx, structured_ir.entry)

    finalize_function!(func_buf, cb, writer.debug_info)
end

#=============================================================================
 Structured IR Emission
=============================================================================#

"""
    emit_block!(ctx, block::Block)

Emit bytecode for a structured IR block.
"""
function emit_block!(ctx::CodegenContext, block::Block)
    code_stmts = code(ctx.target)
    types = ssatypes(ctx.target)

    # Emit body items (interleaved statements and control flow ops)
    for item in block.body
        if item isa Int
            stmt = code_stmts[item]
            result_type = types[item]
            emit_statement!(ctx, stmt, item, result_type)
        else
            emit_control_flow_op!(ctx, item)
        end
    end

    # Emit terminator
    if block.terminator !== nothing
        emit_terminator!(ctx, block.terminator)
    end
end

"""
    emit_control_flow_op!(ctx, op::ControlFlowOp)

Emit bytecode for a structured control flow operation.
"""
function emit_control_flow_op!(ctx::CodegenContext, op::IfOp)
    cb = ctx.cb

    # Get condition value
    cond_tv = emit_value!(ctx, op.condition)
    cond_tv === nothing && error("Cannot resolve condition for IfOp")

    # Determine result types from the result_vars
    result_types = TypeId[]
    for result_var in op.result_vars
        result_type = ssatypes(ctx.target)[result_var.id]
        type_id = tile_type_for_julia!(ctx, result_type)
        push!(result_types, type_id)
    end
    # Add token type as additional result (for memory ordering)
    push!(result_types, ctx.token_type)

    # Save token before branches
    token_before = ctx.token

    # Emit IfOp with callback-based region building
    # Each branch will yield its final token via emit_terminator!
    then_body = function(_)
        ctx.token = token_before  # Reset to pre-branch token
        emit_block!(ctx, op.then_block)
    end
    else_body = function(_)
        ctx.token = token_before  # Reset to pre-branch token
        emit_block!(ctx, op.else_block)
    end
    results = encode_IfOp!(then_body, else_body, cb, result_types, cond_tv.v)

    # Last result is the merged token from both branches
    ctx.token = results[end]

    # Map result values to SSA values (excluding the token)
    for (i, result_var) in enumerate(op.result_vars)
        if i <= length(results) - 1  # Exclude token
            result_type = ssatypes(ctx.target)[result_var.id]
            type_id = tile_type_for_julia!(ctx, result_type)
            ctx[result_var] = TileValue(results[i], type_id, result_type)
        end
    end
end

function emit_control_flow_op!(ctx::CodegenContext, op::ForOp)
    cb = ctx.cb
    tt = ctx.tt

    # Get bounds values
    lower_tv = emit_value!(ctx, op.lower)
    upper_tv = emit_value!(ctx, op.upper)
    step_tv = emit_value!(ctx, op.step)

    (lower_tv === nothing || upper_tv === nothing || step_tv === nothing) &&
        error("Cannot resolve ForOp bounds")

    # Get init values
    init_values = Value[]
    for init_val in op.init_values
        tv = emit_value!(ctx, init_val)
        (tv === nothing || tv.v === nothing) && error("Cannot resolve ForOp init value")
        push!(init_values, tv.v)
    end
    # Add token as additional init value (for memory ordering)
    push!(init_values, ctx.token)

    # Determine result types
    result_types = TypeId[]
    for result_var in op.result_vars
        result_type = ssatypes(ctx.target)[result_var.id]
        type_id = tile_type_for_julia!(ctx, result_type)
        @debug "ForOp result_var $result_var: type=$result_type, type_id=$type_id"
        push!(result_types, type_id)
    end
    # Add token type as additional result (for memory ordering)
    push!(result_types, ctx.token_type)
    @debug "ForOp result_types: $result_types ($(length(result_types)) types)"

    # Number of user result types (excluding token)
    n_user_results = length(op.result_vars)

    # Emit ForOp with callback-based region building
    body_builder = function(block_args)
        # Map block arguments (induction var + carried values + token)
        if !isempty(op.body.args)
            # First block arg is induction variable
            iv_arg = op.body.args[1]
            iv_type = tile_type!(tt, I32(tt), Int[])
            iv_tv = TileValue(block_args[1], iv_type, Int32)
            ctx[iv_arg] = iv_tv
            # Also map the induction variable phi SSAValue
            ctx[op.iv_ssa] = iv_tv

            # Carried values are block_args[2:end-1] (last is token)
            for (i, body_arg) in enumerate(op.body.args[2:end])
                if i <= length(block_args) - 2 && i <= n_user_results  # -2 for iv and token
                    shape = extract_tile_shape(body_arg.type)
                    tv = TileValue(block_args[i+1], result_types[i], body_arg.type, shape)
                    ctx[body_arg] = tv
                    # Also map the phi SSAValue so body statements can reference it
                    if i <= length(op.result_vars)
                        ctx[op.result_vars[i]] = tv
                    end
                end
            end
        end

        # Set token from last block arg
        ctx.token = block_args[end]

        emit_block!(ctx, op.body)
    end
    results = encode_ForOp!(body_builder, cb, result_types, lower_tv.v, upper_tv.v, step_tv.v, init_values)

    # Last result is the token
    ctx.token = results[end]

    # Map user result values (excluding token)
    for (i, result_var) in enumerate(op.result_vars)
        if i <= length(results) - 1  # Exclude token
            result_type = ssatypes(ctx.target)[result_var.id]
            type_id = tile_type_for_julia!(ctx, result_type)
            shape = extract_tile_shape(result_type)
            ctx[result_var] = TileValue(results[i], type_id, result_type, shape)
        end
    end
end

function emit_control_flow_op!(ctx::CodegenContext, op::LoopOp)
    cb = ctx.cb

    # Get init values
    init_values = Value[]
    for init_val in op.init_values
        tv = emit_value!(ctx, init_val)
        (tv === nothing || tv.v === nothing) && error("Cannot resolve LoopOp init value")
        push!(init_values, tv.v)
    end
    # Add token as additional init value (for memory ordering)
    push!(init_values, ctx.token)

    # Determine result types from init values
    result_types = TypeId[]
    for result_var in op.result_vars
        result_type = ssatypes(ctx.target)[result_var.id]
        type_id = tile_type_for_julia!(ctx, result_type)
        push!(result_types, type_id)
    end
    # Add token type as additional result (for memory ordering)
    push!(result_types, ctx.token_type)

    # Number of user result types (excluding token)
    n_user_results = length(op.result_vars)

    # Emit LoopOp with callback-based region building
    body_builder = function(block_args)
        # Map block arguments (carried values, last is token)
        for (i, body_arg) in enumerate(op.body.args)
            if i <= length(block_args) - 1 && i <= n_user_results  # -1 for token
                shape = extract_tile_shape(body_arg.type)
                ctx[body_arg] = TileValue(block_args[i], result_types[i], body_arg.type, shape)
            end
        end

        # Also map result_vars to block args so references inside loop body work
        # e.g., if %2 is a phi that becomes result_var, references to %2 inside the
        # loop body should resolve to the block argument value
        for (i, result_var) in enumerate(op.result_vars)
            if i <= length(block_args) - 1 && i <= n_user_results  # -1 for token
                result_type = ssatypes(ctx.target)[result_var.id]
                shape = extract_tile_shape(result_type)
                ctx[result_var] = TileValue(block_args[i], result_types[i], result_type, shape)
            end
        end

        # Set token from last block arg
        ctx.token = block_args[end]

        emit_block!(ctx, op.body)

        # In Tile IR, if the loop body ends with an IfOp (even one with continue/break
        # in all branches), the if is NOT a terminator. We need an explicit terminator
        # after the if. Add an unreachable ContinueOp as fallback terminator.
        # This is only reached if the if doesn't cover all paths (which it should).
        if op.body.terminator === nothing
            # Include token in fallback continue
            fallback_operands = copy(block_args)
            fallback_operands[end] = ctx.token
            encode_ContinueOp!(ctx.cb, fallback_operands)
        end
    end
    results = encode_LoopOp!(body_builder, cb, result_types, init_values)

    # Last result is the token
    ctx.token = results[end]

    # Map user result values (excluding token)
    for (i, result_var) in enumerate(op.result_vars)
        if i <= length(results) - 1  # Exclude token
            result_type = ssatypes(ctx.target)[result_var.id]
            type_id = tile_type_for_julia!(ctx, result_type)
            shape = extract_tile_shape(result_type)
            ctx[result_var] = TileValue(results[i], type_id, result_type, shape)
        end
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


#=============================================================================
 Statement Emission
=============================================================================#

"""
    emit_statement!(ctx, stmt, idx, result_type)

Emit bytecode for a single SSA statement.
"""
function emit_statement!(ctx::CodegenContext, @nospecialize(stmt), idx::Int, @nospecialize(result_type))
    if stmt isa ReturnNode
        emit_return!(ctx, stmt)
    elseif stmt isa GotoNode
        # Skip - needs restructuring pass
    elseif stmt isa GotoIfNot
        error("Control flow not yet supported: GotoIfNot")
    elseif stmt isa Expr
        tv = emit_expr!(ctx, stmt, result_type)
        if tv !== nothing
            ctx[SSAValue(idx)] = tv
        end
    elseif stmt isa GlobalRef
        # Function references resolved when used
    elseif stmt isa QuoteNode
        tv = emit_constant!(ctx, stmt.value, result_type)
        if tv !== nothing
            ctx[SSAValue(idx)] = tv
        end
    elseif stmt isa SlotNumber
        slot_tv = ctx[stmt]
        if slot_tv !== nothing
            ctx[SSAValue(idx)] = slot_tv
        end
    elseif stmt isa PiNode
        # PiNode is a type narrowing assertion - handled via extract_constant
        # No bytecode emission needed
    elseif stmt === nothing
        # No-op
    else
        @warn "Unhandled statement type" typeof(stmt) stmt
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
    emit_value!(ctx, ref) -> Union{TileValue, Nothing}

Emit/resolve a value reference to a TileValue using multiple dispatch.
"""
function emit_value!(ctx::CodegenContext, ssa::SSAValue)
    # First try to get from context (already processed)
    tv = ctx[ssa]
    tv !== nothing && return tv
    # Otherwise follow the SSA chain to the defining statement
    stmt = code(ctx.target)[ssa.id]
    emit_value!(ctx, stmt)
end
emit_value!(ctx::CodegenContext, arg::Argument) = ctx[arg]
emit_value!(ctx::CodegenContext, slot::SlotNumber) = ctx[slot]
emit_value!(ctx::CodegenContext, block_arg::BlockArg) = ctx[block_arg]
emit_value!(ctx::CodegenContext, lit::Literal) = emit_value!(ctx, lit.value)

function emit_value!(ctx::CodegenContext, val::Integer)
    type_id = tile_type_for_julia!(ctx, Int32)
    bytes = reinterpret(UInt8, [Int32(val)])
    v = encode_ConstantOp!(ctx.cb, type_id, collect(bytes))
    TileValue(v, type_id, Int32, Int[], nothing, val)  # Preserve original type in constant
end

function emit_value!(ctx::CodegenContext, val::AbstractFloat)
    jltype = typeof(val)
    type_id = tile_type_for_julia!(ctx, jltype)
    bytes = reinterpret(UInt8, [jltype(val)])
    v = encode_ConstantOp!(ctx.cb, type_id, collect(bytes))
    TileValue(v, type_id, jltype, Int[], nothing, val)
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
    emit_expr!(ctx, expr::Expr, result_type) -> Union{TileValue, Nothing}

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
    emit_call!(ctx, expr::Expr, result_type) -> Union{TileValue, Nothing}

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
    emit_invoke!(ctx, expr::Expr, result_type) -> Union{TileValue, Nothing}

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
        elseif stmt isa Expr && stmt.head === :call
            # Handle getproperty: Base.getproperty(obj, :name) -> obj.name
            callee = stmt.args[1]
            if callee isa GlobalRef && callee.mod === Base && callee.name === :getproperty
                obj = resolve_function(ctx, stmt.args[2])
                prop = stmt.args[3]
                prop_name = prop isa QuoteNode ? prop.value : prop
                if obj isa Module && prop_name isa Symbol
                    return getfield(obj, prop_name)
                end
            end
        end
    end
    return ref
end

#=============================================================================
 Intrinsic Dispatch
=============================================================================#

# Generic fallback
emit_intrinsic!(ctx::CodegenContext, @nospecialize(func), args, @nospecialize(result_type)) = missing

# Skip tuple construction
emit_intrinsic!(ctx::CodegenContext, ::typeof(Core.tuple), args, @nospecialize(result_type)) = nothing

# Skip getproperty (module.field access, resolved elsewhere)
emit_intrinsic!(ctx::CodegenContext, ::typeof(Base.getproperty), args, @nospecialize(result_type)) = nothing

# Skip isa type assertions (inserted by Julia during inlining)
emit_intrinsic!(ctx::CodegenContext, ::typeof(isa), args, @nospecialize(result_type)) = nothing

#-----------------------------------------------------------------------------
# cuTile intrinsics
#-----------------------------------------------------------------------------

function emit_intrinsic!(ctx::CodegenContext, ::typeof(bid), args, @nospecialize(result_type))
    axis = @something get_constant(ctx, args[1]) error("bid() axis must be a compile-time constant")
    axis in (0, 1, 2) || error("bid() axis must be 0, 1, or 2, got $axis")

    res_type = tile_type!(ctx.tt, I32(ctx.tt), Int[])
    bid_x, bid_y, bid_z = encode_GetTileBlockIdOp!(ctx.cb, res_type, res_type, res_type)
    result = (bid_x, bid_y, bid_z)[axis + 1]

    TileValue(result, res_type, Int32)
end

function emit_intrinsic!(ctx::CodegenContext, ::typeof(num_blocks), args, @nospecialize(result_type))
    axis = @something get_constant(ctx, args[1]) error("num_blocks() axis must be a compile-time constant")
    axis in (0, 1, 2) || error("num_blocks() axis must be 0, 1, or 2, got $axis")

    res_type = tile_type!(ctx.tt, I32(ctx.tt), Int[])
    nb_x, nb_y, nb_z = encode_GetNumTileBlocksOp!(ctx.cb, res_type, res_type, res_type)

    TileValue((nb_x, nb_y, nb_z)[axis + 1], res_type, Int32)
end

function emit_intrinsic!(ctx::CodegenContext, ::typeof(_load), args, @nospecialize(result_type))
    emit_load!(ctx, args, result_type)
end

function emit_intrinsic!(ctx::CodegenContext, ::typeof(store), args, @nospecialize(result_type))
    emit_store!(ctx, args, result_type)
    nothing
end

function emit_intrinsic!(ctx::CodegenContext, ::typeof(_atomic_cas), args, @nospecialize(result_type))
    emit_atomic_cas!(ctx, args, result_type)
end

function emit_intrinsic!(ctx::CodegenContext, ::typeof(_atomic_xchg), args, @nospecialize(result_type))
    emit_atomic_rmw!(ctx, args, result_type, AtomicXCHG)
end

function emit_intrinsic!(ctx::CodegenContext, ::typeof(_atomic_add), args, @nospecialize(result_type))
    emit_atomic_rmw!(ctx, args, result_type, AtomicADD)
end

function emit_intrinsic!(ctx::CodegenContext, ::typeof(transpose), args, @nospecialize(result_type))
    emit_transpose!(ctx, args, result_type)
end

function emit_intrinsic!(ctx::CodegenContext, ::typeof(mma), args, @nospecialize(result_type))
    emit_mma!(ctx, args, result_type)
end

function emit_intrinsic!(ctx::CodegenContext, ::typeof(full), args, @nospecialize(result_type))
    emit_full!(ctx, args, result_type)
end

function emit_intrinsic!(ctx::CodegenContext, ::typeof(num_tiles), args, @nospecialize(result_type))
    emit_num_tiles!(ctx, args, result_type)
end

function emit_intrinsic!(ctx::CodegenContext, ::typeof(cdiv), args, @nospecialize(result_type))
    emit_cdiv!(ctx, args, result_type)
end

function emit_intrinsic!(ctx::CodegenContext, ::typeof(floordiv), args, @nospecialize(result_type))
    emit_floordiv!(ctx, args, result_type)
end

function emit_intrinsic!(ctx::CodegenContext, ::typeof(Base.rem), args, @nospecialize(result_type))
    emit_rem!(ctx, args, result_type)
end

function emit_intrinsic!(ctx::CodegenContext, ::typeof(Base.min), args, @nospecialize(result_type))
    emit_min!(ctx, args, result_type)
end

function emit_intrinsic!(ctx::CodegenContext, ::typeof(astype), args, @nospecialize(result_type))
    emit_astype!(ctx, args, result_type)
end

function emit_intrinsic!(ctx::CodegenContext, ::typeof(broadcast_to), args, @nospecialize(result_type))
    emit_broadcast_to!(ctx, args, result_type)
end

# Tile(scalar) constructor - creates a 0D tile from a scalar value
function emit_intrinsic!(ctx::CodegenContext, ::Type{<:Tile}, args, @nospecialize(result_type))
    cb = ctx.cb
    tt = ctx.tt

    # Get element type from result type
    result_type_unwrapped = unwrap_type(result_type)
    elem_type = if result_type_unwrapped <: Tile
        result_type_unwrapped.parameters[1]
    else
        nothing  # Will be determined later
    end

    # Emit the value - this handles both compile-time constants and runtime values
    source = emit_value!(ctx, args[1])
    source === nothing && error("Cannot resolve source operand for Tile(scalar)")

    # If we have a compile-time constant, determine elem_type from it
    if source.constant !== nothing && elem_type === nothing
        elem_type = typeof(source.constant)
    end

    # Already a tile value, just return it (might need reshape to 0D)
    if isempty(source.shape)
        return source
    else
        # Reshape to 0D if it's a scalar-sized tile
        error("Tile(scalar) called with non-scalar tile")
    end
end

#-----------------------------------------------------------------------------
# Tile arithmetic
#-----------------------------------------------------------------------------

"""
    compute_broadcast_shape(s1::Vector{Int}, s2::Vector{Int}) -> Vector{Int}

Compute the NumPy-style broadcast shape from two shapes.
Shapes are compared from the right, dimensions must be equal or 1.
"""
function compute_broadcast_shape(s1::Vector{Int}, s2::Vector{Int})
    max_ndim = max(length(s1), length(s2))
    result = Vector{Int}(undef, max_ndim)
    for i in 1:max_ndim
        idx1 = length(s1) - max_ndim + i
        idx2 = length(s2) - max_ndim + i
        d1 = idx1 > 0 ? s1[idx1] : 1
        d2 = idx2 > 0 ? s2[idx2] : 1
        if d1 != d2 && d1 != 1 && d2 != 1
            error("Shapes $s1 and $s2 are not broadcastable")
        end
        result[i] = max(d1, d2)
    end
    result
end

"""
    broadcast_tile_to_shape!(cb, tt, tv::TileValue, target_shape::Vector{Int}, dtype::TypeId) -> Value

Broadcast a tile to a target shape by inserting ReshapeOp (for leading 1s) and BroadcastOp.
Returns the value after broadcasting, or the original value if shapes already match.
"""
function broadcast_tile_to_shape!(cb::CodeBuilder, tt::TypeTable, tv::TileValue,
                                   target_shape::Vector{Int}, dtype::TypeId)
    src_shape = tv.shape

    # Already the right shape?
    if src_shape == target_shape
        return tv.v
    end

    current_val = tv.v
    current_shape = src_shape

    # Step 1: Add leading 1s via ReshapeOp if needed (dimension mismatch)
    if length(current_shape) < length(target_shape)
        # Prepend 1s to match target ndim
        n_extra = length(target_shape) - length(current_shape)
        new_shape = vcat(fill(1, n_extra), current_shape)
        reshaped_type = tile_type!(tt, dtype, new_shape)
        current_val = encode_ReshapeOp!(cb, reshaped_type, current_val)
        current_shape = new_shape
    end

    # Step 2: Broadcast dimensions that are 1 to target size
    if current_shape != target_shape
        broadcast_type = tile_type!(tt, dtype, target_shape)
        current_val = encode_BroadcastOp!(cb, broadcast_type, current_val)
    end

    current_val
end

"""
    emit_binop!(ctx, args, float_encoder, int_encoder)

Binary operation emitter.

Handles:
- Tile + Tile (same shapes - broadcasting is done at intrinsic level via broadcast_to)
- Scalar + Scalar (for integer intrinsics on index calculations)

Note: tile+scalar operations are handled at the intrinsic level via Tile(scalar) and
broadcast_to(), so by the time we reach tile_add etc., both operands are already tiles.
"""
function emit_binop!(ctx::CodegenContext, args, float_encoder::Function, int_encoder::Function)
    cb = ctx.cb
    tt = ctx.tt

    # Emit both operands
    lhs_tv = emit_value!(ctx, args[1])
    rhs_tv = emit_value!(ctx, args[2])

    # Both operands must resolve to TileValues
    if lhs_tv === nothing || rhs_tv === nothing
        return missing
    end

    # Determine what kind of operands we have
    lhs_is_tile = unwrap_type(lhs_tv.jltype) <: Tile
    rhs_is_tile = unwrap_type(rhs_tv.jltype) <: Tile

    if lhs_is_tile && rhs_is_tile
        # Tile + Tile: shapes should be identical (broadcasting via broadcast_to at intrinsic level)
        elem_type = unwrap_type(lhs_tv.jltype).parameters[1]
        result_shape = lhs_tv.shape
        lhs_v = lhs_tv.v
        rhs_v = rhs_tv.v
        result_jltype = Tile{elem_type, Tuple(result_shape)}
    elseif !lhs_is_tile && !rhs_is_tile
        # Scalar + Scalar: for integer intrinsics on index calculations
        elem_type = unwrap_type(lhs_tv.jltype)
        result_shape = Int[]
        lhs_v = lhs_tv.v
        rhs_v = rhs_tv.v
        result_jltype = elem_type
    else
        error("Mixed tile/scalar operations should be handled at intrinsic level via Tile() and broadcast_to()")
    end

    dtype = julia_to_tile_dtype!(tt, elem_type)
    result_type_id = tile_type!(tt, dtype, result_shape)

    # Emit the binary operation
    if elem_type <: AbstractFloat
        result_v = float_encoder(cb, result_type_id, lhs_v, rhs_v)
    else
        result_v = int_encoder(cb, result_type_id, lhs_v, rhs_v)
    end

    TileValue(result_v, result_type_id, result_jltype, result_shape)
end

# Same-shape tile operations - these emit the raw binary op
emit_intrinsic!(ctx::CodegenContext, ::typeof(tile_add), args, @nospecialize(_)) =
    emit_binop!(ctx, args, encode_AddFOp!, encode_AddIOp!)

emit_intrinsic!(ctx::CodegenContext, ::typeof(tile_sub), args, @nospecialize(_)) =
    emit_binop!(ctx, args, encode_SubFOp!, encode_SubIOp!)

emit_intrinsic!(ctx::CodegenContext, ::typeof(tile_mul), args, @nospecialize(_)) =
    emit_binop!(ctx, args, encode_MulFOp!, encode_MulIOp!)

emit_intrinsic!(ctx::CodegenContext, ::typeof(tile_div), args, @nospecialize(_)) =
    emit_binop!(ctx, args, encode_DivFOp!, encode_DivIOp!)

# Power operation (float only)
function emit_intrinsic!(ctx::CodegenContext, ::typeof(tile_pow), args, @nospecialize(_))
    cb = ctx.cb
    tt = ctx.tt

    lhs_tv = emit_value!(ctx, args[1])
    rhs_tv = emit_value!(ctx, args[2])

    (lhs_tv === nothing || rhs_tv === nothing) && error("Cannot resolve operands for tile_pow")

    # Power is float-only, so we expect tiles with float element types
    elem_type = unwrap_type(lhs_tv.jltype)
    if elem_type <: Tile
        elem_type = elem_type.parameters[1]
    end
    elem_type <: AbstractFloat || error("tile_pow only supports float types, got $elem_type")

    result_shape = lhs_tv.shape
    result_jltype = Tile{elem_type, Tuple(result_shape)}

    dtype = julia_to_tile_dtype!(tt, elem_type)
    result_type_id = tile_type!(tt, dtype, result_shape)

    result_v = encode_PowOp!(cb, result_type_id, lhs_tv.v, rhs_tv.v)

    TileValue(result_v, result_type_id, result_jltype, result_shape)
end

# Julia integer intrinsics (all are Core.IntrinsicFunction, so dispatch by value)
function emit_intrinsic!(ctx::CodegenContext, func::Core.IntrinsicFunction, args, @nospecialize(_))
    if func === Base.add_int
        return emit_binop!(ctx, args, encode_AddFOp!, encode_AddIOp!)
    elseif func === Base.sub_int
        return emit_binop!(ctx, args, encode_SubFOp!, encode_SubIOp!)
    elseif func === Base.mul_int
        return emit_binop!(ctx, args, encode_MulFOp!, encode_MulIOp!)
    elseif func === Base.sitofp
        return emit_sitofp!(ctx, args)
    elseif func === Base.uitofp
        return emit_uitofp!(ctx, args)
    # Integer comparison intrinsics (signed and unsigned use same predicate, signedness is separate)
    elseif func === Base.slt_int
        return emit_int_cmp!(ctx, args, CmpLessThan, SignednessSigned)
    elseif func === Base.sle_int
        return emit_int_cmp!(ctx, args, CmpLessThanOrEqual, SignednessSigned)
    elseif func === Base.ult_int
        return emit_int_cmp!(ctx, args, CmpLessThan, SignednessUnsigned)
    elseif func === Base.ule_int
        return emit_int_cmp!(ctx, args, CmpLessThanOrEqual, SignednessUnsigned)
    elseif func === Base.eq_int
        return emit_int_cmp!(ctx, args, CmpEqual, SignednessSigned)
    elseif func === Base.ne_int
        return emit_int_cmp!(ctx, args, CmpNotEqual, SignednessSigned)
    end
    missing  # Unknown intrinsic
end

# Signed integer to floating point conversion
function emit_sitofp!(ctx::CodegenContext, args)
    cb = ctx.cb
    tt = ctx.tt

    # args[1] is the target type (e.g., Float32), args[2] is the value
    target_type = args[1]
    source = emit_value!(ctx, args[2])
    source === nothing && error("Cannot resolve source operand for sitofp")

    # Get the target float type
    dtype = julia_to_tile_dtype!(tt, target_type)
    result_shape = source isa TileValue ? source.shape : Int[]
    result_type = tile_type!(tt, dtype, result_shape)

    result_v = encode_IToFOp!(cb, result_type, source isa TileValue ? source.v : source;
                              signedness=SignednessSigned)
    TileValue(result_v, result_type, target_type, result_shape)
end

# Unsigned integer to floating point conversion
function emit_uitofp!(ctx::CodegenContext, args)
    cb = ctx.cb
    tt = ctx.tt

    # args[1] is the target type (e.g., Float32), args[2] is the value
    target_type = args[1]
    source = emit_value!(ctx, args[2])
    source === nothing && error("Cannot resolve source operand for uitofp")

    # Get the target float type
    dtype = julia_to_tile_dtype!(tt, target_type)
    result_shape = source isa TileValue ? source.shape : Int[]
    result_type = tile_type!(tt, dtype, result_shape)

    result_v = encode_IToFOp!(cb, result_type, source isa TileValue ? source.v : source;
                              signedness=SignednessUnsigned)
    TileValue(result_v, result_type, target_type, result_shape)
end

#-----------------------------------------------------------------------------
# Comparison operators
#-----------------------------------------------------------------------------

function emit_cmp!(ctx::CodegenContext, args, predicate::ComparisonPredicate)
    emit_int_cmp!(ctx, args, predicate, SignednessSigned)
end

function emit_int_cmp!(ctx::CodegenContext, args, predicate::ComparisonPredicate, signedness::Signedness)
    cb = ctx.cb
    tt = ctx.tt

    lhs = emit_value!(ctx, args[1])
    rhs = emit_value!(ctx, args[2])

    lhs === nothing && error("Cannot resolve LHS operand for comparison")
    rhs === nothing && error("Cannot resolve RHS operand for comparison")

    # Result type is a boolean (i1) scalar
    result_type = tile_type!(tt, I1(tt), Int[])

    lhs_v = lhs isa TileValue ? lhs.v : lhs
    rhs_v = rhs isa TileValue ? rhs.v : rhs

    result_v = encode_CmpIOp!(cb, result_type, lhs_v, rhs_v;
                              predicate=predicate, signedness=signedness)

    TileValue(result_v, result_type, Bool, Int[])
end

emit_intrinsic!(ctx::CodegenContext, ::typeof(Base.:(>)), args, @nospecialize(_)) =
    emit_cmp!(ctx, args, CmpGreaterThan)

emit_intrinsic!(ctx::CodegenContext, ::typeof(Base.:(<)), args, @nospecialize(_)) =
    emit_cmp!(ctx, args, CmpLessThan)

emit_intrinsic!(ctx::CodegenContext, ::typeof(Base.:(>=)), args, @nospecialize(_)) =
    emit_cmp!(ctx, args, CmpGreaterThanOrEqual)

emit_intrinsic!(ctx::CodegenContext, ::typeof(Base.:(<=)), args, @nospecialize(_)) =
    emit_cmp!(ctx, args, CmpLessThanOrEqual)

emit_intrinsic!(ctx::CodegenContext, ::typeof(===), args, @nospecialize(_)) =
    emit_cmp!(ctx, args, CmpEqual)

emit_intrinsic!(ctx::CodegenContext, ::typeof(Base.:(==)), args, @nospecialize(_)) =
    emit_cmp!(ctx, args, CmpEqual)

emit_intrinsic!(ctx::CodegenContext, ::typeof(Base.:(!=)), args, @nospecialize(_)) =
    emit_cmp!(ctx, args, CmpNotEqual)

#-----------------------------------------------------------------------------
# getfield for destructured arguments (lazy chain extension)
#-----------------------------------------------------------------------------

function emit_intrinsic!(ctx::CodegenContext, ::typeof(Base.getfield), args, @nospecialize(result_type))
    length(args) >= 2 || return nothing

    obj_arg = args[1]
    field_arg = args[2]

    # Extract field name or index
    field = get_constant(ctx, field_arg)

    # Try to get the object as a TileValue
    obj_tv = emit_value!(ctx, obj_arg)

    # If obj is a lazy arg_ref, extend the chain
    if obj_tv !== nothing && is_arg_ref(obj_tv)
        arg_idx, chain = obj_tv.arg_ref

        if field isa Symbol
            # Field access: extend chain with symbol
            new_chain = Union{Symbol, Int}[chain..., field]
            # Check if this resolves to a scalar field (auto-materialize leaf)
            values = get_arg_flat_values(ctx, arg_idx, field)
            if values !== nothing && length(values) == 1
                # Scalar field - materialize immediately
                type_id = tile_type_for_julia!(ctx, unwrap_type(result_type))
                return TileValue(values[1], type_id, unwrap_type(result_type))
            end
            return arg_ref_value(arg_idx, new_chain, unwrap_type(result_type))
        elseif field isa Integer && !isempty(chain) && chain[end] isa Symbol
            # Tuple indexing: chain ends with field name, now indexing into it
            # This is a leaf - materialize immediately
            field_name = chain[end]
            values = get_arg_flat_values(ctx, arg_idx, field_name)
            if values !== nothing && 1 <= field <= length(values)
                type_id = tile_type_for_julia!(ctx, unwrap_type(result_type))
                return TileValue(values[field], type_id, unwrap_type(result_type))
            end
        end
    end

    nothing
end

function extract_argument_index(@nospecialize(arg))
    if arg isa SlotNumber
        return arg.id - 1
    elseif arg isa Argument
        return arg.n - 1
    end
    nothing
end

#-----------------------------------------------------------------------------
# getindex for tuple field access (lazy chain extension)
#-----------------------------------------------------------------------------

function emit_intrinsic!(ctx::CodegenContext, ::typeof(Base.getindex), args, @nospecialize(result_type))
    length(args) >= 2 || return nothing

    obj_arg = args[1]
    index_arg = args[2]

    # Extract constant index
    index = get_constant(ctx, index_arg)
    index isa Integer || return nothing

    # Try to get the object as a TileValue
    obj_tv = emit_value!(ctx, obj_arg)
    obj_tv === nothing && return nothing

    # If obj is a lazy arg_ref, extend the chain with the index
    if is_arg_ref(obj_tv)
        arg_idx, chain = obj_tv.arg_ref
        new_chain = Union{Symbol, Int}[chain..., Int(index)]
        return arg_ref_value(arg_idx, new_chain, unwrap_type(result_type))
    end

    # Not an arg_ref - not handled here
    nothing
end

#=============================================================================
 Load/Store Operations
=============================================================================#

"""
Convert integer padding mode value to bytecode PaddingValue enum.
Maps from cuTile.PaddingMode constants to bytecode PaddingValue.
"""
function padding_mode_to_padding_value(mode::Int)
    if mode == 0  # Undetermined
        PaddingMissing
    elseif mode == 1  # Zero
        PaddingZero
    elseif mode == 2  # NegZero
        PaddingNegZero
    elseif mode == 3  # Nan
        PaddingNan
    elseif mode == 4  # PosInf
        PaddingPosInf
    else  # 5 = NegInf
        PaddingNegInf
    end
end

function emit_load!(ctx::CodegenContext, args::AbstractVector, @nospecialize(result_type))
    cb = ctx.cb
    tt = ctx.tt

    array_arg = args[1]

    # Check if TileArray argument
    arg_idx = extract_argument_index(array_arg)
    is_tilearray = arg_idx !== nothing && is_destructured_arg(ctx, arg_idx)

    # Get pointer and element type
    array_spec = nothing
    if is_tilearray
        ptr_vals = get_arg_flat_values(ctx, arg_idx, :ptr)
        isempty(ptr_vals) && error("Cannot get ptr from TileArray argument")
        array_val = ptr_vals[1]
        tilearray_type = get_arg_type(ctx, arg_idx)
        elem_type = eltype(tilearray_type)
        array_spec = get_array_spec(tilearray_type)
    else
        tv = emit_value!(ctx, array_arg)
        tv === nothing && error("Cannot resolve array argument for load()")
        array_val = tv.v
        elem_type = extract_pointer_elem_type(tv.jltype)
    end

    # Parse shape from Val{shape} argument
    tile_shape = Int[16]  # Default
    if length(args) >= 3
        shape = get_constant(ctx, args[3])
        if shape isa Tuple
            tile_shape = collect(Int, shape)
        end
    end

    ndim = length(tile_shape)

    # Parse padding_mode from args[4] (default: PaddingMode.Undetermined = 0)
    padding_mode_int = 0  # Default: Undetermined
    if length(args) >= 4
        pm = get_constant(ctx, args[4])
        if pm isa Integer
            padding_mode_int = Int(pm)
        end
    end
    padding_value = padding_mode_to_padding_value(padding_mode_int)

    # Parse index argument
    index_vals = extract_index_values(ctx, args, 2, ndim)

    dtype = julia_to_tile_dtype!(tt, elem_type)

    # Create types
    tile_type = tile_type!(tt, dtype, tile_shape)
    token_type = Token(tt)

    # TensorView type - use static strides where known from ArraySpec
    tv_shape = fill(DYNAMIC_SHAPE, ndim)
    tv_strides = compute_tensor_view_strides(array_spec, ndim)
    tv_type = tensor_view_type!(tt, dtype, tv_shape, tv_strides)

    # PartitionView type
    dim_map = collect(0:ndim-1)
    pv_type = partition_view_type!(tt, tile_shape, tv_type, dim_map, padding_value)

    scalar_i32 = tile_type!(tt, I32(tt), Int[])

    # Get size and stride values
    size_vals, stride_vals = get_size_stride_vals(ctx, arg_idx, is_tilearray, ndim, tile_shape, index_vals, scalar_i32)

    # Emit AssumeOps for optimization hints (skip stride assumes for static strides)
    if array_spec !== nothing
        array_val, size_vals, stride_vals = emit_assume_ops!(ctx, array_val, size_vals, stride_vals, array_spec, dtype, scalar_i32; tv_strides)
    end

    # Filter strides to only pass dynamic ones as operands
    dynamic_stride_vals = filter_dynamic_strides(stride_vals, tv_strides)

    # Create tensor view
    tensor_view = encode_MakeTensorViewOp!(cb, tv_type, array_val, size_vals, dynamic_stride_vals)

    # Create partition view
    partition = encode_MakePartitionViewOp!(cb, pv_type, tensor_view)

    # Pad indices if needed
    index_vals = pad_indices(ctx, index_vals, ndim, scalar_i32)

    # Load tile with token
    tile_val, new_token = encode_LoadViewTkoOp!(cb, tile_type, token_type, partition, index_vals; token=ctx.token)
    ctx.token = new_token

    TileValue(tile_val, tile_type, Tile{elem_type, Tuple(tile_shape)}, tile_shape)
end

function emit_store!(ctx::CodegenContext, args::AbstractVector, @nospecialize(result_type))
    cb = ctx.cb
    tt = ctx.tt

    array_arg = args[1]

    # Check if TileArray argument
    arg_idx = extract_argument_index(array_arg)
    is_tilearray = arg_idx !== nothing && is_destructured_arg(ctx, arg_idx)

    # Get pointer and element type
    array_spec = nothing
    if is_tilearray
        ptr_vals = get_arg_flat_values(ctx, arg_idx, :ptr)
        isempty(ptr_vals) && error("Cannot get ptr from TileArray argument")
        array_val = ptr_vals[1]
        tilearray_type = get_arg_type(ctx, arg_idx)
        elem_type = eltype(tilearray_type)
        array_spec = get_array_spec(tilearray_type)
    else
        tv = emit_value!(ctx, array_arg)
        tv === nothing && error("Cannot resolve array argument for store()")
        array_val = tv.v
        elem_type = extract_pointer_elem_type(tv.jltype)
    end

    # Get tile value and shape
    tile_tv = emit_value!(ctx, args[3])
    tile_tv === nothing && error("store() requires a tile argument")
    tile_shape = tile_tv.shape
    isempty(tile_shape) && error("Cannot determine tile shape for store()")

    dtype = julia_to_tile_dtype!(tt, elem_type)
    ndim = length(tile_shape)

    # Parse index argument
    index_vals = extract_index_values(ctx, args, 2, ndim)

    # TensorView type - use static strides where known from ArraySpec
    tv_shape = fill(DYNAMIC_SHAPE, ndim)
    tv_strides = compute_tensor_view_strides(array_spec, ndim)
    tv_type = tensor_view_type!(tt, dtype, tv_shape, tv_strides)

    # PartitionView type
    dim_map = collect(0:ndim-1)
    pv_type = partition_view_type!(tt, tile_shape, tv_type, dim_map, PaddingZero)

    scalar_i32 = tile_type!(tt, I32(tt), Int[])

    # Get size and stride values
    size_vals, stride_vals = get_size_stride_vals(ctx, arg_idx, is_tilearray, ndim, tile_shape, index_vals, scalar_i32)

    # Emit AssumeOps (skip stride assumes for static strides)
    if array_spec !== nothing
        array_val, size_vals, stride_vals = emit_assume_ops!(ctx, array_val, size_vals, stride_vals, array_spec, dtype, scalar_i32; tv_strides)
    end

    # Filter strides to only pass dynamic ones as operands
    dynamic_stride_vals = filter_dynamic_strides(stride_vals, tv_strides)

    # Create tensor view
    tensor_view = encode_MakeTensorViewOp!(cb, tv_type, array_val, size_vals, dynamic_stride_vals)

    # Create partition view
    partition = encode_MakePartitionViewOp!(cb, pv_type, tensor_view)

    # Pad indices
    index_vals = pad_indices(ctx, index_vals, ndim, scalar_i32)

    # Store tile with token
    token_type = Token(tt)
    new_token = encode_StoreViewTkoOp!(cb, token_type, tile_tv.v, partition, index_vals; token=ctx.token)
    ctx.token = new_token

    nothing
end

#=============================================================================
 Atomic Operations
=============================================================================#

"""
Convert integer memory order value to bytecode MemoryOrderingSemantics enum
"""
function memory_order_to_semantics(order::Int)
    if order == 0  # Weak
        MemoryWeak
    elseif order == 1  # Relaxed
        MemoryRelaxed
    elseif order == 2  # Acquire
        MemoryAcquire
    elseif order == 3  # Release
        MemoryRelease
    else  # 4 = AcqRel
        MemoryAcqRel
    end
end

"""
Convert integer memory scope value to bytecode MemoryScope enum
"""
function memory_scope_to_scope(scope::Int)
    if scope == 0  # Block
        ScopeTLBlock
    elseif scope == 1  # Device
        ScopeDevice
    else  # 2 = System
        ScopeSystem
    end
end

function emit_atomic_cas!(ctx::CodegenContext, args::AbstractVector, @nospecialize(result_type))
    cb = ctx.cb
    tt = ctx.tt

    # args: (array, index, expected, desired, memory_order, memory_scope)
    array_arg = args[1]

    # Get array info
    arg_idx = extract_argument_index(array_arg)
    is_tilearray = arg_idx !== nothing && is_destructured_arg(ctx, arg_idx)

    if !is_tilearray
        error("atomic_cas requires a TileArray argument")
    end

    ptr_vals = get_arg_flat_values(ctx, arg_idx, :ptr)
    isempty(ptr_vals) && error("Cannot get ptr from TileArray argument")
    array_val = ptr_vals[1]
    tilearray_type = get_arg_type(ctx, arg_idx)
    elem_type = eltype(tilearray_type)

    # Get expected and desired values
    expected_tv = emit_value!(ctx, args[3])
    expected_tv === nothing && error("atomic_cas requires expected value")
    desired_tv = emit_value!(ctx, args[4])
    desired_tv === nothing && error("atomic_cas requires desired value")

    # Get memory order and scope from args
    memory_order = @something get_constant(ctx, args[5]) error("atomic_cas requires constant memory_order")
    memory_scope = @something get_constant(ctx, args[6]) error("atomic_cas requires constant memory_scope")

    # Create result type (0D tile of element type)
    dtype = julia_to_tile_dtype!(tt, elem_type)
    result_tile_type = tile_type!(tt, dtype, Int[])
    token_type = Token(tt)

    # Create pointer tile - compute pointer to index
    scalar_i64_type = tile_type!(tt, I64(tt), Int[])
    index_tv = emit_value!(ctx, args[2])

    # Create pointer type for element
    ptr_type = pointer_type!(tt, dtype)
    ptr_tile_type = tile_type!(tt, ptr_type, Int[])

    # Compute pointer: base + index * sizeof(elem)
    # Convert pointer to int64, do arithmetic, convert back
    ptr_as_int = encode_PtrToIntOp!(cb, scalar_i64_type, array_val)

    elem_size = sizeof(elem_type)
    elem_size_const_tv = emit_constant!(ctx, Int64(elem_size), Int64)
    elem_size_const = elem_size_const_tv.v

    # Extend index to 64-bit if needed
    index_i64 = encode_ExtIOp!(cb, scalar_i64_type, index_tv.v; signedness=SignednessSigned)
    index_scaled = encode_MulIOp!(cb, scalar_i64_type, index_i64, elem_size_const)
    ptr_int_offset = encode_AddIOp!(cb, scalar_i64_type, ptr_as_int, index_scaled)
    pointers = encode_IntToPtrOp!(cb, ptr_tile_type, ptr_int_offset)

    # Emit atomic CAS
    mem_ordering = memory_order_to_semantics(memory_order)
    mem_scope = memory_scope_to_scope(memory_scope)

    old_val, new_token = encode_AtomicCASPtrOp!(cb, result_tile_type, token_type, pointers,
                                         expected_tv.v, desired_tv.v;
                                         token=ctx.token,
                                         memory_ordering=mem_ordering,
                                         memory_scope=mem_scope)
    ctx.token = new_token

    # Return scalar type (not Tile) to match the intrinsic signature
    TileValue(old_val, result_tile_type, elem_type, Int[])
end

function emit_atomic_rmw!(ctx::CodegenContext, args::AbstractVector, @nospecialize(result_type), mode::AtomicRMWMode)
    cb = ctx.cb
    tt = ctx.tt

    # args: (array, index, val, memory_order, memory_scope)
    array_arg = args[1]

    # Get array info
    arg_idx = extract_argument_index(array_arg)
    is_tilearray = arg_idx !== nothing && is_destructured_arg(ctx, arg_idx)

    if !is_tilearray
        error("atomic operations require a TileArray argument")
    end

    ptr_vals = get_arg_flat_values(ctx, arg_idx, :ptr)
    isempty(ptr_vals) && error("Cannot get ptr from TileArray argument")
    array_val = ptr_vals[1]
    tilearray_type = get_arg_type(ctx, arg_idx)
    elem_type = eltype(tilearray_type)

    # Get update value
    val_tv = emit_value!(ctx, args[3])
    val_tv === nothing && error("atomic operation requires value")

    # Get memory order and scope from args
    memory_order = @something get_constant(ctx, args[4]) error("atomic operation requires constant memory_order")
    memory_scope = @something get_constant(ctx, args[5]) error("atomic operation requires constant memory_scope")

    # Create result type (0D tile of element type)
    dtype = julia_to_tile_dtype!(tt, elem_type)
    result_tile_type = tile_type!(tt, dtype, Int[])
    token_type = Token(tt)

    # Create pointer tile
    scalar_i64_type = tile_type!(tt, I64(tt), Int[])
    index_tv = emit_value!(ctx, args[2])

    ptr_type = pointer_type!(tt, dtype)
    ptr_tile_type = tile_type!(tt, ptr_type, Int[])

    # Compute pointer: base + index * sizeof(elem)
    # Convert pointer to int64, do arithmetic, convert back
    ptr_as_int = encode_PtrToIntOp!(cb, scalar_i64_type, array_val)

    elem_size = sizeof(elem_type)
    elem_size_const_tv = emit_constant!(ctx, Int64(elem_size), Int64)
    elem_size_const = elem_size_const_tv.v

    # Extend index to 64-bit if needed
    index_i64 = encode_ExtIOp!(cb, scalar_i64_type, index_tv.v; signedness=SignednessSigned)
    index_scaled = encode_MulIOp!(cb, scalar_i64_type, index_i64, elem_size_const)
    ptr_int_offset = encode_AddIOp!(cb, scalar_i64_type, ptr_as_int, index_scaled)
    pointers = encode_IntToPtrOp!(cb, ptr_tile_type, ptr_int_offset)

    # Use float add mode for floating point types
    actual_mode = mode
    if mode == AtomicADD && elem_type <: AbstractFloat
        actual_mode = AtomicADDF
    end

    # Emit atomic RMW
    mem_ordering = memory_order_to_semantics(memory_order)
    mem_scope = memory_scope_to_scope(memory_scope)

    old_val, new_token = encode_AtomicRMWPtrOp!(cb, result_tile_type, token_type, pointers,
                                         val_tv.v, actual_mode;
                                         token=ctx.token,
                                         memory_ordering=mem_ordering,
                                         memory_scope=mem_scope)
    ctx.token = new_token

    # Return scalar type (not Tile) to match the intrinsic signature
    TileValue(old_val, result_tile_type, elem_type, Int[])
end

#=============================================================================
 Load/Store Helpers
=============================================================================#

function extract_pointer_elem_type(@nospecialize(jltype))
    jltype <: Ptr ? eltype(jltype) : Float32
end

function get_array_spec(@nospecialize(T))
    if T <: TileArray && length(T.parameters) >= 3
        S = T.parameters[3]
        S isa ArraySpec && return S
    end
    nothing
end

"""
    compute_tensor_view_strides(array_spec, ndim) -> Vector{Int64}

Compute the stride values for a TensorView type based on ArraySpec.
Returns static stride values where known, DYNAMIC_SHAPE where dynamic.

For contiguous arrays (array_spec.contiguous == true), stride[1] = 1 is statically known.
Higher dimensions are typically dynamic unless we have explicit info.
"""
function compute_tensor_view_strides(array_spec::Union{ArraySpec, Nothing}, ndim::Int)
    strides = fill(DYNAMIC_SHAPE, ndim)

    if array_spec !== nothing && array_spec.contiguous && ndim >= 1
        # Contiguous array: first stride is statically known to be 1
        strides[1] = 1
    end

    return strides
end

"""
    filter_dynamic_strides(stride_vals, tv_strides) -> Vector{Value}

Filter stride values to only include those corresponding to dynamic dimensions.
Only pass operands for dimensions where tv_strides[i] == DYNAMIC_SHAPE.
"""
function filter_dynamic_strides(stride_vals::Vector{Value}, tv_strides::Vector{Int64})
    dynamic_vals = Value[]
    for (i, stride_type_val) in enumerate(tv_strides)
        if stride_type_val == DYNAMIC_SHAPE && i <= length(stride_vals)
            push!(dynamic_vals, stride_vals[i])
        end
    end
    return dynamic_vals
end

"""
    extract_tile_shape(T) -> Vector{Int}

Extract shape from a Tile{T, Shape} type, returning Int[] if not a Tile type.
"""
function extract_tile_shape(@nospecialize(T))
    T = unwrap_type(T)
    if T <: Tile && length(T.parameters) >= 2
        shape = T.parameters[2]
        if shape isa Tuple
            return collect(Int, shape)
        end
    end
    Int[]
end

function extract_index_values(ctx::CodegenContext, args::AbstractVector, idx_pos::Int, ndim::Int)
    index_vals = Value[]
    length(args) < idx_pos && return index_vals

    index_arg = args[idx_pos]
    if index_arg isa SSAValue
        tuple_stmt = code(ctx.target)[index_arg.id]
        if tuple_stmt isa Expr && tuple_stmt.head === :call
            callee = tuple_stmt.args[1]
            if callee isa GlobalRef && callee.mod === Core && callee.name === :tuple
                for elem_arg in tuple_stmt.args[2:end]
                    tv = emit_value!(ctx, elem_arg)
                    tv.v !== nothing && push!(index_vals, tv.v)
                end
                return index_vals
            end
        end
        # Single value
        tv = emit_value!(ctx, index_arg)
        tv.v !== nothing && push!(index_vals, tv.v)
    else
        tv = emit_value!(ctx, index_arg)
        tv.v !== nothing && push!(index_vals, tv.v)
    end

    return index_vals
end

function get_size_stride_vals(ctx::CodegenContext, arg_idx, is_tilearray::Bool, ndim::Int,
                               tile_shape::Vector{Int}, index_vals::Vector{Value}, scalar_i32::TypeId)
    cb = ctx.cb
    tt = ctx.tt
    size_vals = Value[]
    stride_vals = Value[]

    if is_tilearray
        sizes_from_arg = get_arg_flat_values(ctx, arg_idx, :sizes)
        strides_from_arg = get_arg_flat_values(ctx, arg_idx, :strides)

        if sizes_from_arg !== nothing && length(sizes_from_arg) >= ndim
            size_vals = sizes_from_arg[1:ndim]
        end
        if strides_from_arg !== nothing && length(strides_from_arg) >= ndim
            stride_vals = strides_from_arg[1:ndim]
        end
    end

    # Compute from grid if not available
    if isempty(size_vals)
        nb_x, nb_y, nb_z = encode_GetNumTileBlocksOp!(cb, scalar_i32, scalar_i32, scalar_i32)
        grid_sizes = [nb_x, nb_y, nb_z]

        for dim in 1:ndim
            tile_size_bytes = reinterpret(UInt8, [Int32(tile_shape[dim])])
            tile_size_val = encode_ConstantOp!(cb, scalar_i32, collect(tile_size_bytes))
            size_val = encode_MulIOp!(cb, scalar_i32, grid_sizes[dim], tile_size_val)
            push!(size_vals, size_val)
        end
    end

    if isempty(stride_vals)
        for dim in 1:ndim
            if dim == 1
                stride_bytes = reinterpret(UInt8, [Int32(1)])
                stride_val = encode_ConstantOp!(cb, scalar_i32, collect(stride_bytes))
            else
                stride_val = encode_MulIOp!(cb, scalar_i32, stride_vals[end], size_vals[dim-1])
            end
            push!(stride_vals, stride_val)
        end
    end

    return size_vals, stride_vals
end

function emit_assume_ops!(ctx::CodegenContext, array_val::Value, size_vals::Vector{Value},
                          stride_vals::Vector{Value}, array_spec::ArraySpec, dtype::TypeId, scalar_i32::TypeId;
                          tv_strides::Union{Vector{Int64}, Nothing}=nothing)
    cb = ctx.cb
    tt = ctx.tt

    # Pointer alignment
    if array_spec.alignment > 0
        ptr_dtype = pointer_type!(tt, dtype)
        ptr_tile_type = tile_type!(tt, ptr_dtype, Int[])
        array_val = encode_AssumeOp!(cb, ptr_tile_type, array_val, DivBy(array_spec.alignment))
    end

    # Bounds assumes for sizes
    size_vals = [encode_AssumeOp!(cb, scalar_i32, v, Bounded(0, nothing)) for v in size_vals]

    # Bounds assumes for strides - only for dynamic strides
    if tv_strides !== nothing
        stride_vals = [tv_strides[i] == DYNAMIC_SHAPE ?
                       encode_AssumeOp!(cb, scalar_i32, v, Bounded(0, nothing)) : v
                       for (i, v) in enumerate(stride_vals)]
    else
        stride_vals = [encode_AssumeOp!(cb, scalar_i32, v, Bounded(0, nothing)) for v in stride_vals]
    end

    # Divisibility assumes for sizes
    if hasproperty(array_spec, :shape_div_by)
        for (i, div_by) in enumerate(array_spec.shape_div_by)
            if div_by > 0 && i <= length(size_vals)
                size_vals[i] = encode_AssumeOp!(cb, scalar_i32, size_vals[i], DivBy(div_by))
            end
        end
    end

    # Divisibility assumes for strides - only for dynamic strides
    if hasproperty(array_spec, :stride_div_by)
        for (i, div_by) in enumerate(array_spec.stride_div_by)
            if div_by > 0 && i <= length(stride_vals)
                # Skip if this stride is static (not DYNAMIC_SHAPE)
                if tv_strides === nothing || tv_strides[i] == DYNAMIC_SHAPE
                    stride_vals[i] = encode_AssumeOp!(cb, scalar_i32, stride_vals[i], DivBy(div_by))
                end
            end
        end
    end

    return array_val, size_vals, stride_vals
end

function pad_indices(ctx::CodegenContext, index_vals::Vector{Value}, ndim::Int, idx_type::TypeId)
    while length(index_vals) < ndim
        idx_bytes = reinterpret(UInt8, [Int32(0)])
        push!(index_vals, encode_ConstantOp!(ctx.cb, idx_type, collect(idx_bytes)))
    end
    return index_vals
end

#=============================================================================
 Transpose Operation
=============================================================================#

function emit_transpose!(ctx::CodegenContext, args::AbstractVector, @nospecialize(result_type))
    cb = ctx.cb
    tt = ctx.tt

    source = emit_value!(ctx, args[1])
    source === nothing && error("Cannot resolve operand for transpose()")

    input_shape = source.shape
    isempty(input_shape) && error("Cannot determine tile shape for transpose()")

    output_shape = reverse(input_shape)

    elem_type = unwrap_type(source.jltype)
    if elem_type <: Tile
        elem_type = elem_type.parameters[1]
    end

    dtype = julia_to_tile_dtype!(tt, elem_type)
    output_tile_type = tile_type!(tt, dtype, output_shape)

    ndim = length(output_shape)
    permutation = collect(ndim-1:-1:0)

    result = encode_PermuteOp!(cb, output_tile_type, source.v, permutation)

    TileValue(result, output_tile_type, Tile{elem_type, Tuple(output_shape)}, output_shape)
end

#=============================================================================
 Matrix Multiply-Accumulate
=============================================================================#

function emit_mma!(ctx::CodegenContext, args::AbstractVector, @nospecialize(result_type))
    cb = ctx.cb

    lhs = emit_value!(ctx, args[1])
    rhs = emit_value!(ctx, args[2])
    acc = emit_value!(ctx, args[3])

    (lhs === nothing || rhs === nothing || acc === nothing) && error("Cannot resolve operands for mma()")

    result = encode_MmaFOp!(cb, acc.type_id, lhs.v, rhs.v, acc.v)

    TileValue(result, acc.type_id, acc.jltype, acc.shape)
end

#=============================================================================
 Type Conversion
=============================================================================#

function emit_astype!(ctx::CodegenContext, args::AbstractVector, @nospecialize(result_type))
    cb = ctx.cb
    tt = ctx.tt

    # Get source tile
    source = emit_value!(ctx, args[1])
    source === nothing && error("Cannot resolve source operand for astype()")

    # Get source element type and shape
    source_type = unwrap_type(source.jltype)
    source_elem = source_type <: Tile ? source_type.parameters[1] : source_type
    tile_shape = source.shape

    # Get target element type from the Type argument
    target_elem = @something get_constant(ctx, args[2]) error("astype() requires a compile-time constant type")
    target_elem isa Type || error("astype() second argument must be a Type")

    # Same type? Return source unchanged
    if source_elem === target_elem
        return source
    end

    # Create target type
    target_dtype = julia_to_tile_dtype!(tt, target_elem)
    target_tile_type = tile_type!(tt, target_dtype, tile_shape)

    # Determine signedness for integer types
    function is_signed_int(T)
        T <: Signed || T === Int32 || T === Int64 || T === Int16 || T === Int8
    end

    # Emit conversion based on source and target types
    result = if source_elem <: AbstractFloat && target_elem <: AbstractFloat
        # Float -> Float
        encode_FToFOp!(cb, target_tile_type, source.v)
    elseif source_elem <: Integer && target_elem <: AbstractFloat
        # Integer -> Float
        signedness = is_signed_int(source_elem) ? SignednessSigned : SignednessUnsigned
        encode_IToFOp!(cb, target_tile_type, source.v; signedness)
    elseif source_elem <: AbstractFloat && target_elem <: Integer
        # Float -> Integer
        signedness = is_signed_int(target_elem) ? SignednessSigned : SignednessUnsigned
        encode_FToIOp!(cb, target_tile_type, source.v; signedness)
    elseif source_elem <: Integer && target_elem <: Integer
        # Integer -> Integer
        source_size = sizeof(source_elem)
        target_size = sizeof(target_elem)
        if source_size == target_size
            # Same size - no conversion needed (just reinterpret)
            source.v
        elseif target_size > source_size
            # Extension (upsize)
            signedness = is_signed_int(source_elem) ? SignednessSigned : SignednessUnsigned
            encode_ExtIOp!(cb, target_tile_type, source.v; signedness)
        else
            # Truncation (downsize)
            encode_TruncIOp!(cb, target_tile_type, source.v)
        end
    else
        error("astype() unsupported conversion: $source_elem -> $target_elem")
    end

    TileValue(result, target_tile_type, Tile{target_elem, Tuple(tile_shape)}, tile_shape)
end

#=============================================================================
 Explicit Broadcasting
=============================================================================#

function emit_broadcast_to!(ctx::CodegenContext, args::AbstractVector, @nospecialize(result_type))
    cb = ctx.cb
    tt = ctx.tt

    # Get source tile
    source = emit_value!(ctx, args[1])
    source === nothing && error("Cannot resolve source operand for broadcast_to()")

    # Get source element type
    source_type = unwrap_type(source.jltype)
    source_elem = source_type <: Tile ? source_type.parameters[1] : source_type

    # Extract target shape from the constant tuple argument
    target_shape_tuple = get_constant(ctx, args[2])
    target_shape_tuple isa Tuple || error("broadcast_to() shape must be a compile-time constant tuple")
    target_shape = collect(Int, target_shape_tuple)

    # If already the right shape, return unchanged
    if source.shape == target_shape
        return source
    end

    # Use the existing broadcast helper
    dtype = julia_to_tile_dtype!(tt, source_elem)
    result_v = broadcast_tile_to_shape!(cb, tt, source, target_shape, dtype)
    result_type_id = tile_type!(tt, dtype, target_shape)

    TileValue(result_v, result_type_id, Tile{source_elem, Tuple(target_shape)}, target_shape)
end

#=============================================================================
 Tile Construction
=============================================================================#

function emit_full!(ctx::CodegenContext, args::AbstractVector, @nospecialize(result_type))
    cb = ctx.cb
    tt = ctx.tt

    # Extract shape
    shape = get_constant(ctx, args[1])
    shape isa Tuple || error("full() shape must be a compile-time constant tuple")
    tile_shape = collect(Int, shape)

    # Extract value
    value = @something get_constant(ctx, args[2]) error("full() value must be a compile-time constant")

    # Extract dtype from result type
    result_type_unwrapped = unwrap_type(result_type)
    elem_type = Float32
    if result_type_unwrapped <: Tile
        elem_type = result_type_unwrapped.parameters[1]
    end

    dtype = julia_to_tile_dtype!(tt, elem_type)
    tile_type = tile_type!(tt, dtype, tile_shape)

    # Create scalar constant
    scalar_type = tile_type!(tt, dtype, Int[])
    value_bytes = constant_to_bytes(value, elem_type)
    scalar_val = encode_ConstantOp!(cb, scalar_type, value_bytes)

    # Reshape and broadcast
    ndims = length(tile_shape)
    if ndims > 0
        ones_shape = fill(1, ndims)
        reshaped_type = tile_type!(tt, dtype, ones_shape)
        reshaped_val = encode_ReshapeOp!(cb, reshaped_type, scalar_val)
    else
        reshaped_val = scalar_val
    end

    result = encode_BroadcastOp!(cb, tile_type, reshaped_val)

    TileValue(result, tile_type, Tile{elem_type, Tuple(tile_shape)}, tile_shape)
end

#=============================================================================
 Integer Arithmetic
=============================================================================#

function emit_num_tiles!(ctx::CodegenContext, args::AbstractVector, @nospecialize(result_type))
    cb = ctx.cb
    tt = ctx.tt

    array_arg = args[1]
    axis = @something get_constant(ctx, args[2]) error("num_tiles() axis must be a compile-time constant")

    shape = get_constant(ctx, args[3])
    shape isa Tuple || error("num_tiles() shape must be a compile-time constant tuple")

    tile_size = shape[axis + 1]

    arg_idx = extract_argument_index(array_arg)
    (arg_idx === nothing || !is_destructured_arg(ctx, arg_idx)) && error("num_tiles() requires a TileArray argument")

    sizes_vals = get_arg_flat_values(ctx, arg_idx, :sizes)
    (sizes_vals === nothing || length(sizes_vals) <= axis) && error("Cannot get size for axis $axis")

    array_size = sizes_vals[axis + 1]
    scalar_i32 = tile_type!(tt, I32(tt), Int[])

    # cdiv: (array_size + tile_size - 1) / tile_size
    tile_size_val = encode_ConstantOp!(cb, scalar_i32, collect(reinterpret(UInt8, [Int32(tile_size)])))
    one_val = encode_ConstantOp!(cb, scalar_i32, collect(reinterpret(UInt8, [Int32(1)])))

    sum1 = encode_AddIOp!(cb, scalar_i32, array_size, tile_size_val)
    sum2 = encode_SubIOp!(cb, scalar_i32, sum1, one_val)
    result = encode_DivIOp!(cb, scalar_i32, sum2, tile_size_val; signedness=SignednessSigned, rounding=RoundingZero)

    TileValue(result, scalar_i32, Int32)
end

function emit_cdiv!(ctx::CodegenContext, args::AbstractVector, @nospecialize(result_type))
    cb = ctx.cb
    tt = ctx.tt

    scalar_i32 = tile_type!(tt, I32(tt), Int[])

    a = resolve_or_constant(ctx, args[1], scalar_i32)
    b = resolve_or_constant(ctx, args[2], scalar_i32)

    one_val = encode_ConstantOp!(cb, scalar_i32, collect(reinterpret(UInt8, [Int32(1)])))

    sum1 = encode_AddIOp!(cb, scalar_i32, a, b)
    sum2 = encode_SubIOp!(cb, scalar_i32, sum1, one_val)
    result = encode_DivIOp!(cb, scalar_i32, sum2, b; signedness=SignednessSigned, rounding=RoundingZero)

    TileValue(result, scalar_i32, Int32)
end

function emit_floordiv!(ctx::CodegenContext, args::AbstractVector, @nospecialize(result_type))
    cb = ctx.cb
    tt = ctx.tt

    scalar_i32 = tile_type!(tt, I32(tt), Int[])

    a = resolve_or_constant(ctx, args[1], scalar_i32)
    b = resolve_or_constant(ctx, args[2], scalar_i32)

    result = encode_DivIOp!(cb, scalar_i32, a, b; signedness=SignednessSigned, rounding=RoundingZero)

    TileValue(result, scalar_i32, Int32)
end

function emit_rem!(ctx::CodegenContext, args::AbstractVector, @nospecialize(result_type))
    cb = ctx.cb
    tt = ctx.tt

    scalar_i32 = tile_type!(tt, I32(tt), Int[])

    a = resolve_or_constant(ctx, args[1], scalar_i32)
    b = resolve_or_constant(ctx, args[2], scalar_i32)

    result = encode_RemIOp!(cb, scalar_i32, a, b; signedness=SignednessSigned)

    TileValue(result, scalar_i32, Int32)
end

function emit_min!(ctx::CodegenContext, args::AbstractVector, @nospecialize(result_type))
    cb = ctx.cb
    tt = ctx.tt

    scalar_i32 = tile_type!(tt, I32(tt), Int[])

    a = resolve_or_constant(ctx, args[1], scalar_i32)
    b = resolve_or_constant(ctx, args[2], scalar_i32)

    result = encode_MinIOp!(cb, scalar_i32, a, b; signedness=SignednessSigned)

    TileValue(result, scalar_i32, Int32)
end

function resolve_or_constant(ctx::CodegenContext, @nospecialize(arg), type_id::TypeId)
    tv = emit_value!(ctx, arg)
    # If we have a runtime value, use it
    tv.v !== nothing && return tv.v
    # Otherwise emit a constant from the compile-time value
    val = @something tv.constant error("Cannot resolve argument")
    bytes = reinterpret(UInt8, [Int32(val)])
    encode_ConstantOp!(ctx.cb, type_id, collect(bytes))
end

#=============================================================================
 Math Operations
=============================================================================#

function emit_intrinsic!(ctx::CodegenContext, ::typeof(Base.sqrt), args, @nospecialize(result_type))
    cb = ctx.cb

    source = emit_value!(ctx, args[1])
    source === nothing && error("Cannot resolve operand for sqrt()")

    result = encode_SqrtOp!(cb, source.type_id, source.v)

    TileValue(result, source.type_id, source.jltype, source.shape)
end

#=============================================================================
 Tile Factory Operations
=============================================================================#

function emit_intrinsic!(ctx::CodegenContext, ::typeof(arange), args, @nospecialize(result_type))
    cb = ctx.cb
    tt = ctx.tt

    # Extract shape
    shape = get_constant(ctx, args[1])
    shape isa Tuple || error("arange() shape must be a compile-time constant tuple")
    tile_shape = collect(Int, shape)

    # Extract dtype from result type
    result_type_unwrapped = unwrap_type(result_type)
    elem_type = Int32
    if result_type_unwrapped <: Tile
        elem_type = result_type_unwrapped.parameters[1]
    end

    dtype = julia_to_tile_dtype!(tt, elem_type)
    tile_type = tile_type!(tt, dtype, tile_shape)

    # Emit IotaOp
    result = encode_IotaOp!(cb, tile_type)

    TileValue(result, tile_type, Tile{elem_type, Tuple(tile_shape)}, tile_shape)
end

#=============================================================================
 Reduction Operations
=============================================================================#

function emit_intrinsic!(ctx::CodegenContext, ::typeof(reduce_sum), args, @nospecialize(result_type))
    emit_reduce!(ctx, args, result_type, :add)
end

function emit_intrinsic!(ctx::CodegenContext, ::typeof(reduce_max), args, @nospecialize(result_type))
    emit_reduce!(ctx, args, result_type, :max)
end

function emit_reduce!(ctx::CodegenContext, args, @nospecialize(result_type), reduce_fn::Symbol)
    cb = ctx.cb
    tt = ctx.tt

    # Get input tile
    input_tv = emit_value!(ctx, args[1])
    input_tv === nothing && error("Cannot resolve input tile for reduction")

    # Get reduction axis
    axis = @something get_constant(ctx, args[2]) error("Reduction axis must be a compile-time constant")

    # Get element type and shapes
    input_type = unwrap_type(input_tv.jltype)
    elem_type = input_type <: Tile ? input_type.parameters[1] : input_type
    input_shape = input_tv.shape
    isempty(input_shape) && error("Cannot reduce scalar tile")

    # Compute output shape (dimension at axis is removed)
    output_shape = Int[input_shape[i] for i in eachindex(input_shape) if i != axis + 1]

    dtype = julia_to_tile_dtype!(tt, elem_type)

    # Output tile type
    output_tile_type = tile_type!(tt, dtype, output_shape)

    # Scalar type for reduction body (0D tile)
    scalar_tile_type = tile_type!(tt, dtype, Int[])

    # Create identity value - use simple dtype (f32), not tile type
    identity_val = reduce_fn == :add ? -0.0 : (reduce_fn == :max ? -Inf : 0.0)
    identity = FloatIdentity(identity_val, dtype, elem_type)

    # Emit ReduceOp
    results = encode_ReduceOp!(cb, [output_tile_type], [input_tv.v], axis, [identity], [scalar_tile_type]) do block_args
        acc, elem = block_args[1], block_args[2]

        if reduce_fn == :add
            res = encode_AddFOp!(cb, scalar_tile_type, acc, elem)
        elseif reduce_fn == :max
            res = encode_MaxFOp!(cb, scalar_tile_type, acc, elem)
        else
            error("Unsupported reduction function: $reduce_fn")
        end

        encode_YieldOp!(cb, [res])
    end

    TileValue(results[1], output_tile_type, Tile{elem_type, Tuple(output_shape)}, output_shape)
end

#=============================================================================
 Conditional Selection
=============================================================================#

function emit_intrinsic!(ctx::CodegenContext, ::typeof(where), args, @nospecialize(result_type))
    cb = ctx.cb

    cond_tv = emit_value!(ctx, args[1])
    x_tv = emit_value!(ctx, args[2])
    y_tv = emit_value!(ctx, args[3])

    (cond_tv === nothing || x_tv === nothing || y_tv === nothing) &&
        error("Cannot resolve operands for where()")

    result = encode_SelectOp!(cb, x_tv.type_id, cond_tv.v, x_tv.v, y_tv.v)

    TileValue(result, x_tv.type_id, x_tv.jltype, x_tv.shape)
end

#=============================================================================
 Tile Comparison Operations
=============================================================================#

function emit_tile_cmp!(ctx::CodegenContext, args, predicate::ComparisonPredicate)
    cb = ctx.cb
    tt = ctx.tt

    lhs = emit_value!(ctx, args[1])
    rhs = emit_value!(ctx, args[2])

    (lhs === nothing || rhs === nothing) && error("Cannot resolve operands for tile comparison")

    # Result type is boolean tile with same shape
    tile_shape = lhs.shape
    bool_tile_type = tile_type!(tt, I1(tt), tile_shape)

    # Determine element type to choose CmpFOp vs CmpIOp
    elem_type = unwrap_type(lhs.jltype)
    if elem_type <: Tile
        elem_type = elem_type.parameters[1]
    end

    result_v = if elem_type <: AbstractFloat
        encode_CmpFOp!(cb, bool_tile_type, lhs.v, rhs.v;
                       predicate=predicate, ordering=CmpOrdered)
    else
        encode_CmpIOp!(cb, bool_tile_type, lhs.v, rhs.v;
                       predicate=predicate, signedness=SignednessSigned)
    end

    TileValue(result_v, bool_tile_type, Tile{Bool, Tuple(tile_shape)}, tile_shape)
end

emit_intrinsic!(ctx::CodegenContext, ::typeof(tile_lt), args, @nospecialize(_)) =
    emit_tile_cmp!(ctx, args, CmpLessThan)

emit_intrinsic!(ctx::CodegenContext, ::typeof(tile_gt), args, @nospecialize(_)) =
    emit_tile_cmp!(ctx, args, CmpGreaterThan)

emit_intrinsic!(ctx::CodegenContext, ::typeof(tile_le), args, @nospecialize(_)) =
    emit_tile_cmp!(ctx, args, CmpLessThanOrEqual)

emit_intrinsic!(ctx::CodegenContext, ::typeof(tile_ge), args, @nospecialize(_)) =
    emit_tile_cmp!(ctx, args, CmpGreaterThanOrEqual)

emit_intrinsic!(ctx::CodegenContext, ::typeof(tile_eq), args, @nospecialize(_)) =
    emit_tile_cmp!(ctx, args, CmpEqual)

emit_intrinsic!(ctx::CodegenContext, ::typeof(tile_ne), args, @nospecialize(_)) =
    emit_tile_cmp!(ctx, args, CmpNotEqual)

#=============================================================================
 Constants
=============================================================================#

function emit_constant!(ctx::CodegenContext, @nospecialize(value), @nospecialize(result_type))
    result_type_unwrapped = unwrap_type(result_type)

    # Ghost types have no runtime representation
    if is_ghost_type(result_type_unwrapped)
        return ghost_value(result_type_unwrapped)
    end

    # Skip non-primitive types
    if !(result_type_unwrapped <: Number || result_type_unwrapped === Bool)
        return nothing
    end

    type_id = tile_type_for_julia!(ctx, result_type_unwrapped)
    bytes = constant_to_bytes(value, result_type_unwrapped)
    v = encode_ConstantOp!(ctx.cb, type_id, bytes)

    TileValue(v, type_id, result_type_unwrapped)
end

function constant_to_bytes(@nospecialize(value), @nospecialize(T::Type))
    if T === Bool
        return UInt8[value ? 0xff : 0x00]
    elseif T === Int32 || T === UInt32
        return collect(reinterpret(UInt8, [Int32(value)]))
    elseif T === Int64 || T === UInt64
        return collect(reinterpret(UInt8, [Int64(value)]))
    elseif T === Float32
        return collect(reinterpret(UInt8, [Float32(value)]))
    elseif T === Float64
        return collect(reinterpret(UInt8, [Float64(value)]))
    else
        error("Cannot convert $T to constant bytes")
    end
end

#=============================================================================
 Public API
=============================================================================#

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
