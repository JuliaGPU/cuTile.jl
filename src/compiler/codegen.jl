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
    cond_tv = emit_irvalue!(ctx, op.condition)
    cond_tv === nothing && error("Cannot resolve condition for IfOp")

    # Determine result types from the result_vars
    result_types = TypeId[]
    for result_var in op.result_vars
        result_type = ssatypes(ctx.target)[result_var.id]
        type_id = tile_type_for_julia!(ctx, result_type)
        push!(result_types, type_id)
    end

    # Emit IfOp with callback-based region building
    then_body = _ -> emit_block!(ctx, op.then_block)
    else_body = _ -> emit_block!(ctx, op.else_block)
    results = encode_IfOp!(then_body, else_body, cb, result_types, cond_tv.v)

    # Map result values to SSA values
    for (i, result_var) in enumerate(op.result_vars)
        if i <= length(results)
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
    lower_tv = emit_irvalue!(ctx, op.lower)
    upper_tv = emit_irvalue!(ctx, op.upper)
    step_tv = emit_irvalue!(ctx, op.step)

    (lower_tv === nothing || upper_tv === nothing || step_tv === nothing) &&
        error("Cannot resolve ForOp bounds")

    # Get init values
    init_values = Value[]
    for init_val in op.init_values
        tv = emit_irvalue!(ctx, init_val)
        tv === nothing && error("Cannot resolve ForOp init value")
        push!(init_values, tv.v)
    end

    # Determine result types
    result_types = TypeId[]
    for result_var in op.result_vars
        result_type = ssatypes(ctx.target)[result_var.id]
        type_id = tile_type_for_julia!(ctx, result_type)
        @debug "ForOp result_var $result_var: type=$result_type, type_id=$type_id"
        push!(result_types, type_id)
    end
    @debug "ForOp result_types: $result_types ($(length(result_types)) types)"

    # Emit ForOp with callback-based region building
    body_builder = function(block_args)
        # Map block arguments (induction var + carried values)
        if !isempty(op.body.args)
            # First block arg is induction variable
            iv_arg = op.body.args[1]
            iv_type = tile_type!(tt, I32(tt), Int[])
            iv_tv = TileValue(block_args[1], iv_type, Int32)
            ctx[iv_arg] = iv_tv
            # Also map the induction variable phi SSAValue
            ctx[op.iv_ssa] = iv_tv

            # Remaining are carried values - map both BlockArg and corresponding SSAValue (phi)
            for (i, body_arg) in enumerate(op.body.args[2:end])
                if i <= length(block_args) - 1 && i <= length(result_types)
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

        emit_block!(ctx, op.body)
    end
    results = encode_ForOp!(body_builder, cb, result_types, lower_tv.v, upper_tv.v, step_tv.v, init_values)

    # Map result values
    for (i, result_var) in enumerate(op.result_vars)
        if i <= length(results)
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
        tv = emit_irvalue!(ctx, init_val)
        tv === nothing && error("Cannot resolve LoopOp init value")
        push!(init_values, tv.v)
    end

    # Determine result types from init values
    result_types = TypeId[]
    for result_var in op.result_vars
        result_type = ssatypes(ctx.target)[result_var.id]
        type_id = tile_type_for_julia!(ctx, result_type)
        push!(result_types, type_id)
    end

    # Emit LoopOp with callback-based region building
    body_builder = function(block_args)
        # Map block arguments (carried values)
        for (i, body_arg) in enumerate(op.body.args)
            if i <= length(block_args) && i <= length(result_types)
                shape = extract_tile_shape(body_arg.type)
                ctx[body_arg] = TileValue(block_args[i], result_types[i], body_arg.type, shape)
            end
        end

        emit_block!(ctx, op.body)
    end
    results = encode_LoopOp!(body_builder, cb, result_types, init_values)

    # Map result values
    for (i, result_var) in enumerate(op.result_vars)
        if i <= length(results)
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
        tv = emit_irvalue!(ctx, val)
        @debug "YieldOp value $val => $(tv !== nothing ? tv : "nothing")"
        tv !== nothing && push!(operands, tv.v)
    end
    @debug "YieldOp operands: $operands ($(length(operands)) values)"
    encode_YieldOp!(ctx.cb, operands)
end

function emit_terminator!(ctx::CodegenContext, op::ContinueOp)
    # Collect continue operands (updated carried values)
    operands = Value[]
    for val in op.values
        tv = emit_irvalue!(ctx, val)
        tv !== nothing && push!(operands, tv.v)
    end
    encode_ContinueOp!(ctx.cb, operands)
end

function emit_terminator!(ctx::CodegenContext, op::BreakOp)
    # Collect break operands (final values)
    operands = Value[]
    for val in op.values
        tv = emit_irvalue!(ctx, val)
        tv !== nothing && push!(operands, tv.v)
    end
    encode_BreakOp!(ctx.cb, operands)
end

function emit_terminator!(ctx::CodegenContext, ::Nothing)
    # No terminator, nothing to emit
end

"""
    emit_irvalue!(ctx, ref::IRValue) -> Union{TileValue, Nothing}

Emit/resolve an IRValue reference.
"""
emit_irvalue!(ctx::CodegenContext, ssa::SSAValue) = ctx[ssa]
emit_irvalue!(ctx::CodegenContext, arg::Argument) = ctx[arg]
emit_irvalue!(ctx::CodegenContext, slot::SlotNumber) = ctx[slot]
emit_irvalue!(ctx::CodegenContext, block_arg::BlockArg) = ctx[block_arg]
emit_irvalue!(ctx::CodegenContext, lit::Literal) = emit_value!(ctx, lit.value)

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
emit_value!(ctx::CodegenContext, ssa::SSAValue) = ctx[ssa]
emit_value!(ctx::CodegenContext, arg::Argument) = ctx[arg]
emit_value!(ctx::CodegenContext, slot::SlotNumber) = ctx[slot]

function emit_value!(ctx::CodegenContext, val::Integer)
    type_id = tile_type_for_julia!(ctx, Int32)
    bytes = reinterpret(UInt8, [Int32(val)])
    v = encode_ConstantOp!(ctx.cb, type_id, collect(bytes))
    TileValue(v, type_id, Int32)
end

function emit_value!(ctx::CodegenContext, val::AbstractFloat)
    jltype = typeof(val)
    type_id = tile_type_for_julia!(ctx, jltype)
    bytes = reinterpret(UInt8, [jltype(val)])
    v = encode_ConstantOp!(ctx.cb, type_id, collect(bytes))
    TileValue(v, type_id, jltype)
end

function emit_value!(ctx::CodegenContext, node::QuoteNode)
    emit_value!(ctx, node.value)
end

emit_value!(ctx::CodegenContext, ::GlobalRef) = nothing
emit_value!(ctx::CodegenContext, ::Nothing) = nothing

# Fallback for other types (constants embedded in IR)
function emit_value!(ctx::CodegenContext, @nospecialize(val))
    # Try to interpret as a constant
    if val isa Type
        return nothing  # Type values are not runtime values
    end
    @warn "Unhandled value type in emit_value!" typeof(val)
    nothing
end

#=============================================================================
 Constant Extraction
=============================================================================#

"""
    extract_constant(ctx, ref) -> Union{Any, Nothing}

Extract a compile-time constant from an IR reference.
"""
extract_constant(ctx::CodegenContext, val::Integer) = val
extract_constant(ctx::CodegenContext, val::AbstractFloat) = val
extract_constant(ctx::CodegenContext, val::Symbol) = val
extract_constant(ctx::CodegenContext, val::Type) = val
extract_constant(ctx::CodegenContext, val::Tuple) = val

function extract_constant(ctx::CodegenContext, node::QuoteNode)
    extract_constant(ctx, node.value)
end

function extract_constant(ctx::CodegenContext, node::PiNode)
    # PiNode narrows the type. If the narrowed type is Type{T}, extract T
    if node.typ isa Type{<:Type}
        # Type{TFloat32} -> TFloat32
        return node.typ.parameters[1]
    end
    # Otherwise try to extract from the value
    extract_constant(ctx, node.val)
end

function extract_constant(ctx::CodegenContext, ssa::SSAValue)
    stmt = code(ctx.target)[ssa.id]
    extract_constant(ctx, stmt)
end

function extract_constant(ctx::CodegenContext, @nospecialize(val))
    # Handle Val{V} instances
    T = typeof(val)
    if T <: Val && length(T.parameters) == 1
        return T.parameters[1]
    end
    # Handle Constant{T, V} instances
    if T <: Constant && length(T.parameters) >= 2
        return T.parameters[2]
    end
    nothing
end

"""
    extract_tuple(ctx, ref) -> Union{Tuple, Nothing}

Extract a constant tuple from an IR reference.
"""
function extract_tuple(ctx::CodegenContext, @nospecialize(ref))
    val = extract_constant(ctx, ref)
    val isa Tuple ? val : nothing
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

    # Handle kwcall
    if func === Core.kwcall
        return emit_kwcall!(ctx, args, result_type)
    end

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

#-----------------------------------------------------------------------------
# cuTile intrinsics
#-----------------------------------------------------------------------------

function emit_intrinsic!(ctx::CodegenContext, ::typeof(bid), args, @nospecialize(result_type))
    axis = extract_constant(ctx, args[1])
    axis === nothing && error("bid() axis must be a compile-time constant")
    axis in (0, 1, 2) || error("bid() axis must be 0, 1, or 2, got $axis")

    res_type = tile_type!(ctx.tt, I32(ctx.tt), Int[])
    bid_x, bid_y, bid_z = encode_GetTileBlockIdOp!(ctx.cb, res_type, res_type, res_type)
    result = (bid_x, bid_y, bid_z)[axis + 1]

    TileValue(result, res_type, Int32)
end

function emit_intrinsic!(ctx::CodegenContext, ::typeof(num_blocks), args, @nospecialize(result_type))
    axis = extract_constant(ctx, args[1])
    axis === nothing && error("num_blocks() axis must be a compile-time constant")
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

function emit_binop!(ctx::CodegenContext, args, float_encoder::Function, int_encoder::Function)
    lhs = emit_value!(ctx, args[1])
    rhs = emit_value!(ctx, args[2])

    lhs === nothing && error("Cannot resolve LHS operand")
    rhs === nothing && error("Cannot resolve RHS operand")

    elem_type = unwrap_type(lhs.jltype)
    if elem_type <: Tile
        elem_type = elem_type.parameters[1]
    end

    cb = ctx.cb
    tt = ctx.tt

    # Compute broadcast shape
    result_shape = compute_broadcast_shape(lhs.shape, rhs.shape)
    dtype = julia_to_tile_dtype!(tt, elem_type)

    # Broadcast each operand to result shape if needed
    lhs_v = broadcast_tile_to_shape!(cb, tt, lhs, result_shape, dtype)
    rhs_v = broadcast_tile_to_shape!(cb, tt, rhs, result_shape, dtype)

    # Create result type
    result_type_id = tile_type!(tt, dtype, result_shape)

    # Perform the operation
    if elem_type <: AbstractFloat
        result_v = float_encoder(cb, result_type_id, lhs_v, rhs_v)
    else
        result_v = int_encoder(cb, result_type_id, lhs_v, rhs_v)
    end

    # Compute result Julia type
    result_jltype = Tile{elem_type, Tuple(result_shape)}

    TileValue(result_v, result_type_id, result_jltype, result_shape)
end

emit_intrinsic!(ctx::CodegenContext, ::typeof(tile_add), args, @nospecialize(_)) =
    emit_binop!(ctx, args, encode_AddFOp!, encode_AddIOp!)

emit_intrinsic!(ctx::CodegenContext, ::typeof(tile_sub), args, @nospecialize(_)) =
    emit_binop!(ctx, args, encode_SubFOp!, encode_SubIOp!)

emit_intrinsic!(ctx::CodegenContext, ::typeof(tile_mul), args, @nospecialize(_)) =
    emit_binop!(ctx, args, encode_MulFOp!, encode_MulIOp!)

# Base operators on Tiles
emit_intrinsic!(ctx::CodegenContext, ::typeof(Base.:(+)), args, @nospecialize(_)) =
    emit_binop!(ctx, args, encode_AddFOp!, encode_AddIOp!)

# Julia integer intrinsics (all are Core.IntrinsicFunction, so dispatch by value)
function emit_intrinsic!(ctx::CodegenContext, func::Core.IntrinsicFunction, args, @nospecialize(_))
    if func === Base.add_int
        return emit_binop!(ctx, args, encode_AddFOp!, encode_AddIOp!)
    elseif func === Base.sub_int
        return emit_binop!(ctx, args, encode_SubFOp!, encode_SubIOp!)
    elseif func === Base.mul_int
        return emit_binop!(ctx, args, encode_MulFOp!, encode_MulIOp!)
    end
    missing  # Unknown intrinsic
end

emit_intrinsic!(ctx::CodegenContext, ::typeof(Base.:(-)), args, @nospecialize(_)) =
    emit_binop!(ctx, args, encode_SubFOp!, encode_SubIOp!)

emit_intrinsic!(ctx::CodegenContext, ::typeof(Base.:(*)), args, @nospecialize(_)) =
    emit_binop!(ctx, args, encode_MulFOp!, encode_MulIOp!)

#-----------------------------------------------------------------------------
# Comparison operators
#-----------------------------------------------------------------------------

function emit_cmp!(ctx::CodegenContext, args, predicate::ComparisonPredicate)
    cb = ctx.cb
    tt = ctx.tt

    lhs = emit_value!(ctx, args[1])
    rhs = emit_value!(ctx, args[2])

    lhs === nothing && error("Cannot resolve LHS operand for comparison")
    rhs === nothing && error("Cannot resolve RHS operand for comparison")

    # Result type is a boolean (i1) scalar
    result_type = tile_type!(tt, I1(tt), Int[])

    result_v = encode_CmpIOp!(cb, result_type, lhs.v, rhs.v;
                              predicate=predicate, signedness=SignednessSigned)

    TileValue(result_v, result_type, Bool)
end

emit_intrinsic!(ctx::CodegenContext, ::typeof(Base.:(>)), args, @nospecialize(_)) =
    emit_cmp!(ctx, args, CmpGreaterThan)

emit_intrinsic!(ctx::CodegenContext, ::typeof(Base.:(<)), args, @nospecialize(_)) =
    emit_cmp!(ctx, args, CmpLessThan)

emit_intrinsic!(ctx::CodegenContext, ::typeof(Base.:(>=)), args, @nospecialize(_)) =
    emit_cmp!(ctx, args, CmpGreaterThanOrEqual)

emit_intrinsic!(ctx::CodegenContext, ::typeof(Base.:(<=)), args, @nospecialize(_)) =
    emit_cmp!(ctx, args, CmpLessThanOrEqual)

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
    field = extract_constant(ctx, field_arg)

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
    index = extract_constant(ctx, index_arg)
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
 Keyword Call Handling
=============================================================================#

function emit_kwcall!(ctx::CodegenContext, args::Vector{Any}, @nospecialize(result_type))
    # args[1] = Core.kwcall
    # args[2] = kwargs NamedTuple
    # args[3] = the actual function
    # args[4:end] = positional arguments

    kwargs_ref = args[2]
    func_ref = args[3]
    positional_args = args[4:end]

    func = resolve_function(ctx, func_ref)
    kwargs = extract_kwargs(ctx, kwargs_ref)

    if func === load || func === _load
        return emit_load_kwcall!(ctx, positional_args, kwargs, result_type)
    elseif func === store
        return emit_store_kwcall!(ctx, positional_args, kwargs, result_type)
    else
        error("Unknown kwcall: $func")
    end
end

function extract_kwargs(ctx::CodegenContext, @nospecialize(ref))
    kwargs = Dict{Symbol, Any}()

    if ref isa SSAValue
        stmt = code(ctx.target)[ref.id]
        if stmt isa Expr && stmt.head === :new
            nt_type = stmt.args[1]
            values = stmt.args[2:end]
            if nt_type isa Type && nt_type <: NamedTuple
                names = fieldnames(nt_type)
                for (name, val) in zip(names, values)
                    kwargs[name] = val
                end
            end
        end
    end

    return kwargs
end

function emit_load_kwcall!(ctx::CodegenContext, positional_args::Vector{Any},
                           kwargs::Dict{Symbol, Any}, @nospecialize(result_type))
    array_arg = positional_args[1]
    index_arg = get(kwargs, :index, nothing)
    shape_arg = get(kwargs, :shape, nothing)

    index_arg === nothing && error("load() requires index keyword argument")
    shape_arg === nothing && error("load() requires shape keyword argument")

    emit_load!(ctx, Any[array_arg, index_arg, shape_arg], result_type)
end

function emit_store_kwcall!(ctx::CodegenContext, positional_args::Vector{Any},
                            kwargs::Dict{Symbol, Any}, @nospecialize(result_type))
    array_arg = positional_args[1]
    index_arg = get(kwargs, :index, nothing)
    tile_arg = get(kwargs, :tile, nothing)

    index_arg === nothing && error("store() requires index keyword argument")
    tile_arg === nothing && error("store() requires tile keyword argument")

    emit_store!(ctx, Any[array_arg, index_arg, tile_arg], result_type)
    nothing
end

#=============================================================================
 Load/Store Operations
=============================================================================#

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
        shape = extract_constant(ctx, args[3])
        if shape isa Tuple
            tile_shape = collect(Int, shape)
        end
    end

    ndim = length(tile_shape)

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
    pv_type = partition_view_type!(tt, tile_shape, tv_type, dim_map, PaddingZero)

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
    tile_val, _ = encode_LoadViewTkoOp!(cb, tile_type, token_type, partition, index_vals; token=ctx.token)

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
    encode_StoreViewTkoOp!(cb, token_type, tile_tv.v, partition, index_vals; token=ctx.token)

    nothing
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
                    tv !== nothing && push!(index_vals, tv.v)
                end
                return index_vals
            end
        end
        # Single value
        tv = emit_value!(ctx, index_arg)
        tv !== nothing && push!(index_vals, tv.v)
    else
        tv = emit_value!(ctx, index_arg)
        tv !== nothing && push!(index_vals, tv.v)
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
    target_elem = extract_constant(ctx, args[2])
    target_elem === nothing && error("astype() requires a compile-time constant type")
    target_elem isa Type || error("astype() second argument must be a Type")

    # Same type? Return source unchanged
    if source_elem === target_elem
        return source
    end

    # Create target type
    target_dtype = julia_to_tile_dtype!(tt, target_elem)
    target_tile_type = tile_type!(tt, target_dtype, tile_shape)

    # Emit conversion
    result = if source_elem <: AbstractFloat && target_elem <: AbstractFloat
        encode_FToFOp!(cb, target_tile_type, source.v)
    else
        error("astype() only supports float-to-float conversion currently (got $source_elem -> $target_elem)")
    end

    TileValue(result, target_tile_type, Tile{target_elem, Tuple(tile_shape)}, tile_shape)
end

#=============================================================================
 Tile Construction
=============================================================================#

function emit_full!(ctx::CodegenContext, args::AbstractVector, @nospecialize(result_type))
    cb = ctx.cb
    tt = ctx.tt

    # Extract shape
    shape = extract_constant(ctx, args[1])
    shape isa Tuple || error("full() shape must be a compile-time constant tuple")
    tile_shape = collect(Int, shape)

    # Extract value
    value = extract_constant(ctx, args[2])
    value === nothing && error("full() value must be a compile-time constant")

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
    axis = extract_constant(ctx, args[2])
    axis === nothing && error("num_tiles() axis must be a compile-time constant")

    shape = extract_tuple(ctx, args[3])
    shape === nothing && error("num_tiles() shape must be a compile-time constant tuple")

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
    if tv !== nothing
        tv.v !== nothing && return tv.v
    end

    val = extract_constant(ctx, arg)
    val !== nothing || error("Cannot resolve argument")

    bytes = reinterpret(UInt8, [Int32(val)])
    encode_ConstantOp!(ctx.cb, type_id, collect(bytes))
end

#=============================================================================
 Floating-point Division
=============================================================================#

emit_intrinsic!(ctx::CodegenContext, ::typeof(tile_div), args, @nospecialize(_)) =
    emit_binop!(ctx, args, encode_DivFOp!, encode_DivIOp!)

function emit_intrinsic!(ctx::CodegenContext, ::typeof(tile_div_scalar), args, @nospecialize(result_type))
    cb = ctx.cb
    tt = ctx.tt

    tile_tv = emit_value!(ctx, args[1])
    tile_tv === nothing && error("Cannot resolve tile operand for division")

    # Get the scalar value
    scalar_val = extract_constant(ctx, args[2])
    scalar_val === nothing && error("Division by scalar requires compile-time constant")

    # Get element type and shape
    elem_type = unwrap_type(tile_tv.jltype)
    if elem_type <: Tile
        elem_type = elem_type.parameters[1]
    end
    tile_shape = tile_tv.shape

    dtype = julia_to_tile_dtype!(tt, elem_type)

    # Create scalar constant
    scalar_type = tile_type!(tt, dtype, Int[])
    value_bytes = constant_to_bytes(scalar_val, elem_type)
    scalar_const = encode_ConstantOp!(cb, scalar_type, value_bytes)

    # Broadcast scalar to tile shape
    if !isempty(tile_shape)
        ones_shape = fill(1, length(tile_shape))
        reshaped_type = tile_type!(tt, dtype, ones_shape)
        reshaped_val = encode_ReshapeOp!(cb, reshaped_type, scalar_const)
        broadcast_type = tile_type!(tt, dtype, tile_shape)
        broadcasted = encode_BroadcastOp!(cb, broadcast_type, reshaped_val)
    else
        broadcasted = scalar_const
    end

    # Perform division
    result = encode_DivFOp!(cb, tile_tv.type_id, tile_tv.v, broadcasted)

    TileValue(result, tile_tv.type_id, tile_tv.jltype, tile_shape)
end

# tile / integer - convert integer to float, then broadcast and divide
function emit_intrinsic!(ctx::CodegenContext, ::typeof(tile_div_int), args, @nospecialize(result_type))
    cb = ctx.cb
    tt = ctx.tt

    tile_tv = emit_value!(ctx, args[1])
    tile_tv === nothing && error("Cannot resolve tile operand for division")

    # Get the integer value
    int_val = extract_constant(ctx, args[2])
    int_val === nothing && error("Division by integer requires compile-time constant")

    # Get element type and shape
    elem_type = unwrap_type(tile_tv.jltype)
    if elem_type <: Tile
        elem_type = elem_type.parameters[1]
    end
    tile_shape = tile_tv.shape

    dtype = julia_to_tile_dtype!(tt, elem_type)

    # Create scalar constant (convert int to float)
    scalar_type = tile_type!(tt, dtype, Int[])
    float_val = elem_type(int_val)
    value_bytes = constant_to_bytes(float_val, elem_type)
    scalar_const = encode_ConstantOp!(cb, scalar_type, value_bytes)

    # Broadcast scalar to tile shape
    broadcasted = broadcast_scalar_to_tile!(cb, tt, scalar_const, dtype, tile_shape)

    # Perform division
    result = encode_DivFOp!(cb, tile_tv.type_id, tile_tv.v, broadcasted)

    TileValue(result, tile_tv.type_id, tile_tv.jltype, tile_shape)
end

# scalar / tile - broadcast scalar and divide
function emit_intrinsic!(ctx::CodegenContext, ::typeof(scalar_div_tile), args, @nospecialize(result_type))
    cb = ctx.cb
    tt = ctx.tt

    # Get the scalar value
    scalar_val = extract_constant(ctx, args[1])
    scalar_val === nothing && error("Scalar division requires compile-time constant numerator")

    tile_tv = emit_value!(ctx, args[2])
    tile_tv === nothing && error("Cannot resolve tile operand for division")

    # Get element type and shape
    elem_type = unwrap_type(tile_tv.jltype)
    if elem_type <: Tile
        elem_type = elem_type.parameters[1]
    end
    tile_shape = tile_tv.shape

    dtype = julia_to_tile_dtype!(tt, elem_type)

    # Create scalar constant
    scalar_type = tile_type!(tt, dtype, Int[])
    value_bytes = constant_to_bytes(scalar_val, elem_type)
    scalar_const = encode_ConstantOp!(cb, scalar_type, value_bytes)

    # Broadcast scalar to tile shape
    broadcasted = broadcast_scalar_to_tile!(cb, tt, scalar_const, dtype, tile_shape)

    # Perform division (scalar / tile)
    result = encode_DivFOp!(cb, tile_tv.type_id, broadcasted, tile_tv.v)

    TileValue(result, tile_tv.type_id, tile_tv.jltype, tile_shape)
end

# tile + scalar
function emit_intrinsic!(ctx::CodegenContext, ::typeof(tile_add_scalar), args, @nospecialize(result_type))
    cb = ctx.cb
    tt = ctx.tt

    tile_tv = emit_value!(ctx, args[1])
    tile_tv === nothing && error("Cannot resolve tile operand for addition")

    # Get the scalar value
    scalar_val = extract_constant(ctx, args[2])
    scalar_val === nothing && error("Scalar addition requires compile-time constant")

    # Get element type and shape
    elem_type = unwrap_type(tile_tv.jltype)
    if elem_type <: Tile
        elem_type = elem_type.parameters[1]
    end
    tile_shape = tile_tv.shape

    dtype = julia_to_tile_dtype!(tt, elem_type)

    # Create scalar constant
    scalar_type = tile_type!(tt, dtype, Int[])
    value_bytes = constant_to_bytes(scalar_val, elem_type)
    scalar_const = encode_ConstantOp!(cb, scalar_type, value_bytes)

    # Broadcast scalar to tile shape
    broadcasted = broadcast_scalar_to_tile!(cb, tt, scalar_const, dtype, tile_shape)

    # Perform addition
    result = encode_AddFOp!(cb, tile_tv.type_id, tile_tv.v, broadcasted)

    TileValue(result, tile_tv.type_id, tile_tv.jltype, tile_shape)
end

# tile - scalar
function emit_intrinsic!(ctx::CodegenContext, ::typeof(tile_sub_scalar), args, @nospecialize(result_type))
    cb = ctx.cb
    tt = ctx.tt

    tile_tv = emit_value!(ctx, args[1])
    tile_tv === nothing && error("Cannot resolve tile operand for subtraction")

    # Get the scalar value
    scalar_val = extract_constant(ctx, args[2])
    scalar_val === nothing && error("Scalar subtraction requires compile-time constant")

    # Get element type and shape
    elem_type = unwrap_type(tile_tv.jltype)
    if elem_type <: Tile
        elem_type = elem_type.parameters[1]
    end
    tile_shape = tile_tv.shape

    dtype = julia_to_tile_dtype!(tt, elem_type)

    # Create scalar constant
    scalar_type = tile_type!(tt, dtype, Int[])
    value_bytes = constant_to_bytes(scalar_val, elem_type)
    scalar_const = encode_ConstantOp!(cb, scalar_type, value_bytes)

    # Broadcast scalar to tile shape
    broadcasted = broadcast_scalar_to_tile!(cb, tt, scalar_const, dtype, tile_shape)

    # Perform subtraction
    result = encode_SubFOp!(cb, tile_tv.type_id, tile_tv.v, broadcasted)

    TileValue(result, tile_tv.type_id, tile_tv.jltype, tile_shape)
end

# scalar - tile
function emit_intrinsic!(ctx::CodegenContext, ::typeof(scalar_sub_tile), args, @nospecialize(result_type))
    cb = ctx.cb
    tt = ctx.tt

    # Get the scalar value
    scalar_val = extract_constant(ctx, args[1])
    scalar_val === nothing && error("Scalar subtraction requires compile-time constant")

    tile_tv = emit_value!(ctx, args[2])
    tile_tv === nothing && error("Cannot resolve tile operand for subtraction")

    # Get element type and shape
    elem_type = unwrap_type(tile_tv.jltype)
    if elem_type <: Tile
        elem_type = elem_type.parameters[1]
    end
    tile_shape = tile_tv.shape

    dtype = julia_to_tile_dtype!(tt, elem_type)

    # Create scalar constant
    scalar_type = tile_type!(tt, dtype, Int[])
    value_bytes = constant_to_bytes(scalar_val, elem_type)
    scalar_const = encode_ConstantOp!(cb, scalar_type, value_bytes)

    # Broadcast scalar to tile shape
    broadcasted = broadcast_scalar_to_tile!(cb, tt, scalar_const, dtype, tile_shape)

    # Perform subtraction (scalar - tile)
    result = encode_SubFOp!(cb, tile_tv.type_id, broadcasted, tile_tv.v)

    TileValue(result, tile_tv.type_id, tile_tv.jltype, tile_shape)
end

# tile * scalar
function emit_intrinsic!(ctx::CodegenContext, ::typeof(tile_mul_scalar), args, @nospecialize(result_type))
    cb = ctx.cb
    tt = ctx.tt

    tile_tv = emit_value!(ctx, args[1])
    tile_tv === nothing && error("Cannot resolve tile operand for multiplication")

    # Get the scalar value
    scalar_val = extract_constant(ctx, args[2])
    scalar_val === nothing && error("Scalar multiplication requires compile-time constant")

    # Get element type and shape
    elem_type = unwrap_type(tile_tv.jltype)
    if elem_type <: Tile
        elem_type = elem_type.parameters[1]
    end
    tile_shape = tile_tv.shape

    dtype = julia_to_tile_dtype!(tt, elem_type)

    # Create scalar constant
    scalar_type = tile_type!(tt, dtype, Int[])
    value_bytes = constant_to_bytes(scalar_val, elem_type)
    scalar_const = encode_ConstantOp!(cb, scalar_type, value_bytes)

    # Broadcast scalar to tile shape
    broadcasted = broadcast_scalar_to_tile!(cb, tt, scalar_const, dtype, tile_shape)

    # Perform multiplication
    result = encode_MulFOp!(cb, tile_tv.type_id, tile_tv.v, broadcasted)

    TileValue(result, tile_tv.type_id, tile_tv.jltype, tile_shape)
end

# Helper to broadcast a scalar constant to tile shape
function broadcast_scalar_to_tile!(cb::CodeBuilder, tt::TypeTable, scalar_const::Value,
                                    dtype::TypeId, tile_shape::Vector{Int})
    if isempty(tile_shape)
        return scalar_const
    end

    ones_shape = fill(1, length(tile_shape))
    reshaped_type = tile_type!(tt, dtype, ones_shape)
    reshaped_val = encode_ReshapeOp!(cb, reshaped_type, scalar_const)
    broadcast_type = tile_type!(tt, dtype, tile_shape)
    encode_BroadcastOp!(cb, broadcast_type, reshaped_val)
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
    shape = extract_constant(ctx, args[1])
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
    axis = extract_constant(ctx, args[2])
    axis === nothing && error("Reduction axis must be a compile-time constant")

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

function emit_tile_cmpf!(ctx::CodegenContext, args, predicate::ComparisonPredicate)
    cb = ctx.cb
    tt = ctx.tt

    lhs = emit_value!(ctx, args[1])
    rhs = emit_value!(ctx, args[2])

    (lhs === nothing || rhs === nothing) && error("Cannot resolve operands for tile comparison")

    # Result type is boolean tile with same shape
    tile_shape = lhs.shape
    bool_tile_type = tile_type!(tt, I1(tt), tile_shape)

    result_v = encode_CmpFOp!(cb, bool_tile_type, lhs.v, rhs.v;
                              predicate=predicate, ordering=CmpOrdered)

    TileValue(result_v, bool_tile_type, Tile{Bool, Tuple(tile_shape)}, tile_shape)
end

emit_intrinsic!(ctx::CodegenContext, ::typeof(tile_gt), args, @nospecialize(_)) =
    emit_tile_cmpf!(ctx, args, CmpGreaterThan)

emit_intrinsic!(ctx::CodegenContext, ::typeof(tile_lt), args, @nospecialize(_)) =
    emit_tile_cmpf!(ctx, args, CmpLessThan)

emit_intrinsic!(ctx::CodegenContext, ::typeof(tile_ge), args, @nospecialize(_)) =
    emit_tile_cmpf!(ctx, args, CmpGreaterThanOrEqual)

emit_intrinsic!(ctx::CodegenContext, ::typeof(tile_le), args, @nospecialize(_)) =
    emit_tile_cmpf!(ctx, args, CmpLessThanOrEqual)

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
