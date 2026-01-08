# structured IR pretty printing

function Base.show(io::IO, ::MIME"text/plain", sci::StructuredIRCode)
    color = get(io, :color, false)::Bool

    # Print header
    println(io, "StructuredIRCode(")

    # Create printer with │ prefix for the entry block (2 chars, not 4)
    base_p = IRPrinter(io, sci.max_ssa_idx)
    p = child_printer(base_p, sci.entry, "│ ")

    # Print entry block body - use is_closing_self=true so the last item
    # replaces │ with └── instead of adding └── after │
    print_block_body(p, sci.entry; is_closing_self=true)

    print(io, ")")
end

# Use detailed format for string() as well (like CodeInfo does)
function Base.show(io::IO, sci::StructuredIRCode)
    show(io, MIME"text/plain"(), sci)
end

"""
    IRPrinter

Context for printing structured IR with proper indentation and value formatting.
Uses Julia's IRCode style with box-drawing characters.
"""
mutable struct IRPrinter
    io::IO
    indent::Int
    line_prefix::String    # Prefix for continuation lines (│, spaces)
    max_idx_width::Int     # Max width of "%N = " for alignment
    color::Bool            # Whether to use colors
end

function IRPrinter(io::IO, max_ssa_idx::Int)
    # Compute max index width for alignment: "%N = " where N is the max index
    max_idx_width = ndigits(max_ssa_idx) + 4  # % + digits + space + = + space
    color = get(io, :color, false)::Bool
    IRPrinter(io, 0, "", max_idx_width, color)
end

function indent(p::IRPrinter, n::Int=1)
    new_prefix = p.line_prefix * "  "  # 2 spaces per indent level
    return IRPrinter(p.io, p.indent + n, new_prefix, p.max_idx_width, p.color)
end

# Create a child printer for a nested Block
function child_printer(p::IRPrinter, nested_block::Block, cont_prefix::String)
    # Use parent's max_idx_width since SSA indices are global
    IRPrinter(p.io, p.indent + 1, p.line_prefix * cont_prefix, p.max_idx_width, p.color)
end

# Print region header: "├ label:" or "└ label:" for last/empty region
function print_region_header(p::IRPrinter, label::String, args::Vector{BlockArg}; is_last::Bool=false)
    print_indent(p)
    print_colored(p, is_last ? "└" : "├", :light_black)
    print(p.io, " ", label)
    if !isempty(args)
        print(p.io, "(")
        for (i, arg) in enumerate(args)
            i > 1 && print(p.io, ", ")
            print(p.io, "%arg", arg.id)
            print_colored(p, string("::", format_type(arg.type)), :cyan)
        end
        print(p.io, ")")
    end
    println(p.io, ":")
end

function print_indent(p::IRPrinter)
    # Color the line prefix (box-drawing characters from parent blocks)
    print_colored(p, p.line_prefix, :light_black)
end

# Helper for colored output
function print_colored(p::IRPrinter, s, color::Symbol)
    if p.color
        printstyled(p.io, s; color=color)
    else
        print(p.io, s)
    end
end

# Print an IR value (no special coloring, like Julia's code_typed)
function print_value(p::IRPrinter, v::SSAValue)
    print(p.io, "%", v.id)
end

function print_value(p::IRPrinter, v::BlockArg)
    print(p.io, "%arg", v.id)
end

function print_value(p::IRPrinter, v::Argument)
    # Print argument as _N (IRCode doesn't have slot names)
    print(p.io, "_", v.n)
end

function print_value(p::IRPrinter, v::SlotNumber)
    print(p.io, "slot#", v.id)
end

function print_value(p::IRPrinter, v::QuoteNode)
    print(p.io, repr(v.value))
end

function print_value(p::IRPrinter, v::GlobalRef)
    print(p.io, v.mod, ".", v.name)
end

function print_value(p::IRPrinter, v)
    print(p.io, repr(v))
end

# String versions for places that need strings (e.g., join)
function format_value(p::IRPrinter, v::SSAValue)
    string("%", v.id)
end
function format_value(p::IRPrinter, v::BlockArg)
    string("%arg", v.id)
end
function format_value(p::IRPrinter, v::Argument)
    string("_", v.n)
end
function format_value(p::IRPrinter, v::SlotNumber)
    string("slot#", v.id)
end
function format_value(p::IRPrinter, v::QuoteNode)
    repr(v.value)
end
function format_value(p::IRPrinter, v::GlobalRef)
    string(v.mod, ".", v.name)
end
function format_value(p::IRPrinter, v)
    repr(v)
end

# Format type for printing (compact form)
function format_type(t)
    if t isa Core.Const
        string("Const(", repr(t.val), ")")
    elseif t isa Type
        string(t)
    else
        string(t)
    end
end

# Print type annotation
function print_type_annotation(p::IRPrinter, t)
    print_colored(p, string("::", format_type(t)), :cyan)
end

# Check if a function reference is an intrinsic
function is_intrinsic_call(func)
    if func isa GlobalRef
        try
            f = getfield(func.mod, func.name)
            return f isa Core.IntrinsicFunction
        catch
            return false
        end
    end
    return false
end

# Print an expression (RHS of a statement)
function print_expr(p::IRPrinter, expr::Expr)
    if expr.head == :call
        func = expr.args[1]
        args = expr.args[2:end]
        # Check if this is an intrinsic call
        if is_intrinsic_call(func)
            print_colored(p, "intrinsic ", :light_black)
        end
        print_value(p, func)
        print(p.io, "(")
        for (i, a) in enumerate(args)
            i > 1 && print(p.io, ", ")
            print_value(p, a)
        end
        print(p.io, ")")
    elseif expr.head == :invoke
        mi = expr.args[1]
        func = expr.args[2]
        args = expr.args[3:end]
        print_colored(p, "invoke ", :light_black)
        if mi isa Core.MethodInstance
            print(p.io, mi.def.name)
            # Get argument types from MethodInstance signature
            sig = mi.specTypes isa DataType ? mi.specTypes.parameters : ()
            print(p.io, "(")
            for (i, a) in enumerate(args)
                i > 1 && print(p.io, ", ")
                print_value(p, a)
                # Print type annotation if available (sig includes function type at position 1)
                if i + 1 <= length(sig)
                    print_colored(p, string("::", sig[i + 1]), :cyan)
                end
            end
            print(p.io, ")")
        else
            print_value(p, func)
            print(p.io, "(")
            for (i, a) in enumerate(args)
                i > 1 && print(p.io, ", ")
                print_value(p, a)
            end
            print(p.io, ")")
        end
    elseif expr.head == :new
        print(p.io, "new ", expr.args[1], "(")
        for (i, a) in enumerate(expr.args[2:end])
            i > 1 && print(p.io, ", ")
            print_value(p, a)
        end
        print(p.io, ")")
    elseif expr.head == :foreigncall
        print(p.io, "foreigncall ", repr(expr.args[1]))
    elseif expr.head == :boundscheck
        print(p.io, "boundscheck")
    else
        print(p.io, expr.head, " ")
        for (i, a) in enumerate(expr.args)
            i > 1 && print(p.io, ", ")
            print_value(p, a)
        end
    end
end

function print_expr(p::IRPrinter, v)
    print_value(p, v)
end

# Print block arguments (for loops and structured control flow)
function print_block_args(p::IRPrinter, args::Vector{BlockArg})
    if isempty(args)
        return
    end
    print(p.io, "(")
    for (i, a) in enumerate(args)
        i > 1 && print(p.io, ", ")
        print(p.io, "%arg", a.id)
        # Block args are always "used" within their scope
        print_colored(p, string("::", format_type(a.type)), :cyan)
    end
    print(p.io, ")")
end

# Print initial values (carries) with their types
function print_init_values(p::IRPrinter, carry_args::Vector{BlockArg}, init_values::Vector{IRValue})
    if isempty(carry_args)
        return
    end
    print(p.io, " init(")
    for (i, (arg, init)) in enumerate(zip(carry_args, init_values))
        i > 1 && print(p.io, ", ")
        # Use same naming as BlockArg print_value for consistency
        print(p.io, "%arg", arg.id, " = ")
        print_value(p, init)
        print_colored(p, string("::", format_type(arg.type)), :cyan)
    end
    print(p.io, ")")
end

function print_loop_args(p::IRPrinter, block_args::Vector{BlockArg}, init_values::Vector{IRValue})
    print_init_values(p, block_args, init_values)
end

# Print a terminator
# is_last_in_block: if true, replace trailing │ with └ to close the block
function print_terminator(p::IRPrinter, term::ReturnNode; is_last_in_block::Bool=false)
    if is_last_in_block && endswith(p.line_prefix, "│   ")
        closing_prefix = chop(p.line_prefix; tail=4)
        print_colored(p, closing_prefix, :light_black)
        print_colored(p, "└   ", :light_black)
    else
        print_indent(p)
    end
    print(p.io, "return")
    if isdefined(term, :val)
        print(p.io, " ")
        print_value(p, term.val)
    end
    println(p.io)
end

function print_terminator(p::IRPrinter, term::Union{YieldOp,ContinueOp,BreakOp,ConditionOp}; is_last_in_block::Bool=false)
    if is_last_in_block && endswith(p.line_prefix, "│   ")
        closing_prefix = chop(p.line_prefix; tail=4)
        print_colored(p, closing_prefix, :light_black)
        print_colored(p, "└   ", :light_black)
    else
        print_indent(p)
    end
    print_terminator_content(p, term)
    println(p.io)
end

function print_terminator(p::IRPrinter, ::Nothing; is_last_in_block::Bool=false)
    # No terminator
end

# Print terminator when closing the block itself (replace trailing │ with └)
function print_terminator_closing_self(p::IRPrinter, term)
    # Replace trailing "│ " or "│   " with "└ " in the prefix
    if endswith(p.line_prefix, "│   ")
        closing_prefix = chop(p.line_prefix; tail=4)
        print_colored(p, closing_prefix, :light_black)
        print_colored(p, "└ ", :light_black)
    elseif endswith(p.line_prefix, "│ ")
        closing_prefix = chop(p.line_prefix; tail=2)
        print_colored(p, closing_prefix, :light_black)
        print_colored(p, "└ ", :light_black)
    else
        print_indent(p)
    end
    print_terminator_content(p, term)
    println(p.io)
end

# Print expression with type annotation (no box-drawing, just indent)
function print_expr_with_type(p::IRPrinter, idx::Int, expr, typ)
    print_indent(p)

    # Print %N = assignment
    idx_s = string(idx)
    pad = " "^(p.max_idx_width - length(idx_s) - 4)  # -4 for "% = "
    print(p.io, "%", idx_s, pad, " = ")
    print_expr(p, expr)

    # Print type annotation
    print_colored(p, string("::", format_type(typ)), :cyan)
    println(p.io)
end

# Print a Block's contents
# is_last_in_parent: if true, the terminator should close with └── (added after prefix)
# is_closing_self: if true, the last item replaces the trailing │ with └── (for entry block)
function print_block_body(p::IRPrinter, block::Block; is_last_in_parent::Bool=false, is_closing_self::Bool=false)
    items = []
    for (i, (idx, entry)) in enumerate(block.body)
        if entry.stmt isa ControlFlowOp
            push!(items, (:nested, idx, entry.stmt, entry.typ))
        else
            push!(items, (:expr, idx, entry.stmt, entry.typ))
        end
    end
    if block.terminator !== nothing
        push!(items, (:term, block.terminator))
    end

    for (i, item) in enumerate(items)
        is_last = (i == length(items))
        if item[1] == :expr
            # No box-drawing for expressions, just print with indent
            print_expr_with_type(p, item[2], item[3], item[4])
        elseif item[1] == :nested
            # Control flow ops handle their own box-drawing
            print_control_flow(p, item[3], item[2], item[4]; is_last=is_last && is_last_in_parent)
        else  # :term
            if is_last && is_closing_self
                # Closing this block itself: replace trailing │ with └──
                print_terminator_closing_self(p, item[2])
            else
                # Terminators get └── only if this is the last item AND we're closing the parent block
                print_terminator(p, item[2]; is_last_in_block=is_last && is_last_in_parent)
            end
        end
    end
end

# Print just the terminator content (keyword + values), no prefix or newline
function print_terminator_content(p::IRPrinter, term::YieldOp)
    print_colored(p, "yield", :yellow)
    print_terminator_values(p, term.values)
end
function print_terminator_content(p::IRPrinter, term::ContinueOp)
    print_colored(p, "continue", :yellow)
    print_terminator_values(p, term.values)
end
function print_terminator_content(p::IRPrinter, term::BreakOp)
    print_colored(p, "break", :yellow)
    print_terminator_values(p, term.values)
end
function print_terminator_content(p::IRPrinter, term::ConditionOp)
    print_colored(p, "condition", :yellow)
    print(p.io, "(")
    print_value(p, term.condition)
    print(p.io, ")")
    print_terminator_values(p, term.args)
end
function print_terminator_content(p::IRPrinter, term::ReturnNode)
    print(p.io, "return")
    if isdefined(term, :val)
        print(p.io, " ")
        print_value(p, term.val)
    end
end

function print_terminator_values(p::IRPrinter, values)
    if !isempty(values)
        print(p.io, " ")
        for (i, v) in enumerate(values)
            i > 1 && print(p.io, ", ")
            print_value(p, v)
        end
    end
end

# Print ControlFlowOp (final type, dispatches via multiple dispatch)
print_control_flow(p::IRPrinter, op::IfOp, pos::Int, @nospecialize(result_type); is_last::Bool=false) = print_if_op_final(p, op, pos, result_type; is_last)
print_control_flow(p::IRPrinter, op::ForOp, pos::Int, @nospecialize(result_type); is_last::Bool=false) = print_for_op_final(p, op, pos, result_type; is_last)
print_control_flow(p::IRPrinter, op::WhileOp, pos::Int, @nospecialize(result_type); is_last::Bool=false) = print_while_op_final(p, op, pos, result_type; is_last)
print_control_flow(p::IRPrinter, op::LoopOp, pos::Int, @nospecialize(result_type); is_last::Bool=false) = print_loop_op_final(p, op, pos, result_type; is_last)

function print_if_op_final(p::IRPrinter, op::IfOp, pos::Int, @nospecialize(result_type); is_last::Bool=false)
    # Print the if header (no box-drawing prefix for the op itself)
    print_indent(p)
    idx_s = string(pos)
    pad = " "^(p.max_idx_width - length(idx_s) - 4)
    print(p.io, "%", idx_s, pad, " = if ")
    print_value(p, op.condition)

    # Show return type annotation
    if result_type !== nothing
        print_colored(p, " -> ", :light_black)
        print_colored(p, string(result_type), :cyan)
    end

    println(p.io)

    # Check if else region is empty (no body, no terminator)
    else_is_empty = isempty(op.else_region.body) && op.else_region.terminator === nothing
    # Check if else has only a terminator (no body statements)
    else_is_just_terminator = isempty(op.else_region.body) && op.else_region.terminator !== nothing

    # Print "then:" region header (at same level as the if)
    print_region_header(p, "then", op.then_region.args; is_last=false)
    # Print then region body (indented one level)
    # If else is empty, then's last item closes the if with └──
    then_body_p = child_printer(p, op.then_region, "│   ")
    print_block_body(then_body_p, op.then_region; is_last_in_parent=else_is_empty)

    # Handle else region
    if else_is_empty
        # Empty else: just print └ else: to close the if
        print_region_header(p, "else", op.else_region.args; is_last=true)
    else
        # Else has content: print region header and body
        print_region_header(p, "else", op.else_region.args; is_last=false)
        else_body_p = child_printer(p, op.else_region, "│   ")
        print_block_body(else_body_p, op.else_region; is_last_in_parent=true)
    end
end

function print_for_op_final(p::IRPrinter, op::ForOp, pos::Int, @nospecialize(result_type); is_last::Bool=false)
    cont_prefix = is_last ? "    " : "│   "

    # Print the for header (no box-drawing prefix)
    print_indent(p)
    idx_s = string(pos)
    pad = " "^(p.max_idx_width - length(idx_s) - 4)
    print(p.io, "%", idx_s, pad, " = ")
    print_colored(p, "for", :yellow)
    print(p.io, " %arg", op.iv_arg.id, " = ")
    print_value(p, op.lower)
    print(p.io, ":")
    print_value(p, op.step)
    print(p.io, ":")
    print_value(p, op.upper)

    if !isempty(op.body.args)
        print_loop_args(p, op.body.args, op.init_values)
    end

    # Show return type annotation
    if result_type !== nothing
        print_colored(p, " -> ", :light_black)
        print_colored(p, string(result_type), :cyan)
    end

    println(p.io)

    # Body content - the terminator (continue) closes the for block with └──
    body_p = child_printer(p, op.body, cont_prefix)
    print_block_body(body_p, op.body; is_last_in_parent=true)
end

function print_loop_op_final(p::IRPrinter, op::LoopOp, pos::Int, @nospecialize(result_type); is_last::Bool=false)
    cont_prefix = is_last ? "    " : "│   "

    # Print the loop header (no box-drawing prefix)
    print_indent(p)
    idx_s = string(pos)
    pad = " "^(p.max_idx_width - length(idx_s) - 4)
    print(p.io, "%", idx_s, pad, " = ")
    print_colored(p, "loop", :yellow)
    print_loop_args(p, op.body.args, op.init_values)
    # Show return type annotation
    if result_type !== nothing
        print_colored(p, " -> ", :light_black)
        print_colored(p, string(result_type), :cyan)
    end
    println(p.io)

    # Body content - the terminator closes the loop block with └──
    body_p = child_printer(p, op.body, cont_prefix)
    print_block_body(body_p, op.body; is_last_in_parent=true)
end

function print_while_op_final(p::IRPrinter, op::WhileOp, pos::Int, @nospecialize(result_type); is_last::Bool=false)
    # Print the while header (no box-drawing prefix)
    print_indent(p)
    idx_s = string(pos)
    pad = " "^(p.max_idx_width - length(idx_s) - 4)
    print(p.io, "%", idx_s, pad, " = ")
    print_colored(p, "while", :yellow)
    print_loop_args(p, op.before.args, op.init_values)
    # Show return type annotation
    if result_type !== nothing
        print_colored(p, " -> ", :light_black)
        print_colored(p, string(result_type), :cyan)
    end
    println(p.io)

    # Check if "do" region is empty
    do_is_empty = isempty(op.after.body) && op.after.terminator === nothing

    # Print "before" region header (at same level as the while)
    print_region_header(p, "before", op.before.args; is_last=false)
    # Print before region body (indented one level)
    # If do is empty, before's last item closes the while with └──
    before_body_p = child_printer(p, op.before, "│   ")
    print_block_body(before_body_p, op.before; is_last_in_parent=do_is_empty)

    # Print "do" region header
    # Use └ if empty (nothing follows), ├ if has content
    print_region_header(p, "do", op.after.args; is_last=do_is_empty)
    # Print do region body
    if !do_is_empty
        after_body_p = child_printer(p, op.after, "│   ")
        print_block_body(after_body_p, op.after; is_last_in_parent=true)
    end
end
