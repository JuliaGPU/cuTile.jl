# Pretty Printing (Julia CodeInfo-style with colors)

_scan_expr_uses!(used::BitSet, v::SSAValue) = push!(used, v.id)
_scan_expr_uses!(used::BitSet, v::Expr) = _scan_expr_uses!.(Ref(used), v.args)
_scan_expr_uses!(used::BitSet, v::PhiNode) = _scan_expr_uses!.(Ref(used), v.values)
_scan_expr_uses!(used::BitSet, v::PiNode) = _scan_expr_uses!(used, v.val)
_scan_expr_uses!(used::BitSet, v::GotoIfNot) = _scan_expr_uses!(used, v.cond)
_scan_expr_uses!(::BitSet, _) = nothing  # constants, GlobalRefs, etc.

function _scan_terminator_uses!(used::BitSet, term::ReturnNode)
    if isdefined(term, :val)
        _scan_expr_uses!(used, term.val)
    end
end

_scan_terminator_uses!(used::BitSet, term::Union{YieldOp,ContinueOp,BreakOp}) =
    _scan_expr_uses!.(Ref(used), term.values)

function _scan_terminator_uses!(used::BitSet, term::ConditionOp)
    _scan_expr_uses!(used, term.condition)
    _scan_expr_uses!.(Ref(used), term.args)
end

_scan_terminator_uses!(::BitSet, ::Nothing) = nothing

# Block usage scanning (for type coloring)
function compute_used_ssas(block::Block)
    used = BitSet()
    _scan_block_uses!(used, block)
    return used
end

function _scan_block_uses!(used::BitSet, block::Block)
    for stmt in statements(block.body)
        if stmt isa ControlFlowOp
            _scan_control_flow_uses!(used, stmt)
        else
            _scan_expr_uses!(used, stmt)
        end
    end
    if block.terminator !== nothing
        _scan_terminator_uses!(used, block.terminator)
    end
end

function _scan_control_flow_uses!(used::BitSet, op::IfOp)
    _scan_expr_uses!(used, op.condition)
    _scan_block_uses!(used, op.then_region)
    _scan_block_uses!(used, op.else_region)
end

function _scan_control_flow_uses!(used::BitSet, op::ForOp)
    _scan_expr_uses!(used, op.lower)
    _scan_expr_uses!(used, op.upper)
    _scan_expr_uses!(used, op.step)
    for v in op.init_values
        _scan_expr_uses!(used, v)
    end
    _scan_block_uses!(used, op.body)
end

function _scan_control_flow_uses!(used::BitSet, op::WhileOp)
    for v in op.init_values
        _scan_expr_uses!(used, v)
    end
    _scan_block_uses!(used, op.before)
    _scan_block_uses!(used, op.after)
end

function _scan_control_flow_uses!(used::BitSet, op::LoopOp)
    for v in op.init_values
        _scan_expr_uses!(used, v)
    end
    _scan_block_uses!(used, op.body)
end

"""
    IRPrinter

Context for printing structured IR with proper indentation and value formatting.
Uses Julia's CodeInfo style with box-drawing characters and colors.
"""
mutable struct IRPrinter
    io::IO
    code::CodeInfo
    indent::Int
    line_prefix::String    # Prefix for continuation lines (│, spaces)
    is_last_stmt::Bool     # Whether current stmt is last in block
    used::BitSet           # Which SSA values are used anywhere in the IR
    max_idx_width::Int     # Max width of "%N = " for alignment (like Julia's SSA printer)
    color::Bool            # Whether to use colors
end

function IRPrinter(io::IO, code::CodeInfo, entry::Block)
    used = compute_used_ssas(entry)
    # Compute max index width for alignment: "%N = " where N is the max index
    max_idx_width = ndigits(length(entry.body)) + 4  # % + digits + space + = + space
    color = get(io, :color, false)::Bool
    IRPrinter(io, code, 0, "", false, used, max_idx_width, color)
end

function indent(p::IRPrinter, n::Int=1)
    new_prefix = p.line_prefix * "    "  # 4 spaces per indent level
    return IRPrinter(p.io, p.code, p.indent + n, new_prefix, false, p.used, p.max_idx_width, p.color)
end

# Create a child printer for a nested Block
function child_printer(p::IRPrinter, nested_block::Block, cont_prefix::String)
    nested_max_idx_width = ndigits(length(nested_block.body)) + 4
    IRPrinter(p.io, p.code, p.indent + 1, p.line_prefix * cont_prefix, false, p.used, nested_max_idx_width, p.color)
end

# Print region header: "├ label(%arg1::Type):" (regions always use ├, closing is on last line of content)
function print_region_header(p::IRPrinter, label::String, args::Vector{BlockArg})
    print_indent(p)
    print_colored(p, "├", :light_black)
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
    # Use slot names if available from CodeInfo
    if v.n <= length(p.code.slotnames)
        name = p.code.slotnames[v.n]
        print(p.io, name)
    else
        print(p.io, "_", v.n)
    end
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
    if v.n <= length(p.code.slotnames)
        name = p.code.slotnames[v.n]
        return string(name)
    else
        return string("_", v.n)
    end
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

# Print type annotation with color based on whether the value is used
# Like Julia's code_typed: both :: and type share the same color
function print_type_annotation(p::IRPrinter, idx::Int, t)
    is_used = idx in p.used
    color = is_used ? :cyan : :light_black
    print_colored(p, string("::", format_type(t)), color)
end

# Format result variables (string version for backwards compat)
function format_results(p::IRPrinter, results::Vector{SSAValue})
    if isempty(results)
        ""
    elseif length(results) == 1
        r = results[1]
        typ = p.code.ssavaluetypes[r.id]
        string(format_value(p, r), "::", format_type(typ))
    else
        parts = [string(format_value(p, r), "::", format_type(p.code.ssavaluetypes[r.id]))
                 for r in results]
        string("(", join(parts, ", "), ")")
    end
end

# Print result variables with type colors
function print_results(p::IRPrinter, results::Vector{SSAValue})
    if isempty(results)
        return
    elseif length(results) == 1
        r = results[1]
        print(p.io, "%", r.id)
        is_used = r.id in p.used
        color = is_used ? :cyan : :light_black
        print_colored(p, string("::", format_type(p.code.ssavaluetypes[r.id])), color)
    else
        print(p.io, "(")
        for (i, r) in enumerate(results)
            i > 1 && print(p.io, ", ")
            print(p.io, "%", r.id)
            is_used = r.id in p.used
            color = is_used ? :cyan : :light_black
            print_colored(p, string("::", format_type(p.code.ssavaluetypes[r.id])), color)
        end
        print(p.io, ")")
    end
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

function print_expr(p::IRPrinter, node::PhiNode)
    print(p.io, "φ (")
    first = true
    for (edge, val) in zip(node.edges, node.values)
        first || print(p.io, ", ")
        first = false
        print(p.io, "#", edge, " => ")
        if isassigned(node.values, findfirst(==(val), node.values))
            print_value(p, val)
        else
            print_colored(p, "#undef", :red)
        end
    end
    print(p.io, ")")
end

function print_expr(p::IRPrinter, node::PiNode)
    print(p.io, "π (")
    print_value(p, node.val)
    print(p.io, ", ", node.typ, ")")
end

function print_expr(p::IRPrinter, node::GotoNode)
    print(p.io, "goto #", node.label)
end

function print_expr(p::IRPrinter, node::GotoIfNot)
    print(p.io, "goto #", node.dest, " if not ")
    print_value(p, node.cond)
end

function print_expr(p::IRPrinter, node::ReturnNode)
    print(p.io, "return")
    if isdefined(node, :val)
        print(p.io, " ")
        print_value(p, node.val)
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
function print_terminator(p::IRPrinter, term::ReturnNode; prefix::String="└──")
    print_indent(p)
    if !isempty(prefix)
        print_colored(p, prefix, :light_black)
        print(p.io, " ")
    end
    print(p.io, "return")
    if isdefined(term, :val)
        print(p.io, " ")
        print_value(p, term.val)
    end
    println(p.io)
end

function print_terminator(p::IRPrinter, term::Union{YieldOp,ContinueOp,BreakOp,ConditionOp}; prefix::String="└──")
    print_indent(p)
    if !isempty(prefix)
        print_colored(p, prefix, :light_black)
        print(p.io, " ")
    end
    print_terminator_content(p, term)
    println(p.io)
end

function print_terminator(p::IRPrinter, ::Nothing; prefix::String="└──")
    # No terminator
end

#=============================================================================
 Pretty Printing for Final Block/ControlFlowOp
=============================================================================#

# Print expression with type annotation (for final Block)
# prefix can be "│  " for continuation, "└──" for last, or "" for no prefix (inside regions)
function print_expr_with_type(p::IRPrinter, idx::Int, expr, typ; prefix::String="│  ")
    print_indent(p)
    if !isempty(prefix)
        print_colored(p, prefix, :light_black)
        print(p.io, " ")
    end

    # Show %N = if value is used anywhere in the IR (global usage)
    # Type color: cyan if used, grey if unused (matches Julia's SSA IR printer)
    is_used = idx in p.used
    if is_used
        idx_s = string(idx)
        pad = " "^(p.max_idx_width - length(idx_s) - 4)  # -4 for "% = "
        print(p.io, "%", idx_s, pad, " = ")
    else
        # Pad to align with "%N = " (like Julia's SSA printer)
        print(p.io, " "^p.max_idx_width)
    end
    print_expr(p, expr)

    # Print type annotation
    color = is_used ? :cyan : :light_black
    print_colored(p, string("::", format_type(typ)), color)
    println(p.io)
end

# Print a Block's contents (final type)
# If inside_region=true, don't show statement-level box drawing (just use line_prefix)
# If is_last_region=true, modify the last line to show outer block closing
function print_block_body(p::IRPrinter, block::Block; inside_region::Bool=false, is_last_region::Bool=false, cascade_depth::Int=0)
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
            if inside_region
                prefix = ""
            else
                prefix = is_last ? "└──" : "│  "
            end
            if is_last && is_last_region
                # Replace trailing │ characters in line_prefix with └ for outer block closing
                print_expr_with_type_closing(p, item[2], item[3], item[4]; cascade_depth=cascade_depth)
            else
                print_expr_with_type(p, item[2], item[3], item[4]; prefix=prefix)
            end
        elseif item[1] == :nested
            if is_last && is_last_region
                # Cascade the closing - don't use └── here (is_last=false), and increment cascade_depth
                # so the nested op knows to close this level too
                print_control_flow(p, item[3], item[2], item[4]; is_last=false, cascade_depth=cascade_depth + 1)
            else
                print_control_flow(p, item[3], item[2], item[4]; is_last=is_last, cascade_depth=cascade_depth)
            end
        else  # :term
            if inside_region
                prefix = ""
            else
                prefix = "└──"
            end
            if is_last && is_last_region
                # Replace trailing │ characters in line_prefix with └ for outer block closing
                print_terminator_closing(p, item[2]; cascade_depth=cascade_depth)
            else
                print_terminator(p, item[2]; prefix=prefix)
            end
        end
    end
end

"""
    close_trailing_boxes(prefix::String, count::Int) -> String

Replace `count` trailing │ characters in prefix: the last one becomes └, the rest become spaces.
E.g., close_trailing_boxes("│   │   │   ", 2) → "│   └       " (closes 2 levels)
"""
function close_trailing_boxes(prefix::String, count::Int)
    count <= 0 && return prefix

    # Find all │ positions (working backwards)
    chars = collect(prefix)
    box_positions = Int[]
    for i in length(chars):-1:1
        if chars[i] == '│'
            push!(box_positions, i)
        end
    end

    # Replace up to `count` trailing │ characters
    n_to_replace = min(count, length(box_positions))
    for (i, pos) in enumerate(box_positions[1:n_to_replace])
        if i == 1
            # Last │ becomes └
            chars[pos] = '└'
        else
            # Other trailing │ become space
            chars[pos] = ' '
        end
    end

    return String(chars)
end

# Print expression for last line of last region (replace trailing │ with └)
function print_expr_with_type_closing(p::IRPrinter, idx::Int, expr, typ; cascade_depth::Int=0)
    # Replace (cascade_depth + 1) trailing │ characters: last one becomes └, rest become spaces
    closing_prefix = close_trailing_boxes(p.line_prefix, cascade_depth + 1)
    print_colored(p, closing_prefix, :light_black)

    # Show %N = if value is used anywhere in the IR (global usage)
    # Type color: cyan if used, grey if unused (matches Julia's SSA IR printer)
    is_used = idx in p.used
    if is_used
        idx_s = string(idx)
        pad = " "^(p.max_idx_width - length(idx_s) - 4)  # -4 for "% = "
        print(p.io, "%", idx_s, pad, " = ")
    else
        # Pad to align with "%N = " (like Julia's SSA printer)
        print(p.io, " "^p.max_idx_width)
    end
    print_expr(p, expr)

    # Print type annotation
    color = is_used ? :cyan : :light_black
    print_colored(p, string("::", format_type(typ)), color)
    println(p.io)
end

# Print terminator for last line of last region (replace trailing │ with └)
function print_terminator_closing(p::IRPrinter, term; cascade_depth::Int=0)
    closing_prefix = close_trailing_boxes(p.line_prefix, cascade_depth + 1)
    print_colored(p, closing_prefix, :light_black)
    print_terminator_content(p, term)
    println(p.io)
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
print_control_flow(p::IRPrinter, op::IfOp, pos::Int, @nospecialize(result_type); is_last::Bool=false, cascade_depth::Int=0) = print_if_op_final(p, op, pos, result_type; is_last, cascade_depth)
print_control_flow(p::IRPrinter, op::ForOp, pos::Int, @nospecialize(result_type); is_last::Bool=false, cascade_depth::Int=0) = print_for_op_final(p, op, pos, result_type; is_last, cascade_depth)
print_control_flow(p::IRPrinter, op::WhileOp, pos::Int, @nospecialize(result_type); is_last::Bool=false, cascade_depth::Int=0) = print_while_op_final(p, op, pos, result_type; is_last, cascade_depth)
print_control_flow(p::IRPrinter, op::LoopOp, pos::Int, @nospecialize(result_type); is_last::Bool=false, cascade_depth::Int=0) = print_loop_op_final(p, op, pos, result_type; is_last, cascade_depth)

function print_if_op_final(p::IRPrinter, op::IfOp, pos::Int, @nospecialize(result_type); is_last::Bool=false, cascade_depth::Int=0)
    # When cascade_depth > 0, keep │ going (don't use └──) so inner content can close it
    use_closing = is_last && cascade_depth == 0
    prefix = use_closing ? "└──" : "├──"
    cont_prefix = use_closing ? "    " : "│   "

    print_indent(p)
    print_colored(p, prefix, :light_black)
    print(p.io, " ")

    # Show assignment if this position is used anywhere in the IR
    if pos in p.used
        print(p.io, "%", pos, " = ")
    end

    print(p.io, "if ")
    print_value(p, op.condition)

    # Show return type annotation
    if result_type !== nothing
        print_colored(p, " -> ", :light_black)
        print_colored(p, string(result_type), :cyan)
    end

    println(p.io)

    # Print "then:" region header
    then_p = child_printer(p, op.then_region, cont_prefix)
    print_region_header(then_p, "then", op.then_region.args)
    # Print then region body (inside_region=true, no statement-level box drawing)
    then_body_p = child_printer(then_p, op.then_region, "│   ")
    print_block_body(then_body_p, op.then_region; inside_region=true)

    # Print "else:" region header
    else_p = child_printer(p, op.else_region, cont_prefix)
    print_region_header(else_p, "else", op.else_region.args)
    # Print else region body (last region, show closing on last line)
    # Pass cascade_depth so innermost item can close all cascaded levels
    else_body_p = child_printer(else_p, op.else_region, "│   ")
    print_block_body(else_body_p, op.else_region; inside_region=true, is_last_region=true, cascade_depth=cascade_depth)
end

function print_for_op_final(p::IRPrinter, op::ForOp, pos::Int, @nospecialize(result_type); is_last::Bool=false, cascade_depth::Int=0)
    # When cascade_depth > 0, keep │ going so inner content can close it
    use_closing = is_last && cascade_depth == 0
    prefix = use_closing ? "└──" : "├──"
    cont_prefix = use_closing ? "    " : "│   "

    print_indent(p)
    print_colored(p, prefix, :light_black)
    print(p.io, " ")

    # Show assignment if this position is used anywhere in the IR
    if pos in p.used
        print(p.io, "%", pos, " = ")
    end

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

    # Body is inline (no region header for single-region ops)
    # Use normal block body printing - the last item will get └── prefix automatically
    body_p = child_printer(p, op.body, cont_prefix)
    print_block_body(body_p, op.body)
end

function print_loop_op_final(p::IRPrinter, op::LoopOp, pos::Int, @nospecialize(result_type); is_last::Bool=false, cascade_depth::Int=0)
    # When cascade_depth > 0, keep │ going so inner content can close it
    use_closing = is_last && cascade_depth == 0
    prefix = use_closing ? "└──" : "├──"
    cont_prefix = use_closing ? "    " : "│   "

    print_indent(p)
    print_colored(p, prefix, :light_black)
    print(p.io, " ")

    # Show assignment if this position is used anywhere in the IR
    if pos in p.used
        print(p.io, "%", pos, " = ")
    end

    print_colored(p, "loop", :yellow)
    print_loop_args(p, op.body.args, op.init_values)
    # Show return type annotation
    if result_type !== nothing
        print_colored(p, " -> ", :light_black)
        print_colored(p, string(result_type), :cyan)
    end
    println(p.io)

    # Body is inline (no region header for single-region ops)
    # Use normal block body printing - the last item will get └── prefix automatically
    body_p = child_printer(p, op.body, cont_prefix)
    print_block_body(body_p, op.body)
end

function print_while_op_final(p::IRPrinter, op::WhileOp, pos::Int, @nospecialize(result_type); is_last::Bool=false, cascade_depth::Int=0)
    # When cascade_depth > 0, keep │ going so inner content can close it
    use_closing = is_last && cascade_depth == 0
    prefix = use_closing ? "└──" : "├──"
    cont_prefix = use_closing ? "    " : "│   "

    print_indent(p)
    print_colored(p, prefix, :light_black)
    print(p.io, " ")

    # Show assignment if this position is used anywhere in the IR
    if pos in p.used
        print(p.io, "%", pos, " = ")
    end

    print_colored(p, "while", :yellow)
    print_loop_args(p, op.before.args, op.init_values)
    # Show return type annotation
    if result_type !== nothing
        print_colored(p, " -> ", :light_black)
        print_colored(p, string(result_type), :cyan)
    end
    println(p.io)

    # Print "before" region header
    before_p = child_printer(p, op.before, cont_prefix)
    print_region_header(before_p, "before", op.before.args)
    # Print before region body (inside_region=true, no statement-level box drawing)
    before_body_p = child_printer(before_p, op.before, "│   ")
    print_block_body(before_body_p, op.before; inside_region=true)

    # Print "do" region header
    after_p = child_printer(p, op.after, cont_prefix)
    print_region_header(after_p, "do", op.after.args)
    # Print do region body (last region, show closing on last line)
    # Pass cascade_depth so innermost item can close all cascaded levels
    after_body_p = child_printer(after_p, op.after, "│   ")
    print_block_body(after_body_p, op.after; inside_region=true, is_last_region=true, cascade_depth=cascade_depth)
end

# Main entry point: show for StructuredCodeInfo
function Base.show(io::IO, ::MIME"text/plain", sci::StructuredCodeInfo)
    # Get return type from last stmt if it's a return
    ret_type = "Any"
    for stmt in reverse(sci.code.code)
        if stmt isa ReturnNode && isdefined(stmt, :val)
            val = stmt.val
            if val isa SSAValue
                ret_type = format_type(sci.code.ssavaluetypes[val.id])
            else
                ret_type = format_type(typeof(val))
            end
            break
        end
    end

    color = get(io, :color, false)::Bool

    # Print header
    println(io, "StructuredCodeInfo(")

    p = IRPrinter(io, sci.code, sci.entry)

    # Print entry block body
    print_block_body(p, sci.entry)

    print(io, ") => ")
    if color
        printstyled(io, ret_type; color=:cyan)
        println(io)
    else
        println(io, ret_type)
    end
end

# Keep the simple show method for compact display
function Base.show(io::IO, sci::StructuredCodeInfo)
    n_body = length(sci.entry.body)
    print(io, "StructuredCodeInfo(", length(sci.code.code), " stmts, entry=Block(", n_body, " items))")
end
