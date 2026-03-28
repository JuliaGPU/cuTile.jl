# IR Normalization Pass
#
# Lowers Julia Core intrinsics and builtins to cuTile Intrinsics in the
# StructuredIRCode. Run immediately after structurization, before all other
# passes.
#
# Core intrinsics appear in the SCI either because:
# - IRStructurizer introduces them for control flow (loop bounds, increments)
# - Julia's type inference inlined Base functions down to Core intrinsics
#   (e.g., Base.:-(x::Int32, y::Int32) → Core.Intrinsics.sub_int(x, y))
#
# This pass replaces them with cuTile equivalents so downstream passes
# (alias analysis, token ordering, DCE) see a uniform IR.

using Core: SSAValue

"""
    normalize_ir!(sci::StructuredIRCode)

Replace Julia Core intrinsics with cuTile Intrinsics equivalents.
"""
function normalize_ir!(sci::StructuredIRCode)
    for block in eachblock(sci)
        normalize_block!(block)
    end
    for block in eachblock(sci)
        for inst in instructions(block)
            call = resolve_call(stmt(inst))
            call === nothing && continue
            func, _ = call
            if func isa Core.IntrinsicFunction
                throw(IRError("Core.Intrinsics.$(nameof(func)) not handled by " *
                              "normalize_ir! — add a mapping in normalize.jl or an overlay"))
            end
        end
    end
end

function normalize_block!(block::Block)
    for inst in instructions(block)
        s = stmt(inst)
        s isa Expr || continue
        (s.head === :call || s.head === :invoke) || continue

        call = resolve_call(s)
        call === nothing && continue
        func, operands = call

        if func isa Core.IntrinsicFunction
            normalize_core_intrinsic!(block, inst, s, func, operands)
        elseif func === (===)
            replace_call!(s, Intrinsics.cmpi,
                          Any[operands..., ComparisonPredicate.Equal, Signedness.Signed])
        elseif func === Core.ifelse
            replace_call!(s, Intrinsics.select, operands)
        end
    end
end

function normalize_core_intrinsic!(block::Block, inst::Instruction, expr::Expr,
                                    func::Core.IntrinsicFunction, operands)
    # Integer arithmetic
    if     func === Core.Intrinsics.add_int;  replace_call!(expr, Intrinsics.addi, operands)
    elseif func === Core.Intrinsics.sub_int;  replace_call!(expr, Intrinsics.subi, operands)
    elseif func === Core.Intrinsics.mul_int;  replace_call!(expr, Intrinsics.muli, operands)
    elseif func === Core.Intrinsics.neg_int;  replace_call!(expr, Intrinsics.negi, operands)
    # Integer comparison
    elseif func === Core.Intrinsics.slt_int
        replace_call!(expr, Intrinsics.cmpi, Any[operands..., ComparisonPredicate.LessThan, Signedness.Signed])
    elseif func === Core.Intrinsics.sle_int
        replace_call!(expr, Intrinsics.cmpi, Any[operands..., ComparisonPredicate.LessThanOrEqual, Signedness.Signed])
    elseif func === Core.Intrinsics.ult_int
        replace_call!(expr, Intrinsics.cmpi, Any[operands..., ComparisonPredicate.LessThan, Signedness.Unsigned])
    # Bitwise
    elseif func === Core.Intrinsics.and_int;  replace_call!(expr, Intrinsics.andi, operands)
    elseif func === Core.Intrinsics.or_int;   replace_call!(expr, Intrinsics.ori, operands)
    elseif func === Core.Intrinsics.xor_int;  replace_call!(expr, Intrinsics.xori, operands)
    elseif func === Core.Intrinsics.not_int;  normalize_not_int!(block, inst, expr, operands)
    # Float arithmetic
    elseif func === Core.Intrinsics.add_float;  replace_call!(expr, Intrinsics.addf, operands)
    elseif func === Core.Intrinsics.sub_float;  replace_call!(expr, Intrinsics.subf, operands)
    elseif func === Core.Intrinsics.mul_float;  replace_call!(expr, Intrinsics.mulf, operands)
    elseif func === Core.Intrinsics.div_float;  replace_call!(expr, Intrinsics.divf, operands)
    elseif func === Core.Intrinsics.neg_float;  replace_call!(expr, Intrinsics.negf, operands)
    # Float comparison
    elseif func === Core.Intrinsics.lt_float
        replace_call!(expr, Intrinsics.cmpf, Any[operands..., ComparisonPredicate.LessThan])
    elseif func === Core.Intrinsics.le_float
        replace_call!(expr, Intrinsics.cmpf, Any[operands..., ComparisonPredicate.LessThanOrEqual])
    elseif func === Core.Intrinsics.eq_float
        replace_call!(expr, Intrinsics.cmpf, Any[operands..., ComparisonPredicate.Equal])
    elseif func === Core.Intrinsics.ne_float
        replace_call!(expr, Intrinsics.cmpf, Any[operands..., ComparisonPredicate.NotEqual])
    end
end

function normalize_not_int!(block::Block, inst::Instruction, expr::Expr, operands)
    length(operands) >= 1 || return
    operand = operands[1]
    operand_type = value_type(block, operand)
    operand_type === nothing && return
    jltype = CC.widenconst(operand_type)
    allones = jltype === Bool ? true : jltype(-1)
    replace_call!(expr, Intrinsics.xori, Any[operand, allones])
end

function replace_call!(expr::Expr, new_func, new_args)
    args = collect(Any, new_args)  # copy before mutating expr.args (may be a view)
    empty!(expr.args)
    expr.head = :call
    push!(expr.args, new_func)
    append!(expr.args, args)
end
