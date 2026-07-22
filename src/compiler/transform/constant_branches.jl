# Fold structured branches whose conditions are known at compile time.

plain_yield(block::Block) =
    terminator(block) === nothing || terminator(block) isa YieldOp

function constant_if_candidates(sci::StructuredIRCode, constants::ConstantInfo)
    candidates = Tuple{Block, SSAValue, Bool}[]
    for block in eachblock(sci)
        for inst in instructions(block)
            op = inst[:stmt]
            op isa IfOp || continue
            # BreakOp and ContinueOp transfer control to an enclosing loop;
            # splicing those arms requires a separate loop-control rewrite.
            plain_yield(op.then_region) && plain_yield(op.else_region) || continue
            condition = const_value(constants, op.condition)
            condition isa Bool || continue
            push!(candidates, (block, SSAValue(inst), condition))
        end
    end
    return candidates
end

function yield_values(block::Block)
    term = terminator(block)
    return term isa YieldOp ? term.values : Any[]
end

function tuple_result_extracts(sci::StructuredIRCode, r::Rewriter,
                               if_ssa::SSAValue, nresults::Int)
    extracts = Tuple{Block, SSAValue, Int}[]
    for block in eachblock(sci)
        for inst in instructions(block)
            is_getfield_of(inst[:stmt], if_ssa) || continue
            field = inst[:stmt].args[3]
            1 <= field <= nresults || return nothing
            push!(extracts, (block, SSAValue(inst), field))
        end
    end
    # Tuple-valued control flow is represented by getfield extractions. Avoid
    # folding an unfamiliar aggregate use until it has an explicit rewrite.
    length(extracts) == use_count(r, if_ssa) || return nothing
    return extracts
end

function fold_constant_if!(sci::StructuredIRCode, parent::Block,
                           if_ssa::SSAValue, condition::Bool)
    haskey(parent, if_ssa.id) || return false
    inst = Instruction(if_ssa.id, parent)
    op = inst[:stmt]
    op isa IfOp || return false
    plain_yield(op.then_region) && plain_yield(op.else_region) || return false

    taken = condition ? op.then_region : op.else_region
    r = Rewriter(sci)
    result_type = CC.widenconst(inst[:type])
    extracts = if result_type <: Tuple
        tuple_result_extracts(sci, r, if_ssa, length(result_type.parameters))
    else
        nothing
    end
    result_type <: Tuple && extracts === nothing && return false

    yielded = yield_values(taken)
    if result_type === Nothing
        isempty(yielded) || return false
    elseif result_type <: Tuple
        length(yielded) == length(result_type.parameters) || return false
    else
        length(yielded) == 1 || return false
    end

    # Preserve SSA indices and source locations while moving the taken body.
    for region_inst in collect(instructions(taken))
        move_before!(region_inst, inst)
    end

    if result_type <: Tuple
        for (block, extract_ssa, field) in extracts
            replace_uses!(r, extract_ssa, yielded[field])
            erase!(r, block, extract_ssa)
        end
    elseif result_type !== Nothing
        replace_uses!(r, if_ssa, only(yielded))
    end

    erase!(r, parent, if_ssa)
    return true
end

function fold_constant_branches!(sci::StructuredIRCode)
    while true
        constants = analyze_constants(sci)
        folded = false
        for (block, if_ssa, condition) in constant_if_candidates(sci, constants)
            if fold_constant_if!(sci, block, if_ssa, condition)
                folded = true
                break
            end
        end
        folded || break
    end
    return sci
end
