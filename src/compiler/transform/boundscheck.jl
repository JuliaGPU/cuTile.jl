# Resolve Julia's bounds-check marker before the rest of the Tile IR pipeline.

function boundscheck_value(expr::Expr)
    @assert expr.head === :boundscheck
    opt = Base.JLOptions().check_bounds
    return opt == 1 ? true :
           opt == 2 ? false :
           isempty(expr.args) ? true : expr.args[1] !== false
end

function resolve_boundscheck!(sci::StructuredIRCode)
    r = Rewriter(sci)
    for block in eachblock(sci)
        for inst in collect(instructions(block))
            stmt = inst[:stmt]
            stmt isa Expr && stmt.head === :boundscheck || continue
            val = SSAValue(inst)
            replace_uses!(r, val, boundscheck_value(stmt))
            erase!(r, block, val)
        end
    end
    return sci
end
