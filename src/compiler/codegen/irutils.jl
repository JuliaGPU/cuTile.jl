# StructuredIRCode / SSAMap mutation utilities
#
# Helpers for passes that modify the structured IR in place.
# Inspired by Julia's IncrementalCompact (Compiler/src/ssair/ir.jl).

"""
    new_ssa_idx!(sci::StructuredIRCode) -> Int

Allocate a fresh SSA index from the StructuredIRCode.
"""
function new_ssa_idx!(sci::StructuredIRCode)
    sci.max_ssa_idx += 1
    return sci.max_ssa_idx
end

"""
    new_block_arg!(block::Block, sci::StructuredIRCode, @nospecialize(typ)) -> BlockArg

Add a new BlockArg to a block, allocating a fresh ID.
"""
function new_block_arg!(block::Block, sci::StructuredIRCode, @nospecialize(typ))
    id = new_ssa_idx!(sci)
    arg = BlockArg(id, typ)
    push!(block.args, arg)
    return arg
end

"""
    Base.pushfirst!(m::SSAMap, (idx, stmt, typ)::Tuple{Int,Any,Any})

Prepend a statement at the beginning of an SSAMap.
"""
function Base.pushfirst!(m::SSAMap, (idx, stmt, typ)::Tuple{Int,Any,Any})
    pushfirst!(m.ssa_idxes, idx)
    pushfirst!(m.stmts, stmt)
    pushfirst!(m.types, typ)
    return nothing
end

"""
    insert_before!(m::SSAMap, before_idx::Int, new_idx::Int, stmt, typ)

Insert a new entry before the entry with SSA index `before_idx`.
"""
function insert_before!(m::SSAMap, before_idx::Int, new_idx::Int, stmt, typ)
    pos = findfirst(==(before_idx), m.ssa_idxes)
    pos === nothing && throw(KeyError(before_idx))
    insert!(m.ssa_idxes, pos, new_idx)
    insert!(m.stmts, pos, stmt)
    insert!(m.types, pos, typ)
    return nothing
end

"""
    insert_after!(m::SSAMap, after_idx::Int, new_idx::Int, stmt, typ)

Insert a new entry after the entry with SSA index `after_idx`.
"""
function insert_after!(m::SSAMap, after_idx::Int, new_idx::Int, stmt, typ)
    pos = findfirst(==(after_idx), m.ssa_idxes)
    pos === nothing && throw(KeyError(after_idx))
    insert!(m.ssa_idxes, pos + 1, new_idx)
    insert!(m.stmts, pos + 1, stmt)
    insert!(m.types, pos + 1, typ)
    return nothing
end

"""
    update_type!(m::SSAMap, ssa_idx::Int, @nospecialize(new_type))

Update the type annotation for an existing SSAMap entry.
"""
function update_type!(m::SSAMap, ssa_idx::Int, @nospecialize(new_type))
    pos = findfirst(==(ssa_idx), m.ssa_idxes)
    pos === nothing && throw(KeyError(ssa_idx))
    m.types[pos] = new_type
    return nothing
end

"""
    resolve_call(stmt) -> (resolved_func, operands) or nothing

Extract the resolved function and operands from a `:call` or `:invoke` Expr.
Shared by alias analysis and token ordering passes.
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
        try; getfield(func_ref.mod, func_ref.name); catch; nothing; end
    else
        func_ref
    end
    resolved === nothing && return nothing
    return (resolved, operands)
end
