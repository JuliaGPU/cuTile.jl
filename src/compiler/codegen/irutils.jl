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
