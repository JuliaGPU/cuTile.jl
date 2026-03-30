# Index Lowering Pass
#
# Converts 1-based Julia indices to 0-based Tile IR indices for
# load_partition_view and store_partition_view.
#
# Uses the RFunc (callable RHS) extension to the @rewrite framework:
# for each index element in the indices tuple, inserts subi(elem, 1).

using Core: SSAValue

#=============================================================================
 Implementation
=============================================================================#

function _lower_indices_rhs(sci, block, inst, match)
    idx = match.bindings[:idx]
    ref = SSAValue(inst)

    # Get tuple type to determine element count and types
    idx_type = CC.widenconst(value_type(block, idx))
    idx_type <: Tuple || return false
    n = fieldcount(idx_type)
    n == 0 && return true

    # Extract each element via getfield, subtract 1, build new tuple
    lowered = Any[]
    for i in 1:n
        ft = fieldtype(idx_type, i)
        elem = SSAValue(insert_before!(block, ref,
            Expr(:call, GlobalRef(Core, :getfield), idx, i), ft))
        sub = SSAValue(insert_before!(block, ref,
            Expr(:call, GlobalRef(Intrinsics, :subi), elem, one(ft)), ft))
        push!(lowered, sub)
    end

    new_tuple = SSAValue(insert_before!(block, ref,
        Expr(:call, GlobalRef(Core, :tuple), lowered...), idx_type))

    s = stmt(inst)
    for i in eachindex(s.args)
        s.args[i] === idx && (s.args[i] = new_tuple)
    end

    return true
end

#=============================================================================
 Rewrite rules
=============================================================================#

# Lower 1-based indices → 0-based for load_partition_view.
# args: (pv, latency, allow_tma, indices_tuple)
const LOAD_INDEX_LOWER = RewriteRule(
    PCall(:load_partition_view, [PBind(:pv), PBind(:lat), PBind(:tma), PBind(:idx)]),
    RFunc(_lower_indices_rhs)
)

# Lower 1-based indices → 0-based for store_partition_view.
# args: (pv, tile, latency, allow_tma, indices_tuple)
const STORE_INDEX_LOWER = RewriteRule(
    PCall(:store_partition_view, [PBind(:pv), PBind(:tile), PBind(:lat), PBind(:tma), PBind(:idx)]),
    RFunc(_lower_indices_rhs)
)

const INDEX_LOWER_RULES = RewriteRule[LOAD_INDEX_LOWER, STORE_INDEX_LOWER]

#=============================================================================
 Driver
=============================================================================#

"""
    index_lower_pass!(sci::StructuredIRCode)

Lower 1-based Julia indices to 0-based Tile IR indices for load/store ops.
"""
index_lower_pass!(sci::StructuredIRCode) = rewrite_patterns!(sci, INDEX_LOWER_RULES)
