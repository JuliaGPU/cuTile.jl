# public API

export code_structured, structurize!, StructuredCodeInfo

"""
    code_structured(f, argtypes; validate=true, kwargs...) -> Pair{StructuredCodeInfo, DataType}

Returns an array of structured IR for the methods matching the given generic function and typesignature.

This is analogous to `code_typed` but returns a `StructuredCodeInfo` with
control flow restructured into nested SCF-style operations (if/for/while).

# Arguments
- `f`: The function to analyze
- `argtypes`: Argument types as a tuple of types (e.g., `(Int, Float64)`)
- `validate`: Whether to validate that all control flow was properly structured (default: true).
  When `true`, throws `UnstructuredControlFlowError` if any unstructured control flow remains.

Other keyword arguments are passed to `code_typed`.

ForOp is created directly during CFG analysis for loops that match counting patterns
(while i < n with i += step, or for i in 1:n). WhileOp is used for condition-at-header
loops that don't match counting patterns. LoopOp is used for general cyclic regions.

# Returns
A `Pair{StructuredCodeInfo, DataType}` where the first element is the structured IR
and the second is the return type. Displays with MLIR SCF-style syntax.

# Example
```julia
julia> f(x) = x > 0 ? x + 1 : x - 1

julia> code_structured(f, Tuple{Int}) |> only
StructuredCodeInfo(
│ %1 = Base.slt_int(0, x)::Bool
│ ...
└ return %3
) => Int64

julia> sci, ret_type = code_structured(f, Tuple{Int}) |> only # destructure
```
"""
function code_structured(@nospecialize(f), @nospecialize(argtypes);
                         validate::Bool=true, kwargs...)
    map(code_typed(f, argtypes; kwargs...)) do (ci, ret)
        sci = StructuredCodeInfo(ci)
        structurize!(sci; validate)
        return sci => ret
    end
end

"""
    structurize!(sci::StructuredCodeInfo; validate=true) -> StructuredCodeInfo

Convert unstructured control flow in `sci` to structured control flow operations
(IfOp, ForOp, WhileOp, LoopOp) in-place.

This transforms GotoNode and GotoIfNot statements into nested structured ops
that can be traversed hierarchically.

ForOp is created directly during CFG analysis for loops matching counting patterns.
WhileOp is used for condition-at-header loops. LoopOp is used for general cyclic regions.

Returns `sci` for convenience (allows chaining).
"""
function structurize!(sci::StructuredCodeInfo; validate::Bool=true)
    code = sci.code
    stmts = code.code
    types = code.ssavaluetypes

    n = length(stmts)
    n == 0 && return sci
    ctx = StructurizationContext(types, n + 1)

    # Build block-level CFG and convert to control tree
    blocks, cfg = build_block_cfg(code)
    ctree = ControlTree(cfg, code, blocks)

    # Build structured IR
    entry = control_tree_to_structured_ir(ctree, code, blocks, ctx)
    validate && validate_scf(entry)
    validate && validate_no_phis(entry)
    sci.entry = entry
    sci.max_ssa_idx = ctx.next_ssa_idx - 1  # Update to include synthesized indices

    return sci
end
