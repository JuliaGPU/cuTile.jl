# public API

export code_structured, StructuredIRCode

"""
    code_structured(f, argtypes; validate=true, kwargs...) -> Pair{StructuredIRCode, DataType}

Get the structured IR for a function with the given argument types.

This is analogous to `code_typed` but returns a `StructuredIRCode` with
control flow restructured into nested SCF-style operations (if/for/while).

# Arguments
- `f`: The function to analyze
- `argtypes`: Argument types as a tuple of types (e.g., `(Int, Float64)`)
- `validate`: Whether to validate that all control flow was properly structured (default: true).
  When `true`, throws `UnstructuredControlFlowError` if any unstructured control flow remains.

Other keyword arguments are passed to `code_ircode`.

ForOp is created directly during CFG analysis for loops that match counting patterns
(while i < n with i += step, or for i in 1:n). WhileOp is used for condition-at-header
loops that don't match counting patterns. LoopOp is used for general cyclic regions.

# Returns
A `Pair{StructuredIRCode, DataType}` where the first element is the structured IR
and the second is the return type. Displays with MLIR SCF-style syntax.

# Example
```julia
julia> f(x) = x > 0 ? x + 1 : x - 1

julia> code_structured(f, Tuple{Int})
StructuredIRCode(
│ %1 = Base.slt_int(0, x)::Bool
│ ...
└ return %3
) => Int64

julia> sci, ret_type = code_structured(f, Tuple{Int})  # destructure
```
"""
function code_structured(@nospecialize(f), @nospecialize(argtypes);
                         validate::Bool=true, kwargs...)
    ir, ret_type = only(code_ircode(f, argtypes; kwargs...))
    sci = StructuredIRCode(ir; validate)
    return sci => ret_type
end
