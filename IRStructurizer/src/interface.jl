# public API

export code_structured

"""
    code_structured(f, argtypes; validate=true, loop_patterning=true, kwargs...) -> StructuredCodeInfo

Get the structured IR for a function with the given argument types.

This is analogous to `code_typed` but returns a `StructuredCodeInfo` with
control flow restructured into nested SCF-style operations (if/for/while).

# Arguments
- `f`: The function to analyze
- `argtypes`: Argument types as a tuple of types (e.g., `(Int, Float64)`)
- `validate`: Whether to validate that all control flow was properly structured (default: true).
  When `true`, throws `UnstructuredControlFlowError` if any unstructured control flow remains.
- `loop_patterning`: Whether to apply loop pattern detection (default: true).
  When `true`, loops are classified as ForOp (bounded counters) or WhileOp (condition-based).
  When `false`, all loops become LoopOp, useful for testing CFG analysis separately.

Other keyword arguments are passed to `code_typed`.

# Returns
A `StructuredCodeInfo` that displays with MLIR SCF-style syntax showing
nested control flow structure.

# Example
```julia
julia> f(x) = x > 0 ? x + 1 : x - 1

julia> code_structured(f, Tuple{Int})
StructuredCodeInfo {
  %1 = Base.slt_int(0, x) : Bool
  scf.if %1 {
    %3 = Base.add_int(x, 1) : Int64
    scf.yield %3
  } else {
    %5 = Base.sub_int(x, 1) : Int64
    scf.yield %5
  }
  return %3
}

julia> code_structured(f, Tuple{Int}; validate=false)  # skip validation
julia> code_structured(f, Tuple{Int}; loop_patterning=false)  # all loops as LoopOp
```
"""
function code_structured(@nospecialize(f), @nospecialize(argtypes);
                         validate::Bool=true, loop_patterning::Bool=true, kwargs...)
    ci, _ = only(code_typed(f, argtypes; kwargs...))
    sci = StructuredCodeInfo(ci)
    structurize!(sci; loop_patterning)
    if validate
        validate_scf(sci)
    end
    return sci
end
