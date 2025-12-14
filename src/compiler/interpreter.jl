# Typed IR utilities for cuTile compilation
#
# Gets fully-typed SSA IR that we can translate to Tile IR bytecode.

"""
    get_typed_ir(f, argtypes; world=Base.get_world_counter(), optimize=true) -> (CodeInfo, return_type)

Get the typed SSA IR for a function with the given argument types.
If optimize=true (default), returns optimized IR with clean SSA form.
If optimize=false, returns IR that preserves all intermediate assignments.
"""
function get_typed_ir(@nospecialize(f), @nospecialize(argtypes);
                      world::UInt = Base.get_world_counter(),
                      optimize::Bool = true)
    sig = Base.signature_type(f, argtypes)
    results = Base.code_typed_by_type(sig; optimize, world)

    if isempty(results)
        error("Type inference failed for $f with argument types $argtypes")
    end

    ci, rettype = results[1]
    return ci, rettype
end

"""
    get_method_instance(f, argtypes; world=Base.get_world_counter()) -> MethodInstance

Get the MethodInstance for a function call.
"""
function get_method_instance(@nospecialize(f), @nospecialize(argtypes); world::UInt = Base.get_world_counter())
    tt = Base.signature_type(f, argtypes)
    match = Base._which(tt; world)
    return CC.specialize_method(match)
end

#=============================================================================
 Public API: code_structured
=============================================================================#

"""
    code_structured(f, argtypes; world=Base.get_world_counter(), optimize=true, validate=false) -> StructuredCodeInfo

Get the structured IR for a function with the given argument types.

This is analogous to `code_typed` but returns a `StructuredCodeInfo` with
control flow restructured into nested SCF-style operations (if/for/while).

# Arguments
- `f`: The function to analyze
- `argtypes`: Argument types as a Tuple type (e.g., `Tuple{Int, Float64}`)
- `world`: World age for method lookup (default: current)
- `optimize`: Whether to use optimized IR (default: true)
- `validate`: Whether to validate that all control flow was properly structured (default: false).
  When `true`, throws `UnstructuredControlFlowError` if any unstructured control flow remains.

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

julia> code_structured(f, Tuple{Int}; validate=true)  # throws if restructuring incomplete
```
"""
function code_structured(@nospecialize(f), @nospecialize(argtypes);
                         world::UInt = Base.get_world_counter(),
                         optimize::Bool = true,
                         validate::Bool = false)
    ci, _ = get_typed_ir(f, argtypes; world, optimize)
    sci = lower_to_structured_ir(ci)
    if validate
        validate_structured_control_flow(sci)
    end
    return sci
end
