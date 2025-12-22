# public API

export code_structured, structurize!, StructuredCodeInfo, LocalSSAValue

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

"""
    structurize!(sci::StructuredCodeInfo; loop_patterning=true) -> StructuredCodeInfo

Convert unstructured control flow in `sci` to structured control flow operations
(IfOp, ForOp, WhileOp, LoopOp) in-place.

This transforms GotoNode and GotoIfNot statements into nested structured ops
that can be traversed hierarchically.

Four-phase approach:
1. Build structure (pure SSAValues, no BlockArgs)
2. Create BlockArgs and substitute SSA→BlockArg
3. Upgrade loop patterns (ForOp/WhileOp) if enabled
4. Convert to LocalSSAValues

When `loop_patterning=true` (default), loops are classified as ForOp (bounded counters)
or WhileOp (condition-based). When `false`, all loops become LoopOp.

Returns `sci` for convenience (allows chaining).
"""
function structurize!(sci::StructuredCodeInfo; loop_patterning::Bool=true)
    code = sci.code
    stmts = code.code
    types = code.ssavaluetypes
    n = length(stmts)

    n == 0 && return sci

    # Check if the code is straight-line (no control flow)
    has_control_flow = any(s -> s isa GotoNode || s isa GotoIfNot, stmts)

    if !has_control_flow
        # Straight-line code - no substitutions needed
        new_entry = PartialBlock(1)
        for i in 1:n
            stmt = stmts[i]
            if stmt isa ReturnNode
                new_entry.terminator = stmt
            elseif !(stmt isa GotoNode || stmt isa GotoIfNot)
                push!(new_entry.body, Statement(i, stmt, types[i]))
            end
        end
        sci.entry = new_entry
        return sci
    end

    # Build block-level CFG
    blocks, cfg = build_block_cfg(code)

    # Build control tree using SPIRV.jl-style graph contraction
    ctree = ControlTree(cfg)

    # Phase 1: Build structure (pure SSAValues, no BlockArgs)
    sci.entry = control_tree_to_structured_ir(ctree, code, blocks)

    # Phase 2: Create BlockArgs and substitute SSA→BlockArg
    apply_block_args!(sci.entry, types)

    # Phase 3: Upgrade loop patterns (optional)
    if loop_patterning
        apply_loop_patterns!(sci.entry)
    end

    # Phase 4: Convert to LocalSSAValues
    convert_to_local_ssa!(sci.entry)

    return sci
end
