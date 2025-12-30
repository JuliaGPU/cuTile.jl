# IRStructurizer

Convert Julia's unstructured SSA IR into structured control flow representation (SCF-style operations).

## Quick Start

```julia-repl
julia> using IRStructurizer

julia> f(x) = x > 0 ? x + 1 : x - 1

julia> code_structured(f, Tuple{Int})
StructuredCodeInfo(
│   %1 = intrinsic Base.slt_int(0, x)::Bool
└── if %1 -> Nothing
    ├ then:
    │   %3 = intrinsic Base.add_int(x, 1)::Int64
    │   return %3
    ├ else:
    │   %5 = intrinsic Base.sub_int(x, 1)::Int64
    └   return %5
) => Int64
```

## API

### `code_structured(f, argtypes; validate=true, loop_patterning=true)`

Get structured IR for function `f` with argument types `argtypes`.

- `validate`: Throw `UnstructuredControlFlowError` if unstructured control flow remains
- `loop_patterning`: Upgrade `WhileOp` to `ForOp` where bounded counter patterns are detected

### `structurize!(sci::StructuredCodeInfo)`

Convert unstructured control flow in-place. Lower-level API if you already have a `StructuredCodeInfo`, which can be constructed using `StructuredCodeInfo(ci::CodeInfo)`.
