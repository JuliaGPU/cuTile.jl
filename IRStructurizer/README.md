# IRStructurizer

Convert Julia's unstructured SSA IR into structured control flow representation (SCF-style operations).

## Quick Start

```julia-repl
julia> using IRStructurizer

julia> f(x) = x > 0 ? x + 1 : x - 1

julia> code_structured(f, Tuple{Int})
StructuredIRCode(
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

### `code_structured(f, argtypes; validate=true)`

Get structured IR for function `f` with argument types `argtypes`.

- `validate`: Throw `UnstructuredControlFlowError` if unstructured control flow remains

### `structurize!(sci::StructuredIRCode)`

Convert unstructured control flow in-place. Lower-level API if you already have a `StructuredIRCode`, which can be constructed from Julia's `IRCode`.


## Implementation

The structurization pipeline converts Julia's unstructured SSA IR (with `GotoNode` and
`GotoIfNot`) into nested control flow operations (`IfOp`, `ForOp`, `WhileOp`, `LoopOp`).

```
Julia IRCode (from code_ircode, includes CFG)
     │
     ▼ control_tree.jl
Control Tree (hierarchical regions)
     │
     ▼ structure.jl
Structured IR (nested Blocks with IfOp/ForOp/etc.)
```

### Control Tree Construction

`ControlTree()` pattern-matches on the CFG (from `ir.cfg.blocks`) to identify structured
regions. Back edges are detected using `Core.Compiler.construct_domtree()`.

| Region Type | Pattern |
|-------------|---------|
| `REGION_BLOCK` | Linear chain of blocks |
| `REGION_IF_THEN` | Conditional with one branch |
| `REGION_IF_THEN_ELSE` | Diamond pattern (two branches merge) |
| `REGION_WHILE_LOOP` | Header with back edge from body |
| `REGION_FOR_LOOP` | While loop with detected counter pattern |
| `REGION_NATURAL_LOOP` | General cyclic region |

Matched regions are contracted into single nodes, and the process repeats until the entire
CFG reduces to a single control tree.

For-loop detection analyzes phi nodes in loop headers to find induction variables with
patterns like `===(iv, bound)` or `slt_int(iv, bound)`.

### Structured IR Generation

`control_tree_to_structured_ir()` converts the control tree into nested `Block` structures:

- **`IfOp`**: Condition + then/else blocks, results via `YieldOp`
- **`ForOp`**: Lower/upper/step bounds + body block with induction variable as `BlockArg`
- **`WhileOp`**: Before (condition) + after (body) regions
- **`LoopOp`**: General loop with `ContinueOp`/`BreakOp` terminators

Phi nodes become explicit `BlockArg` values (like MLIR block arguments).


## Acknowledgements

Most of this package is based on [Cédric Belmant](https://github.com/serenity4)'s
[SPIRV.jl](https://github.com/serenity4/SPIRV.jl) structurization code.
