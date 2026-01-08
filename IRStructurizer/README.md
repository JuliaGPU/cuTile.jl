# IRStructurizer

Convert Julia's unstructured SSA IR into structured control flow representation (SCF-style operations).

## Quick Start

```julia-repl
julia> using IRStructurizer

julia> f(x) = x > 0 ? x + 1 : x - 1

julia> code_structured(f, Tuple{Int})
1-element Vector{Pair{StructuredCodeInfo, DataType}}:
 StructuredCodeInfo(
│ %1 = intrinsic Base.slt_int(0, x)::Bool
│ %2 = if %1 -> Nothing
│ ├ then:
│ │   %3 = intrinsic Base.add_int(x, 1)::Int64
│ │   return %3
│ ├ else:
│ │   %5 = intrinsic Base.sub_int(x, 1)::Int64
│ └   return %5
) => Int64
```

## API

### `code_structured(f, argtypes; validate=true, loop_patterning=true)`

Get structured IR for function `f` with argument types `argtypes`.

- `validate`: Throw `UnstructuredControlFlowError` if unstructured control flow remains
- `loop_patterning`: Upgrade `WhileOp` to `ForOp` where bounded counter patterns are detected

### `structurize!(sci::StructuredCodeInfo)`

Convert unstructured control flow in-place. Lower-level API if you already have a `StructuredCodeInfo`, which can be constructed using `StructuredCodeInfo(ci::CodeInfo)`.


## Implementation

The structurization pipeline converts Julia's unstructured SSA IR (with `GotoNode` and
`GotoIfNot`) into nested control flow operations (`IfOp`, `ForOp`, `WhileOp`, `LoopOp`).

```
Julia CodeInfo
     │
     ▼ blocks.jl
Basic Blocks + CFG (SimpleDiGraph)
     │
     ▼ dominance.jl
Dominators + Edge Classification
     │
     ▼ control_tree.jl
Control Tree (hierarchical regions)
     │
     ▼ structure.jl
Structured IR (nested Blocks with IfOp/ForOp/etc.)
```

### Phase 1: Block Extraction

`build_block_cfg()` scans the IR for jump targets (`GotoNode`, `GotoIfNot`) and splits the
code into basic blocks. Each block has a range of statement indices and
predecessor/successor relationships.

### Phase 2: CFG Analysis

`dominators()` computes dominator sets for each block. `EdgeClassification` categorizes
edges as tree, forward, retreating, or cross edges. Back edges (retreating edges where the
target dominates the source) identify loops.

### Phase 3: Control Tree Construction

`ControlTree()` iteratively pattern-matches on the CFG to identify structured regions:

| Region Type | Pattern |
|-------------|---------|
| `REGION_BLOCK` | Linear chain of blocks |
| `REGION_IF_THEN` | Conditional with one branch |
| `REGION_IF_THEN_ELSE` | Diamond pattern (two branches merge) |
| `REGION_NATURAL_LOOP` | General cyclic region |
| `REGION_WHILE_LOOP` | Header with back edge from body |
| `REGION_FOR_LOOP` | While loop with detected counter pattern |

Matched regions are contracted into single nodes, and the process repeats until the entire
CFG reduces to a single control tree.

For-loop detection analyzes phi nodes in loop headers to find induction variables with
patterns like `===(iv, bound)` or `slt_int(iv, bound)`.

### Phase 4: Structured IR Generation

`control_tree_to_structured_ir()` recursively converts the control tree into nested `Block`
structures containing:

- **`IfOp`**: Condition + then/else blocks, results via `YieldOp`
- **`LoopOp`**: General loop with `ContinueOp`/`BreakOp` terminators
- **`WhileOp`**: Before (condition) + after (body) regions
- **`ForOp`**: Lower/upper/step bounds + body block with induction variable as `BlockArg`

Phi nodes become explicit `BlockArg` values (like MLIR block arguments), preserving SSA
semantics.


## Acknowledgements

Most of this package is based on [Cédric Belmant](https://github.com/serenity4)'s
[SPIRV.jl](https://github.com/serenity4/SPIRV.jl) structurization code.
