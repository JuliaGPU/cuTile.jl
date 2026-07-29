# Debugging

## Printing and assertions

| Operation | Description |
|-----------|-------------|
| `print(args...)` | Print values |
| `println(args...)` | Print values with newline |
| `ct.@assert cond [msg]` | Abort kernel if condition is false |

Standard Julia `print`/`println` work inside kernels. String constants and tiles
can be mixed freely; format specifiers are inferred from element types at
compile time. String interpolation is supported.

```julia
println("Block ", ct.bid(1), ": tile=", tile)
println("result=$result")  # string interpolation
ct.@assert idx <= n "index out of bounds"
```

These are debugging aids and are not optimized for performance.


## Inspecting generated Tile IR

The generated Tile IR can be inspected with `ct.code_tiled`:

```julia
ct.code_tiled(vadd, Tuple{ct.TileArray{Float32, 1, ct.ArraySpec{1}(128, true, (0,), (32,))},
                          ct.TileArray{Float32, 1, ct.ArraySpec{1}(128, true, (0,), (32,))},
                          ct.TileArray{Float32, 1, ct.ArraySpec{1}(128, true, (0,), (32,))},
                          ct.Constant{Int64, 16}})
```

Spelling out those types is only worth it when you have no GPU, since
`code_tiled` does not need CUDA.jl. Otherwise let the launch site derive them
for you with `ct.@device_code_tiled`, described next.


## Intercepting a launch

`@device_code_*` macros intercept compilation during a kernel launch, deriving
the argument types from the actual `CuArray`s:

```julia-repl
julia> ct.@device_code_tiled @cuda backend=cuTile blocks=cld(vector_size, tile_size) vadd(a, b, c, ct.Constant(tile_size))
// vadd(cuTile.TileArray{Float32, 1, cuTile.ArraySpec{1, 128, true, (0,), (16,), false}()}, …)

cuda_tile.module @kernels {
  entry @vadd(%arg0: tile<ptr<f32>>, …) {
    ...
    return
  }
}
```

Three are available, corresponding to successive stages of the pipeline:

| Macro | Output |
|-------|--------|
| `ct.@device_code_tiled` | Final Tile IR (MLIR textual format) |
| `ct.@device_code_typed` | Typed Julia IR after overlay resolution |
| `ct.@device_code_structured` | Structured IR (after control-flow structurization) |

Read top to bottom, they run backwards through the pipeline: Julia IR with Tile
IR intrinsics substituted by the overlay method table, then structured control
flow, then emitted Tile IR.


## Dumping bytecode

Setting `JULIA_CUTILE_DUMP_BYTECODE` writes the emitted Tile IR bytecode to disk:

```
❯ JULIA_CUTILE_DUMP_BYTECODE=/tmp/julia_tiles julia --project -e 'using cuTile; ...'
Dumping TILEIR bytecode to file: /tmp/julia_tiles/example.ln42.cutile
```

The resulting files can be disassembled with NVIDIA's `tileirdisasm`:

```
❯ tileirdisasm /tmp/julia_tiles/example.ln42.cutile
```

CUDA toolkits older than 13.4 ship `cuda-tile-translate` instead, which needs an
explicit flag and cannot read bytecode newer than its own version:

```
❯ cuda-tile-translate --cudatilebc-to-mlir /tmp/julia_tiles/example.ln42.cutile
```

This is the same mechanism cuTile Python exposes through
`CUDA_TILE_DUMP_BYTECODE`, so bytecode from both can be compared directly.
