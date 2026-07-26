# Debugging

## Printing and assertions

| Operation | Description |
|-----------|-------------|
| `print(args...)` | Print values |
| `println(args...)` | Print values with newline |
| `ct.@assert cond [msg]` | Abort kernel if condition is false |

Standard Julia `print`/`println` work inside kernels. String constants and tiles can be mixed
freely; format specifiers are inferred from element types at compile time. String
interpolation is supported.

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

Since these types are verbose, and are derived from the runtime properties of arrays, it is
often easier to use the `ct.@code_tiled` macro:

```julia-repl
julia> ct.@code_tiled @cuda backend=cuTile blocks=(cld(vector_size, tile_size), 1, 1) vadd(a, b, c, ct.Constant(tile_size))
// vadd(cuTile.TileArray{Float32, 1, cuTile.ArraySpec{1}(128, true, (0,), (32,))}, cuTile.TileArray{Float32, 1, cuTile.ArraySpec{1}(128, true, (0,), (32,))}, cuTile.TileArray{Float32, 1, cuTile.ArraySpec{1}(128, true, (0,), (32,))}, cuTile.Constant{Int64, 16})

cuda_tile.module @kernels {
  entry @vadd(...) {
    ...
    return
  }
}
```

The former works on systems without a GPU, since it does not require CUDA.jl; the latter
needs valid `CuArray`s to pass to the kernel.


## Intercepting a launch

`@device_code_*` macros intercept compilation during a kernel launch:

```julia
ct.@device_code_tiled @cuda backend=cuTile blocks=grid vadd(a, b, c, ct.Constant(16))
ct.@device_code_typed @cuda backend=cuTile blocks=grid vadd(a, b, c, ct.Constant(16))
ct.@device_code_structured @cuda backend=cuTile blocks=grid vadd(a, b, c, ct.Constant(16))
```

| Macro | Output |
|-------|--------|
| `ct.@device_code_tiled` | Final Tile IR (MLIR textual format) |
| `ct.@device_code_typed` | Typed Julia IR after overlay resolution |
| `ct.@device_code_structured` | Structured IR (after control-flow structurization) |

These correspond to successive stages of the compilation pipeline: Julia IR with Tile IR
intrinsics substituted by the overlay method table, then structured control flow, then
emitted Tile IR.


## Dumping bytecode

Setting `JULIA_CUTILE_DUMP_BYTECODE` writes the emitted Tile IR bytecode to disk:

```
❯ JULIA_CUTILE_DUMP_BYTECODE=/tmp/julia_tiles julia --project -e 'using cuTile; ...'
Dumping TILEIR bytecode to file: /tmp/julia_tiles/example.ln42.cutile
```

The resulting files can be disassembled with NVIDIA's `cuda-tile-translate`:

```
❯ cuda-tile-translate --cudatilebc-to-mlir /tmp/julia_tiles/example.ln42.cutile
```

This is the same mechanism cuTile Python exposes through `CUDA_TILE_DUMP_BYTECODE`, so
bytecode from both can be compared directly.
