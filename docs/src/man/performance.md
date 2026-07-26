# Performance

## Kernel configuration

`ct.@compiler_options` sets optimization hints inside a kernel function body:

```julia
function matmul(A, B, C, ...)
    ct.@compiler_options num_ctas=ct.ByTarget(v"10.0" => 2) occupancy=8
    ...
end
```

| Option | Description | Valid values |
|--------|-------------|--------------|
| `num_ctas` | Number of CTAs in a CGA | Powers of 2 |
| `occupancy` | Target concurrent CTAs per SM | 1–32 |
| `opt_level` | Optimization level | 0–3 |
| `num_worker_warps` | Worker warps per CTA in a warp-specialized kernel | 4 or 8 |

Values can be plain scalars or `ct.ByTarget(...)` for per-architecture dispatch. `ByTarget`
maps compute capabilities to values, with an optional default:

```julia
ct.@compiler_options num_ctas=ct.ByTarget(v"10.0" => 4, v"12.0" => 2; default=1)
```

Hints can also be passed as keyword arguments to `@cuda backend=cuTile` or `ct.code_tiled`,
which take precedence over `@compiler_options`.

These correspond to the hints described in the Tile IR
[optimization guide](https://docs.nvidia.com/cuda/tile-ir/latest/sections/optimization_guide.html),
which documents what each one does to code generation.


## Load/store hints

`ct.load` and `ct.store` accept optional keyword arguments that influence memory traffic
scheduling:

| Hint | Description |
|------|-------------|
| `latency` | DRAM traffic weight hint, integer 1 (low) to 10 (high). Default: compiler-inferred. |
| `allow_tma` | Whether to allow Tensor Memory Accelerator lowering. Default: allowed. |


## Array specialization

Kernels are specialized on each array's layout, as described under
[what makes a distinct kernel](execution.md#What-makes-a-distinct-kernel). The performance
consequence is worth stating separately: a contiguous, 128-byte-aligned array whose dimensions
divide evenly by the tile shape compiles to code without tile-boundary handling and with wider
vectorized accesses. Views and unusual strides weaken those guarantees, which shows up as a
different and slower specialization rather than as an error — so a kernel that got slower after
you started passing it a `@view` has not been mis-tuned, it has been re-specialized.
