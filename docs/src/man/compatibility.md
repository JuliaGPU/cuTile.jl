# Compatibility

cuTile.jl emits [Tile IR](https://docs.nvidia.com/cuda/tile-ir/) bytecode, which
`tileiras` compiles for a specific GPU architecture. Feature availability
therefore depends on two independent things: the **bytecode version** in use,
and the **compute capability** of the device. This page collects both.

For what Tile IR itself guarantees across versions, see the specification's
[stability chapter](https://docs.nvidia.com/cuda/tile-ir/latest/sections/stability.html).


## Baseline requirements

| Requirement | Minimum |
|-------------|---------|
| Julia | 1.11 |
| NVIDIA driver | 580 (CUDA 13) |
| Compute capability | 8.0 (Ampere) |

CUDA.jl must be imported to launch kernels; it supplies the CUDA toolkit
artifacts, including `tileiras`.


## Bytecode versions

cuTile.jl can emit bytecode versions **v13.1 through v13.4**. By default it
probes the `tileiras` binary in use and emits the newest version that binary
accepts; the `bytecode_version` preference overrides this.
`cuTile.versioninfo()` reports what will be used.

Each architecture has a minimum bytecode version below which Tile IR is not
supported at all:

| Architecture | Compute capability | Minimum bytecode |
|--------------|--------------------|------------------|
| Blackwell | ≥ 10.0 | v13.1 |
| Hopper | 9.0 | v13.3 |
| Ampere / Ada | 8.0 – 8.9 | v13.2 |
| older | < 8.0 | unsupported |

cuTile checks this at launch and reports an error naming both the detected
version and the requirement.


## Features by bytecode version

Everything not listed here works at v13.1.

### Requires v13.2

| Feature | Notes |
|---------|-------|
| `Float8_E8M0FNU` | Element type, used as an MX block scale |
| `atan2` | |

### Requires v13.3

| Feature | Notes |
|---------|-------|
| `Float4_E2M1FN` | Element type |
| `ct.muladd_scaled` | Also requires Blackwell |
| `muladd(a, b, acc; fast_acc=true)` | FP8 inputs only; only takes effect on Hopper |
| `exp` with approximate rounding | `ct.@fpmode rounding_mode=ct.Rounding.Approx` |
| `ct.atomic_store_*` | View-based atomic reductions |
| Atomic add on `BFloat16` | Also requires Hopper (sm_90) or newer |
| Sparse views | `@view a[tile_of_indices, ...]`, consumed by `ct.load`/`ct.store` |
| `StepRange` views | `@view a[i:s:j, :]` |
| `eachtile` with `step != shape` | Equal shape and step work at v13.1 |
| `num_worker_warps` compiler option | |
| Whole-tile `reinterpret` | Between element types of differing width |

### Requires v13.4

| Feature | Notes |
|---------|-------|
| `ct.insert` | Replace a non-overlapping sub-tile |
| `check_bounds=false` | On `ct.load` and `ct.store` only; on `ct.gather`/`ct.scatter` it merely skips the bounds mask |


## Features by architecture

| Feature | Requirement | Behaviour otherwise |
|---------|-------------|---------------------|
| `ct.muladd_scaled` | Blackwell (≥ sm_100) | Error |
| Atomic add on `BFloat16` | Hopper (≥ sm_90) | Error |
| `fast_acc=true` | Hopper (sm_90) | Silently ignored |

`fast_acc` is the exception to the pattern: it is a throughput hint rather than
a capability, so on architectures where it does nothing, it is accepted and
ignored rather than rejected. It is still an error to pass it with non-FP8
inputs on any architecture.


## API stability

cuTile.jl is in beta. Most Tile IR features are implemented, and the package is
verified against the benchmarks and tests in the repository, but interfaces may
change without notice between releases. Pin a version if you need stability.

Symbols are marked `public` or `export`ed when they are intended for use;
anything else is internal, and will change without a deprecation cycle.
