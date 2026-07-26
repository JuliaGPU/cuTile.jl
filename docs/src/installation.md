# Installation

cuTile.jl is installed like any other Julia package. From the Julia REPL, type `]` to enter
the Pkg REPL mode and run:

```
pkg> add cuTile
```

Or, equivalently, via the `Pkg` API:

```julia
julia> import Pkg; Pkg.add("cuTile")
```


## Requirements

- **Julia 1.11 or newer.**
- **An NVIDIA driver supporting CUDA 13** (version 580 or later). cuTile generates
  [Tile IR](https://docs.nvidia.com/cuda/tile-ir/), which the CUDA 13 toolchain compiles to
  CUBIN.
- **A GPU with [compute capability](https://developer.nvidia.com/cuda/gpus) 8.0 or newer**
  (Ampere and later). Individual features may require newer architectures; see
  [Compatibility](man/compatibility.md).
- **CUDA.jl**, imported before launching a kernel. It provides the `CuArray` type and the
  `@cuda backend=cuTile` launch path.

You do not need to install the CUDA toolkit yourself: CUDA.jl downloads the appropriate
artifacts automatically.


## Checking your setup

Kernel compilation goes through `tileiras`, the Tile IR assembler shipped with the CUDA
compiler artifacts. To see which one is in use, and which Tile IR bytecode version cuTile
will emit by default:

```julia-repl
julia> using cuTile

julia> cuTile.versioninfo()
cuTile toolchain:
- tileiras 13.3.36, artifact installation
- bytecode v13.3, auto-detected
```

The bytecode version is probed from the `tileiras` binary in use, and can be overridden with
the `bytecode_version` preference; `cuTile.bytecode_version()` returns the value on its
own. Which features are available at which version is documented in
[Compatibility](man/compatibility.md).

To run the test suite:

```julia
using Pkg
Pkg.test("cuTile")
```

The `device` and `host` test groups are skipped automatically when CUDA is not functional,
so the suite still runs (in reduced form) on machines without a GPU.
