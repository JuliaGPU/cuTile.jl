# Installation

cuTile.jl is installed like any other Julia package. From the Julia REPL, type
`]` to enter the Pkg REPL mode and run:

```
pkg> add cuTile
```

Or, equivalently, via the `Pkg` API:

```julia
julia> import Pkg; Pkg.add("cuTile")
```


## Requirements

- **Julia 1.11 or newer.**
- **An NVIDIA driver supporting CUDA 13** (version 580 or later). cuTile.jl
  generates [Tile IR](https://docs.nvidia.com/cuda/tile-ir/), which the CUDA 13
  toolchain compiles to CUBIN.
- **A GPU with [compute capability](https://developer.nvidia.com/cuda/gpus) 8.0
  or newer** (Ampere and later). Individual features may require newer
  architectures; see [Compatibility](man/compatibility.md).
- **CUDA.jl**, imported before launching a kernel. It provides the `CuArray`
  type and the `@cuda backend=cuTile` launch path.

You do not need to install the CUDA toolkit yourself: CUDA.jl downloads the
appropriate artifacts automatically. See the [CUDA.jl installation
guide](https://cuda.juliagpu.org/stable/installation/overview/) for driver,
toolkit-selection and troubleshooting details; cuTile.jl does not duplicate
that setup here.


## Checking your setup

Kernel compilation goes through `tileiras`, the Tile IR assembler shipped with
the CUDA compiler artifacts. To see which one is in use, and which Tile IR
bytecode version cuTile will emit by default:

```julia-repl
julia> using cuTile

julia> cuTile.versioninfo()
cuTile toolchain:
- tileiras 13.3.36, artifact installation
- bytecode v13.3, auto-detected
```

The bytecode version is probed from the `tileiras` binary in use, and can be
set to an older supported version with the `bytecode_version` preference;
`cuTile.bytecode_version()` returns the selected target and
`cuTile.tileiras_version()` returns the compiler version. Which features are
available at which version is documented in
[Compatibility](man/compatibility.md).

To run the test suite:

```julia-repl
julia> using Pkg

julia> Pkg.test("cuTile")
```

The test suite requires a functional CUDA GPU and exits immediately when
`CUDA.functional()` is false. Building the documentation and using
`ct.code_tiled` with explicit argument types do not require a GPU.
