# Code Inspection

```@meta
CurrentModule = cuTile
```

## Compiling a signature

```@docs
code_tiled
code_typed
code_ircode
code_structured
code_ptx
code_sass
```

## Jobs

```@docs
TileJob
tile_job
```

## Intercepting a launch

```@docs
@device_code_tiled
@device_code_structured
@device_code_ptx
```

`@device_code_typed` and `@device_code_warntype` are GPUCompiler's, re-exported;
they work for cuTile kernels through the shared compile hook.

`CUDA.@device_code_sass` works for cuTile kernels: it intercepts module loads
at the driver level (via CUPTI), so it captures any backend's kernels without
backend-specific support.
