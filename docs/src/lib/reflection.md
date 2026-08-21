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

## Intercepting a launch

```@docs
@device_code_tiled
@device_code_typed
@device_code_structured
@device_code_ptx
@device_code_sass
```

`CUDA.@device_code_sass` also works for cuTile kernels: it intercepts module
loads at the driver level (via CUPTI), so it captures any backend's kernels
without backend-specific support.
