# Operations

```@meta
CurrentModule = cuTile
```

Most operations on tiles are `Base` functions that cuTile overlays, and are documented in
the [Operations](../man/operations.md) manual page. The entries below are the ones with
cuTile-specific behaviour or no `Base` counterpart.

## Construction

```@docs
arange
```

## Shape

```@docs
cat
broadcast_to
extract
insert
```

## Reinterpretation

```@docs
Base.reinterpret(::Type, ::Tile)
Base.reinterpret(::typeof(reshape), ::Type, ::Tile)
```

## Arithmetic

```@docs
divmod
rsqrt
```

## Matrix multiplication

```@docs
Base.muladd(::Tile, ::Tile, ::Tile)
muladd_scaled
```
