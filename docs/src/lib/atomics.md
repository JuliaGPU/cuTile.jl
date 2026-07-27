# Atomics

```@meta
CurrentModule = cuTile
```

## Read-modify-write

```@docs
atomic_cas
atomic_xchg
atomic_add
atomic_max
atomic_min
atomic_and
atomic_or
atomic_xor
```

## View-based reductions

```@docs
atomic_store_add
atomic_store_max
atomic_store_min
atomic_store_and
atomic_store_or
atomic_store_xor
```

## Macro form

```@docs
@atomic
```

## Ordering

```@docs
MemoryOrder
MemScope
```
