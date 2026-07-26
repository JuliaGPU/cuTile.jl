# Memory Model

Atomic operations take `memory_order` and `memory_scope` keyword arguments that control how
their effects are ordered with respect to other memory operations, and which threads observe
that ordering. Both map directly onto the Tile IR memory model; the
[specification](https://docs.nvidia.com/cuda/tile-ir/13.3/memory_model.html) is the
authoritative reference for the formal semantics, including token ordering and the
definition of a data race.


## Ordering

`ct.MemoryOrder` selects the ordering strength:

| Value | Meaning |
|-------|---------|
| `Weak` | No concurrent accesses to the location |
| `Relaxed` | There may be concurrent accesses, but this one establishes no happens-before relationship |
| `Release` | If this release is observed by an acquire, happens-before is established |
| `Acquire` | If this acquire observes a release, happens-before is established |
| `AcqRel` | Both a release and an acquire |

Synchronizing through memory is a two-party process: it takes a releaser and an acquirer
observing the same location. Any ordering other than `Weak` requires a scope.

The `atomic_*` functions default to `ct.MemoryOrder.AcqRel`, which is the safe choice: it
orders surrounding memory traffic in both directions. Weakening it to `Relaxed` is worthwhile
when an atomic is used purely as a counter or accumulator whose result nothing else is
synchronized against.


## Scope

`ct.MemScope` selects which threads participate in that ordering:

| Value | Meaning |
|-------|---------|
| `Block` | Threads within the same block |
| `Device` | All threads on the same GPU |
| `System` | All threads in the system, including other GPUs and the host |

The default is `ct.MemScope.Device`. Narrowing to `Block` is cheaper when the communication
is genuinely block-local; widening to `System` is required when the host or a peer GPU reads
the result concurrently.


## Defaults by operation

| Form | Default order | Default scope |
|------|---------------|---------------|
| `ct.atomic_*` | `AcqRel` | `Device` |
| `ct.atomic_store_*` | relaxed (fixed) | device-wide (fixed) |
| `ct.@atomic` statement form | `:monotonic` (relaxed) | `Device` |
| `ct.@atomic` value form | `:acquire_release` | `Device` |

The view-based `atomic_store_*` reductions do not accept ordering arguments: they are always
relaxed and device-wide. The statement form of `ct.@atomic` defaults to relaxed because it
discards the old value, so there is usually nothing to synchronize against; the value form
returns `old => new` and therefore defaults to the stronger ordering.

See [Atomics](atomics.md) for the operations themselves.


## Ordering example

Acquire/release pairing to implement a spin lock:

```julia
# acquire the lock
while ct.atomic_cas(locks, idx, Int32(0), Int32(1);
                    memory_order=ct.MemoryOrder.Acquire) == Int32(1)
end

# ... critical section ...

# release it
ct.atomic_xchg(locks, idx, Int32(0); memory_order=ct.MemoryOrder.Release)
```
