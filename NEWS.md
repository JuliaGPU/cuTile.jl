# cuTile.jl release history

This document lists the noteworthy changes in every release of cuTile.jl, newest
first. It is a hand-written summary aimed at users; for the complete list of
merged pull requests, see the [GitHub
releases](https://github.com/JuliaGPU/cuTile.jl/releases).


## v1.0 (August 2026)

The README's single long tutorial was replaced by a documentation site at
<https://juliagpu.github.io/cuTile.jl>, with tutorials, a manual, an API
reference, a compatibility matrix stating what each feature needs from the Tile
IR version and the GPU architecture, and a comparison with cuTile Python
([#284](https://github.com/JuliaGPU/cuTile.jl/pull/284)).

The rest of the cycle went into catching up with Tile IR 13.4 and with cuTile
Python: view-based atomic reductions, sparse and stepped array views,
`eachtile`, array-construction syntax, multi-dimensional reductions, 64-bit
array indexing, and programmatic dependent launch.

*Breaking changes*:

- `TileArray` gained an index-type parameter: `TileArray{T,N,Spec}` is now
  `TileArray{T,N,I,Spec}`, with `I` either `Int32` or `Int64`. Sizes and strides
  use `Int32` when they fit and `Int64` otherwise, chosen when the argument is
  converted, so crossing the boundary compiles a separate kernel.
  `ct.TileArray(a; index=Int64)` selects the wide path explicitly and
  `ct.indextype` reports it; `Int64` indexing requires bytecode v13.3
  ([#292](https://github.com/JuliaGPU/cuTile.jl/pull/292)).
- `ArraySpec` gained two type parameters, `may_alias_internally` (whether two
  distinct in-bounds indices can name the same memory) and `singleton` (which
  axes have both size and stride one). Both are derived from the runtime sizes
  and strides, and the existing positional constructor still works
  ([#262](https://github.com/JuliaGPU/cuTile.jl/pull/262),
  [#296](https://github.com/JuliaGPU/cuTile.jl/pull/296)).
- `ct.cat` was removed in favor of Julia's own concatenation syntax, described
  under new features below
  ([#289](https://github.com/JuliaGPU/cuTile.jl/pull/289)).
- The bitwise atomics `atomic_or`, `atomic_and` and `atomic_xor` no longer
  convert the update value: it has to have the array's element type already
  ([#283](https://github.com/JuliaGPU/cuTile.jl/pull/283)).
- The disk cache is configured through package preferences instead of
  environment variables: `disk_cache = false` disables it, `cache_dir` overrides
  its directory and `cache_size_bytes` its 1 GiB limit.
  `JULIA_CUTILE_CACHE_DIR` and `JULIA_CUTILE_CACHE_SIZE` are gone
  ([#296](https://github.com/JuliaGPU/cuTile.jl/pull/296)).

*New features*:

- Added support for programmatic dependent launch, which overlaps the tail of
  one kernel with the preamble of the next one on the same stream. The consumer
  is launched with `@cuda backend=cuTile dependent=true`, the producer calls
  `ct.grid_dependency_control_launch_dependents()`, and the consumer calls
  `ct.grid_dependency_control_wait()` before reading the producer's results.
  Requires Tile IR 13.4 and compute capability 9.0 or newer. `launch` also
  gained a `stream` keyword argument
  ([#299](https://github.com/JuliaGPU/cuTile.jl/pull/299)).
- Added view-based atomic reductions: `atomic_store_add`, `atomic_store_max`,
  `atomic_store_min`, `atomic_store_or`, `atomic_store_and` and
  `atomic_store_xor` reduce a tile into a `TileArray` or into a window of an
  `eachtile` view without returning the old value. On top of them sits
  `ct.@atomic`, a `Base`-style macro with statement forms (`ct.@atomic a[i] +=
  v`) and a value form returning `old => new`. Requires bytecode v13.3
  ([#283](https://github.com/JuliaGPU/cuTile.jl/pull/283)).
- The read-modify-write `atomic_*` functions gained a `mask`, AND'd with the
  bounds mask; masked-out elements are not modified, and `atomic_cas` returns
  `expected` for them while the others return an implementation-defined value
  ([#297](https://github.com/JuliaGPU/cuTile.jl/pull/297)).
- Added `eachtile(a, shape; step=shape)`, an indexable device-side collection of
  fixed-shape tile windows: `tiles[i, j]` loads and `tiles[i, j] = tile` stores.
  A `step` smaller than the tile shape produces overlapping windows and a larger
  one gapped windows; unequal shape and step require bytecode v13.3
  ([#281](https://github.com/JuliaGPU/cuTile.jl/pull/281)).
- `view`, `@view` and `@views` on a `TileArray` accept more index kinds.
  Positive step ranges scale that axis' element stride. A 1D integer `Tile` in
  exactly one dimension, combined with unit ranges or `:` in the others, creates
  a sparse view that `ct.load` and `ct.store` consume as a Tile IR
  gather/scatter view. Both require bytecode v13.3
  ([#280](https://github.com/JuliaGPU/cuTile.jl/pull/280),
  [#282](https://github.com/JuliaGPU/cuTile.jl/pull/282)).
- Julia's array-construction syntax builds tiles inside kernels: `[a, b, c]` and
  typed forms like `Float32[a, b]`, bracket concatenation (`[A; B]`, `[a b; C]`,
  `[A;;; B]`) of scalars and tiles, and `cat(A, B...; dims)`. Typed empty
  literals such as `Float32[]` produce zero-volume tiles, which are folded away
  by concatenation and rejected with an explanatory error where Tile IR cannot
  represent them ([#289](https://github.com/JuliaGPU/cuTile.jl/pull/289),
  [#291](https://github.com/JuliaGPU/cuTile.jl/pull/291)).
- Reductions take a multi-dimensional `dims`: an integer, an iterable of
  integers, or `:` for a scalar result. This covers the tile-level and the
  host-level `ct.Tiled` reductions alike, and `dropdims` accepts a tuple
  ([#290](https://github.com/JuliaGPU/cuTile.jl/pull/290)). `maximum`,
  `minimum`, `argmax` and `argmin` additionally take a `propagate_nan` keyword
  argument, whose default of ignoring NaNs matches cuTile Python
  ([#278](https://github.com/JuliaGPU/cuTile.jl/pull/278)).
- Added support for Tile IR bytecode v13.4, emitted by default when `tileiras`
  accepts it. With it come `ct.insert`, which replaces a non-overlapping subtile
  and inverts `ct.extract`, and `check_bounds=false` on `ct.load` and
  `ct.store`, an explicit promise that the whole tile lies inside the array
  ([#278](https://github.com/JuliaGPU/cuTile.jl/pull/278),
  [#287](https://github.com/JuliaGPU/cuTile.jl/pull/287)).
- Float-to-float conversions accept an explicit rounding mode:
  `Float32(x, RoundDown)` and `Float32.(tile, RoundUp)`, with `RoundNearest`,
  `RoundToZero`, `RoundDown`, `RoundUp` and `RoundNearestTiesAway`. Anything
  other than round-to-nearest requires Tile IR 13.4
  ([#294](https://github.com/JuliaGPU/cuTile.jl/pull/294)).
- `ct.load` and `ct.store` accept `memory_order` and `memory_scope`. Loads take
  `Weak`, `Relaxed` or `Acquire` and stores `Weak`, `Relaxed` or `Release`;
  anything but `Weak` requires a scope
  ([#296](https://github.com/JuliaGPU/cuTile.jl/pull/296)).
- `ct.code_tiled` and `ct.@device_code_tiled` take `remarks=true` to run
  `tileiras` and print its optimization remarks, which report tensor-core
  selection and memory alignment issues. Requires `tileiras` 13.4
  ([#293](https://github.com/JuliaGPU/cuTile.jl/pull/293)).
- `ct.arange` takes `start` and `step` keyword arguments, and Julia's `mod` and
  a `ct.divmod` are available in kernels, both with Julia's floored sign
  convention ([#278](https://github.com/JuliaGPU/cuTile.jl/pull/278)).

*Minor changes*:

- Values crossing into and out of arrays convert like they do in Julia:
  `ct.store` and `ct.scatter` convert to the array's element type instead of
  demanding an exact match, and a scalar in a broadcast takes the element type
  of the tile it is combined with, so `tile .* 2.0` over a `Float32` tile stays
  in `Float32` ([#296](https://github.com/JuliaGPU/cuTile.jl/pull/296)).
- `^` with an integer exponent goes through `pow` and restores the sign from the
  exponent's parity, rather than the squaring chain it used for small literal
  exponents ([#287](https://github.com/JuliaGPU/cuTile.jl/pull/287)).
- `@inbounds` and `--check-bounds` are resolved at compile time, so ordinary
  Julia `@boundscheck` blocks in a kernel cost nothing. They deliberately do not
  affect the bounds handling of `ct.load`, `ct.store`, `eachtile`, views,
  gathers, scatters or atomics: Tile IR's checked memory operations define what
  partial edge tiles do rather than protect against a mistake, so padding and
  clipping are part of a correct kernel's result. Use `check_bounds=false` for
  the stronger promise ([#279](https://github.com/JuliaGPU/cuTile.jl/pull/279),
  [#288](https://github.com/JuliaGPU/cuTile.jl/pull/288)).
- Toolchain selection was unified and its inputs validated: an unsupported
  `bytecode_version` preference, a version the selected `tileiras` cannot
  compile, and a version too old for the target architecture are all reported as
  errors naming what was selected. `cuTile.versioninfo()` shows the selected
  version next to the highest one `tileiras` accepts, and a new
  `compiler_timeout_seconds` preference bounds each `tileiras` invocation
  ([#298](https://github.com/JuliaGPU/cuTile.jl/pull/298),
  [#296](https://github.com/JuliaGPU/cuTile.jl/pull/296),
  [#300](https://github.com/JuliaGPU/cuTile.jl/pull/300)).
- Several invalid uses are now rejected rather than silently miscompiled: an
  `order` argument that is not a permutation of the axes, an
  `assume_divisible_by` that contradicts a known constant, and `reinterpret` to
  or from `Bool` ([#278](https://github.com/JuliaGPU/cuTile.jl/pull/278),
  [#296](https://github.com/JuliaGPU/cuTile.jl/pull/296)).

*Bug fixes*:

- Fixed a write-after-write race in the token-ordering pass. Its loop-parallel
  store optimization accepted any index transitively derived from the induction
  variable as injective, including expressions like `(i + 1) ÷ 2`, so stores
  from different iterations could hit the same address with no ordering between
  them ([#262](https://github.com/JuliaGPU/cuTile.jl/pull/262)).
- Fixed soundness holes in the bounds and divisibility analyses, which drive
  bounds masks, `nsw`/`nuw` flags and alignment assumptions: stepped loops could
  get a bound the loop overshoots, unrefuted facts survived a join through
  never-visited statements, zero-extension was treated as value-preserving for
  possibly-negative ranges, and arithmetic ranges were not clamped to the result
  width ([#265](https://github.com/JuliaGPU/cuTile.jl/pull/265),
  [#277](https://github.com/JuliaGPU/cuTile.jl/pull/277)).
- `atomic_max` and `atomic_min` on arrays with an unsigned element type compared
  as signed ([#283](https://github.com/JuliaGPU/cuTile.jl/pull/283)).
- Host-level reductions over `ct.Tiled` arrays handle a `dims` region that is
  empty or reaches past the array's rank, and reject a non-integer or
  non-positive dimension with an error instead of failing later
  ([#290](https://github.com/JuliaGPU/cuTile.jl/pull/290)).


## v0.3 (May 2026)

Kernels are now launched through CUDA.jl's `@cuda`, and compilation results
persist across sessions. cuTile.jl no longer keeps its CUDA support in a package
extension: it depends on CUDACore directly and registers a back-end with
CUDACore's launch protocol, so the same `@cuda`, `cufunction` and kernel-object
structure that the LLVM back-end uses also applies to Tile IR kernels.

*Breaking changes*:

- The combiner passed to `reduce`, `mapreduce` and `accumulate` now receives its
  arguments in `(accumulator, element)` order, matching `Base`. Tile IR delivers
  them the other way around, and the previous binding forwarded them unswapped,
  which silently computed the wrong result for non-commutative combiners
  ([#203](https://github.com/JuliaGPU/cuTile.jl/pull/203)).
- Matrix multiplication of tiles now runs the accumulator in the element type
  Tile IR allows for the input type and converts the result back, so
  `BFloat16 * BFloat16` accumulates in `Float32` and still returns a `BFloat16`
  tile ([#218](https://github.com/JuliaGPU/cuTile.jl/pull/218)).
- Operations that used to emit invalid bytecode now raise an error at compile
  time: `mul_hi` on signed integers (Tile IR's is unsigned-only, so
  `reinterpret` the operands), `memory_order=MemoryOrder.Weak` on atomics, and
  `reduce`/`accumulate` over `TFloat32` tiles
  ([#203](https://github.com/JuliaGPU/cuTile.jl/pull/203),
  [#218](https://github.com/JuliaGPU/cuTile.jl/pull/218)).
- `Constant{T,V}` throws an `InexactError` when `V` does not fit in `T`, rather
  than truncating the value during codegen, and `ArraySpec` rejects
  `contiguous=true` combined with a first stride divisor other than 0 or 1
  ([#212](https://github.com/JuliaGPU/cuTile.jl/pull/212),
  [#218](https://github.com/JuliaGPU/cuTile.jl/pull/218)).

*New features*:

- Kernels can be launched with `@cuda backend=cuTile blocks=grid f(args...)`.
  `cuTile.cufunction(f, tt)` compiles a kernel and returns a `TileKernel` that
  can be called repeatedly without going through method-instance lookup again.
  Launch arguments are converted with Adapt.jl, so a user-defined struct with an
  `Adapt.adapt_structure` method has its arrays converted recursively.
  `cuTile.launch(f, grid, args...)` remains available
  ([#214](https://github.com/JuliaGPU/cuTile.jl/pull/214)).
- Compiled cubins are now cached on disk, in an LMDB database in cuTile's
  scratchspace, keyed on the bytecode, the target architecture, the optimization
  level and the `tileiras` version. A second session running the same kernel
  skips `tileiras` entirely; on the FFT example that cuts time-to-first-result
  from 34 s to 8.3 s. The cache is capped at 1 GiB and evicts the
  least-recently-used entries
  ([#216](https://github.com/JuliaGPU/cuTile.jl/pull/216)).
- Kernels can generate random numbers: `rand`, `randn` and `randexp` return
  scalars or tiles from a tile-vectorized Philox2x32-7 generator.
  `cuTile.DeviceRNG()` opens an independent stream that can be re-seeded with
  `Random.seed!`, and each launch is keyed on a fresh host seed. On the host,
  `cuTile.RNG` fills `CuArray`s through `Random.rand!`, `randn!` and `randexp!`
  ([#193](https://github.com/JuliaGPU/cuTile.jl/pull/193),
  [#204](https://github.com/JuliaGPU/cuTile.jl/pull/204)).
- `view` and `@view` derive a sub-range `TileArray` from an existing one. Every
  index has to be `:` or a `UnitRange`; other index types are rejected at
  compile time ([#191](https://github.com/JuliaGPU/cuTile.jl/pull/191)).
- `permutedims`, `transpose` and `reshape` now also apply to a `TileArray`,
  adjusting sizes and strides without touching memory. `reshape` is column-major
  and requires a contiguous source
  ([#194](https://github.com/JuliaGPU/cuTile.jl/pull/194)).
- Integer matrix multiplication is supported: `Int8`/`UInt8` operands with an
  `Int32` accumulator, with per-operand signedness derived from the Julia types
  ([#203](https://github.com/JuliaGPU/cuTile.jl/pull/203)).
- Additional operations work in kernels: `atan(y, x)` (requires Tile IR 13.2),
  `fld` and `div(x, y, RoundDown)` on floats, and `^` with an integer exponent.
  The `rounding_mode` of the enclosing `@fpmode` scope is now forwarded to
  `tanh` ([#193](https://github.com/JuliaGPU/cuTile.jl/pull/193),
  [#203](https://github.com/JuliaGPU/cuTile.jl/pull/203),
  [#218](https://github.com/JuliaGPU/cuTile.jl/pull/218)).

*Minor changes*:

- CUDACore is a regular dependency now instead of a weak one, so `launch` no
  longer errors out asking for CUDA.jl to be imported first. Requires CUDACore
  6.1 and IRStructurizer 0.6
  ([#214](https://github.com/JuliaGPU/cuTile.jl/pull/214)).
- First-kernel latency was reduced by precompiling the codegen pipeline with a
  representative workload, and by running that pipeline in the world age
  captured at `__init__` so packages loaded later cannot invalidate it. The
  vector-addition example went from about 26 s to 14 s
  ([#216](https://github.com/JuliaGPU/cuTile.jl/pull/216)).
- Launch overhead was reduced: argument flattening is generated at compile time
  instead of splatting at run time, the compilation cache is keyed on an
  `isbits` value, and the support check is memoized per device
  ([#214](https://github.com/JuliaGPU/cuTile.jl/pull/214)).
- The optimizer gained a common-subexpression pass, an interval analysis that
  attaches no-wrap flags to integer arithmetic, and a divisibility analysis that
  emits `assume` operations for pointer alignment and for shape and stride
  divisors. Rewrites now preserve the effect flags that gate CSE and
  loop-invariant code motion, and normalization runs to a fixed point before FMA
  fusion so that `a*b - c*d` fuses the same way every time, which closed a 9%
  gap on the FFT example
  ([#190](https://github.com/JuliaGPU/cuTile.jl/pull/190),
  [#205](https://github.com/JuliaGPU/cuTile.jl/pull/205),
  [#207](https://github.com/JuliaGPU/cuTile.jl/pull/207),
  [#211](https://github.com/JuliaGPU/cuTile.jl/pull/211),
  [#220](https://github.com/JuliaGPU/cuTile.jl/pull/220)).
- `gather` and `scatter` on contiguous arrays now vectorize. The unit stride of
  the contiguous axis is folded away instead of being multiplied in as a
  run-time value, and no alignment is asserted on the resulting tile of
  pointers; either one made `tileiras` fall back to 2-byte stores where the
  equivalent cuTile Python kernel got 16-byte ones. That cut the
  mixture-of-experts example's kernel time by about 25%
  ([#212](https://github.com/JuliaGPU/cuTile.jl/pull/212),
  [#213](https://github.com/JuliaGPU/cuTile.jl/pull/213),
  [#219](https://github.com/JuliaGPU/cuTile.jl/pull/219)).

*Bug fixes*:

- Fixed `print` and `println` of 64-bit and unsigned integers, which used format
  specifiers that made the device print the wrong values. Printing a tuple no
  longer raises an error, and on Tile IR 13.1 a print no longer breaks the
  memory ordering of the stores and atomics that follow it
  ([#203](https://github.com/JuliaGPU/cuTile.jl/pull/203),
  [#218](https://github.com/JuliaGPU/cuTile.jl/pull/218)).
- `!=` on `BFloat16` values now uses the unordered comparison, so `NaN != NaN`
  is `true`; it previously took a path that also tripped over a shape mismatch
  ([#218](https://github.com/JuliaGPU/cuTile.jl/pull/218)).
- Kernels that capture a value of non-inferrable type in a closure, and
  element-wise subtraction of `BFloat16` tiles, no longer fail to compile
  ([#201](https://github.com/JuliaGPU/cuTile.jl/pull/201),
  [#202](https://github.com/JuliaGPU/cuTile.jl/pull/202)).
- Referring to a global through a `Module` value in kernel code now resolves the
  binding at compile time instead of reporting an unsupported value type.
  Conversions of a value whose type could not be inferred are no longer assumed
  to produce a scalar, which used to select the scalar `store` method for what
  was actually a multi-element tile and crash `tileiras`
  ([#201](https://github.com/JuliaGPU/cuTile.jl/pull/201)).
- `iota`, `cat`, `reshape` and pointer offsetting validate their shapes and
  element types before emitting bytecode, so invalid uses produce a Julia-level
  error instead of a `tileiras` failure
  ([#201](https://github.com/JuliaGPU/cuTile.jl/pull/201),
  [#203](https://github.com/JuliaGPU/cuTile.jl/pull/203)).

### v0.3.1 (May 2026)

- Restricted the `CUDA_Compiler_jll` compatibility bound to `0.4 - 0.4.3`, so
  that v0.3 does not resolve against newer compiler releases it was not tested
  with.

### v0.3.2 (July 2026)

- Added support for Tile IR 13.3, which is now the minimum version of
  `CUDA_Tile_jll`, and with it Hopper (sm_90) GPUs. The device check reports the
  bytecode version cuTile emits rather than the CUDA toolkit version: Blackwell
  requires 13.1, Ampere and Ada 13.2, and Hopper 13.3
  ([#234](https://github.com/JuliaGPU/cuTile.jl/pull/234),
  [#236](https://github.com/JuliaGPU/cuTile.jl/pull/236)).
- The bytecode version to emit is no longer derived from `CUDA_Compiler_jll`'s
  CUDA version, but probed by having `tileiras` compile an empty module at each
  supported version. A local `tileiras` binary can be selected with the
  `tileiras` preference and the probe overridden with the `bytecode_version`
  preference; `cuTile.versioninfo()` reports both
  ([#222](https://github.com/JuliaGPU/cuTile.jl/pull/222)).
- `muladd` on tiles gained a `fast_acc` keyword argument, which enables
  lower-precision accumulation for FP8 inputs on Hopper, and
  `cuTile.muladd_scaled` was added for block-scaled multiply-accumulate. Both
  require Tile IR 13.3, `muladd_scaled` also Blackwell
  ([#239](https://github.com/JuliaGPU/cuTile.jl/pull/239)).
- Added a Microfloats.jl extension, making `Float8_E4M3FN`, `Float8_E5M2`,
  `Float8_E8M0FNU` and `Float4_E2M1FN` usable as kernel element types
  ([#223](https://github.com/JuliaGPU/cuTile.jl/pull/223),
  [#234](https://github.com/JuliaGPU/cuTile.jl/pull/234)).
- `repeat(tile, counts...)` and `repeat(tile; inner, outer)` implement
  `Base.repeat` for tiles. `reinterpret(T, tile)` and `reinterpret(reshape, T,
  tile)` view a whole tile at a different element width, lowering to `bitcast`,
  `pack` or `unpack`; this is how sub-byte formats travel through memory
  ([#244](https://github.com/JuliaGPU/cuTile.jl/pull/244),
  [#238](https://github.com/JuliaGPU/cuTile.jl/pull/238)).
- Added the `num_worker_warps` entry hint (4 or 8, Tile IR 13.3 and up), and
  `cuTile.assume_divisible_by(x, divisor)`, which declares that an index or
  offset is a multiple of `divisor` so the compiler can prove alignment and
  widen memory operations
  ([#245](https://github.com/JuliaGPU/cuTile.jl/pull/245),
  [#252](https://github.com/JuliaGPU/cuTile.jl/pull/252)).
- Code generation no longer stops at the first unsupported construct:
  independent errors are collected and reported together in source order as a
  `CodegenErrors` exception. Julia `throw`s are handled as well: an unavoidable
  one becomes a compile-time diagnostic, while a throw on a conditional path
  becomes a device-side assertion carrying the original message
  ([#248](https://github.com/JuliaGPU/cuTile.jl/pull/248)).
- Host-side `Tiled` broadcast now follows Base's shape semantics: size-1
  dimensions expand to the destination's size, a scalar right-hand side fills
  the destination, incompatible shapes throw a `DimensionMismatch`, and empty
  and 0-dimensional arrays no longer launch malformed grids. In-place `ct.@.`
  evaluates to the destination array instead of its `Tiled` wrapper, and host
  memory is rejected with an `ArgumentError` instead of faulting on the device
  ([#264](https://github.com/JuliaGPU/cuTile.jl/pull/264)).
- Broadcast arguments that Base wraps in a `Ref` now work, so expressions like
  `x .^ 2` compile ([#255](https://github.com/JuliaGPU/cuTile.jl/pull/255)).
- Views of device arrays can be passed to kernels: a `SubArray` is converted to
  a single `TileArray` instead of being recursed into by Adapt. Kernel array
  arguments also gained a supertype, `AbstractTileArray{T,N}`, and the host-side
  entry points convert with `cuTileconvert`, so other device array types can
  hook in ([#228](https://github.com/JuliaGPU/cuTile.jl/pull/228),
  [#258](https://github.com/JuliaGPU/cuTile.jl/pull/258)).
- Compiling large kernels is much faster: the IR rewriting passes cache their
  use lookups instead of walking the entire function for every query. Compiling
  `randn` for `Float64` went from 13 s to 1.3 s
  ([#235](https://github.com/JuliaGPU/cuTile.jl/pull/235)).
- The disk cache was rebuilt on LMDB.jl instead of hand-written `liblmdb`
  bindings. Its location and size can be set with `JULIA_CUTILE_CACHE_DIR` and
  `JULIA_CUTILE_CACHE_SIZE`, entries are keyed by a SHA-256 of the compilation
  inputs so entries written by earlier versions no longer match, and the cache
  is skipped entirely when the `tileiras` version cannot be read
  ([#217](https://github.com/JuliaGPU/cuTile.jl/pull/217),
  [#266](https://github.com/JuliaGPU/cuTile.jl/pull/266)).
- Several previously accepted but invalid uses are now rejected: `ct.extract`
  validates that the slice shape divides the tile shape and that the slice index
  is in bounds, `num_ctas > 1` throws on non-Blackwell targets, and reading a
  non-`const` global from a kernel errors with a message naming the binding
  ([#252](https://github.com/JuliaGPU/cuTile.jl/pull/252),
  [#234](https://github.com/JuliaGPU/cuTile.jl/pull/234),
  [#226](https://github.com/JuliaGPU/cuTile.jl/pull/226)).
- Fixed kernels whose entire body sits inside control flow being rejected by the
  Tile IR verifier, and kernels involving `Vararg` tuple types tripping up
  scalar elimination
  ([#246](https://github.com/JuliaGPU/cuTile.jl/pull/246),
  [#237](https://github.com/JuliaGPU/cuTile.jl/pull/237)).
- Device code now shows up in coverage reports when Julia is run with
  `--code-coverage`. Lines are marked while the kernel is compiled, not while it
  executes ([#249](https://github.com/JuliaGPU/cuTile.jl/pull/249),
  [#259](https://github.com/JuliaGPU/cuTile.jl/pull/259)).


## v0.2 (April 2026)

Kernels now go through an optimization pipeline before Tile IR is emitted: the
structured IR is canonicalized, constant-folded and rewritten with algebraic and
strength-reduction rules, and then handed to an alias-aware token ordering pass,
loop-invariant code motion and dead code elimination. Tile shapes are also kept
in Tile IR's native row-major order, so `reshape` and batched matrix
multiplication no longer wrap their operands in permutations. Measured on an RTX
5080, this closes the gap with cuTile Python and overtakes it on several
kernels: layer normalization, which trailed Python by 63% in the previous
release, now runs its forward pass at 931 GB/s against Python's 716 GB/s
([#89](https://github.com/JuliaGPU/cuTile.jl/pull/89),
[#142](https://github.com/JuliaGPU/cuTile.jl/pull/142),
[#147](https://github.com/JuliaGPU/cuTile.jl/pull/147),
[#158](https://github.com/JuliaGPU/cuTile.jl/pull/158),
[#165](https://github.com/JuliaGPU/cuTile.jl/pull/165),
[#177](https://github.com/JuliaGPU/cuTile.jl/pull/177)).

*Breaking changes*:

- `ct.where(cond, x, y)` was removed. Use `ifelse.(cond, x, y)`, which
  broadcasts and accepts scalar arguments the same way
  ([#180](https://github.com/JuliaGPU/cuTile.jl/pull/180)).
- The element type of `arange` moved from a positional argument to a `dtype`
  keyword argument that defaults to `Int32`: write `ct.arange(16)` or
  `ct.arange(16; dtype=Int64)` instead of `ct.arange(16, Int32)`
  ([#139](https://github.com/JuliaGPU/cuTile.jl/pull/139)).
- Batched matrix multiplication takes its batch dimensions last,
  `(M, K, batch...) * (K, N, batch...)`, instead of first, following Julia's
  column-major convention. The batch dimensions of the two operands broadcast
  against each other, and matrix-vector and vector-matrix products are now
  supported ([#132](https://github.com/JuliaGPU/cuTile.jl/pull/132)).
- `PaddingMode`, `MemoryOrder` and `MemScope` are EnumX enumerations instead of
  modules of `Int` constants. Spellings such as `ct.PaddingMode.Zero` are
  unchanged, but passing a bare integer no longer works
  ([#140](https://github.com/JuliaGPU/cuTile.jl/pull/140)).

*New features*:

- Julia's `for` loops work in kernels: `for i in start:stop` and
  `for i in start:step:stop` are structurized into Tile IR `ForOp`s, so the
  hand-written counting `while` loop that the README used to prescribe is no
  longer needed ([#174](https://github.com/JuliaGPU/cuTile.jl/pull/174)).
- `print` and `println` work inside kernels. Format strings are assembled at
  compile time from string constants and tile arguments, and string
  interpolation (`println("bid=$bid")`) is supported
  ([#173](https://github.com/JuliaGPU/cuTile.jl/pull/173)).
- Added `ct.@fpmode rounding_mode=... flush_to_zero=... begin ... end`, which
  sets the floating-point rounding mode and flush-to-zero behavior for every
  operation in the block, including operations in inlined callees. The rounding
  modes are `ct.Rounding.NearestEven`, `Zero`, `NegInf`, `PosInf` and `Approx`
  ([#172](https://github.com/JuliaGPU/cuTile.jl/pull/172)).
- Added the atomic read-modify-write operations `ct.atomic_max`,
  `ct.atomic_min`, `ct.atomic_or`, `ct.atomic_and` and `ct.atomic_xor`
  ([#136](https://github.com/JuliaGPU/cuTile.jl/pull/136)).
- `gather` and `scatter` accept a `mask` that is combined with the automatic
  bounds check, a `check_bounds=false` to skip that check when the indices are
  known to be in range, and (for `gather`) a `padding_value` for masked-out
  elements. The atomics take `check_bounds` as well
  ([#139](https://github.com/JuliaGPU/cuTile.jl/pull/139),
  [#166](https://github.com/JuliaGPU/cuTile.jl/pull/166)).
- `load` and `store` accept their operands as keyword arguments as well,
  `ct.load(a; index, shape)` and `ct.store(c; index, tile)`, matching how the
  same calls are spelled in cuTile Python
  ([#139](https://github.com/JuliaGPU/cuTile.jl/pull/139)).
- Reductions of a whole `CuArray` can be run through cuTile by wrapping it in
  `ct.Tiled`: `reduce`, `mapreduce`, `sum`, `prod`, `maximum`, `minimum`, `any`
  and `all`, with or without `dims`, plus the in-place `sum!`, `prod!`,
  `maximum!` and `minimum!`. When the reduction produces one element per block
  and a hardware atomic exists for the operator, the result is accumulated
  atomically instead of in a second pass
  ([#134](https://github.com/JuliaGPU/cuTile.jl/pull/134),
  [#141](https://github.com/JuliaGPU/cuTile.jl/pull/141)).
- Kernels accept more kinds of compile-time arguments: types, either as
  `ct.Constant(Int)` or passed directly to `launch`; the zero-size values Julia
  has no runtime representation for, such as function singletons and `Val`s; and
  `ct.Constant(nothing)`
  ([#133](https://github.com/JuliaGPU/cuTile.jl/pull/133),
  [#138](https://github.com/JuliaGPU/cuTile.jl/pull/138),
  [#181](https://github.com/JuliaGPU/cuTile.jl/pull/181)).
- The emitted bytecode carries source locations, and `tileiras` is invoked with
  `--lineinfo`, so `CUDA.@device_code_sass` and profilers can attribute SASS
  back to Julia source lines. `code_tiled` and `@device_code_tiled` print the
  locations with `debuginfo=true`
  ([#175](https://github.com/JuliaGPU/cuTile.jl/pull/175)).
- `isnan` is now supported in kernels, and `reinterpret` maps onto a Tile IR
  bitcast ([#166](https://github.com/JuliaGPU/cuTile.jl/pull/166),
  [#179](https://github.com/JuliaGPU/cuTile.jl/pull/179)).

*Minor changes*:

- The reflection entry points `code_typed`, `code_structured` and `code_tiled`
  accept `Constant{T,V}` argument types and run the same constant-seeded
  inference that `launch` does, so the IR they show matches what is compiled.
  `code_structured`'s `validate` keyword argument was replaced by `optimize`
  ([#157](https://github.com/JuliaGPU/cuTile.jl/pull/157)).
- Julia 1.13 is supported. Calls without a Tile IR equivalent report a single
  "Unsupported function call during Tile IR compilation" error naming the
  function and its argument types, whether the compiler or Julia's inference
  found no method ([#182](https://github.com/JuliaGPU/cuTile.jl/pull/182)).
- Added Mixture-of-Experts and fused multi-head attention examples, with Python
  counterparts for comparison
  ([#163](https://github.com/JuliaGPU/cuTile.jl/pull/163),
  [#170](https://github.com/JuliaGPU/cuTile.jl/pull/170)).

*Bug fixes*:

- `!=` on floating-point values uses an unordered comparison, so comparisons
  involving `NaN` return `true` as IEEE 754 requires; they returned `false`
  before ([#179](https://github.com/JuliaGPU/cuTile.jl/pull/179)).
- The alias analysis matched `getfield` against the wrong `GlobalRef` and did
  not look through `offset`, so every pointer derived from a kernel argument
  ended up in one alias set. Gathers and scatters that touch different arrays
  were serialized by token dependencies that dead code elimination could then
  not remove, which also tripped a `tileiras` crash at `-O1` and above
  ([#164](https://github.com/JuliaGPU/cuTile.jl/pull/164)).
- The divisibility computed for array shapes could exceed the maximum divisor,
  emitting a `DivBy(32)` assumption where `DivBy(16)` was intended and
  specializing kernels more often than necessary
  ([#153](https://github.com/JuliaGPU/cuTile.jl/pull/153)).
- `@device_code_tiled` no longer fails on kernels containing a `reduce`, where
  it tried to compile the combiner region as a standalone entry point
  ([#135](https://github.com/JuliaGPU/cuTile.jl/pull/135)).

### v0.2.1 (April 2026)

- Fixed the encoding of reduction identity values in the bytecode. Floating
  point identities were written as raw bit patterns instead of `ap_int`, so
  `maximum` recorded `NaN` instead of `-Inf` and `minimum` recorded `1.5`
  instead of `+Inf`; integer identities were zigzag-encoded where the format
  stores raw values, and overflowed for `typemin(Int32)`
  ([#186](https://github.com/JuliaGPU/cuTile.jl/pull/186)).
- Kernel names containing non-ASCII characters no longer fail to compile:
  characters outside `[a-zA-Z0-9_]` are escaped by codepoint instead of being
  forced into a single byte
  ([#183](https://github.com/JuliaGPU/cuTile.jl/pull/183)).
- Added a softmax example, with a Python counterpart
  ([#185](https://github.com/JuliaGPU/cuTile.jl/pull/185)).

### v0.2.2 (April 2026)

- Adapted to CUDA.jl v6: the extension is triggered by CUDACore instead of CUDA,
  and CUDA.jl 6 is now required
  ([#178](https://github.com/JuliaGPU/cuTile.jl/pull/178)).
- Fixed the `@device_code_*` macros on kernels that take a type argument, which
  reconstructed it as `Constant{DataType,T}` and then rejected the resulting
  signature as non-concrete
  ([#184](https://github.com/JuliaGPU/cuTile.jl/pull/184)).


## v0.1 (February 2026)

The initial release, published as a call for testing. cuTile.jl compiles Julia
functions to NVIDIA Tile IR bytecode, which the Tile IR assembler then turns
into a CUBIN: kernels are written in terms of whole tiles instead of individual
threads, and the Tile IR compiler decides how to map those onto the hardware.
The model follows NVIDIA's cuTile Python, but the surface is Julia's: `Base`
operations are routed to Tile IR through an overlay method table, so `+`, `sum`,
`reshape` and broadcasting mean on a tile what they mean on an array. Not all of
Tile IR is implemented, and the supported subset of Julia is only what the
bundled examples exercise; notably, Julia's iterator-based `for` loops are not
recognized yet, so counted loops have to be written as `while` loops.

*New features*:

- A kernel is a plain Julia function that returns `nothing`; there is no macro
  or decorator. Array arguments arrive as a `TileArray{T,N,S}` carrying the base
  pointer, sizes and strides, and `ct.load(arr, index, shape)` and
  `ct.store(arr, index, tile)` move a `Tile{T,Shape}` between the array and
  registers. `ct.bid`, `ct.num_blocks` and `ct.num_tiles` describe the grid.
  Indices are 1-based throughout, following Julia rather than Python.
- `launch(f, grid, args...)` compiles a kernel and launches it on the task-bound
  CUDA stream. `CuArray` arguments are converted to `TileArray`s and flattened
  into kernel parameters automatically, and `name`, `sm_arch`, `opt_level`,
  `num_ctas` and `occupancy` are available as keyword arguments. Compilation
  results are cached, so relaunching a kernel does not recompile it. CUDA.jl has
  to be imported for `launch` to be available.
- Tile operations are spelled the Julia way: `+` and `-` for element-wise
  arithmetic, `*` for matrix multiplication and for scaling by a scalar,
  `muladd`, broadcasting (`.+`, `exp.(x)`, `ifelse.(...)`, `Float16.(x)`) and
  `map`, `reduce`, `mapreduce` and `accumulate` along with the derived `sum`,
  `prod`, `maximum`, `minimum`, `any`, `all`, `count`, `argmax`, `argmin`,
  `cumsum` and `cumprod`, and `reshape`, `permutedims`, `transpose` and
  `dropdims`. Operations with no Julia counterpart keep a `ct.` prefix:
  `ct.arange`, `ct.full`, `ct.zeros`, `ct.broadcast_to`, `ct.extract`, `ct.cat`
  and `ct.where`.
- `ct.gather` and `ct.scatter` index 1D and 2D arrays with tiles of indices, and
  `ct.atomic_cas`, `ct.atomic_xchg` and `ct.atomic_add` perform atomic
  operations with a configurable memory order and scope. `ct.@assert` compiles
  to a Tile IR assertion, which aborts the kernel when it fails.
- Element types follow the Tile IR type system: 8- to 64-bit signed and unsigned
  integers, `Float16`, `BFloat16`, `Float32`, `Float64`, and `TFloat32` to feed
  `Float32` data to the tensor cores at reduced precision. Loading the
  DLFP8Types package additionally makes `Float8_E4M3FN` and `Float8_E5M2` usable
  as element types.
- `ct.Constant(value)` marks a launch argument whose value should be baked into
  the code: it generates no kernel parameter, and distinct values compile to
  distinct kernels. Array arguments are further specialized on an `ArraySpec`
  that records pointer alignment, contiguity, and stride and shape divisibility,
  which the Tile IR compiler uses to select vectorized and TMA-based accesses.
- `code_tiled(f, argtypes)` prints the Tile IR for a signature as textual MLIR,
  and works without a GPU or CUDA.jl. `@device_code_tiled` and
  `@device_code_structured` print the Tile IR and the structured Julia IR for
  every kernel compiled while evaluating an expression.
- Requires Julia 1.11, a CUDA 13.1 toolkit and a Blackwell GPU (compute
  capability 10.0 or higher). `examples/` holds vector addition, transpose,
  matrix multiplication, batched matrix multiplication, layer normalization and
  a three-stage Cooley-Tukey FFT, each next to the cuTile Python kernel it was
  modeled on, plus a script that benchmarks the two against each other.

### v0.1.1 (March 2026)

- cuTile.jl and its dependencies are registered, so the package installs with
  `Pkg.add("cuTile")` instead of a manual clone.
- Broadcasting a lower-rank tile against a higher-rank one now pads the shape
  with trailing ones, matching Julia's left-aligned broadcast rules: `(64,)`
  broadcasts against `(64, 128)` along the first dimension. Leading ones were
  inserted before, which is NumPy's right-aligned convention
  ([#91](https://github.com/JuliaGPU/cuTile.jl/pull/91)).
- `ct.atomic_cas`, `ct.atomic_xchg` and `ct.atomic_add` accept tiles of indices
  in addition to scalar ones, operating on many elements at once and returning a
  tile of the previous values. Out-of-bounds indices are masked off
  ([#96](https://github.com/JuliaGPU/cuTile.jl/pull/96)).
- Kernels can return early from an `if`. Such returns are hoisted out of the
  conditional before code generation, which the Tile IR assembler requires
  ([#92](https://github.com/JuliaGPU/cuTile.jl/pull/92)).
- `ct.full` accepts a runtime value or a 0-D tile, not just a compile-time
  constant ([#100](https://github.com/JuliaGPU/cuTile.jl/pull/100)).
- Indexing a `TileArray` with scalars (`arr[i]`, useful to read a loop bound
  from an array) compiles again; the call had gone stale against the intrinsic
  it uses ([#94](https://github.com/JuliaGPU/cuTile.jl/pull/94)).
- `BFloat16` is supported as a scalar type in kernels, not just as a tile
  element type ([#90](https://github.com/JuliaGPU/cuTile.jl/pull/90)).
- `launch` drops every zero-size argument instead of only `Constant`, so any
  ghost type can be passed to a kernel
  ([#93](https://github.com/JuliaGPU/cuTile.jl/pull/93)).

### v0.1.2 (March 2026)

- Added support for CUDA 13.2, which extends Tile IR to Ampere and Ada GPUs:
  compute capability 8.0 and higher is now usable, while Blackwell still only
  needs CUDA 13.1. Hopper remains unsupported. `launch` checks the toolkit and
  the device up front and reports what is missing instead of failing further
  down ([#110](https://github.com/JuliaGPU/cuTile.jl/pull/110),
  [#111](https://github.com/JuliaGPU/cuTile.jl/pull/111),
  [#113](https://github.com/JuliaGPU/cuTile.jl/pull/113)).
- Added host-level broadcast, which uses cuTile without writing a kernel:
  wrapping `CuArray`s in `ct.Tiled` routes a broadcast expression through
  Julia's broadcast machinery into a single generated and launched cuTile
  kernel. `ct.@.` is a variant of `Base.@.` that wraps every value-position
  leaf, so `ct.@. C = A + sin(B)` and the allocating `D = ct.@. A + B` work
  directly on `CuArray`s
  ([#129](https://github.com/JuliaGPU/cuTile.jl/pull/129)).
- Optimization hints can be attached to a kernel from within its body with
  `ct.@compiler_options num_ctas=... occupancy=... opt_level=...`, and
  `ct.ByTarget(v"10.0" => 2, v"12.0" => 4; default=1)` selects a value per
  compute capability. The corresponding keyword arguments of `launch` and
  `code_tiled` still work and take precedence; `sm_arch` now takes a
  `VersionNumber` such as `v"10.0"` instead of a string such as `"sm_100"`
  ([#122](https://github.com/JuliaGPU/cuTile.jl/pull/122),
  [#111](https://github.com/JuliaGPU/cuTile.jl/pull/111)).
- `ct.zeros` and `ct.full` were removed in favor of overlays on `Base`, so tiles
  are constructed with `zeros(T, dims)`, `ones(T, dims)` and `fill(value, dims)`
  the way arrays are. `ct.arange` also accepts a plain length, and `reshape`
  accepts the new dimensions as separate arguments
  ([#123](https://github.com/JuliaGPU/cuTile.jl/pull/123)).
- `transpose` on a tile is now defined for 1D and 2D tiles only, reshaping a 1D
  tile to `1 × N` and swapping the dimensions of a 2D one; higher ranks throw
  and should use `permutedims`. The `ct.transpose` intrinsic was removed
  ([#124](https://github.com/JuliaGPU/cuTile.jl/pull/124)).
- `sum`, `prod`, `maximum`, `minimum`, `any`, `all` and `count` can be called
  without `dims`, reducing over every dimension and returning a scalar. `store`
  accepts a scalar in place of a tile
  ([#118](https://github.com/JuliaGPU/cuTile.jl/pull/118)).
- Kernel arguments can be arbitrary isbits structs, which are recursively
  flattened into kernel parameters, so a struct holding `TileArray`s and scalars
  can be passed to a kernel and have its fields accessed there. Destructuring
  used to be specific to `TileArray`
  ([#128](https://github.com/JuliaGPU/cuTile.jl/pull/128)).
- The atomic operations convert their value argument to the array's element
  type, instead of failing with a code generation error when the types differ
  ([#111](https://github.com/JuliaGPU/cuTile.jl/pull/111)).
- Shift-left was encoded without its overflow field, which made the assembler
  reject kernels using `<<`. Bitwise `~` on integers and `!` on booleans are
  supported as well ([#106](https://github.com/JuliaGPU/cuTile.jl/pull/106),
  [#107](https://github.com/JuliaGPU/cuTile.jl/pull/107)).
- Code generation no longer fails with a `MethodError` when the optimizer leaves
  a bare SSA reference as a statement, and `@device_code_tiled` works on kernels
  taking `Constant` arguments, which used to report an unsupported Julia type
  ([#125](https://github.com/JuliaGPU/cuTile.jl/pull/125),
  [#127](https://github.com/JuliaGPU/cuTile.jl/pull/127)).
- A failing Tile IR compilation now reports the assembler's output and keeps the
  bytecode around to attach to a bug report, and the assembler is pointed at the
  `ptxas` from its own artifact, so a `CUDA_ROOT` in the environment no longer
  makes it pick up a mismatched one
  ([#111](https://github.com/JuliaGPU/cuTile.jl/pull/111)).
