# Random number generation intrinsics
#
# Low-level building blocks for the user-facing `rand` API. They exist as
# placeholders in the IR and are rewritten to concrete SSA operations by
# `rng_state_pass!` before codegen — so they have no `emit_intrinsic!`
# methods.
#
# Handles are `cuTile.RNGHandle` values: a ghost type returned by the two
# stream-creating intrinsics. The pass distinguishes streams by the *defining
# intrinsic* of the operand's SSA:
#
#   rng_stream()   → fresh stream; each call site is its own key
#   rng_default() → shared default stream; all calls map to `:default`
#
# The identity-per-call for `rng_stream` is preserved by the combination of
# `@intrinsic`'s noinline + compilerbarrier + our `effect_free=ALWAYS_FALSE`
# convention — the optimizer can't CSE distinct alloc sites.
#
# State-accessor intrinsics take a stream as their first operand:
#
#   rng_counter(h)      -> Tile{UInt32, ()}  — read h's counter
#   rng_advance(h, n)   -> Nothing           — add n to h's counter
#   rng_seed(h)         -> Tile{UInt32, ()}  — read h's seed
#   rng_set_seed(h, s)  -> Nothing           — overwrite h's seed

# ---- allocation ------------------------------------------------------------

# Handles are `Int` IDs, assigned at compile time by `rng_assign_ids_pass!`:
#   rng_default() → 0   (shared default stream)
#   rng_stream()   → 1, 2, 3, ...   (one unique ID per call site in each kernel)
# The pass rewrites every alloc/default call to its literal ID via
# `replace_uses!` before `rng_state_pass!` runs, so state-accessor
# intrinsics see a plain integer operand. `effect_free=ALWAYS_FALSE`
# keeps each alloc site a distinct IR node until the assignment pass
# replaces it.

@intrinsic rng_stream()
tfunc(𝕃, ::typeof(Intrinsics.rng_stream)) = Int
efunc(::typeof(Intrinsics.rng_stream), effects::CC.Effects) =
    CC.Effects(effects; effect_free=CC.ALWAYS_FALSE)

@intrinsic rng_default()
tfunc(𝕃, ::typeof(Intrinsics.rng_default)) = Int
efunc(::typeof(Intrinsics.rng_default), effects::CC.Effects) =
    CC.Effects(effects; effect_free=CC.ALWAYS_FALSE)

# ---- counter ---------------------------------------------------------------

@intrinsic rng_counter(stream)
tfunc(𝕃, ::typeof(Intrinsics.rng_counter), @nospecialize(h)) = Tile{UInt32, Tuple{}}
efunc(::typeof(Intrinsics.rng_counter), effects::CC.Effects) =
    CC.Effects(effects; effect_free=CC.ALWAYS_FALSE)

@intrinsic rng_advance(stream, n)
tfunc(𝕃, ::typeof(Intrinsics.rng_advance), @nospecialize(h), @nospecialize(n)) = Nothing
efunc(::typeof(Intrinsics.rng_advance), effects::CC.Effects) =
    CC.Effects(effects; effect_free=CC.ALWAYS_FALSE)

# ---- seed ------------------------------------------------------------------

@intrinsic rng_seed(stream)
tfunc(𝕃, ::typeof(Intrinsics.rng_seed), @nospecialize(h)) = Tile{UInt32, Tuple{}}
efunc(::typeof(Intrinsics.rng_seed), effects::CC.Effects) =
    CC.Effects(effects; effect_free=CC.ALWAYS_FALSE)

@intrinsic rng_set_seed(stream, s)
tfunc(𝕃, ::typeof(Intrinsics.rng_set_seed), @nospecialize(h), @nospecialize(s)) = Nothing
efunc(::typeof(Intrinsics.rng_set_seed), effects::CC.Effects) =
    CC.Effects(effects; effect_free=CC.ALWAYS_FALSE)

# `kernel_state()` lives in `intrinsics/kernel_state.jl` — `lower_rng_state!`
# uses it to seed the default stream from the host-supplied `KernelState.seed`.
