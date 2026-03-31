# Julia built-in intrinsics
#
# Handles Julia built-ins that survive into the StructuredIRCode and are NOT
# lowered by rewrite_passes! (because they have no direct cuTile equivalent
# or are compile-time-only constructs).
#
# Julia Core.Intrinsics (add_int, sub_int, slt_int, etc.) and Core.ifelse /
# === are lowered to cuTile Intrinsics by rewrite_passes! and should not
# appear here.

# built-in: tuple (ghost — no runtime representation)
emit_intrinsic!(ctx::CGCtx, ::typeof(Core.tuple), args) = nothing

# built-in: isa (compile-time type narrowing)
emit_intrinsic!(ctx::CGCtx, ::typeof(isa), args) = nothing

# built-in: donotdelete (keep-alive barrier — no Tile IR emission)
emit_intrinsic!(ctx::CGCtx, ::typeof(donotdelete), args) = nothing
