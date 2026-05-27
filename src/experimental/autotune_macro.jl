# `@autotune` — surface syntax for `autotune_launch`.
#
# Desugars
#
#     @autotune(
#         key = (T, Dk_pow2),
#         space = (TILE_M=(32, 64), TILE_N=(32, 64), occupancy=(1, 2)),
#         blocks = (cld(SeqLen_Q, $TILE_M), Heads * Batch),
#         mha_fwd(Q, K, V, ..., Constant($TILE_M), Constant($TILE_N)),
#     )
#
# into a call to `autotune_launch`, with `$X` interpolated as `cfg.X` in
# `blocks` and the kernel-call args. `$X` is the parser's `Expr(:$, :X)`
# node, which is normally rejected by lowering; the macro intercepts and
# rewrites it before that happens.

# Walk an expression replacing `Expr(:$, :X)` with `cfg.X`.
function _autotune_interp(ex, cfg::Symbol)
    if Meta.isexpr(ex, :$)
        length(ex.args) == 1 ||
            error("@autotune: \$ takes exactly one argument")
        sym = ex.args[1]
        sym isa Symbol ||
            error("@autotune: \$ argument must be a symbol, got `$sym`")
        return :($cfg.$sym)
    elseif ex isa Expr
        return Expr(ex.head, [_autotune_interp(a, cfg) for a in ex.args]...)
    else
        return ex
    end
end

# Try to extract axis names from a literal NT space expression.
# Returns Vector{Symbol} on success, or `nothing` if the expression
# isn't a recognizable NT literal (e.g. it's a variable reference, a
# function call, or a constructor like `FixedSpace([...])`).
function _autotune_space_axes(space_expr)
    Meta.isexpr(space_expr, :tuple) || return nothing
    axes = Symbol[]
    for a in space_expr.args
        if Meta.isexpr(a, :(=)) && a.args[1] isa Symbol
            push!(axes, a.args[1])
        elseif Meta.isexpr(a, :parameters)
            for kv in a.args
                if Meta.isexpr(kv, :kw) && kv.args[1] isa Symbol
                    push!(axes, kv.args[1])
                else
                    return nothing
                end
            end
        else
            return nothing
        end
    end
    return axes
end

const _AUTOTUNE_KWARGS = (:key, :space, :blocks, :tuning, :verify, :setup,
                          :sm_arch, :opt_level, :launch_args,
                          :num_ctas, :occupancy)

"""
    @autotune key=... space=... blocks=... [kwargs...] kernel(args...)

Tune `kernel` over `space` and launch the best config. Inside `blocks=`
and the kernel-call args, `\$X` interpolates `cfg.X` (where `cfg` is the
tuning configuration being evaluated).

# Required kwargs
- `space` — a `NamedTuple` like `(A=(...), B=(...))` (becomes a
  `CartesianSpace`), a `Vector` of `NamedTuple`s (becomes a `FixedSpace`),
  or any `AbstractSearchSpace` (passed through — useful for
  `CartesianSpace(constraint; ...)`).
- `blocks` — grid dimensions, an `Int` or `Tuple`. May reference `\$X`.

# Optional kwargs
- `key`         — cache key (any value)
- `tuning`      — `NamedTuple` of tuning knobs (`preset`, `force`, etc.)
- `verify`      — `() -> (() -> Bool)` factory; the returned checker is
                  called after each warmup pass to reject incorrect cfgs
- `setup`       — `() -> (() -> Nothing)` factory; reset between reps
- `launch_args` — final-launch args (or `cfg -> args` if it should differ
                  from the kernel-call args). Use this when the timed args
                  are throwaway copies (in-place kernels) and the final
                  launch should hit the real buffers
- `sm_arch`, `opt_level` — forwarded to `cufunction`
- `num_ctas`, `occupancy` — **static** hints applied uniformly to every
                  cfg. May not coexist with same-named axes in `space`
                  (the macro flags the conflict at expansion time when
                  `space` is a literal NT; otherwise `autotune_launch`
                  catches it at run time)

# Example

```julia
@autotune(
    key = (eltype(A), size(A, 2)),
    space = (TILE_M=(64, 128), TILE_N=(64, 128), occupancy=(1, 2, 4)),
    blocks = (cld(M, \$TILE_M), cld(N, \$TILE_N)),
    matmul(A, B, C, Constant(\$TILE_M), Constant(\$TILE_N))
)
```
"""
macro autotune(args...)
    kwargs = Dict{Symbol, Any}()
    call = nothing
    for arg in args
        if Meta.isexpr(arg, :(=)) || Meta.isexpr(arg, :kw)
            k, v = arg.args
            k isa Symbol || error("@autotune: kwarg key must be a symbol, got `$k`")
            k in _AUTOTUNE_KWARGS ||
                error("@autotune: unknown kwarg `$k`. Valid: $(join(_AUTOTUNE_KWARGS, ", "))")
            haskey(kwargs, k) && error("@autotune: duplicate kwarg `$k`")
            kwargs[k] = v
        elseif Meta.isexpr(arg, :call)
            call === nothing || error("@autotune: only one kernel call allowed")
            call = arg
        else
            error("@autotune: unexpected argument `$arg` — expected `kwarg=val` or a kernel call")
        end
    end

    call === nothing && error("@autotune: missing kernel call (e.g. `kernel(args...)`)")
    haskey(kwargs, :space)  || error("@autotune: missing required kwarg `space=`")
    haskey(kwargs, :blocks) || error("@autotune: missing required kwarg `blocks=`")

    space_expr  = kwargs[:space]
    blocks_expr = kwargs[:blocks]

    # Macro-time conflict check: if `space` is a literal NT and contains
    # `num_ctas`/`occupancy` as an axis, the same name can't also appear
    # as a static kwarg. (Opaque spaces are caught at run time.)
    space_axes = _autotune_space_axes(space_expr)
    if space_axes !== nothing
        for hint in (:num_ctas, :occupancy)
            if hint in space_axes && haskey(kwargs, hint)
                error("@autotune: `$hint` appears both as an axis in `space=` and " *
                      "as a static kwarg. Pick one.")
            end
        end
    end

    # Extract the kernel call (positional only — no kernel kwargs).
    Meta.isexpr(call, :call) ||
        error("@autotune: kernel must be a function-call expression")
    f_expr = call.args[1]
    arg_exprs = call.args[2:end]
    for a in arg_exprs
        if Meta.isexpr(a, :parameters) || Meta.isexpr(a, :kw)
            error("@autotune: kernel-side kwargs are not supported; pass values " *
                  "as positional args (typically wrapped in `Constant(...)`)")
        end
    end

    # Substitute `$X` -> `cfg.X` inside blocks/args.
    cfg = gensym(:cfg)
    grid_body = _autotune_interp(blocks_expr, cfg)
    arg_subs  = [_autotune_interp(a, cfg) for a in arg_exprs]

    grid_fn = :($cfg -> $grid_body)
    args_fn = :($cfg -> ($(arg_subs...),))

    # Forward all macro kwargs (except space/blocks, which are positional
    # / lifted into the closures) to `autotune_launch`.
    forwarded_keys = (:key, :tuning, :verify, :setup, :sm_arch, :opt_level,
                      :launch_args, :num_ctas, :occupancy)
    kw_exprs = [Expr(:kw, k, kwargs[k]) for k in forwarded_keys if haskey(kwargs, k)]

    return esc(quote
        $autotune_launch($f_expr, $space_expr, $grid_fn, $args_fn; $(kw_exprs...))
    end)
end
