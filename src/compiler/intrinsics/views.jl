# views



"""
    Intrinsics.get_index_space_shape(pv::PartitionView, axis::Integer) -> Int32

Returns the size of `pv`'s index space along `axis` (i.e. how many tiles
fit along that dimension); lowers to `cuda_tile.get_index_space_shape`.

`axis` is 0-indexed in Julia order and must be a compile-time constant.
The Tile IR op returns the full shape; the codegen picks the requested
axis (in row-major order).
"""
@intrinsic get_index_space_shape(pv, axis)
tfunc(𝕃, ::typeof(Intrinsics.get_index_space_shape), @nospecialize(pv), @nospecialize(axis)) = Int32
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.get_index_space_shape), args)
    cb = ctx.cb
    tt = ctx.tt

    # args: (partition_view, axis)
    pv_arg = emit_value!(ctx, args[1])
    pv_arg === nothing && throw(IRError("get_index_space_shape() requires a PartitionView argument"))
    pv_arg.v === nothing && throw(IRError("get_index_space_shape() requires a materialized PartitionView"))

    # Get axis (0-indexed Julia) and flip to Tile IR order
    axis = @something get_constant(ctx, args[2]) throw(IRError("get_index_space_shape() axis must be a compile-time constant"))
    axis = Int(axis)

    # Get ndim from the PartitionView constant field
    pv_arg.constant === nothing && throw(IRError("get_index_space_shape(): PartitionView missing ndim info"))
    ndim = something(pv_arg.constant)

    # Flip axis for row-major Tile IR: Julia dim 0 → Tile IR dim ndim-1
    tileir_axis = ndim - 1 - axis

    # Create result types for all dimensions
    scalar_i32 = tile_type!(tt, I32(tt), RowMajorShape(()))
    result_types = fill(scalar_i32, ndim)

    # Emit GetIndexSpaceShapeOp
    shape_vals = encode_GetIndexSpaceShapeOp!(cb, result_types, pv_arg.v)

    # Return the value for the requested axis (in Tile IR order)
    # shape_vals is a single Value when ndim == 1, otherwise a Tuple
    result_val = ndim == 1 ? shape_vals : shape_vals[tileir_axis + 1]
    CGVal(result_val, scalar_i32, Tile{Int32, Tuple{}})
end

# TODO: cuda_tile.get_tensor_shape

"""
    Intrinsics.load_partition_view(pv::PartitionView{T,N,Shape},
                                   latency::Union{Int,Nothing},
                                   allow_tma::Bool,
                                   indices::NTuple{M,<:Integer}) -> Tile{T,Shape}

Token-ordered load of a tile from a `PartitionView`; lowers to
`cuda_tile.load_view_tko`.

`latency` and `allow_tma` are compile-time hints. `indices` is in Julia
order and is reversed/zero-padded to match `pv`'s index space rank
before emission. The token argument is appended by `token_order_pass!`
and is not part of the user-visible signature.
"""
@intrinsic load_partition_view(pv, latency, allow_tma, indices)
function tfunc(𝕃, ::typeof(Intrinsics.load_partition_view), @nospecialize(pv), @nospecialize args...)
    pv_type = CC.widenconst(pv)
    pv_type <: PartitionView || return nothing
    pv_type isa DataType || return nothing
    length(pv_type.parameters) >= 3 || return nothing
    T = eltype(pv_type)
    Shape = pv_type.parameters[3]
    Shape isa Type || return nothing
    return Tile{T, Shape}
end
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.load_partition_view), args)
    cb = ctx.cb
    tt = ctx.tt

    # Extract input token from last arg (added by token_order_pass!)
    input_token = extract_token_arg!(ctx, args)

    # args: (partition_view, latency, allow_tma, indices)
    pv_arg = emit_value!(ctx, args[1])
    pv_arg === nothing && throw(IRError("load_partition_view() requires a PartitionView argument"))
    pv_arg.v === nothing && throw(IRError("load_partition_view() requires a materialized PartitionView"))

    # Get ndim from PartitionView constant field
    pv_arg.constant === nothing && throw(IRError("load_partition_view(): PartitionView missing ndim info"))
    ndim = something(pv_arg.constant)

    # Extract tile shape from PartitionView type (PartitionView{T, N, Shape})
    # Reverse to Tile IR row-major order
    pv_type = CC.widenconst(pv_arg.jltype)
    elem_type = eltype(pv_type)
    tile_shape = RowMajorShape(ColMajorShape(size(pv_type)))

    dtype = lookup_dtype!(tt, elem_type)
    tile_type = tile_type!(tt, dtype, tile_shape)
    token_type = Token(tt)

    latency = @something get_constant(ctx, args[2]) throw(IRError("load_partition_view(): latency must be a compile-time constant"))
    allow_tma = @something get_constant(ctx, args[3]) throw(IRError("load_partition_view(): allow_tma must be a compile-time constant"))
    allow_tma_val = allow_tma isa Bool ? allow_tma : true

    # Extract indices
    index_tvs = resolve_tuple(ctx, args[4], "load_partition_view indices")
    index_vals = Value[tv.v for tv in index_tvs]
    index_jl_types = Type[tv.jltype for tv in index_tvs]

    unique_types = unique(index_jl_types)
    length(unique_types) <= 1 || throw(IRError("All index types must match, got: $unique_types"))
    isempty(unique_types) && ndim > 0 && throw(IRError("load_partition_view(): indices required for $(ndim)D view"))
    index_jl_type = isempty(unique_types) ? Int32 : unique_types[1]  # Int32 only for 0D case
    index_type = tile_type_for_julia!(ctx, index_jl_type)

    # Pad indices if needed, then reverse for Tile IR row-major order
    index_vals = pad_indices(ctx, index_vals, ndim, index_type, index_jl_type)
    reverse!(index_vals)

    # Create optimization hints if provided
    optimization_hints = create_optimization_hints(ctx, latency, allow_tma_val)

    tile_val, result_token = encode_LoadViewTkoOp!(
        cb, tile_type, token_type, pv_arg.v, index_vals;
        token = input_token, optimization_hints
    )

    # Store result token for TokenResultNode
    ctx.result_tokens[ctx.current_ssa_idx] = result_token

    julia_shape = ColMajorShape(tile_shape)
    return CGVal(tile_val, tile_type, Tile{elem_type, TupleType(julia_shape)}, tile_shape)
end

function pad_indices(ctx::CGCtx, index_vals::Vector{Value}, ndim::Int, idx_type::TypeId, idx_jl_type::Type)
    while length(index_vals) < ndim
        idx_bytes = reinterpret(UInt8, [eltype(idx_jl_type)(0)])
        push!(index_vals, encode_ConstantOp!(ctx.cb, idx_type, collect(idx_bytes)))
    end
    return index_vals
end

"""
    Intrinsics.make_partition_view(tv::TensorView{T,N},
                                   shape::Tuple,
                                   padding_mode::PaddingValue.T,
                                   order::Union{Tuple,Nothing}) -> PartitionView{T,N,Tuple{shape...}}

Constructs a `PartitionView` over `tv` with per-tile `shape`; lowers to
`cuda_tile.make_partition_view`.

`shape`, `padding_mode`, and `order` are compile-time constants. `shape`
is in Julia (column-major) order, reversed for Tile IR's row-major
layout. `order` may be `nothing` (identity) or a Julia-order 1-indexed
permutation tuple, converted to Tile IR's 0-indexed row-major dim_map.
"""
@intrinsic make_partition_view(tv, shape, padding_mode, order)
function tfunc(𝕃, ::typeof(Intrinsics.make_partition_view), @nospecialize(tv), @nospecialize(shape_arg), @nospecialize args...)
    tv_type = CC.widenconst(tv)
    tv_type <: TensorView || return nothing
    isa(shape_arg, CC.Const) || return nothing
    shape = shape_arg.val
    T = eltype(tv_type)
    N = ndims(tv_type)
    return PartitionView{T, N, Tuple{shape...}}
end
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.make_partition_view), args)
    tv = emit_value!(ctx, args[1])
    tv === nothing && throw(IRError("make_partition_view() requires a TensorView argument"))

    # Shape from user call (e.g., load(arr, idx, (16,)))
    # Reverse to Tile IR row-major order
    shape = @something get_constant(ctx, args[2]) throw(IRError("make_partition_view() shape must be a compile-time constant"))
    shape isa Tuple || throw(IRError("make_partition_view() shape must be a tuple, got $(typeof(shape))"))
    validate_tile_shape(collect(Int, shape), "load")
    tile_shape = RowMajorShape(ColMajorShape(shape))

    padding_value = if length(args) >= 3
        convert_enum(PaddingValue, @something get_constant(ctx, args[3]) throw(IRError("padding_mode must be a compile-time constant")))
    else
        PaddingValue.Missing
    end

    tensor_view = tv.v
    tv_type = tv.type_id
    elem_type = eltype(tv.jltype)
    ndim = length(tile_shape)

    # Extract order (arg 4) and reverse for Tile IR row-major order
    # nothing → identity dim_map, (2,1) → [1, 0] (1-indexed → 0-indexed)
    order_val = @something get_constant(ctx, args[4]) throw(IRError("make_partition_view() order must be a compile-time constant"))
    if order_val === nothing
        dim_map = collect(0:ndim-1)
    else
        # Convert Julia dim_map to Tile IR: reverse and remap indices
        julia_dim_map = collect(Int, map(p -> p - 1, order_val))
        dim_map = [ndim - 1 - julia_dim_map[ndim - i] for i in 0:ndim-1]
    end

    pv_type = partition_view_type!(ctx.tt, tile_shape, tv_type, dim_map, padding_value)
    partition = encode_MakePartitionViewOp!(ctx.cb, pv_type, tensor_view)

    CGVal(partition, pv_type, PartitionView{elem_type, ndim, Tuple{shape...}}, RowMajorShape(()), nothing, Some(ndim), nothing)
end

"""
    compute_tensor_view_strides(array_spec, ndim) -> Vector{Int64}

Compute the stride values for a TensorView type based on ArraySpec.
Returns static stride values where known, DYNAMIC_SHAPE where dynamic.

For contiguous column-major arrays (matching Julia's memory layout),
stride[1] = 1 is statically known. Higher dimensions are typically dynamic.
"""
function compute_tensor_view_strides(array_spec::Union{ArraySpec, Nothing}, ndim::Int)
    strides = fill(DYNAMIC_SHAPE, ndim)

    if array_spec !== nothing && array_spec.contiguous && ndim >= 1
        # Contiguous column-major array: Julia stride[1]=1 becomes Tile IR stride[ndim]=1
        strides[ndim] = 1
    end

    return strides
end

"""
    filter_dynamic_strides(stride_vals, tv_strides) -> Vector{Value}

Filter stride values to only include those corresponding to dynamic dimensions.
Only pass operands for dimensions where tv_strides[i] == DYNAMIC_SHAPE.
"""
function filter_dynamic_strides(stride_vals::Vector{Value}, tv_strides::Vector{Int64})
    dynamic_vals = Value[]
    for (i, stride_type_val) in enumerate(tv_strides)
        if stride_type_val == DYNAMIC_SHAPE && i <= length(stride_vals)
            push!(dynamic_vals, stride_vals[i])
        end
    end
    return dynamic_vals
end

"""
    Intrinsics.make_tensor_view(::Type{<:TileArray{T,N}}, ptr, sizes::Tuple, strides::Tuple) -> TensorView{T,N}

Constructs a `TensorView` from a destructured `TileArray`; lowers to
`cuda_tile.make_tensor_view`.

The first argument is a compile-time constant `TileArray` type. Its
`ArraySpec` (alignment, contiguity, per-axis divisibility) plus the
divisibility / bounds dataflow analyses feed `op_predicates`
(analysis/assume.jl) at codegen time to derive an `AssumePredicate`
chain per operand; `wrap_for` consults the per-`Value` cache so each
source `Value` is wrapped at most once across all consumers, then the
wrapped operands are fed to `encode_MakeTensorViewOp!`. `sizes` and
`strides` are tuples in Julia (column-major) order; they are reversed
for Tile IR's row-major layout.
"""
@intrinsic make_tensor_view(::Type{T}, ptr, sizes, strides) where {T}
function tfunc(𝕃, ::typeof(Intrinsics.make_tensor_view),
               @nospecialize(T_arg), @nospecialize args...)
    T_outer = CC.widenconst(T_arg)
    T_outer isa DataType && T_outer <: Type || return nothing
    T = T_outer.parameters[1]
    T isa Type && T <: TileArray || return nothing
    TensorView{eltype(T), ndims(T)}
end
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.make_tensor_view), args)
    cb = ctx.cb
    tt = ctx.tt

    T_arg, ptr_arg, sizes_arg, strides_arg = args

    T = @something get_constant(ctx, T_arg) throw(IRError("make_tensor_view: TileArray type must be a compile-time constant"))
    T isa Type && T <: TileArray ||
        throw(IRError("make_tensor_view: first arg must be a TileArray type, got $T"))

    elem_T = eltype(T)
    ndim = ndims(T)
    spec = array_spec(T)
    dtype = lookup_dtype!(tt, elem_T)

    # Resolve operands. ptr is a single Value; sizes/strides expand to N values.
    ptr_tv = emit_value!(ctx, ptr_arg)
    ptr_tv === nothing && throw(IRError("make_tensor_view: cannot resolve ptr"))
    base_ptr = ptr_tv.v::Value

    size_tvs = resolve_tuple(ctx, sizes_arg, "make_tensor_view: sizes")
    stride_tvs = resolve_tuple(ctx, strides_arg, "make_tensor_view: strides")
    length(size_tvs) == ndim ||
        throw(IRError("make_tensor_view: expected $ndim sizes, got $(length(size_tvs))"))
    length(stride_tvs) == ndim ||
        throw(IRError("make_tensor_view: expected $ndim strides, got $(length(stride_tvs))"))

    # Wrap each operand `Value` with the `AssumeOp` chain derived
    # on demand from the divby/bounds dataflow plus this MTV's spec.
    # `wrap_for` consults `ctx.assume_wrapped` so a `Value` shared
    # with another consumer (e.g. a gather over the same kernel-arg
    # ptr) — or with this kernel's entry-time slot wrap — is wrapped
    # exactly once. For tuple-typed sizes/strides we walk back via
    # `tuple_element_source` to the per-axis source SSA so the
    # dataflow query has the right anchor; when the source is opaque
    # (wholesale `getfield(arg, :sizes)`) `nothing` falls through to
    # spec-only facts.
    block = ctx.current_block::Block

    # Spec-derived divisor hints (1 = "no info") combine with the
    # dataflow via `lcm` inside `op_predicates`, so a missing spec
    # collapses cleanly to dataflow-only facts.
    align_hint = spec === nothing ? 1 : Int(spec.alignment)
    shape_hint(i) = spec === nothing ? 1 : Int(spec.shape_div_by[i])
    stride_hint(i) = spec === nothing ? 1 : Int(spec.stride_div_by[i])

    base_ptr = wrap_for(ctx, base_ptr, ptr_tv.type_id::TypeId,
                        op_predicates(ctx.divby_info, ctx.bounds_info,
                                      ptr_arg, :ptr, align_hint))

    size_vals = Value[
        let elem_op = tuple_element_source(block, sizes_arg, i)
            wrap_for(ctx, tv.v::Value, tv.type_id::TypeId,
                     op_predicates(ctx.divby_info, ctx.bounds_info,
                                   elem_op, :size, shape_hint(i)))
        end
        for (i, tv) in enumerate(size_tvs)
    ]
    stride_vals = Value[
        let elem_op = tuple_element_source(block, strides_arg, i),
            # Skip the contiguous axis: its stride is statically `1`
            # and never enters the bytecode kernel signature
            # (`filter_dynamic_strides`).
            chain = (spec !== nothing && spec.contiguous && i == 1) ? EMPTY_PREDS :
                    op_predicates(ctx.divby_info, ctx.bounds_info,
                                  elem_op, :stride, stride_hint(i))
            wrap_for(ctx, tv.v::Value, tv.type_id::TypeId, chain)
        end
        for (i, tv) in enumerate(stride_tvs)
    ]

    # Julia column-major order: (stride/size for dim 1, dim 2, ...). Reverse to
    # Tile IR's row-major order at the IR boundary.
    reverse!(size_vals)
    reverse!(stride_vals)

    tv_shape = RowMajorShape(fill(DYNAMIC_SHAPE, ndim))
    tv_strides = compute_tensor_view_strides(spec, ndim)
    tv_type = tensor_view_type!(tt, dtype, tv_shape, tv_strides)

    dynamic_stride_vals = filter_dynamic_strides(stride_vals, tv_strides)

    tensor_view = encode_MakeTensorViewOp!(cb, tv_type, base_ptr, size_vals, dynamic_stride_vals)
    result_jltype = TensorView{elem_T, ndim}
    return CGVal(tensor_view, tv_type, result_jltype)
end

"""
    wrap_for(ctx, value, type_id, preds) -> Value

Apply an `AssumePredicate` chain to a `Value` at most once across all
consumers. `ctx.assume_wrapped` records the first wrap so subsequent
consumers of the same source `Value` reuse it instead of emitting a
parallel `AssumeOp` chain. Empty chain returns the input unchanged.
Mirrors the role of cuTile Python's `var_map` in
`_passes/propagate_divby.py::_add_assume_divby`.

Cache invariant: the cache keys on `Value` only, *not* on the chain
contents. This is sound only when every consumer-derived chain on a
given `Value` is a subset of the first-seen chain — i.e. the first
wrap establishes an upper bound on the facts that any later consumer
would derive. The pipeline arranges this in two ways:

- **Kernel-arg slots:** `apply_arg_assume_predicates!` runs at kernel
  entry and seeds the cache with the spec-tightest chain for each
  TileArray-derived flat slot. Consumer-site `op_predicates` calls on
  SSAs sourced from the same slot can only re-derive a subset (same
  spec hints, equally-tight or looser dataflow), so the cache hit
  drops no information.
- **Per-`Value` consistency of structural priors:** `op_predicates`'s
  `kind` selector (`:ptr` vs. `:size`/`:stride`) is determined by the
  operand's tile type. A single `Value` has one tile type, so all
  consumers see the same `kind` and the same structural prior.

If you ever introduce a consumer that derives a *tighter* chain on a
`Value` already wrapped at kernel entry, the cache will silently drop
the extra facts. Either route the new consumer through a fresh `Value`
(common — the post-offset gather ptr already does this) or refine the
cache key.
"""
@inline function wrap_for(ctx::CGCtx, value::Value, type_id::TypeId,
                          preds::Vector{AssumePredicate})
    isempty(preds) && return value
    cached = get(ctx.assume_wrapped, value, nothing)
    cached !== nothing && return cached
    wrapped = value
    for p in preds
        wrapped = encode_AssumeOp!(ctx.cb, type_id, wrapped, p)
    end
    ctx.assume_wrapped[value] = wrapped
    return wrapped
end

"""
    Intrinsics.store_partition_view(pv::PartitionView{T,N,Shape}, tile::Tile{T},
                                    latency::Union{Int,Nothing},
                                    allow_tma::Bool,
                                    indices::NTuple{M,<:Integer}) -> Nothing  where {T,N,Shape,M}

Token-ordered store of a tile into a `PartitionView`; lowers to
`cuda_tile.store_view_tko`.

`latency` and `allow_tma` are compile-time hints. `indices` is in Julia
order and is reversed/zero-padded to match `pv`'s index space rank
before emission. 0-D tiles are reshaped to 1-D (size 1) since partition
views require at least 1-D. The token argument is appended by
`token_order_pass!` and is not part of the user-visible signature.
"""
@intrinsic store_partition_view(pv::PartitionView{T, N, Shape},
                                          tile::Tile{T},
                                          latency::Union{Int, Nothing},
                                          allow_tma::Bool,
                                          indices::NTuple{M, <:Integer}) where {T, N, Shape, M}
tfunc(𝕃, ::typeof(Intrinsics.store_partition_view), @nospecialize args...) = Nothing
efunc(::typeof(Intrinsics.store_partition_view), effects::CC.Effects) =
    CC.Effects(effects; effect_free=CC.ALWAYS_FALSE)
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.store_partition_view), args)
    cb = ctx.cb
    tt = ctx.tt

    # Extract input token from last arg (added by token_order_pass!)
    input_token = extract_token_arg!(ctx, args)

    # args: (partition_view, tile, latency, allow_tma, indices)
    pv_arg = emit_value!(ctx, args[1])
    pv_arg === nothing && throw(IRError("store_partition_view() requires a PartitionView argument"))
    pv_arg.v === nothing && throw(IRError("store_partition_view() requires a materialized PartitionView"))

    # Get ndim from PartitionView constant field
    pv_arg.constant === nothing && throw(IRError("store_partition_view(): PartitionView missing ndim info"))
    ndim = something(pv_arg.constant)

    # Get tile value
    tile_tv = emit_value!(ctx, args[2])
    tile_tv === nothing && throw(IRError("store_partition_view() requires a tile argument"))
    tile_shape = tile_tv.shape
    tile_shape === nothing && throw(IRError("Cannot determine tile shape for store_partition_view()"))

    elem_type = eltype(CC.widenconst(tile_tv.jltype))
    dtype = lookup_dtype!(tt, elem_type)

    # Handle 0D scalar stores by reshaping to 1D (partition views require at least 1D)
    tile_val = tile_tv.v
    actual_ndim = ndim
    actual_tile_shape = tile_shape
    if length(tile_shape) == 0
        actual_ndim = 1
        actual_tile_shape = RowMajorShape([1])
        tile_1d_type = tile_type!(tt, dtype, actual_tile_shape)
        tile_val = encode_ReshapeOp!(cb, tile_1d_type, tile_val)
    end

    # Extract optimization hints (args[3] = latency, args[4] = allow_tma)
    latency = @something get_constant(ctx, args[3]) throw(IRError("store_partition_view(): latency must be a compile-time constant"))
    allow_tma = @something get_constant(ctx, args[4]) throw(IRError("store_partition_view(): allow_tma must be a compile-time constant"))
    allow_tma_val = allow_tma isa Bool ? allow_tma : true

    # Extract indices
    index_tvs = resolve_tuple(ctx, args[5], "store_partition_view indices")
    index_vals = Value[tv.v for tv in index_tvs]
    index_jl_types = Type[tv.jltype for tv in index_tvs]

    unique_types = unique(index_jl_types)
    length(unique_types) <= 1 || throw(IRError("All index types must match, got: $unique_types"))
    isempty(unique_types) && actual_ndim > 0 && throw(IRError("store_partition_view(): indices required for $(actual_ndim)D view"))
    index_jl_type = isempty(unique_types) ? Int32 : unique_types[1]  # Int32 only for 0D case
    index_type = tile_type_for_julia!(ctx, index_jl_type)

    # Pad indices if needed, then reverse for Tile IR row-major order
    index_vals = pad_indices(ctx, index_vals, actual_ndim, index_type, index_jl_type)
    reverse!(index_vals)

    # Create optimization hints if provided
    optimization_hints = create_optimization_hints(ctx, latency, allow_tma_val)

    token_type = Token(tt)

    result_token = encode_StoreViewTkoOp!(
        cb, token_type, tile_val, pv_arg.v, index_vals;
        token = input_token, optimization_hints
    )

    # Store result token for TokenResultNode
    ctx.result_tokens[ctx.current_ssa_idx] = result_token

    return nothing
end
