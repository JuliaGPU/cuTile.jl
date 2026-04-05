using GPUArrays: neutral_element

# Map reduction ops to their atomic counterparts for a given element type.
# Returns nothing when no hardware atomic is available for the (op, dtype) combo.
_atomic_op(::typeof(+), ::Type{<:Union{Float32, Int32, Int64}}) = atomic_add
_atomic_op(::typeof(Base.add_sum), ::Type{<:Union{Float32, Int32, Int64}}) = atomic_add
_atomic_op(::typeof(max), ::Type{<:Integer}) = atomic_max
_atomic_op(::typeof(min), ::Type{<:Integer}) = atomic_min
_atomic_op(::typeof(|), ::Type{<:Integer}) = atomic_or
_atomic_op(::typeof(&), ::Type{<:Integer}) = atomic_and
_atomic_op(_, ::Type) = nothing

@generated function mapreduce_kernel(
    dest::TileArray{TD, N}, src::TileArray{TS, N},
    f, op, tile_size, reduce_dims, overflow_grids, init_val, pad_mode,
    reduce_stride, atomic_op
) where {TD, TS, N}
    quote
        bids = _unflatten_bids(Val{$N}(), overflow_grids)

        acc = fill(init_val, tile_size)

        # Per-dimension loop bounds: reduced dims iterate all tiles; others run once
        @nexprs $N d -> begin
            idx_d = bids[d]
            n_d = d in reduce_dims ? num_tiles(src, d, tile_size) : idx_d
            start_d = idx_d
        end

        @nloops($N, idx, d -> (start_d:reduce_stride[d]:n_d),
            begin
                tile = load(src, (@ntuple $N d -> idx_d), tile_size; padding_mode=pad_mode)
                acc = op.(acc, f.(tile))
            end)

        # Collapse each reduced dimension within the accumulated tile
        @nexprs $N d -> if d in reduce_dims
            acc = reduce(op, acc; dims=d, init=init_val)
        end

        if atomic_op !== nothing
            scalar = reshape(acc, ())
            @nexprs $N d -> out_d = d in reduce_dims ? Int32(1) : bids[d]
            linear_idx = Int32(1)
            stride = Int32(1)
            @nexprs $N d -> begin
                linear_idx = linear_idx + (out_d - Int32(1)) * stride
                stride = stride * size(dest, d)
            end
            atomic_op(dest, linear_idx, scalar)
        else
            store(dest, bids, acc)
        end

        return
    end
end

function _padding_for_neutral(neutral)
    iszero(neutral)  && return PaddingMode.Zero
    neutral == -Inf  && return PaddingMode.NegInf
    neutral == Inf   && return PaddingMode.PosInf
    return nothing  # no native padding; needs aligned sizes
end

function _mapreducedim!(f, op, R::AbstractArray, A::AbstractArray, reduce_dims::Tuple; init)
    N = ndims(A)
    src_ta = TileArray(A)

    # Reduced dims first (larger tiles => better hardware reduction)
    dim_order = (filter(d -> d in reduce_dims, 1:N)..., filter(d -> !(d in reduce_dims), 1:N)...)
    ts = _compute_tile_sizes(size(A), dim_order)

    pad_mode = _padding_for_neutral(init)
    if pad_mode === nothing
        has_oob = any(i -> size(A, i) % ts[i] != 0, 1:N)
        has_oob && error("cuTile mapreduce: $op on $(eltype(R)) requires aligned array sizes " *
                         "(each dimension divisible by tile size) because no safe padding mode exists. " *
                         "Supported without alignment: +, max/min (float), |.")
        pad_mode = PaddingMode.Zero
    end

    # Pick the largest reduced dim for potential parallelization
    non_reduce_blocks = prod(cld(size(A, d), ts[d]) for d in 1:N if !(d in reduce_dims); init=1)
    par_dim = reduce_dims[argmax(map(d -> cld(size(A, d), ts[d]), reduce_dims))]
    max_tiles = cld(size(A, par_dim), ts[par_dim])

    target = max(1, 128 ÷ non_reduce_blocks)
    par_blocks = min(max_tiles, target)

    _dim_size(d) = d == par_dim ? par_blocks : d in reduce_dims ? 1 : size(A, d)

    # Atomics only work when each block produces one scalar output element,
    # i.e., all non-reduce dims have tile size 1
    scalar_output = all(d -> d in reduce_dims || ts[d] == 1, 1:N)
    atomic_op = scalar_output ? _atomic_op(op, eltype(R)) : nothing

    if par_blocks > 1
        grid = ntuple(N) do d
            d == par_dim ? par_blocks : d in reduce_dims ? 1 : cld(size(A, d), ts[d])
        end
        reduce_stride = ntuple(d -> d == par_dim ? Int32(par_blocks) : Int32(1), N)

        if atomic_op !== nothing
            fill!(R, init)
            _launch_mapreduce!(grid, TileArray(R), src_ta, f, op, ts, reduce_dims,
                               init, pad_mode, reduce_stride, atomic_op)
        else
            # Two-pass: parallelize along par_dim, then reduce partials
            tmp = similar(A, eltype(R), ntuple(_dim_size, N))
            _launch_mapreduce!(grid, TileArray(tmp), src_ta, f, op, ts, reduce_dims,
                               init, pad_mode, reduce_stride, nothing)
            _mapreducedim!(identity, op, R, tmp, (par_dim,); init)
        end
    else
        grid = ntuple(d -> d in reduce_dims ? 1 : cld(size(A, d), ts[d]), N)
        reduce_stride = ntuple(d -> Int32(1), N)
        _launch_mapreduce!(grid, TileArray(R), src_ta, f, op, ts, reduce_dims,
                           init, pad_mode, reduce_stride, nothing)
    end
end

function _launch_mapreduce!(grid, dest_ta, src_ta, f, op, ts, reduce_dims,
                            init, pad_mode, reduce_stride, atomic_op)
    launch_grid, overflow = _flatten_grid(grid)
    launch(mapreduce_kernel, launch_grid, dest_ta, src_ta,
           f, op, Constant(ts), Constant(reduce_dims), Constant(overflow),
           Constant(init), Constant(pad_mode), Constant(reduce_stride),
           atomic_op === nothing ? Constant(nothing) : atomic_op)
end

function _mapreduce(f, op, A::AbstractArray; dims, init)
    T = eltype(A)
    ET = f === identity ? T : Base.promote_op(f, T)
    ET = Base.promote_op(op, ET, ET)
    (ET === Union{} || ET === Any) && (ET = T)

    # The kernel always uses the neutral element internally.
    # User-provided init is applied separately via op(init, result).
    neutral = neutral_element(op, ET)
    N = ndims(A)

    reduce_dims = dims === Colon() ? ntuple(identity, N) :
                  dims isa Integer ? (dims,) : Tuple(dims)
    out_size = ntuple(d -> d in reduce_dims ? 1 : size(A, d), N)
    R = similar(A, ET, out_size)
    _mapreducedim!(f, op, R, A, reduce_dims; init=neutral)

    if dims === Colon()
        result = Array(R)[1]
        return init !== nothing ? op(ET(init), result) : result
    else
        init !== nothing && (R .= op.(ET(init), R))
        return R
    end
end


## user API

Base.mapreduce(f, op, A::Tiled; dims=:, init=nothing) =
    _mapreduce(f, op, parent(A); dims, init)

Base.reduce(op, A::Tiled; dims=:, init=nothing) =
    _mapreduce(identity, op, parent(A); dims, init)

# any/all use short-circuiting iteration in Base, not mapreduce
Base.any(A::Tiled{<:AbstractArray{Bool}}; dims=:) =
    _mapreduce(identity, |, parent(A); dims, init=nothing)
Base.all(A::Tiled{<:AbstractArray{Bool}}; dims=:) =
    _mapreduce(identity, &, parent(A); dims, init=nothing)
Base.any(f::Function, A::Tiled; dims=:) =
    _mapreduce(f, |, parent(A); dims, init=nothing)
Base.all(f::Function, A::Tiled; dims=:) =
    _mapreduce(f, &, parent(A); dims, init=nothing)

# In-place variants
function _mapreducedim_inplace!(f, op, R::AbstractArray, A::AbstractArray)
    T = eltype(R)
    N = ndims(A)
    reduce_dims = Tuple(d for d in 1:N if size(R, d) == 1 && size(A, d) > 1)
    init_val = neutral_element(op, T)
    _mapreducedim!(f, op, R, A, reduce_dims; init=init_val)
    return R
end

Base.sum!(R, A::Tiled) = _mapreducedim_inplace!(identity, +, R, parent(A))
Base.prod!(R, A::Tiled) = _mapreducedim_inplace!(identity, *, R, parent(A))
Base.maximum!(R, A::Tiled) = _mapreducedim_inplace!(identity, max, R, parent(A))
Base.minimum!(R, A::Tiled) = _mapreducedim_inplace!(identity, min, R, parent(A))
