#=============================================================================
 cuTile.Intrinsics - Compiler-recognized intrinsic stubs

 This module contains the low-level intrinsics that the compiler recognizes
 and transforms into Tile IR operations. All intrinsics use 0-indexed axes
 and indices (matching Tile IR convention).

 User-facing wrappers in operations.jl provide 1-indexed Julia-style APIs
 that call these intrinsics with appropriate index conversions.

 Organized per Tile IR Operations documentation:
 https://docs.nvidia.com/cuda/tile-ir/
=============================================================================#

module Intrinsics

using ..cuTile: Tile, TileArray, Constant, TensorView, PartitionView

#=============================================================================
 8.3. Core
 cuda_tile.broadcast, cuda_tile.cat, cuda_tile.cmpf, cuda_tile.cmpi,
 cuda_tile.constant, cuda_tile.extract, cuda_tile.get_num_tile_blocks,
 cuda_tile.get_tile_block_id, cuda_tile.iota, cuda_tile.mmaf, cuda_tile.mmai,
 cuda_tile.offset, cuda_tile.permute, cuda_tile.reduce, cuda_tile.reshape,
 cuda_tile.select
=============================================================================#

"""
    broadcast(tile, shape_val)

Explicitly broadcast a tile to a target shape.
Compiled to cuda_tile.broadcast.
"""
@noinline function broadcast(tile::Tile{T, S}, ::Val{Shape}) where {T, S, Shape}
    Base.donotdelete(tile)
    Tile{T, Shape}()
end

"""
    cat(tiles, axis_val)

Concatenate two tiles along 0-indexed axis.
Compiled to cuda_tile.cat.
"""
@noinline function cat(tiles::Tuple{Tile{T, S1}, Tile{T, S2}}, ::Val{Axis}) where {T, S1, S2, Axis}
    Base.donotdelete(tiles)
    ndims = length(S1)
    axis = Axis < 0 ? ndims + Axis : Axis
    result_shape = ntuple(ndims) do i
        if i == axis + 1  # 0-indexed axis, 1-indexed tuple access
            S1[i] + S2[i]
        else
            S1[i]
        end
    end
    Tile{T, result_shape}()
end

"""
    cmp(a, b, comparator)

Element-wise comparison. Comparator is <, >, <=, >=, ==, or !=.
Compiled to cuda_tile.cmpf for floats, cuda_tile.cmpi for integers.
"""
@noinline function cmp(a::Tile{T, S}, b::Tile{T, S}, ::F) where {T, S, F<:Function}
    Base.donotdelete(a, b)
    Tile{Bool, S}()
end

"""
    constant(shape, value, T)

Create a tile filled with a constant value.
Compiled to cuda_tile.constant.
"""
@noinline function constant(shape::NTuple{N, Int}, value, ::Type{T}) where {N, T}
    Base.donotdelete(value)
    Tile{T, shape}()
end

"""
    extract(tile, index_val, shape_val)

Extract a sub-tile from tile at 0-indexed slice indices.
Compiled to cuda_tile.extract.
"""
@noinline function extract(tile::Tile{T, S}, ::Val{Index}, ::Val{Shape}) where {T, S, Index, Shape}
    Base.donotdelete(tile)
    Tile{T, Shape}()
end

"""
    get_num_tile_blocks(axis)::Int32

Get the grid size along the given axis (0=x, 1=y, 2=z).
Compiled to cuda_tile.get_num_tile_blocks.
"""
@noinline get_num_tile_blocks(axis::Integer)::Int32 = Base.inferencebarrier(zero(Int32))

"""
    get_tile_block_id(axis)::Int32

Get the block ID along the given axis (0=x, 1=y, 2=z).
Compiled to cuda_tile.get_tile_block_id.
"""
@noinline get_tile_block_id(axis::Integer)::Int32 = Base.inferencebarrier(zero(Int32))

"""
    iota(shape, T)

Create a 1D tile with values [0, 1, 2, ..., shape[1]-1] (0-indexed).
Compiled to cuda_tile.iota.
"""
@noinline function iota(shape::NTuple{1, Int}, ::Type{T}) where {T}
    Tile{T, shape}()
end

"""
    mma(a, b, acc)

Matrix-multiply-accumulate: result = a @ b + acc.
Compiled to cuda_tile.mmaf or cuda_tile.mmai.
"""
@noinline function mma(a::Tile{T1, SA}, b::Tile{T2, SB}, acc::Tile{T3, SC}) where {T1, T2, T3, SA, SB, SC}
    Base.donotdelete(a, b, acc)
    Tile{T3, SC}()
end

"""
    permute(tile, perm_val)

Permute tile dimensions according to 0-indexed permutation.
Compiled to cuda_tile.permute.
"""
@noinline function permute(tile::Tile{T, S}, ::Val{Perm}) where {T, S, Perm}
    Base.donotdelete(tile)
    # Compute permuted shape: for each position i in output, take S[Perm[i]+1]
    permuted_shape = ntuple(i -> S[Perm[i] + 1], length(Perm))
    Tile{T, permuted_shape}()
end

"""
    transpose(tile)

Transpose a 2D tile, swapping its dimensions.
Compiled to cuda_tile.permute with perm=(1, 0).
"""
@noinline function transpose(tile::Tile{T, S}) where {T, S}
    Tile{T, reverse(S)}()
end

"""
    reduce_sum(tile, axis_val)

Sum reduction along 0-indexed axis.
Compiled to cuda_tile.reduce with ADD.
"""
@noinline function reduce_sum(tile::Tile{T, S}, ::Val{axis}) where {T <: AbstractFloat, S, axis}
    reduced_shape = ntuple(i -> S[i < axis + 1 ? i : i + 1], length(S) - 1)
    Base.donotdelete(tile)
    Tile{T, reduced_shape}()
end

"""
    reduce_max(tile, axis_val)

Maximum reduction along 0-indexed axis.
Compiled to cuda_tile.reduce with MAX.
"""
@noinline function reduce_max(tile::Tile{T, S}, ::Val{axis}) where {T <: AbstractFloat, S, axis}
    reduced_shape = ntuple(i -> S[i < axis + 1 ? i : i + 1], length(S) - 1)
    Base.donotdelete(tile)
    Tile{T, reduced_shape}()
end

"""
    reshape(tile, shape_val)

Reshape a tile to a new shape (same total elements).
Compiled to cuda_tile.reshape.
"""
@noinline function reshape(tile::Tile{T, S}, ::Val{Shape}) where {T, S, Shape}
    Base.donotdelete(tile)
    Tile{T, Shape}()
end

"""
    select(cond, x, y)

Element-wise conditional selection.
Compiled to cuda_tile.select.
"""
@noinline function select(cond::Tile{Bool, S}, x::Tile{T, S}, y::Tile{T, S}) where {T, S}
    Base.donotdelete(cond, x, y)
    Tile{T, S}()
end

"""
    offset(base, offsets)

Compute base_ptr + offsets for each element of offsets tile (element-scaled).
Returns a tile of pointers. Compiled to cuda_tile.offset.
"""
@noinline function offset(base::Ptr{T}, offsets::Tile{I, S}) where {T, I <: Integer, S}
    Base.donotdelete(base, offsets)
    Tile{Ptr{T}, S}()
end

#=============================================================================
 8.4. Conversions
 cuda_tile.bitcast, cuda_tile.exti, cuda_tile.ftof, cuda_tile.ftoi,
 cuda_tile.itof, cuda_tile.trunci
=============================================================================#

"""
    astype(tile, T2)

Convert tile element type from T1 to T2.
Compiled to cuda_tile.ftof, cuda_tile.ftoi, cuda_tile.itof, etc.
"""
@noinline function astype(tile::Tile{T1, Shape}, ::Type{T2}) where {T1, Shape, T2}
    Base.donotdelete(tile)
    Tile{T2, Shape}()
end

#=============================================================================
 8.6. Memory
 cuda_tile.load_ptr_tko, cuda_tile.store_ptr_tko
=============================================================================#

"""
    load(ptr, shape_val, indices...)

Load a tile from a pointer at the given indices.
Shape must be wrapped in Val for compile-time access.
Compiled to cuda_tile.load_ptr_tko.
"""
@noinline function load(ptr::Ptr{T}, ::Val{shape}, indices...) where {T, shape}
    Tile{T, shape}()
end

"""
    store(ptr, tile, indices...)

Store a tile to a pointer at the given indices.
Compiled to cuda_tile.store_ptr_tko.
"""
@noinline function store(ptr::Ptr{T}, tile::Tile{T}, indices...) where T
    Base.donotdelete(ptr, tile, indices...)
    nothing
end

"""
    load_ptr_tko(ptrs, mask=nothing, padding=nothing)

Load values from a tile of pointers.
If mask is provided, masked-out positions return the padding value.
Compiled to cuda_tile.load_ptr_tko.
"""
@noinline function load_ptr_tko(ptrs::Tile{Ptr{T}, S},
                                 mask::Union{Tile{Bool, S}, Nothing}=nothing,
                                 padding::Union{Tile{T, S}, Nothing}=nothing) where {T, S}
    Base.donotdelete(ptrs, mask, padding)
    Tile{T, S}()
end

"""
    store_ptr_tko(ptrs, values, mask=nothing)

Store values to a tile of pointers.
If mask is provided, masked-out positions are not written.
Compiled to cuda_tile.store_ptr_tko.
"""
@noinline function store_ptr_tko(ptrs::Tile{Ptr{T}, S}, values::Tile{T, S},
                                  mask::Union{Tile{Bool, S}, Nothing}=nothing) where {T, S}
    Base.donotdelete(ptrs, values, mask)
    nothing
end

#=============================================================================
 8.7. Floating Point
 cuda_tile.addf, cuda_tile.divf, cuda_tile.mulf, cuda_tile.pow,
 cuda_tile.rsqrt, cuda_tile.sqrt, cuda_tile.subf
=============================================================================#

"""
    arith(a, b, op)

Element-wise arithmetic. Op is +, -, *, /, or ^.
Compiled to cuda_tile.addf/addi, cuda_tile.subf/subi, etc. based on element type.
"""
@noinline function arith(a::Tile{T, S}, b::Tile{T, S}, ::F) where {T, S, F<:Function}
    Base.donotdelete(a, b)
    Tile{T, S}()
end

"""Element-wise reciprocal square root. Compiled to cuda_tile.rsqrt."""
@noinline function rsqrt(tile::Tile{T, S}) where {T <: AbstractFloat, S}
    Base.donotdelete(tile)
    Tile{T, S}()
end

"""Element-wise square root. Compiled to cuda_tile.sqrt."""
@noinline function sqrt(tile::Tile{T, S}) where {T <: AbstractFloat, S}
    Base.donotdelete(tile)
    Tile{T, S}()
end

#=============================================================================
 8.8. Integer
 cuda_tile.addi, cuda_tile.divi, cuda_tile.muli, cuda_tile.subi
=============================================================================#

"""Ceiling division: ceil(a/b). Compiled to cuda_tile.divi."""
@noinline cdiv(a::Integer, b::Integer)::Int32 = Base.inferencebarrier(zero(Int32))

"""Floor division: floor(a/b). Compiled to cuda_tile.divi."""
@noinline floordiv(a::Integer, b::Integer)::Int32 = Base.inferencebarrier(zero(Int32))

#=============================================================================
 8.8. Bitwise
 cuda_tile.andi
=============================================================================#

"""Element-wise logical AND for boolean tiles. Compiled to cuda_tile.andi."""
@noinline function logical_and(a::Tile{Bool, S}, b::Tile{Bool, S}) where {S}
    Base.donotdelete(a, b)
    Tile{Bool, S}()
end

#=============================================================================
 8.10. Atomics
 cuda_tile.atomic_cas_tko, cuda_tile.atomic_rmw_tko
=============================================================================#

"""
    atomic_cas(array, index, expected, desired, memory_order, memory_scope)

Atomic compare-and-swap at 0-indexed position.
Returns the original value.
Compiled to cuda_tile.atomic_cas_tko.
"""
@noinline function atomic_cas(array::TileArray{T, N}, index, expected, desired,
                               memory_order::Int, memory_scope::Int) where {T, N}
    Base.donotdelete(array, index, expected, desired)
    Base.inferencebarrier(zero(T))::T
end

"""
    atomic_xchg(array, index, val, memory_order, memory_scope)

Atomic exchange at 0-indexed position.
Returns the original value.
Compiled to cuda_tile.atomic_rmw_tko with XCHG.
"""
@noinline function atomic_xchg(array::TileArray{T, N}, index, val,
                                memory_order::Int, memory_scope::Int) where {T, N}
    Base.donotdelete(array, index, val)
    Base.inferencebarrier(zero(T))::T
end

"""
    atomic_add(array, index, val, memory_order, memory_scope)

Atomic addition at 0-indexed position.
Returns the original value.
Compiled to cuda_tile.atomic_rmw_tko with ADD.
"""
@noinline function atomic_add(array::TileArray{T, N}, index, val,
                               memory_order::Int, memory_scope::Int) where {T, N}
    Base.donotdelete(array, index, val)
    Base.inferencebarrier(zero(T))::T
end

#=============================================================================
 8.11. Views
 cuda_tile.make_tensor_view, cuda_tile.make_partition_view,
 cuda_tile.get_index_space_shape, cuda_tile.load_view_tko, cuda_tile.store_view_tko
=============================================================================#

"""
    make_tensor_view(arr::TileArray) -> TensorView

Create a TensorView from a TileArray.
Compiled to cuda_tile.make_tensor_view.
"""
@noinline function make_tensor_view(arr::TileArray{T, N})::TensorView{T, N} where {T, N}
    Base.donotdelete(arr)
    Base.inferencebarrier(TensorView{T, N}())
end

"""
    make_partition_view(tv::TensorView, shape_val, padding_mode) -> PartitionView

Create a PartitionView from a TensorView with the given tile shape.
Compiled to cuda_tile.make_partition_view.
"""
@noinline function make_partition_view(tv::TensorView{T, N}, ::Val{Shape}, padding_mode::Int)::PartitionView{T, N, Shape} where {T, N, Shape}
    Base.donotdelete(tv)
    Base.inferencebarrier(PartitionView{T, N, Shape}())
end

"""
    get_index_space_shape(pv::PartitionView, axis) -> Int32

Get the number of tiles along the given axis (0-indexed).
Compiled to cuda_tile.get_index_space_shape.
"""
@noinline function get_index_space_shape(pv::PartitionView{T, N, Shape}, axis::Integer)::Int32 where {T, N, Shape}
    Base.donotdelete(pv)
    Base.inferencebarrier(zero(Int32))
end

"""
    load_partition_view(pv::PartitionView, index...) -> Tile

Load a tile from a partition view at the given 0-indexed tile coordinates.
Compiled to cuda_tile.load_view_tko.
"""
@noinline function load_partition_view(pv::PartitionView{T, N, Shape}, index::Vararg{Integer})::Tile{T, Shape} where {T, N, Shape}
    Base.donotdelete(pv)
    Base.inferencebarrier(Tile{T, Shape}())
end

"""
    store_partition_view(pv::PartitionView, tile, index...) -> Nothing

Store a tile to a partition view at the given 0-indexed tile coordinates.
Compiled to cuda_tile.store_view_tko.
"""
@noinline function store_partition_view(pv::PartitionView{T, N, Shape}, tile::Tile{T, Shape}, index::Vararg{Integer})::Nothing where {T, N, Shape}
    Base.donotdelete(pv)
    Base.donotdelete(tile)
    nothing
end

"""
    load(arr, shape_val, padding_mode, indices...)

Load a tile from a TileArray at the given indices.
Shape must be wrapped in Val for compile-time access.
Compiled to cuda_tile.load_view_tko with TensorView.
"""
@noinline function load(arr::TileArray{T, N}, ::Val{shape}, padding_mode::Int, indices...) where {T, N, shape}
    Base.donotdelete(arr, indices..., padding_mode)
    Tile{T, shape}()
end

"""
    store(arr, tile, indices...)

Store a tile to a TileArray at the given indices.
Compiled to cuda_tile.store_view_tko with TensorView.
"""
@noinline function store(arr::TileArray{T, N}, tile::Tile{T}, indices...) where {T, N}
    Base.donotdelete(arr, tile, indices...)
    nothing
end

end # module Intrinsics
