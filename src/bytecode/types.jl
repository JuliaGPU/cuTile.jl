# Type system for Tile IR bytecode

# Type ID wrapper
struct TypeId
    id::Int
end

function encode_typeid!(buf::Vector{UInt8}, type_id::TypeId)
    encode_varint!(buf, type_id.id)
end

function encode_typeid_seq!(buf::Vector{UInt8}, type_ids::AbstractVector{TypeId})
    encode_varint!(buf, length(type_ids))
    for tid in type_ids
        encode_varint!(buf, tid.id)
    end
end

# Predefined type IDs (must match Python's order)
const I1_TYPE_ID = TypeId(0)
const I32_TYPE_ID = TypeId(1)

# Simple type tags (from Python's SimpleType enum)
module SimpleType
    const I1    = UInt8(0x00)
    const I8    = UInt8(0x01)
    const I16   = UInt8(0x02)
    const I32   = UInt8(0x03)
    const I64   = UInt8(0x04)
    const F16   = UInt8(0x05)
    const BF16  = UInt8(0x06)
    const F32   = UInt8(0x07)
    const TF32  = UInt8(0x08)
    const F64   = UInt8(0x09)
    const F8E4M3FN = UInt8(0x0a)
    const F8E5M2   = UInt8(0x0b)
    const Token    = UInt8(0x11)
    const Unknown  = UInt8(0x12)
end

# Composite type tags
module CompositeType
    const Pointer       = UInt8(0x0c)
    const Tile          = UInt8(0x0d)
    const TensorView    = UInt8(0x0e)
    const PartitionView = UInt8(0x0f)
    const Func          = UInt8(0x10)
end

# Dynamic shape marker
const DYNAMIC_SHAPE = typemin(Int64)

# Padding values for loads
@enum PaddingValue begin
    PaddingMissing = 0
    PaddingZero = 1
    PaddingNegZero = 2
    PaddingNan = 3
    PaddingPosInf = 4
    PaddingNegInf = 5
end

function encode_padding_value!(buf::Vector{UInt8}, pv::PaddingValue)
    if pv == PaddingMissing
        push!(buf, 0x00)
    else
        push!(buf, 0x01)
        push!(buf, UInt8(Int(pv) - 1))
    end
end

"""
    TypeTable

Table of type definitions. Maps encoded type bytes to TypeIds.
"""
mutable struct TypeTable
    types::Dict{Vector{UInt8}, TypeId}
    next_id::Int
end

function TypeTable()
    table = TypeTable(Dict{Vector{UInt8}, TypeId}(), 0)
    # Pre-register I1 and I32 at fixed positions
    _predefine!(table, [SimpleType.I1], I1_TYPE_ID)
    _predefine!(table, [SimpleType.I32], I32_TYPE_ID)
    return table
end

function _predefine!(table::TypeTable, tag::Vector{UInt8}, expected_id::TypeId)
    actual = _get_or_create!(table, tag)
    if actual.id != expected_id.id
        error("Type registration order mismatch: expected $(expected_id.id), got $(actual.id)")
    end
end

function _get_or_create!(table::TypeTable, encoded::Vector{UInt8})
    get!(table.types, encoded) do
        id = table.next_id
        table.next_id += 1
        TypeId(id)
    end
end

Base.length(table::TypeTable) = length(table.types)

function items(table::TypeTable)
    pairs = collect(table.types)
    sort!(pairs, by = p -> p[2].id)
    return pairs
end

# Type constructors

function simple_type!(table::TypeTable, tag::UInt8)
    _get_or_create!(table, [tag])
end

# Convenience accessors
I1(table::TypeTable) = simple_type!(table, SimpleType.I1)
I8(table::TypeTable) = simple_type!(table, SimpleType.I8)
I16(table::TypeTable) = simple_type!(table, SimpleType.I16)
I32(table::TypeTable) = simple_type!(table, SimpleType.I32)
I64(table::TypeTable) = simple_type!(table, SimpleType.I64)
F16(table::TypeTable) = simple_type!(table, SimpleType.F16)
BF16(table::TypeTable) = simple_type!(table, SimpleType.BF16)
F32(table::TypeTable) = simple_type!(table, SimpleType.F32)
TF32(table::TypeTable) = simple_type!(table, SimpleType.TF32)
F64(table::TypeTable) = simple_type!(table, SimpleType.F64)
F8E4M3FN(table::TypeTable) = simple_type!(table, SimpleType.F8E4M3FN)
F8E5M2(table::TypeTable) = simple_type!(table, SimpleType.F8E5M2)
Token(table::TypeTable) = simple_type!(table, SimpleType.Token)

function tile_type!(table::TypeTable, dtype::TypeId, shape::AbstractVector{<:Integer})
    buf = UInt8[CompositeType.Tile]
    encode_varint!(buf, dtype.id)
    encode_int_list!(buf, shape, 8)  # 8-byte integers
    _get_or_create!(table, buf)
end

function pointer_type!(table::TypeTable, pointee::TypeId)
    buf = UInt8[CompositeType.Pointer]
    encode_varint!(buf, pointee.id)
    _get_or_create!(table, buf)
end

function tensor_view_type!(table::TypeTable, dtype::TypeId,
                           shape::AbstractVector{<:Integer},
                           strides::AbstractVector{<:Integer})
    buf = UInt8[CompositeType.TensorView]
    encode_varint!(buf, dtype.id)
    encode_int_list!(buf, shape, 8)
    encode_int_list!(buf, strides, 8)
    _get_or_create!(table, buf)
end

function partition_view_type!(table::TypeTable,
                              tile_shape::AbstractVector{<:Integer},
                              tensor_view::TypeId,
                              dim_map::AbstractVector{<:Integer},
                              padding_value::PaddingValue)
    buf = UInt8[CompositeType.PartitionView]
    encode_int_list!(buf, tile_shape, 4)  # 4-byte integers
    encode_varint!(buf, tensor_view.id)
    encode_int_list!(buf, dim_map, 4)
    encode_padding_value!(buf, padding_value)
    _get_or_create!(table, buf)
end

function function_type!(table::TypeTable,
                        param_types::AbstractVector{TypeId},
                        result_types::AbstractVector{TypeId})
    buf = UInt8[CompositeType.Func]
    encode_typeid_seq!(buf, param_types)
    encode_typeid_seq!(buf, result_types)
    _get_or_create!(table, buf)
end

# Julia type to Tile type mapping
# Note: TFloat32 is defined in cuTile.jl before this file is included
function julia_to_tile_dtype!(table::TypeTable, ::Type{T}) where T
    if T === Bool
        I1(table)
    elseif T === Int8 || T === UInt8
        I8(table)
    elseif T === Int16 || T === UInt16
        I16(table)
    elseif T === Int32 || T === UInt32
        I32(table)
    elseif T === Int64 || T === UInt64
        I64(table)
    elseif T === Float16
        F16(table)
    elseif T === BFloat16
        BF16(table)
    elseif T === Float32
        F32(table)
    elseif T === TFloat32
        TF32(table)
    elseif T === Float64
        F64(table)
    elseif T === Float8_E4M3FN
        F8E4M3FN(table)
    elseif T === Float8_E5M2
        F8E5M2(table)
    elseif T <: Ptr
        elem_dtype = julia_to_tile_dtype!(table, eltype(T))
        pointer_type!(table, elem_dtype)
    else
        error("Unsupported Julia type for Tile IR: $T")
    end
end
