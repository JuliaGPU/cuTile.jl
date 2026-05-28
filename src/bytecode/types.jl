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
    const F8E8M0FNU = UInt8(0x12)  # since 13.2
    const F4E2M1FN  = UInt8(0x13)  # since 13.3
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
@enumx PaddingValue begin
    Missing = 0
    Zero = 1
    NegZero = 2
    Nan = 3
    PosInf = 4
    NegInf = 5
end

function encode_padding_value!(buf::Vector{UInt8}, pv::PaddingValue.T)
    if pv == PaddingValue.Missing
        push!(buf, 0x00)
    else
        push!(buf, 0x01)
        push!(buf, UInt8(Int(pv) - 1))
    end
end

# v13.3 unified-bitfield view types encode optional fields via a leading
# `optional_flags` varint plus bare value bytes at the tail. Bit 0 marks a
# present padding value.
function encode_optional_flags!(buf::Vector{UInt8}, pv::PaddingValue.T)
    flags = pv == PaddingValue.Missing ? 0 : (1 << 0)
    encode_varint!(buf, flags)
end

function encode_optional_padding_byte!(buf::Vector{UInt8}, pv::PaddingValue.T)
    pv == PaddingValue.Missing && return
    push!(buf, UInt8(Int(pv) - 1))
end

"""
    TypeTable

Table of type definitions. Maps encoded type bytes to TypeIds. Carries the
target bytecode version so version-gated type accessors (e.g. `F8E8M0FNU`)
can validate at registration time.
"""
mutable struct TypeTable
    types::Dict{Vector{UInt8}, TypeId}
    next_id::Int
    version::VersionNumber
end

function TypeTable(; version::VersionNumber)
    table = TypeTable(Dict{Vector{UInt8}, TypeId}(), 0, version)
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
function F8E8M0FNU(table::TypeTable)
    table.version >= v"13.2" ||
        throw(ArgumentError("Float8_E8M0FNU requires Tile IR bytecode v13.2+, got v$(table.version)"))
    simple_type!(table, SimpleType.F8E8M0FNU)
end
function F4E2M1FN(table::TypeTable)
    table.version >= v"13.3" ||
        throw(ArgumentError("Float4_E2M1FN requires Tile IR bytecode v13.3+, got v$(table.version)"))
    simple_type!(table, SimpleType.F4E2M1FN)
end
Token(table::TypeTable) = simple_type!(table, SimpleType.Token)

function tile_type!(table::TypeTable, dtype::TypeId, shape::TileShape)
    buf = UInt8[CompositeType.Tile]
    encode_varint!(buf, dtype.id)
    encode_int_list!(buf, collect(shape), 8)  # 8-byte integers
    _get_or_create!(table, buf)
end

function pointer_type!(table::TypeTable, pointee::TypeId)
    buf = UInt8[CompositeType.Pointer]
    encode_varint!(buf, pointee.id)
    _get_or_create!(table, buf)
end

function tensor_view_type!(table::TypeTable, dtype::TypeId,
                           shape::TileShape,
                           strides::AbstractVector{<:Integer})
    buf = UInt8[CompositeType.TensorView]
    encode_varint!(buf, dtype.id)
    encode_int_list!(buf, collect(shape), 8)
    encode_int_list!(buf, strides, 8)
    _get_or_create!(table, buf)
end

function partition_view_type!(table::TypeTable,
                              tile_shape::TileShape,
                              tensor_view::TypeId,
                              dim_map::AbstractVector{<:Integer},
                              padding_value::PaddingValue.T)
    buf = UInt8[CompositeType.PartitionView]
    if table.version >= v"13.3"
        # Unified bitfield encoding: an `optional_flags` varint up front
        # gates a bare padding byte at the tail (no separate present flag).
        encode_optional_flags!(buf, padding_value)
        encode_int_list!(buf, collect(tile_shape), 4)  # 4-byte integers
        encode_varint!(buf, tensor_view.id)
        encode_int_list!(buf, dim_map, 4)
        encode_optional_padding_byte!(buf, padding_value)
    else
        encode_int_list!(buf, collect(tile_shape), 4)  # 4-byte integers
        encode_varint!(buf, tensor_view.id)
        encode_int_list!(buf, dim_map, 4)
        encode_padding_value!(buf, padding_value)
    end
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

# Julia type to Tile type mapping. Extensions add methods here for their
# own scalar types (e.g. DLFP8TypesExt for Float8_E4M3FN, MicrofloatsExt
# for Float8_E8M0FNU); codegen reaches these via `lookup_dtype!`.
# Note: TFloat32 is defined in cuTile.jl before this file is included.
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
    elseif T <: Ptr
        elem_dtype = lookup_dtype!(table, eltype(T))
        pointer_type!(table, elem_dtype)
    else
        error("Unsupported Julia type for Tile IR: $T")
    end
end

"""
    lookup_dtype!(table, T) -> TypeId

Codegen-facing wrapper around [`julia_to_tile_dtype!`](@ref) that resolves
dispatch in the latest world. The compilation pipeline runs in the world
frozen at `cuTile.__init__` (via `invoke_frozen`, see `cuTile.jl`); a
direct `julia_to_tile_dtype!` call from there cannot see methods added by
extensions loaded after init (e.g. `Float8_E4M3FN` from DLFP8Types,
`Float8_E8M0FNU` from Microfloats). Bouncing through `Base.invokelatest`
mirrors Julia's compiler bootstrap, where compiler infrastructure runs in
a frozen world but user-extensible boundaries (`CompilerPlugins.typeinf`)
hop to the latest.
"""
@inline lookup_dtype!(table::TypeTable, @nospecialize(T::Type)) =
    Base.invokelatest(julia_to_tile_dtype!, table, T)::TypeId
