# Basic bytecode primitives: varint encoding, tables

#=============================================================================
 Varint Encoding
=============================================================================#

"""
    encode_varint!(buf, x)

Encode an unsigned integer using variable-length encoding (LEB128-style).
Each byte uses 7 bits for data and 1 bit to indicate continuation.
"""
function encode_varint!(buf::Vector{UInt8}, x::Integer)
    if x < 0
        throw(ArgumentError("Varint encoding requires non-negative integers, got $x"))
    end

    # Handle zero specially
    if x == 0
        push!(buf, 0x00)
        return buf
    end
    while x > 0x7f
        push!(buf, UInt8((x & 0x7f) | 0x80))
        x >>= 7
    end
    push!(buf, UInt8(x))
    return buf
end

"""
    encode_signed_varint!(buf, x)

Encode a signed integer using zigzag + varint encoding.
"""
function encode_signed_varint!(buf::Vector{UInt8}, x::Integer)
    # Zigzag encoding: map signed to unsigned
    x = x << 1
    if x < 0
        x = ~x
    end
    encode_varint!(buf, x)
end

"""
    encode_int_list!(buf, lst, byte_width)

Encode a list of integers with fixed byte width (little-endian).
Format: [length as varint] [elements as fixed-width little-endian]
"""
function encode_int_list!(buf::Vector{UInt8}, lst::AbstractVector{<:Integer}, byte_width::Int)
    encode_varint!(buf, length(lst))
    for x in lst
        for i in 0:byte_width-1
            push!(buf, UInt8((x >> (8*i)) & 0xff))
        end
    end
    return buf
end

"""
    encode_varint_list!(buf, lst)

Encode a list of unsigned integers as varints.
Format: [length as varint] [elements as varints]
"""
function encode_varint_list!(buf::Vector{UInt8}, lst::AbstractVector{<:Integer})
    encode_varint!(buf, length(lst))
    for x in lst
        encode_varint!(buf, x)
    end
    return buf
end

# String ID wrapper
struct StringId
    id::Int
end

"""
    StringTable

Interned string table. Maps byte sequences to unique IDs.
"""
mutable struct StringTable
    strings::Dict{Vector{UInt8}, StringId}
    next_id::Int
end

StringTable() = StringTable(Dict{Vector{UInt8}, StringId}(), 0)

function Base.getindex(table::StringTable, s::Union{String, Vector{UInt8}})
    bytes = s isa String ? Vector{UInt8}(s) : s
    get!(table.strings, bytes) do
        id = table.next_id
        table.next_id += 1
        StringId(id)
    end
end

Base.length(table::StringTable) = length(table.strings)

function items(table::StringTable)
    # Return sorted by ID
    pairs = collect(table.strings)
    sort!(pairs, by = p -> p[2].id)
    return pairs
end

# Constant ID wrapper
struct ConstantId
    id::Int
end

"""
    ConstantTable

Table of dense constant values (byte arrays).
"""
mutable struct ConstantTable
    constants::Dict{Vector{UInt8}, ConstantId}
    next_id::Int
end

ConstantTable() = ConstantTable(Dict{Vector{UInt8}, ConstantId}(), 0)

function dense_constant!(table::ConstantTable, data::Vector{UInt8})
    # Encode constant with length prefix (matches Python's encoding)
    encoded = UInt8[]
    encode_varint!(encoded, length(data))
    append!(encoded, data)

    get!(table.constants, encoded) do
        id = table.next_id
        table.next_id += 1
        ConstantId(id)
    end
end

Base.length(table::ConstantTable) = length(table.constants)

function items(table::ConstantTable)
    pairs = collect(table.constants)
    sort!(pairs, by = p -> p[2].id)
    return pairs
end

# Padding helper
function pad_to!(buf::Vector{UInt8}, alignment::Int)
    padding = (-length(buf)) % alignment
    if padding < 0
        padding += alignment
    end
    for _ in 1:padding
        push!(buf, 0xcb)  # Padding byte (same as Python)
    end
    return buf
end
