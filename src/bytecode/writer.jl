# Bytecode file writer - handles sections and overall structure

# Bytecode version
const BYTECODE_VERSION = (13, 1, 0)

# Magic number
const MAGIC = UInt8[0x7f, 0x54, 0x69, 0x6c, 0x65, 0x49, 0x52, 0x00]  # "\x7fTileIR\x00"

# Section IDs
module Section
    const EndOfBytecode = UInt8(0x00)
    const String        = UInt8(0x01)
    const Func          = UInt8(0x02)
    const Debug         = UInt8(0x03)
    const Constant      = UInt8(0x04)
    const Type          = UInt8(0x05)
    const Global        = UInt8(0x06)
end

# Debug attribute ID
struct DebugAttrId
    id::Int
end

"""
    DebugAttrTable

Table of debug attributes (file/line info).
"""
mutable struct DebugAttrTable
    attrs::Dict{Vector{UInt8}, DebugAttrId}
    string_table::StringTable
    next_id::Int
end

DebugAttrTable(string_table::StringTable) = DebugAttrTable(
    Dict{Vector{UInt8}, DebugAttrId}(),
    string_table,
    1  # Start at 1, 0 is reserved for "no debug info"
)

function Base.getindex(table::DebugAttrTable, encoded::Vector{UInt8})
    get!(table.attrs, encoded) do
        id = table.next_id
        table.next_id += 1
        DebugAttrId(id)
    end
end

Base.length(table::DebugAttrTable) = length(table.attrs)

function items(table::DebugAttrTable)
    pairs = collect(table.attrs)
    sort!(pairs, by = p -> p[2].id)
    return pairs
end

# SSA Value wrapper
struct Value
    id::Int
end


function encode_operand!(buf::Vector{UInt8}, val::Value)
    encode_varint!(buf, val.id)
end

function encode_optional_operand!(buf::Vector{UInt8}, val::Union{Value, Nothing})
    if val !== nothing
        encode_varint!(buf, val.id)
    end
end

function encode_operands!(buf::Vector{UInt8}, vals::AbstractVector{Value})
    for v in vals
        encode_varint!(buf, v.id)
    end
end

function encode_sized_operands!(buf::Vector{UInt8}, vals::AbstractVector{Value})
    encode_varint!(buf, length(vals))
    encode_operands!(buf, vals)
end

"""
    CodeBuilder

Builds bytecode for a single function's body.
Tracks SSA value numbering and operation count.
"""
mutable struct CodeBuilder
    buf::Vector{UInt8}
    string_table::StringTable
    constant_table::ConstantTable
    type_table::TypeTable
    debug_attrs::Vector{DebugAttrId}
    next_value_id::Int
    cur_debug_attr::DebugAttrId
    num_ops::Int
end

function CodeBuilder(string_table::StringTable, constant_table::ConstantTable, type_table::TypeTable)
    CodeBuilder(
        UInt8[],
        string_table,
        constant_table,
        type_table,
        DebugAttrId[],
        0,
        DebugAttrId(0),  # No debug info
        0
    )
end

"""
Create a new SSA value(s) for an operation result.
"""
function new_op!(cb::CodeBuilder, num_results::Int=1)
    push!(cb.debug_attrs, cb.cur_debug_attr)
    cb.num_ops += 1

    if num_results == 0
        return nothing
    elseif num_results == 1
        val = Value(cb.next_value_id)
        cb.next_value_id += 1
        return val
    else
        vals = [Value(cb.next_value_id + i) for i in 0:num_results-1]
        cb.next_value_id += num_results
        return Tuple(vals)
    end
end

"""
Create block arguments (for function parameters or nested blocks).
"""
function make_block_args!(cb::CodeBuilder, count::Int)
    vals = [Value(cb.next_value_id + i) for i in 0:count-1]
    cb.next_value_id += count
    return vals
end

"""
    with_region(f, cb, arg_type_ids) -> Vector{Value}

Execute `f(block_args)` in a new region/block context.
Handles buffer management automatically (MLIR-style callback pattern).

The callback receives block arguments and should emit operations into `cb`.
"""
function with_region(f::Function, cb::CodeBuilder, arg_type_ids::Vector{TypeId})
    # Number of blocks in region (always 1)
    push!(cb.buf, 0x01)

    # Number of block arguments
    encode_varint!(cb.buf, length(arg_type_ids))

    # Encode block argument types
    for tid in arg_type_ids
        encode_typeid!(cb.buf, tid)
    end

    # Save current state (buffer, num_ops, and next_value_id)
    parent_buf = cb.buf
    parent_num_ops = cb.num_ops
    parent_next_value_id = cb.next_value_id

    # Create block arguments (allocates value IDs in region scope)
    block_args = make_block_args!(cb, length(arg_type_ids))

    # Create fresh buffer for block body
    cb.buf = UInt8[]
    cb.num_ops = 0

    # Execute the callback to populate the block
    f(block_args)

    # Capture block body
    block_body = cb.buf
    block_num_ops = cb.num_ops

    # Restore parent state (including next_value_id - region values are ephemeral)
    cb.buf = parent_buf
    cb.num_ops = parent_num_ops
    cb.next_value_id = parent_next_value_id

    # Encode: num_ops, then block body
    encode_varint!(cb.buf, block_num_ops)
    append!(cb.buf, block_body)

    return block_args
end

# Attribute encoders for operations
function encode_opattr_bool!(cb::CodeBuilder, val::Bool)
    push!(cb.buf, val ? 0x01 : 0x00)
end

function encode_opattr_int!(cb::CodeBuilder, val::Integer)
    encode_varint!(cb.buf, val)
end

function encode_opattr_str!(cb::CodeBuilder, val::String)
    sid = cb.string_table[val]
    encode_varint!(cb.buf, sid.id)
end

function encode_opattr_typeid!(cb::CodeBuilder, tid::TypeId)
    encode_typeid!(cb.buf, tid)
end

function encode_opattr_dense!(cb::CodeBuilder, data::Vector{UInt8})
    cid = dense_constant!(cb.constant_table, data)
    encode_varint!(cb.buf, cid.id)
end

"""
    encode_dense_int32_array!(cb, values)

Encode a sequence of integers as a dense int32 array attribute.
Format: varint(length) followed by each value as 4 little-endian signed bytes.
"""
function encode_dense_int32_array!(cb::CodeBuilder, values::Vector{Int})
    encode_varint!(cb.buf, length(values))
    for v in values
        # Encode as 4 bytes, little-endian, signed
        append!(cb.buf, reinterpret(UInt8, [Int32(v)]))
    end
end

#=============================================================================
 Tagged attributes for ReduceOp identity values
=============================================================================#

"""
    ReduceIdentity

Abstract type for reduce identity attributes.
"""
abstract type ReduceIdentity end

"""
    FloatIdentity(value, type_id, dtype)

Float identity value for reduce operations.
"""
struct FloatIdentity <: ReduceIdentity
    value::Float64
    type_id::TypeId
    dtype::Type  # Float16, Float32, Float64, etc.
end

"""
    encode_tagged_float!(cb, identity::FloatIdentity)

Encode a tagged float attribute for reduce identity.
Format: tag(Float=0x02) + typeid + ap_int(value_bits)
"""
function encode_tagged_float!(cb::CodeBuilder, identity::FloatIdentity)
    # Tag for Float attribute
    push!(cb.buf, 0x02)
    # Type ID
    encode_typeid!(cb.buf, identity.type_id)
    # Value as bits (using signed varint encoding for values <= 64 bits)
    bits = float_to_bits(identity.value, identity.dtype)
    encode_signed_varint!(cb.buf, bits)
end

"""
    float_to_bits(value, dtype)

Convert a float value to its bit representation.
"""
function float_to_bits(value::Float64, ::Type{Float16})
    reinterpret(UInt16, Float16(value))
end

function float_to_bits(value::Float64, ::Type{Float32})
    reinterpret(UInt32, Float32(value))
end

function float_to_bits(value::Float64, ::Type{Float64})
    reinterpret(UInt64, value)
end

# For TFloat32, use Float32 representation
function float_to_bits(value::Float64, ::Type{T}) where T
    # Fallback to Float32 for special types like TFloat32
    reinterpret(UInt32, Float32(value))
end

# Float8 types (from DLFP8Types)
function float_to_bits(value::Float64, ::Type{Float8_E4M3FN})
    reinterpret(UInt8, Float8_E4M3FN(value))
end

function float_to_bits(value::Float64, ::Type{Float8_E5M2})
    reinterpret(UInt8, Float8_E5M2(value))
end

"""
    encode_signed_varint!(buf, value)

Encode a signed integer as a variable-length integer.
Uses zigzag encoding for signed values.
"""
function encode_signed_varint!(buf::Vector{UInt8}, value::Union{UInt8, UInt16, UInt32, UInt64, Int64})
    # For float bits, encode as unsigned varint
    encode_varint!(buf, UInt64(value))
end

"""
    encode_identity_array!(cb, identities)

Encode an array of reduce identity attributes.
"""
function encode_identity_array!(cb::CodeBuilder, identities::Vector{<:ReduceIdentity})
    encode_varint!(cb.buf, length(identities))
    for identity in identities
        encode_tagged_float!(cb, identity)
    end
end

"""
    BytecodeWriter

Top-level writer for bytecode files.
"""
mutable struct BytecodeWriter
    buf::Vector{UInt8}
    string_table::StringTable
    constant_table::ConstantTable
    type_table::TypeTable
    debug_attr_table::DebugAttrTable
    debug_info::Vector{Vector{DebugAttrId}}
    num_functions::Int
end

function BytecodeWriter()
    string_table = StringTable()
    BytecodeWriter(
        UInt8[],
        string_table,
        ConstantTable(),
        TypeTable(),
        DebugAttrTable(string_table),
        Vector{Vector{DebugAttrId}}[],
        0
    )
end

"""
Write the bytecode header.
"""
function write_header!(buf::Vector{UInt8})
    append!(buf, MAGIC)
    major, minor, tag = BYTECODE_VERSION
    push!(buf, UInt8(major))
    push!(buf, UInt8(minor))
    # Tag as 2-byte little-endian
    push!(buf, UInt8(tag & 0xff))
    push!(buf, UInt8((tag >> 8) & 0xff))
end

"""
Write a section with optional alignment.
"""
function write_section!(buf::Vector{UInt8}, section_id::UInt8, content::Vector{UInt8}, alignment::Int=1)
    has_alignment = alignment > 1
    push!(buf, section_id | (has_alignment ? 0x80 : 0x00))
    encode_varint!(buf, length(content))
    if has_alignment
        encode_varint!(buf, alignment)
        pad_to!(buf, alignment)
    end
    append!(buf, content)
end

"""
Write a table (string, type, or constant table).
Format: [count] [offsets...] [data...]
"""
function write_table!(buf::Vector{UInt8}, table_items, index_size::Int)
    encode_varint!(buf, length(table_items))

    # Compute offsets
    pad_to!(buf, index_size)
    offset = 0
    for (encoded, _) in table_items
        for i in 0:index_size-1
            push!(buf, UInt8((offset >> (8*i)) & 0xff))
        end
        offset += length(encoded)
    end

    # Write data
    for (encoded, _) in table_items
        append!(buf, encoded)
    end
end

"""
Write debug info section.
"""
function write_debug_section!(buf::Vector{UInt8}, debug_info::Vector{Vector{DebugAttrId}},
                              debug_attr_table::DebugAttrTable)
    section_buf = UInt8[]

    # Number of functions with debug info
    encode_varint!(section_buf, length(debug_info))

    # Function offsets into index array
    pad_to!(section_buf, 4)
    index_offset = 0
    for func_info in debug_info
        for i in 0:3
            push!(section_buf, UInt8((index_offset >> (8*i)) & 0xff))
        end
        index_offset += length(func_info)
    end

    # Total number of debug attr indices
    encode_varint!(section_buf, index_offset)

    # Debug attr IDs per operation (8-byte each)
    pad_to!(section_buf, 8)
    for func_info in debug_info
        for attr in func_info
            for i in 0:7
                push!(section_buf, UInt8((attr.id >> (8*i)) & 0xff))
            end
        end
    end

    # Write debug attribute table
    # Workaround: empty table needs at least one entry
    if length(debug_attr_table) == 0
        debug_attr_table[UInt8[0x00]]
    end
    write_table!(section_buf, items(debug_attr_table), 4)

    write_section!(buf, Section.Debug, section_buf, 8)
end

"""
Write complete bytecode to a buffer.
Returns the buffer with all sections.
"""
function write_bytecode!(f::Function, num_functions::Int)
    writer = BytecodeWriter()

    # Function section content
    func_buf = UInt8[]
    encode_varint!(func_buf, num_functions)

    # Let user build functions
    f(writer, func_buf)

    @assert writer.num_functions == num_functions "Expected $num_functions functions, got $(writer.num_functions)"

    # Build final output
    buf = UInt8[]
    write_header!(buf)

    # Sections in order: Func, Global (if any), Constant, Debug, Type, String, End
    write_section!(buf, Section.Func, func_buf, 8)

    # Global section (skip if empty)

    # Constant section
    const_buf = UInt8[]
    write_table!(const_buf, items(writer.constant_table), 8)
    write_section!(buf, Section.Constant, const_buf, 8)

    # Debug section
    write_debug_section!(buf, writer.debug_info, writer.debug_attr_table)

    # Type section
    type_buf = UInt8[]
    write_table!(type_buf, items(writer.type_table), 4)
    write_section!(buf, Section.Type, type_buf, 4)

    # String section
    str_buf = UInt8[]
    write_table!(str_buf, items(writer.string_table), 4)
    write_section!(buf, Section.String, str_buf, 4)

    # End marker
    push!(buf, Section.EndOfBytecode)

    return buf
end

"""
Add a function to the bytecode.
"""
function add_function!(writer::BytecodeWriter, func_buf::Vector{UInt8},
                       name::String,
                       param_types::Vector{TypeId},
                       result_types::Vector{TypeId};
                       is_entry::Bool=false,
                       entry_hints::Union{Nothing, Vector{UInt8}}=nothing,
                       func_debug_attr::DebugAttrId=DebugAttrId(0))
    writer.num_functions += 1

    # Function name
    name_id = writer.string_table[name]
    encode_varint!(func_buf, name_id.id)

    # Function signature type
    sig_type = function_type!(writer.type_table, param_types, result_types)
    encode_typeid!(func_buf, sig_type)

    # Entry flags
    flags = 0x00
    if is_entry
        flags |= 0x02
        if entry_hints !== nothing
            flags |= 0x04
        end
    end
    push!(func_buf, UInt8(flags))

    # Debug info: initialize with function-level debug attr
    # Operations will append their attrs to this list
    push!(writer.debug_info, [func_debug_attr])
    encode_varint!(func_buf, length(writer.debug_info))

    # Entry hints if present
    if is_entry && entry_hints !== nothing
        append!(func_buf, entry_hints)
    end

    # Create code builder for function body
    cb = CodeBuilder(writer.string_table, writer.constant_table, writer.type_table)

    return cb
end

"""
Finalize a function's code and append to func_buf.
"""
function finalize_function!(func_buf::Vector{UInt8}, cb::CodeBuilder,
                            debug_info::Vector{Vector{DebugAttrId}})
    # Append operation debug attrs to the function's debug info list
    # (which already contains the function-level debug attr)
    append!(debug_info[end], cb.debug_attrs)

    # Encode code length and append
    encode_varint!(func_buf, length(cb.buf))
    append!(func_buf, cb.buf)
end

#=============================================================================
 Optimization Hints
=============================================================================#

"""
    encode_tagged_value!(cb, value)

Encode a value with its type tag.
"""
function encode_tagged_value!(buf::Vector{UInt8}, type_table::TypeTable, value::Bool)
    push!(buf, AttributeTag.Bool)
    push!(buf, value)
end

function encode_tagged_value!(buf::Vector{UInt8}, type_table::TypeTable, value::Integer)
    push!(buf, AttributeTag.Integer)
    encode_typeid!(buf, I32(type_table))
    encode_varint!(buf, UInt32(value))
end

"""
Optimization hints for load/store operations.
- `latency`: Optional latency hint (1-10), or nothing for default
- `allow_tma`: Whether TMA (Tensor Memory Accelerator) is allowed (default: true)
"""
@kwdef struct LoadStoreHints
    latency::Union{Int, Nothing} = nothing
    allow_tma::Bool = true
end

"""
Optimization hints for load/store operations.
- `hints_by_arch`: List of (SM architecture, load/store hints) pairs
"""
struct OptimizationHints
    hints_by_arch::Vector{Tuple{String, LoadStoreHints}}
end

function make_load_store_hints(sm_arch::Union{String, Nothing}, hints::LoadStoreHints)
    isnothing(sm_arch) && throw(ArgumentError("sm_arch must be explicitly passed when load/store hints are present"))
    OptimizationHints([(sm_arch, hints)])
end

function encode_opattr_optimization_hints!(cb::CodeBuilder, hints::OptimizationHints)
    # Outer dictionary: arch -> hints_dict
    encode_varint!(cb.buf, length(hints.hints_by_arch))
    for (arch, load_store_hints) in hints.hints_by_arch
        arch_id = cb.string_table[arch]
        encode_varint!(cb.buf, arch_id.id)
        # Encode hints as inner dictionary (tagged)
        encode_load_store_hints_dict!(cb, load_store_hints)
    end
end

function encode_load_store_hints_dict!(cb::CodeBuilder, hints::LoadStoreHints)
    # Build list of (key, value) pairs for non-default hints
    items = Tuple{String, Any}[]
    hints.allow_tma || push!(items, ("allow_tma", false))
    isnothing(hints.latency) || push!(items, ("latency", hints.latency))

    # Encode dictionary
    push!(cb.buf, AttributeTag.Dictionary)
    encode_varint!(cb.buf, length(items))
    for (key, value) in items
        key_id = cb.string_table[key]
        encode_varint!(cb.buf, key_id.id)
        encode_tagged_value!(cb.buf, cb.type_table, value)
    end
end

"""
Kernel-level compilation hints (num_ctas, occupancy).
Encoded as a dictionary attribute in bytecode.
"""
@kwdef struct EntryHints
    num_ctas::Union{Int, Nothing} = nothing    # 1, 2, 4, 8, 16
    occupancy::Union{Int, Nothing} = nothing   # 1-32
end

function validate_num_ctas(num_ctas::Union{Int, Nothing})
    isnothing(num_ctas) && return
    1 <= num_ctas <= 16 || throw(ArgumentError("num_ctas must be between 1 and 16, got $num_ctas"))
    ispow2(num_ctas) || throw(ArgumentError("num_ctas must be a power of 2, got $num_ctas"))
end

function validate_occupancy(occupancy::Union{Int, Nothing})
    isnothing(occupancy) && return
    1 <= occupancy <= 32 || throw(ArgumentError("occupancy must be between 1 and 32, got $occupancy"))
end

function encode_entry_hints(writer::BytecodeWriter, sm_arch::Union{String, Nothing}, hints::EntryHints)
    validate_num_ctas(hints.num_ctas)
    validate_occupancy(hints.occupancy)

    # Build items list (only non-nothing values)
    items = Tuple{String, Int}[]
    isnothing(hints.num_ctas) || push!(items, ("num_cta_in_cga", hints.num_ctas))
    isnothing(hints.occupancy) || push!(items, ("occupancy", hints.occupancy))
    isempty(items) && return nothing

    # Use default architecture if not specified and hints are present
    arch = @something sm_arch throw(ArgumentError("sm_arch must be specified when entry hints are present"))

    buf = UInt8[]

    # Start with OptimizationHints tag
    push!(buf, AttributeTag.OptimizationHints)

    # Encode as architecture-specific dictionary
    # Format: num_archs, then for each arch: arch_id, dictionary
    encode_varint!(buf, 1)  # 1 architecture

    # Architecture string ID
    arch_id = writer.string_table[arch]
    encode_varint!(buf, arch_id.id)

    # Encode dictionary
    push!(buf, AttributeTag.Dictionary)
    encode_varint!(buf, length(items))
    for (key, value) in items
        key_id = writer.string_table[key]
        encode_varint!(buf, key_id.id)
        encode_tagged_value!(buf, writer.type_table, value)
    end

    return buf
end
