# Operation encoders for Tile IR bytecode
# Each function encodes a specific operation into the code builder's buffer

# Opcode constants (from Python's encodings.py)
module Opcode
    const AbsFOp = 0
    const AbsIOp = 1
    const AddFOp = 2
    const AddIOp = 3
    const AndIOp = 4
    const AssertOp = 5
    const AssumeOp = 6
    const AtomicCASPtrOp = 7
    const AtomicRMWPtrOp = 8
    const BitcastOp = 9
    const BreakOp = 10
    const BroadcastOp = 11
    const CatOp = 12
    const CeilOp = 13
    const CmpFOp = 14
    const CmpIOp = 15
    const ConstantOp = 16
    const ContinueOp = 17
    const CosOp = 18
    const CosHOp = 19
    const DivFOp = 20
    const DivIOp = 21
    const EntryOp = 22
    const ExpOp = 23
    const Exp2Op = 24
    const ExtIOp = 37
    const ExtractOp = 38
    const FloorOp = 39
    const FmaOp = 40
    const ForOp = 41
    const FToFOp = 42
    const FToIOp = 43
    const GetGlobalOp = 44
    const GetIndexSpaceShapeOp = 45
    const GetNumTileBlocksOp = 46
    const GetTensorShapeOp = 47
    const GetTileBlockIdOp = 48
    const GlobalOp = 49
    const IfOp = 50
    const IntToPtrOp = 51
    const IotaOp = 58
    const IToFOp = 59
    const JoinTokensOp = 60
    const LoadPtrTkoOp = 61
    const LoadViewTkoOp = 62
    const LogOp = 63
    const Log2Op = 64
    const LoopOp = 65
    const MakePartitionViewOp = 66
    const MakeTensorViewOp = 67
    const MakeTokenOp = 68
    const MaxFOp = 69
    const MaxIOp = 70
    const MinFOp = 71
    const MinIOp = 72
    const MmaFOp = 73
    const MmaIOp = 74
    const ModuleOp = 75
    const MulFOp = 76
    const MulhiIOp = 77
    const MulIOp = 78
    const NegFOp = 79
    const NegIOp = 80
    const OffsetOp = 81
    const OrIOp = 82
    const PermuteOp = 83
    const PowOp = 84
    const PrintOp = 85
    const PtrToIntOp = 86
    const PtrToPtrOp = 87
    const ReduceOp = 88
    const RemFOp = 89
    const RemIOp = 90
    const ReshapeOp = 91
    const ReturnOp = 92
    const RsqrtOp = 93
    const ScanOp = 94
    const SelectOp = 95
    const ShLIOp = 96
    const ShRIOp = 97
    const SinOp = 98
    const SinHOp = 99
    const SqrtOp = 100
    const StorePtrTkoOp = 101
    const StoreViewTkoOp = 102
    const SubFOp = 103
    const SubIOp = 104
    const TanOp = 105
    const TanHOp = 106
    const TruncIOp = 107
    const XOrIOp = 108
    const YieldOp = 109
end

# Enums for operation attributes

@enum RoundingMode begin
    RoundingNearestEven = 0
    RoundingZero = 1
    RoundingNegativeInf = 2
    RoundingPositiveInf = 3
    RoundingApprox = 4
    RoundingFull = 5
    RoundingNearestIntToZero = 6
end

@enum IntegerOverflow begin
    OverflowNone = 0
    OverflowNSW = 1
    OverflowNUW = 2
    OverflowNW = 3
end

@enum MemoryOrderingSemantics begin
    MemoryWeak = 0
    MemoryRelaxed = 1
    MemoryAcquire = 2
    MemoryRelease = 3
    MemoryAcqRel = 4
end

@enum MemoryScope begin
    ScopeTLBlock = 0
    ScopeDevice = 1
    ScopeSystem = 2
end

@enum Signedness begin
    SignednessUnsigned = 0
    SignednessSigned = 1
end

@enum ComparisonPredicate begin
    CmpEqual = 0
    CmpNotEqual = 1
    CmpLessThan = 2
    CmpLessThanOrEqual = 3
    CmpGreaterThan = 4
    CmpGreaterThanOrEqual = 5
end

@enum ComparisonOrdering begin
    CmpUnordered = 0
    CmpOrdered = 1
end

@enum AtomicRMWMode begin
    AtomicAND = 0x00
    AtomicOR = 0x01
    AtomicXOR = 0x02
    AtomicADD = 0x03
    AtomicADDF = 0x04
    AtomicMAX = 0x05
    AtomicMIN = 0x06
    AtomicUMAX = 0x07
    AtomicUMIN = 0x08
    AtomicXCHG = 0x09
end

# Helper to encode enum as single byte
function encode_enum!(buf::Vector{UInt8}, e::Enum)
    push!(buf, UInt8(Int(e)))
end

#=============================================================================
 Attribute Tags (for tagged attributes like AssumePredicate)
=============================================================================#

module AttributeTag
    const Integer = 0x01
    const Float = 0x02
    const Bool = 0x03
    const Type = 0x04
    const String = 0x05
    const Array = 0x06
    const DenseElements = 0x07
    const DivBy = 0x08
    const SameElements = 0x09
    const Dictionary = 0x0a
    const OptimizationHints = 0x0b
    const Bounded = 0x0c
end

#=============================================================================
 AssumePredicate types for AssumeOp
=============================================================================#

"""
    AssumePredicate

Abstract type for assume predicates used with AssumeOp.
"""
abstract type AssumePredicate end

"""
    DivBy(divisor; every=nothing, along=nothing)

Predicate asserting a value is divisible by `divisor`.
Optional `every` and `along` specify dimensional constraints.
"""
struct DivBy <: AssumePredicate
    divisor::Int
    every::Union{Int, Nothing}
    along::Union{Int, Nothing}
end
DivBy(divisor::Int) = DivBy(divisor, nothing, nothing)

"""
    Bounded(; lb=nothing, ub=nothing)

Predicate asserting a value is within bounds [lb, ub].
Either bound can be nothing (unbounded).
"""
struct Bounded <: AssumePredicate
    lb::Union{Int, Nothing}
    ub::Union{Int, Nothing}
end
Bounded(; lb=nothing, ub=nothing) = Bounded(lb, ub)

"""
    SameElements(values)

Predicate asserting all elements of a tile have the same value from `values`.
Used for broadcast/splat optimization hints.
"""
struct SameElements <: AssumePredicate
    values::Vector{Int}
end

"""
    encode_assume_predicate!(cb, pred::AssumePredicate)

Encode an assume predicate as a tagged attribute.
"""
function encode_assume_predicate!(cb::CodeBuilder, pred::DivBy)
    push!(cb.buf, AttributeTag.DivBy)
    encode_varint!(cb.buf, pred.divisor)
    flags = (pred.every !== nothing ? 0x01 : 0x00) |
            (pred.along !== nothing ? 0x02 : 0x00)
    push!(cb.buf, flags)
    if pred.every !== nothing
        encode_signed_varint!(cb.buf, pred.every)
    end
    if pred.along !== nothing
        encode_signed_varint!(cb.buf, pred.along)
    end
end

function encode_assume_predicate!(cb::CodeBuilder, pred::Bounded)
    push!(cb.buf, AttributeTag.Bounded)
    flags = (pred.lb !== nothing ? 0x01 : 0x00) |
            (pred.ub !== nothing ? 0x02 : 0x00)
    push!(cb.buf, flags)
    if pred.lb !== nothing
        encode_signed_varint!(cb.buf, pred.lb)
    end
    if pred.ub !== nothing
        encode_signed_varint!(cb.buf, pred.ub)
    end
end

function encode_assume_predicate!(cb::CodeBuilder, pred::SameElements)
    push!(cb.buf, AttributeTag.SameElements)
    encode_varint!(cb.buf, length(pred.values))
    for v in pred.values
        encode_signed_varint!(cb.buf, v)
    end
end

#=============================================================================
 Essential operations for vadd
=============================================================================#

"""
    encode_ConstantOp!(cb, result_type, value_bytes) -> Value

Create a constant value.
Opcode: 16
"""
function encode_ConstantOp!(cb::CodeBuilder, result_type::TypeId, value_bytes::Vector{UInt8})
    encode_varint!(cb.buf, Opcode.ConstantOp)
    encode_typeid!(cb.buf, result_type)
    encode_opattr_dense!(cb, value_bytes)
    return new_op!(cb)
end

"""
    encode_AssumeOp!(cb, result_type, value, predicate) -> Value

Create an assume operation that annotates a value with a predicate.
The predicate tells the compiler it can assume certain properties about
the value (e.g., divisibility, bounds) for optimization.

Opcode: 6
"""
function encode_AssumeOp!(cb::CodeBuilder, result_type::TypeId, value::Value, predicate::AssumePredicate)
    encode_varint!(cb.buf, Opcode.AssumeOp)
    encode_typeid!(cb.buf, result_type)
    encode_assume_predicate!(cb, predicate)
    encode_operand!(cb.buf, value)
    return new_op!(cb)
end

"""
    encode_GetTileBlockIdOp!(cb, x_type, y_type, z_type) -> (Value, Value, Value)

Get the 3D block ID.
Opcode: 48
"""
function encode_GetTileBlockIdOp!(cb::CodeBuilder, x_type::TypeId, y_type::TypeId, z_type::TypeId)
    encode_varint!(cb.buf, Opcode.GetTileBlockIdOp)
    encode_typeid!(cb.buf, x_type)
    encode_typeid!(cb.buf, y_type)
    encode_typeid!(cb.buf, z_type)
    return new_op!(cb, 3)
end

"""
    encode_GetNumTileBlocksOp!(cb, x_type, y_type, z_type) -> (Value, Value, Value)

Get the 3D grid dimensions.
Opcode: 49
"""
function encode_GetNumTileBlocksOp!(cb::CodeBuilder, x_type::TypeId, y_type::TypeId, z_type::TypeId)
    encode_varint!(cb.buf, Opcode.GetNumTileBlocksOp)
    encode_typeid!(cb.buf, x_type)
    encode_typeid!(cb.buf, y_type)
    encode_typeid!(cb.buf, z_type)
    return new_op!(cb, 3)
end

"""
    encode_MakeTensorViewOp!(cb, result_type, base, dynamic_shape, dynamic_strides) -> Value

Create a tensor view from a base pointer and dynamic shape/strides.
Opcode: 67
"""
function encode_MakeTensorViewOp!(cb::CodeBuilder, result_type::TypeId,
                                   base::Value,
                                   dynamic_shape::Vector{Value},
                                   dynamic_strides::Vector{Value})
    encode_varint!(cb.buf, Opcode.MakeTensorViewOp)
    # Variadic result types (just one)
    encode_typeid_seq!(cb.buf, [result_type])
    # Operands
    encode_operand!(cb.buf, base)
    encode_sized_operands!(cb.buf, dynamic_shape)
    encode_sized_operands!(cb.buf, dynamic_strides)
    return new_op!(cb)
end

"""
    encode_MakePartitionViewOp!(cb, result_type, tensor_view) -> Value

Create a partition view from a tensor view.
Opcode: 66
"""
function encode_MakePartitionViewOp!(cb::CodeBuilder, result_type::TypeId, tensor_view::Value)
    encode_varint!(cb.buf, Opcode.MakePartitionViewOp)
    encode_typeid!(cb.buf, result_type)
    encode_operand!(cb.buf, tensor_view)
    return new_op!(cb)
end

"""
    encode_GetIndexSpaceShapeOp!(cb, result_types, partition_view) -> Tuple{Value...}

Get the shape of the index space of a partition view.
Returns one value per dimension.
Opcode: 45
"""
function encode_GetIndexSpaceShapeOp!(cb::CodeBuilder, result_types::AbstractVector{TypeId}, partition_view::Value)
    encode_varint!(cb.buf, Opcode.GetIndexSpaceShapeOp)
    encode_typeid_seq!(cb.buf, result_types)
    encode_operand!(cb.buf, partition_view)
    return new_op!(cb, length(result_types))
end

"""
    encode_MakeTokenOp!(cb, result_type) -> Value

Create an initial memory ordering token.
Opcode: 68
"""
function encode_MakeTokenOp!(cb::CodeBuilder, result_type::TypeId)
    encode_varint!(cb.buf, Opcode.MakeTokenOp)
    encode_typeid!(cb.buf, result_type)
    return new_op!(cb)
end

"""
    encode_JoinTokensOp!(cb, result_type, tokens) -> Value

Join multiple tokens into a single token for synchronization.
Opcode: 60
"""
function encode_JoinTokensOp!(cb::CodeBuilder, result_type::TypeId, tokens::Vector{Value})
    encode_varint!(cb.buf, Opcode.JoinTokensOp)
    encode_typeid_seq!(cb.buf, [result_type])
    encode_varint!(cb.buf, length(tokens))
    for tok in tokens
        encode_operand!(cb.buf, tok)
    end
    return new_op!(cb)
end

"""
    encode_LoadViewTkoOp!(cb, tile_type, token_type, view, index; kwargs...) -> (Value, Value)

Load a tile from a partition view with token ordering.
Opcode: 62
"""
function encode_LoadViewTkoOp!(cb::CodeBuilder,
                               tile_type::TypeId,
                               token_type::TypeId,
                               view::Value,
                               index::Vector{Value};
                               token::Union{Value, Nothing}=nothing,
                               memory_ordering::MemoryOrderingSemantics=MemoryWeak,
                               memory_scope::Union{MemoryScope, Nothing}=nothing,
                               optimization_hints::Union{OptimizationHints, Nothing}=nothing)
    encode_varint!(cb.buf, Opcode.LoadViewTkoOp)
    # Variadic result types
    encode_typeid_seq!(cb.buf, [tile_type, token_type])

    # Flags
    flags = 0
    if memory_scope !== nothing
        flags |= 1
    end
    if optimization_hints !== nothing
        flags |= 2
    end
    if token !== nothing
        flags |= 4
    end
    encode_varint!(cb.buf, flags)

    # Attributes
    encode_enum!(cb.buf, memory_ordering)
    if memory_scope !== nothing
        encode_enum!(cb.buf, memory_scope)
    end
    if optimization_hints !== nothing
        encode_opattr_optimization_hints!(cb, optimization_hints)
    end

    # Operands
    encode_operand!(cb.buf, view)
    encode_sized_operands!(cb.buf, index)
    encode_optional_operand!(cb.buf, token)

    return new_op!(cb, 2)
end

"""
    encode_StoreViewTkoOp!(cb, token_type, tile, view, index; kwargs...) -> Value

Store a tile to a partition view with token ordering.
Opcode: 102
"""
function encode_StoreViewTkoOp!(cb::CodeBuilder,
                                token_type::TypeId,
                                tile::Value,
                                view::Value,
                                index::Vector{Value};
                                token::Union{Value, Nothing}=nothing,
                                memory_ordering::MemoryOrderingSemantics=MemoryWeak,
                                memory_scope::Union{MemoryScope, Nothing}=nothing,
                                optimization_hints::Union{OptimizationHints, Nothing}=nothing)
    encode_varint!(cb.buf, Opcode.StoreViewTkoOp)
    # Variadic result types (just token)
    encode_typeid_seq!(cb.buf, [token_type])

    # Flags
    flags = 0
    if memory_scope !== nothing
        flags |= 1
    end
    if optimization_hints !== nothing
        flags |= 2
    end
    if token !== nothing
        flags |= 4
    end
    encode_varint!(cb.buf, flags)

    # Attributes
    encode_enum!(cb.buf, memory_ordering)
    if memory_scope !== nothing
        encode_enum!(cb.buf, memory_scope)
    end
    if optimization_hints !== nothing
        encode_opattr_optimization_hints!(cb, optimization_hints)
    end

    # Operands
    encode_operand!(cb.buf, tile)
    encode_operand!(cb.buf, view)
    encode_sized_operands!(cb.buf, index)
    encode_optional_operand!(cb.buf, token)

    return new_op!(cb)
end

"""
    encode_OffsetOp!(cb, result_type, ptr, offset) -> Value

Compute pointer offset: ptr + offset.
Opcode: 81

The offset is in elements (not bytes). Broadcasting applies to ptr and offset.
"""
function encode_OffsetOp!(cb::CodeBuilder, result_type::TypeId, ptr::Value, offset::Value)
    encode_varint!(cb.buf, Opcode.OffsetOp)
    encode_typeid!(cb.buf, result_type)
    encode_operand!(cb.buf, ptr)
    encode_operand!(cb.buf, offset)
    return new_op!(cb)
end

"""
    encode_LoadPtrTkoOp!(cb, result_type, token_type, source; kwargs...) -> (Value, Value)

Load from pointer tile with optional masking and token ordering.
Opcode: 61

Returns (loaded_tile, new_token) tuple.
"""
function encode_LoadPtrTkoOp!(cb::CodeBuilder,
                              result_type::TypeId,
                              token_type::TypeId,
                              source::Value;
                              mask::Union{Value, Nothing}=nothing,
                              padding_value::Union{Value, Nothing}=nothing,
                              token::Union{Value, Nothing}=nothing,
                              memory_ordering::MemoryOrderingSemantics=MemoryWeak,
                              memory_scope::Union{MemoryScope, Nothing}=nothing,
                              optimization_hints::Union{OptimizationHints, Nothing}=nothing)
    encode_varint!(cb.buf, Opcode.LoadPtrTkoOp)
    # Result types
    encode_typeid!(cb.buf, result_type)
    encode_typeid!(cb.buf, token_type)

    # Flags
    flags = 0
    if memory_scope !== nothing
        flags |= 1
    end
    if optimization_hints !== nothing
        flags |= 2
    end
    if mask !== nothing
        flags |= 4
    end
    if padding_value !== nothing
        flags |= 8
    end
    if token !== nothing
        flags |= 16
    end
    encode_varint!(cb.buf, flags)

    # Attributes
    encode_enum!(cb.buf, memory_ordering)
    if memory_scope !== nothing
        encode_enum!(cb.buf, memory_scope)
    end
    if optimization_hints !== nothing
        encode_opattr_optimization_hints!(cb, optimization_hints)
    end

    # Operands
    encode_operand!(cb.buf, source)
    encode_optional_operand!(cb.buf, mask)
    encode_optional_operand!(cb.buf, padding_value)
    encode_optional_operand!(cb.buf, token)

    return new_op!(cb, 2)
end

"""
    encode_StorePtrTkoOp!(cb, token_type, destination, value; kwargs...) -> Value

Store to pointer tile with optional masking and token ordering.
Opcode: 101

Returns new_token.
"""
function encode_StorePtrTkoOp!(cb::CodeBuilder,
                               token_type::TypeId,
                               destination::Value,
                               value::Value;
                               mask::Union{Value, Nothing}=nothing,
                               token::Union{Value, Nothing}=nothing,
                               memory_ordering::MemoryOrderingSemantics=MemoryWeak,
                               memory_scope::Union{MemoryScope, Nothing}=nothing,
                               optimization_hints::Union{OptimizationHints, Nothing}=nothing)
    encode_varint!(cb.buf, Opcode.StorePtrTkoOp)
    # Result type (token)
    encode_typeid!(cb.buf, token_type)

    # Flags
    flags = 0
    if memory_scope !== nothing
        flags |= 1
    end
    if optimization_hints !== nothing
        flags |= 2
    end
    if mask !== nothing
        flags |= 4
    end
    if token !== nothing
        flags |= 8
    end
    encode_varint!(cb.buf, flags)

    # Attributes
    encode_enum!(cb.buf, memory_ordering)
    if memory_scope !== nothing
        encode_enum!(cb.buf, memory_scope)
    end
    if optimization_hints !== nothing
        encode_opattr_optimization_hints!(cb, optimization_hints)
    end

    # Operands
    encode_operand!(cb.buf, destination)
    encode_operand!(cb.buf, value)
    encode_optional_operand!(cb.buf, mask)
    encode_optional_operand!(cb.buf, token)

    return new_op!(cb)
end

#=============================================================================
 Arithmetic operations
=============================================================================#

"""
    encode_AddFOp!(cb, result_type, lhs, rhs; kwargs...) -> Value

Float addition.
Opcode: 2
"""
function encode_AddFOp!(cb::CodeBuilder, result_type::TypeId, lhs::Value, rhs::Value;
                        rounding_mode::RoundingMode=RoundingNearestEven,
                        flush_to_zero::Bool=false)
    encode_varint!(cb.buf, Opcode.AddFOp)
    encode_typeid!(cb.buf, result_type)
    encode_varint!(cb.buf, flush_to_zero ? 1 : 0)
    encode_enum!(cb.buf, rounding_mode)
    encode_operand!(cb.buf, lhs)
    encode_operand!(cb.buf, rhs)
    return new_op!(cb)
end

"""
    encode_AddIOp!(cb, result_type, lhs, rhs; overflow) -> Value

Integer addition.
Opcode: 3
"""
function encode_AddIOp!(cb::CodeBuilder, result_type::TypeId, lhs::Value, rhs::Value;
                        overflow::IntegerOverflow=OverflowNone)
    encode_varint!(cb.buf, Opcode.AddIOp)
    encode_typeid!(cb.buf, result_type)
    encode_enum!(cb.buf, overflow)
    encode_operand!(cb.buf, lhs)
    encode_operand!(cb.buf, rhs)
    return new_op!(cb)
end

"""
    encode_SubFOp!(cb, result_type, lhs, rhs; kwargs...) -> Value

Float subtraction.
Opcode: 103
"""
function encode_SubFOp!(cb::CodeBuilder, result_type::TypeId, lhs::Value, rhs::Value;
                        rounding_mode::RoundingMode=RoundingNearestEven,
                        flush_to_zero::Bool=false)
    encode_varint!(cb.buf, Opcode.SubFOp)
    encode_typeid!(cb.buf, result_type)
    encode_varint!(cb.buf, flush_to_zero ? 1 : 0)
    encode_enum!(cb.buf, rounding_mode)
    encode_operand!(cb.buf, lhs)
    encode_operand!(cb.buf, rhs)
    return new_op!(cb)
end

"""
    encode_SubIOp!(cb, result_type, lhs, rhs; overflow) -> Value

Integer subtraction.
Opcode: 104
"""
function encode_SubIOp!(cb::CodeBuilder, result_type::TypeId, lhs::Value, rhs::Value;
                        overflow::IntegerOverflow=OverflowNone)
    encode_varint!(cb.buf, Opcode.SubIOp)
    encode_typeid!(cb.buf, result_type)
    encode_enum!(cb.buf, overflow)
    encode_operand!(cb.buf, lhs)
    encode_operand!(cb.buf, rhs)
    return new_op!(cb)
end

"""
    encode_MulFOp!(cb, result_type, lhs, rhs; kwargs...) -> Value

Float multiplication.
Opcode: 73
"""
function encode_MulFOp!(cb::CodeBuilder, result_type::TypeId, lhs::Value, rhs::Value;
                        rounding_mode::RoundingMode=RoundingNearestEven,
                        flush_to_zero::Bool=false)
    encode_varint!(cb.buf, Opcode.MulFOp)
    encode_typeid!(cb.buf, result_type)
    encode_varint!(cb.buf, flush_to_zero ? 1 : 0)
    encode_enum!(cb.buf, rounding_mode)
    encode_operand!(cb.buf, lhs)
    encode_operand!(cb.buf, rhs)
    return new_op!(cb)
end

"""
    encode_MulIOp!(cb, result_type, lhs, rhs; overflow) -> Value

Integer multiplication.
Opcode: 74
"""
function encode_MulIOp!(cb::CodeBuilder, result_type::TypeId, lhs::Value, rhs::Value;
                        overflow::IntegerOverflow=OverflowNone)
    encode_varint!(cb.buf, Opcode.MulIOp)
    encode_typeid!(cb.buf, result_type)
    encode_enum!(cb.buf, overflow)
    encode_operand!(cb.buf, lhs)
    encode_operand!(cb.buf, rhs)
    return new_op!(cb)
end

"""
    encode_PowOp!(cb, result_type, base, exponent) -> Value

Floating-point power operation (base^exponent).
Opcode: 84
"""
function encode_PowOp!(cb::CodeBuilder, result_type::TypeId, base::Value, exponent::Value)
    encode_varint!(cb.buf, Opcode.PowOp)
    encode_typeid!(cb.buf, result_type)
    encode_operand!(cb.buf, base)
    encode_operand!(cb.buf, exponent)
    return new_op!(cb)
end

"""
    encode_TruncIOp!(cb, result_type, source; overflow) -> Value

Truncate integer to smaller width.
Opcode: 106
"""
function encode_TruncIOp!(cb::CodeBuilder, result_type::TypeId, source::Value;
                          overflow::IntegerOverflow=OverflowNone)
    encode_varint!(cb.buf, Opcode.TruncIOp)
    encode_typeid!(cb.buf, result_type)
    encode_enum!(cb.buf, overflow)
    encode_operand!(cb.buf, source)
    return new_op!(cb)
end

"""
    encode_DivFOp!(cb, result_type, lhs, rhs; rounding_mode, flush_to_zero) -> Value

Floating-point division.
Opcode: 20
"""
function encode_DivFOp!(cb::CodeBuilder, result_type::TypeId, lhs::Value, rhs::Value;
                        rounding_mode::RoundingMode=RoundingNearestEven,
                        flush_to_zero::Bool=false)
    encode_varint!(cb.buf, Opcode.DivFOp)
    encode_typeid!(cb.buf, result_type)
    encode_varint!(cb.buf, flush_to_zero ? 1 : 0)
    encode_enum!(cb.buf, rounding_mode)
    encode_operand!(cb.buf, lhs)
    encode_operand!(cb.buf, rhs)
    return new_op!(cb)
end

"""
    encode_SqrtOp!(cb, result_type, source; rounding_mode, flush_to_zero) -> Value

Square root operation.
Opcode: 100
"""
function encode_SqrtOp!(cb::CodeBuilder, result_type::TypeId, source::Value;
                        rounding_mode::RoundingMode=RoundingNearestEven,
                        flush_to_zero::Bool=false)
    encode_varint!(cb.buf, Opcode.SqrtOp)
    encode_typeid!(cb.buf, result_type)
    encode_varint!(cb.buf, flush_to_zero ? 1 : 0)
    encode_enum!(cb.buf, rounding_mode)
    encode_operand!(cb.buf, source)
    return new_op!(cb)
end

"""
    encode_RSqrtOp!(cb, result_type, source; flush_to_zero) -> Value

Reciprocal square root operation.
Opcode: 93
"""
function encode_RSqrtOp!(cb::CodeBuilder, result_type::TypeId, source::Value;
                         flush_to_zero::Bool=false)
    encode_varint!(cb.buf, Opcode.RsqrtOp)
    encode_typeid!(cb.buf, result_type)
    encode_varint!(cb.buf, flush_to_zero ? 1 : 0)
    encode_operand!(cb.buf, source)
    return new_op!(cb)
end

"""
    encode_ExpOp!(cb, result_type, source) -> Value

Natural exponential (e^x) operation.
Opcode: 23
"""
function encode_ExpOp!(cb::CodeBuilder, result_type::TypeId, source::Value)
    encode_varint!(cb.buf, Opcode.ExpOp)
    encode_typeid!(cb.buf, result_type)
    encode_operand!(cb.buf, source)
    return new_op!(cb)
end

"""
    encode_Exp2Op!(cb, result_type, source; flush_to_zero) -> Value

Base-2 exponential (2^x) operation.
Opcode: 24
"""
function encode_Exp2Op!(cb::CodeBuilder, result_type::TypeId, source::Value;
                        flush_to_zero::Bool=false)
    encode_varint!(cb.buf, Opcode.Exp2Op)
    encode_typeid!(cb.buf, result_type)
    encode_varint!(cb.buf, flush_to_zero ? 1 : 0)
    encode_operand!(cb.buf, source)
    return new_op!(cb)
end

"""
    encode_LogOp!(cb, result_type, source) -> Value

Natural logarithm (ln) operation.
Opcode: 63
"""
function encode_LogOp!(cb::CodeBuilder, result_type::TypeId, source::Value)
    encode_varint!(cb.buf, Opcode.LogOp)
    encode_typeid!(cb.buf, result_type)
    encode_operand!(cb.buf, source)
    return new_op!(cb)
end

"""
    encode_Log2Op!(cb, result_type, source) -> Value

Base-2 logarithm operation.
Opcode: 64
"""
function encode_Log2Op!(cb::CodeBuilder, result_type::TypeId, source::Value)
    encode_varint!(cb.buf, Opcode.Log2Op)
    encode_typeid!(cb.buf, result_type)
    encode_operand!(cb.buf, source)
    return new_op!(cb)
end

"""
    encode_IotaOp!(cb, result_type) -> Value

Create a tile with values [0, 1, 2, ..., size-1] (arange).
The result_type determines the shape and dtype.
Opcode: 58
"""
function encode_IotaOp!(cb::CodeBuilder, result_type::TypeId)
    encode_varint!(cb.buf, Opcode.IotaOp)
    encode_typeid!(cb.buf, result_type)
    return new_op!(cb)
end

"""
    encode_NegFOp!(cb, result_type, source) -> Value

Floating-point negation.
Opcode: 79
"""
function encode_NegFOp!(cb::CodeBuilder, result_type::TypeId, source::Value)
    encode_varint!(cb.buf, Opcode.NegFOp)
    encode_typeid!(cb.buf, result_type)
    encode_operand!(cb.buf, source)
    return new_op!(cb)
end

#=============================================================================
 Shape/layout operations
=============================================================================#

"""
    encode_PermuteOp!(cb, result_type, source, permutation) -> Value

Permute the dimensions of a tile according to the given permutation.
For 2D transpose, permutation would be [1, 0].
Opcode: 83
"""
function encode_PermuteOp!(cb::CodeBuilder, result_type::TypeId, source::Value,
                           permutation::Vector{Int})
    encode_varint!(cb.buf, Opcode.PermuteOp)
    encode_typeid!(cb.buf, result_type)
    # Encode permutation as dense int32 array attribute
    encode_dense_int32_array!(cb, permutation)
    encode_operand!(cb.buf, source)
    return new_op!(cb)
end

#=============================================================================
 Control flow operations
=============================================================================#

"""
    encode_ReturnOp!(cb, operands)

Return from function.
Opcode: 92
"""
function encode_ReturnOp!(cb::CodeBuilder, operands::Vector{Value}=Value[])
    encode_varint!(cb.buf, Opcode.ReturnOp)
    # Variadic result types (none)
    encode_typeid_seq!(cb.buf, TypeId[])
    # Operands
    encode_varint!(cb.buf, length(operands))
    encode_operands!(cb.buf, operands)
    return new_op!(cb, 0)
end

"""
    encode_YieldOp!(cb, operands)

Yield from a nested region (loop body, if branch, etc.).
Opcode: 109
"""
function encode_YieldOp!(cb::CodeBuilder, operands::Vector{Value}=Value[])
    encode_varint!(cb.buf, Opcode.YieldOp)
    # Variadic result types (empty - yield doesn't produce values, it yields to parent)
    encode_typeid_seq!(cb.buf, TypeId[])
    # Operands
    encode_varint!(cb.buf, length(operands))
    encode_operands!(cb.buf, operands)
    return new_op!(cb, 0)
end

"""
    encode_BreakOp!(cb, operands)

Break out of a loop, yielding final result values.
Opcode: 10
"""
function encode_BreakOp!(cb::CodeBuilder, operands::Vector{Value}=Value[])
    encode_varint!(cb.buf, Opcode.BreakOp)
    # Variadic result types (empty - break doesn't produce values, it yields to parent)
    encode_typeid_seq!(cb.buf, TypeId[])
    # Operands
    encode_varint!(cb.buf, length(operands))
    encode_operands!(cb.buf, operands)
    return new_op!(cb, 0)
end

"""
    encode_ContinueOp!(cb, operands)

Continue to next loop iteration with updated carried values.
Opcode: 17
"""
function encode_ContinueOp!(cb::CodeBuilder, operands::Vector{Value}=Value[])
    encode_varint!(cb.buf, Opcode.ContinueOp)
    # Variadic result types (empty)
    encode_typeid_seq!(cb.buf, TypeId[])
    # Operands
    encode_varint!(cb.buf, length(operands))
    encode_operands!(cb.buf, operands)
    return new_op!(cb, 0)
end

"""
    encode_IfOp!(cb, result_types, condition, then_body, else_body) -> Vector{Value}

Create an if-then-else operation with callback-based region building.
Opcode: 50

Arguments:
- `then_body(block_args)`: Callback to emit then branch ops, must end with YieldOp
- `else_body(block_args)`: Callback to emit else branch ops, must end with YieldOp

Returns the result values from the IfOp.

Example:
    results = encode_IfOp!(cb, result_types, cond) do _
        # then body
        encode_YieldOp!(cb, then_results)
    end do _
        # else body
        encode_YieldOp!(cb, else_results)
    end
"""
function encode_IfOp!(then_body::Function, else_body::Function,
                      cb::CodeBuilder, result_types::Vector{TypeId}, condition::Value)
    encode_varint!(cb.buf, Opcode.IfOp)
    encode_typeid_seq!(cb.buf, result_types)
    encode_operand!(cb.buf, condition)

    # Number of regions
    push!(cb.debug_attrs, cb.cur_debug_attr)
    cb.num_ops += 1
    encode_varint!(cb.buf, 2)  # 2 regions: then, else

    # Then region (no block args for if branches)
    with_region(then_body, cb, TypeId[])

    # Else region
    with_region(else_body, cb, TypeId[])

    # Create result values
    num_results = length(result_types)
    if num_results == 0
        return Value[]
    else
        vals = [Value(cb.next_value_id + i) for i in 0:num_results-1]
        cb.next_value_id += num_results
        return vals
    end
end

"""
    encode_LoopOp!(cb, result_types, init_values, body) -> Vector{Value}

Create a general loop operation with callback-based region building.
Opcode: 65

The body callback receives block arguments for loop-carried values.
Use BreakOp to exit the loop and ContinueOp to continue iteration.

Example:
    results = encode_LoopOp!(cb, result_types, init_values) do block_args
        # block_args are the loop-carried values
        # ... compute condition and new values ...
        encode_IfOp!(...) do _
            encode_BreakOp!(cb, final_values)
        end do _
            encode_YieldOp!(cb)
        end
        encode_ContinueOp!(cb, next_values)
    end
"""
function encode_LoopOp!(body::Function, cb::CodeBuilder,
                        result_types::Vector{TypeId}, init_values::Vector{Value})
    encode_varint!(cb.buf, Opcode.LoopOp)
    encode_typeid_seq!(cb.buf, result_types)
    encode_varint!(cb.buf, length(init_values))
    encode_operands!(cb.buf, init_values)

    # Number of regions
    push!(cb.debug_attrs, cb.cur_debug_attr)
    cb.num_ops += 1
    encode_varint!(cb.buf, 1)  # 1 region: body

    # Body region - block args are the loop-carried values
    with_region(body, cb, result_types)

    # Create result values
    num_results = length(result_types)
    if num_results == 0
        return Value[]
    else
        vals = [Value(cb.next_value_id + i) for i in 0:num_results-1]
        cb.next_value_id += num_results
        return vals
    end
end

"""
    encode_ForOp!(body, cb, result_types, iv_type, lower, upper, step, init_values) -> Vector{Value}

Create a for loop operation with callback-based region building.
Opcode: 41

The body callback receives (induction_var, carried_values...) as block arguments.
`iv_type` specifies the type of the induction variable.

Example:
    results = encode_ForOp!(cb, result_types, iv_type, lb, ub, step, init_values) do block_args
        iv = block_args[1]  # induction variable
        carried = block_args[2:end]  # loop-carried values
        # ... loop body ...
        encode_YieldOp!(cb, next_carried_values)
    end
"""
function encode_ForOp!(body::Function, cb::CodeBuilder,
                       result_types::Vector{TypeId}, iv_type::TypeId,
                       lower::Value, upper::Value, step::Value,
                       init_values::Vector{Value})
    encode_varint!(cb.buf, Opcode.ForOp)
    encode_typeid_seq!(cb.buf, result_types)
    # Operands: lower, upper, step, init_values...
    encode_varint!(cb.buf, 3 + length(init_values))
    encode_operand!(cb.buf, lower)
    encode_operand!(cb.buf, upper)
    encode_operand!(cb.buf, step)
    encode_operands!(cb.buf, init_values)

    # Number of regions
    push!(cb.debug_attrs, cb.cur_debug_attr)
    cb.num_ops += 1
    encode_varint!(cb.buf, 1)  # 1 region: body

    # Body region - block args are (induction_var, carried_values...)
    body_arg_types = vcat([iv_type], result_types)
    with_region(body, cb, body_arg_types)

    # Create result values
    num_results = length(result_types)
    if num_results == 0
        return Value[]
    else
        vals = [Value(cb.next_value_id + i) for i in 0:num_results-1]
        cb.next_value_id += num_results
        return vals
    end
end

#=============================================================================
 Matrix multiply-accumulate operations
=============================================================================#

"""
    encode_MmaFOp!(cb, result_type, lhs, rhs, acc) -> Value

Float matrix multiply-accumulate: result = lhs @ rhs + acc.
Opcode: 73
"""
function encode_MmaFOp!(cb::CodeBuilder, result_type::TypeId, lhs::Value, rhs::Value, acc::Value)
    encode_varint!(cb.buf, Opcode.MmaFOp)
    encode_typeid!(cb.buf, result_type)
    encode_operand!(cb.buf, lhs)
    encode_operand!(cb.buf, rhs)
    encode_operand!(cb.buf, acc)
    return new_op!(cb)
end

#=============================================================================
 Integer arithmetic operations
=============================================================================#

"""
    encode_DivIOp!(cb, result_type, lhs, rhs; signedness, rounding) -> Value

Integer division.
Opcode: 21
"""
function encode_DivIOp!(cb::CodeBuilder, result_type::TypeId, lhs::Value, rhs::Value;
                        signedness::Signedness=SignednessSigned,
                        rounding::RoundingMode=RoundingZero)
    encode_varint!(cb.buf, Opcode.DivIOp)
    encode_typeid!(cb.buf, result_type)
    encode_enum!(cb.buf, signedness)
    encode_enum!(cb.buf, rounding)
    encode_operand!(cb.buf, lhs)
    encode_operand!(cb.buf, rhs)
    return new_op!(cb)
end

"""
    encode_RemIOp!(cb, result_type, lhs, rhs; signedness) -> Value

Integer remainder (modulo).
Opcode: 90
"""
function encode_RemIOp!(cb::CodeBuilder, result_type::TypeId, lhs::Value, rhs::Value;
                        signedness::Signedness=SignednessSigned)
    encode_varint!(cb.buf, Opcode.RemIOp)
    encode_typeid!(cb.buf, result_type)
    encode_enum!(cb.buf, signedness)
    encode_operand!(cb.buf, lhs)
    encode_operand!(cb.buf, rhs)
    return new_op!(cb)
end

"""
    encode_MinIOp!(cb, result_type, lhs, rhs; signedness) -> Value

Integer minimum.
Opcode: 72
"""
function encode_MinIOp!(cb::CodeBuilder, result_type::TypeId, lhs::Value, rhs::Value;
                        signedness::Signedness=SignednessSigned)
    encode_varint!(cb.buf, Opcode.MinIOp)
    encode_typeid!(cb.buf, result_type)
    encode_enum!(cb.buf, signedness)
    encode_operand!(cb.buf, lhs)
    encode_operand!(cb.buf, rhs)
    return new_op!(cb)
end

"""
    encode_MaxIOp!(cb, result_type, lhs, rhs; signedness) -> Value

Integer maximum.
Opcode: 70
"""
function encode_MaxIOp!(cb::CodeBuilder, result_type::TypeId, lhs::Value, rhs::Value;
                        signedness::Signedness=SignednessSigned)
    encode_varint!(cb.buf, Opcode.MaxIOp)
    encode_typeid!(cb.buf, result_type)
    encode_enum!(cb.buf, signedness)
    encode_operand!(cb.buf, lhs)
    encode_operand!(cb.buf, rhs)
    return new_op!(cb)
end

"""
    encode_MaxFOp!(cb, result_type, lhs, rhs; propagate_nan, flush_to_zero) -> Value

Floating-point maximum.
Opcode: 69
"""
function encode_MaxFOp!(cb::CodeBuilder, result_type::TypeId, lhs::Value, rhs::Value;
                        propagate_nan::Bool=false, flush_to_zero::Bool=false)
    encode_varint!(cb.buf, Opcode.MaxFOp)
    encode_typeid!(cb.buf, result_type)
    encode_varint!(cb.buf, (propagate_nan ? 1 : 0) | (flush_to_zero ? 2 : 0))
    encode_operand!(cb.buf, lhs)
    encode_operand!(cb.buf, rhs)
    return new_op!(cb)
end

"""
    encode_MinFOp!(cb, result_type, lhs, rhs; propagate_nan, flush_to_zero) -> Value

Floating-point minimum.
Opcode: 71
"""
function encode_MinFOp!(cb::CodeBuilder, result_type::TypeId, lhs::Value, rhs::Value;
                        propagate_nan::Bool=false, flush_to_zero::Bool=false)
    encode_varint!(cb.buf, Opcode.MinFOp)
    encode_typeid!(cb.buf, result_type)
    encode_varint!(cb.buf, (propagate_nan ? 1 : 0) | (flush_to_zero ? 2 : 0))
    encode_operand!(cb.buf, lhs)
    encode_operand!(cb.buf, rhs)
    return new_op!(cb)
end

#=============================================================================
 Reduction operations
=============================================================================#

"""
    encode_ReduceOp!(body, cb, result_types, operands, dim, identities, body_scalar_types) -> Vector{Value}

Create a reduction operation with callback-based region building.
Opcode: 88

The body callback receives pairs of (accumulator, element) values as block arguments
for each operand being reduced.

Example for sum reduction:
    results = encode_ReduceOp!(cb, [output_type], [input_tile], dim, identities, [scalar_0d_type]) do block_args
        acc, elem = block_args[1], block_args[2]  # For single-operand reduce
        result = encode_AddFOp!(cb, scalar_0d_type, acc, elem)
        encode_YieldOp!(cb, [result])
    end

Arguments:
- result_types: Output tile types (shape reduced along dim)
- operands: Input tiles to reduce
- dim: Axis to reduce along (0-indexed)
- identities: Identity values for the reduction (e.g., 0.0 for sum)
- body_scalar_types: 0D tile types for the body block arguments (one per result)
"""
function encode_ReduceOp!(body::Function, cb::CodeBuilder,
                          result_types::Vector{TypeId},
                          operands::Vector{Value},
                          dim::Int,
                          identities::Vector{<:IdentityVal},
                          body_scalar_types::Vector{TypeId})
    encode_varint!(cb.buf, Opcode.ReduceOp)

    # Variadic result types
    encode_typeid_seq!(cb.buf, result_types)

    # Attributes: dim (int) and identities (array)
    encode_opattr_int!(cb, dim)
    encode_identity_array!(cb, identities)

    # Variadic operands
    encode_varint!(cb.buf, length(operands))
    encode_operands!(cb.buf, operands)

    # Number of regions
    push!(cb.debug_attrs, cb.cur_debug_attr)
    cb.num_ops += 1
    encode_varint!(cb.buf, 1)  # 1 region: body

    # Body region - block args are pairs of (acc, elem) for each operand
    # The body operates on 0D tiles (scalars)
    body_arg_types = TypeId[]
    for scalar_type in body_scalar_types
        push!(body_arg_types, scalar_type)  # accumulator
        push!(body_arg_types, scalar_type)  # element
    end
    with_region(body, cb, body_arg_types)

    # Create result values
    num_results = length(result_types)
    if num_results == 0
        return Value[]
    else
        vals = [Value(cb.next_value_id + i) for i in 0:num_results-1]
        cb.next_value_id += num_results
        return vals
    end
end

#=============================================================================
 Comparison and selection operations
=============================================================================#

"""
    encode_CmpFOp!(cb, result_type, lhs, rhs; predicate, ordering) -> Value

Floating-point comparison.
Opcode: 14
"""
function encode_CmpFOp!(cb::CodeBuilder, result_type::TypeId, lhs::Value, rhs::Value;
                        predicate::ComparisonPredicate=CmpEqual,
                        ordering::ComparisonOrdering=CmpOrdered)
    encode_varint!(cb.buf, Opcode.CmpFOp)
    encode_typeid!(cb.buf, result_type)
    encode_enum!(cb.buf, predicate)
    encode_enum!(cb.buf, ordering)
    encode_operand!(cb.buf, lhs)
    encode_operand!(cb.buf, rhs)
    return new_op!(cb)
end

"""
    encode_CmpIOp!(cb, result_type, lhs, rhs; predicate, signedness) -> Value

Integer comparison.
Opcode: 15
"""
function encode_CmpIOp!(cb::CodeBuilder, result_type::TypeId, lhs::Value, rhs::Value;
                        predicate::ComparisonPredicate=CmpEqual,
                        signedness::Signedness=SignednessSigned)
    encode_varint!(cb.buf, Opcode.CmpIOp)
    encode_typeid!(cb.buf, result_type)
    encode_enum!(cb.buf, predicate)
    encode_enum!(cb.buf, signedness)
    encode_operand!(cb.buf, lhs)
    encode_operand!(cb.buf, rhs)
    return new_op!(cb)
end

"""
    encode_SelectOp!(cb, result_type, cond, val_if_true, val_if_false) -> Value

Conditional select (ternary operator).
Opcode: 95
"""
function encode_SelectOp!(cb::CodeBuilder, result_type::TypeId,
                          cond::Value, val_if_true::Value, val_if_false::Value)
    encode_varint!(cb.buf, Opcode.SelectOp)
    encode_typeid!(cb.buf, result_type)
    encode_operand!(cb.buf, cond)
    encode_operand!(cb.buf, val_if_true)
    encode_operand!(cb.buf, val_if_false)
    return new_op!(cb)
end

#=============================================================================
 Bitwise operations
=============================================================================#

"""
    encode_AndIOp!(cb, result_type, lhs, rhs) -> Value

Bitwise AND.
Opcode: 4
"""
function encode_AndIOp!(cb::CodeBuilder, result_type::TypeId, lhs::Value, rhs::Value)
    encode_varint!(cb.buf, Opcode.AndIOp)
    encode_typeid!(cb.buf, result_type)
    encode_operand!(cb.buf, lhs)
    encode_operand!(cb.buf, rhs)
    return new_op!(cb)
end

"""
    encode_OrIOp!(cb, result_type, lhs, rhs) -> Value

Bitwise OR.
Opcode: 82
"""
function encode_OrIOp!(cb::CodeBuilder, result_type::TypeId, lhs::Value, rhs::Value)
    encode_varint!(cb.buf, Opcode.OrIOp)
    encode_typeid!(cb.buf, result_type)
    encode_operand!(cb.buf, lhs)
    encode_operand!(cb.buf, rhs)
    return new_op!(cb)
end

"""
    encode_XOrIOp!(cb, result_type, lhs, rhs) -> Value

Bitwise XOR.
Opcode: 108
"""
function encode_XOrIOp!(cb::CodeBuilder, result_type::TypeId, lhs::Value, rhs::Value)
    encode_varint!(cb.buf, Opcode.XOrIOp)
    encode_typeid!(cb.buf, result_type)
    encode_operand!(cb.buf, lhs)
    encode_operand!(cb.buf, rhs)
    return new_op!(cb)
end

"""
    encode_ShLIOp!(cb, result_type, lhs, rhs) -> Value

Shift left.
Opcode: 96
"""
function encode_ShLIOp!(cb::CodeBuilder, result_type::TypeId, lhs::Value, rhs::Value)
    encode_varint!(cb.buf, Opcode.ShLIOp)
    encode_typeid!(cb.buf, result_type)
    encode_operand!(cb.buf, lhs)
    encode_operand!(cb.buf, rhs)
    return new_op!(cb)
end

"""
    encode_ShRIOp!(cb, result_type, lhs, rhs; signedness) -> Value

Shift right. Signed = arithmetic shift, Unsigned = logical shift.
Opcode: 97
"""
function encode_ShRIOp!(cb::CodeBuilder, result_type::TypeId, lhs::Value, rhs::Value;
                        signedness::Signedness=SignednessSigned)
    encode_varint!(cb.buf, Opcode.ShRIOp)
    encode_typeid!(cb.buf, result_type)
    encode_enum!(cb.buf, signedness)
    encode_operand!(cb.buf, lhs)
    encode_operand!(cb.buf, rhs)
    return new_op!(cb)
end

"""
    encode_NegIOp!(cb, result_type, source) -> Value

Integer negation.
Opcode: 80
"""
function encode_NegIOp!(cb::CodeBuilder, result_type::TypeId, source::Value;
                        overflow::IntegerOverflow=OverflowNone)
    encode_varint!(cb.buf, Opcode.NegIOp)
    encode_typeid!(cb.buf, result_type)
    encode_enum!(cb.buf, overflow)
    encode_operand!(cb.buf, source)
    return new_op!(cb)
end

"""
    encode_AbsFOp!(cb, result_type, source) -> Value

Floating-point absolute value.
Opcode: 0
"""
function encode_AbsFOp!(cb::CodeBuilder, result_type::TypeId, source::Value)
    encode_varint!(cb.buf, Opcode.AbsFOp)
    encode_typeid!(cb.buf, result_type)
    encode_operand!(cb.buf, source)
    return new_op!(cb)
end

#=============================================================================
 Type conversion operations
=============================================================================#

"""
    encode_FToFOp!(cb, result_type, source; rounding_mode) -> Value

Float to float conversion (e.g., fp32 to fp16, fp32 to tf32).
Opcode: 42
"""
function encode_FToFOp!(cb::CodeBuilder, result_type::TypeId, source::Value;
                        rounding_mode::RoundingMode=RoundingNearestEven)
    encode_varint!(cb.buf, Opcode.FToFOp)
    encode_typeid!(cb.buf, result_type)
    encode_enum!(cb.buf, rounding_mode)
    encode_operand!(cb.buf, source)
    return new_op!(cb)
end

"""
    encode_IToFOp!(cb, result_type, source; signedness, rounding_mode) -> Value

Integer to float conversion (e.g., i32 to fp32).
Opcode: 59
"""
function encode_IToFOp!(cb::CodeBuilder, result_type::TypeId, source::Value;
                        signedness::Signedness=SignednessSigned,
                        rounding_mode::RoundingMode=RoundingNearestEven)
    encode_varint!(cb.buf, Opcode.IToFOp)
    encode_typeid!(cb.buf, result_type)
    encode_enum!(cb.buf, signedness)
    encode_enum!(cb.buf, rounding_mode)
    encode_operand!(cb.buf, source)
    return new_op!(cb)
end

"""
    encode_FToIOp!(cb, result_type, source; signedness, rounding_mode) -> Value

Float to integer conversion (e.g., fp32 to i32).
Opcode: 43
"""
function encode_FToIOp!(cb::CodeBuilder, result_type::TypeId, source::Value;
                        signedness::Signedness=SignednessSigned,
                        rounding_mode::RoundingMode=RoundingNearestIntToZero)
    encode_varint!(cb.buf, Opcode.FToIOp)
    encode_typeid!(cb.buf, result_type)
    encode_enum!(cb.buf, signedness)
    encode_enum!(cb.buf, rounding_mode)
    encode_operand!(cb.buf, source)
    return new_op!(cb)
end

"""
    encode_ExtIOp!(cb, result_type, source; signedness) -> Value

Integer extension (e.g., i16 to i32). Sign or zero extends based on signedness.
Opcode: 37
"""
function encode_ExtIOp!(cb::CodeBuilder, result_type::TypeId, source::Value;
                        signedness::Signedness=SignednessSigned)
    encode_varint!(cb.buf, Opcode.ExtIOp)
    encode_typeid!(cb.buf, result_type)
    encode_enum!(cb.buf, signedness)
    encode_operand!(cb.buf, source)
    return new_op!(cb)
end

"""
    encode_BroadcastOp!(cb, result_type, source) -> Value

Broadcast a scalar or smaller tile to a larger tile shape.
Opcode: 11
"""
function encode_BroadcastOp!(cb::CodeBuilder, result_type::TypeId, source::Value)
    encode_varint!(cb.buf, Opcode.BroadcastOp)
    encode_typeid!(cb.buf, result_type)
    encode_operand!(cb.buf, source)
    return new_op!(cb)
end

"""
    encode_ReshapeOp!(cb, result_type, source) -> Value

Reshape a tile to a new shape (with compatible total elements).
Opcode: 91
"""
function encode_ReshapeOp!(cb::CodeBuilder, result_type::TypeId, source::Value)
    encode_varint!(cb.buf, Opcode.ReshapeOp)
    encode_typeid!(cb.buf, result_type)
    encode_operand!(cb.buf, source)
    return new_op!(cb)
end

"""
    encode_ExtractOp!(cb, result_type, source, indices) -> Value

Extract a slice from a tile at the given indices.
The indices specify the starting position for each dimension.
The result shape is determined by result_type.
Opcode: 38
"""
function encode_ExtractOp!(cb::CodeBuilder, result_type::TypeId, source::Value, indices::Vector{Value})
    encode_varint!(cb.buf, Opcode.ExtractOp)
    # Variadic result types (just one)
    encode_typeid_seq!(cb.buf, [result_type])
    # Operands: source + indices
    encode_varint!(cb.buf, 1 + length(indices))
    encode_operand!(cb.buf, source)
    for idx in indices
        encode_operand!(cb.buf, idx)
    end
    return new_op!(cb)
end

"""
    encode_CatOp!(cb, result_type, lhs, rhs, dim) -> Value

Concatenate two tiles along the specified dimension.
Opcode: 12
"""
function encode_CatOp!(cb::CodeBuilder, result_type::TypeId, lhs::Value, rhs::Value, dim::Int)
    encode_varint!(cb.buf, Opcode.CatOp)
    encode_typeid!(cb.buf, result_type)
    # Attribute: dimension (as i32)
    encode_opattr_int!(cb, Int32(dim))
    # Operands
    encode_operand!(cb.buf, lhs)
    encode_operand!(cb.buf, rhs)
    return new_op!(cb)
end

#=============================================================================
 Tensor shape operations
=============================================================================#

"""
    encode_GetTensorShapeOp!(cb, result_types, src) -> Tuple{Value...}

Get the shape of a tensor as multiple scalar values.
Opcode: 47
"""
function encode_GetTensorShapeOp!(cb::CodeBuilder, result_types::Vector{TypeId}, src::Value)
    encode_varint!(cb.buf, Opcode.GetTensorShapeOp)
    encode_typeid_seq!(cb.buf, result_types)
    encode_operand!(cb.buf, src)
    return new_op!(cb, length(result_types))
end

#=============================================================================
 Pointer conversion operations
=============================================================================#

"""
    encode_PtrToIntOp!(cb, result_type, source) -> Value

Convert a pointer tile to an integer tile.
Opcode: 86
"""
function encode_PtrToIntOp!(cb::CodeBuilder, result_type::TypeId, source::Value)
    encode_varint!(cb.buf, Opcode.PtrToIntOp)
    encode_varint!(cb.buf, result_type.id)
    encode_operand!(cb.buf, source)
    return new_op!(cb)
end

"""
    encode_IntToPtrOp!(cb, result_type, source) -> Value

Convert an integer tile to a pointer tile.
Opcode: 51
"""
function encode_IntToPtrOp!(cb::CodeBuilder, result_type::TypeId, source::Value)
    encode_varint!(cb.buf, Opcode.IntToPtrOp)
    encode_varint!(cb.buf, result_type.id)
    encode_operand!(cb.buf, source)
    return new_op!(cb)
end

#=============================================================================
 Atomic operations
=============================================================================#

"""
    encode_AtomicCASPtrOp!(cb, result_type, token_type, pointers, cmp, val;
                           mask, token, memory_ordering, memory_scope) -> (Value, Value)

Atomic compare-and-swap operation.
Opcode: 7

Returns (old_value, new_token) tuple.
"""
function encode_AtomicCASPtrOp!(cb::CodeBuilder,
                                 result_type::TypeId,
                                 token_type::TypeId,
                                 pointers::Value,
                                 cmp::Value,
                                 val::Value;
                                 mask::Union{Value, Nothing}=nothing,
                                 token::Union{Value, Nothing}=nothing,
                                 memory_ordering::MemoryOrderingSemantics=MemoryAcqRel,
                                 memory_scope::MemoryScope=ScopeDevice)
    encode_varint!(cb.buf, Opcode.AtomicCASPtrOp)
    # Result types (no length prefix, per Python reference)
    encode_varint!(cb.buf, result_type.id)
    encode_varint!(cb.buf, token_type.id)

    # Flags
    flags = 0
    if mask !== nothing
        flags |= 1
    end
    if token !== nothing
        flags |= 2
    end
    encode_varint!(cb.buf, flags)

    # Attributes
    encode_enum!(cb.buf, memory_ordering)
    encode_enum!(cb.buf, memory_scope)

    # Operands
    encode_operand!(cb.buf, pointers)
    encode_operand!(cb.buf, cmp)
    encode_operand!(cb.buf, val)
    encode_optional_operand!(cb.buf, mask)
    encode_optional_operand!(cb.buf, token)

    return new_op!(cb, 2)
end

"""
    encode_AtomicRMWPtrOp!(cb, result_type, token_type, pointers, arg, mode;
                           mask, token, memory_ordering, memory_scope) -> (Value, Value)

Atomic read-modify-write operation (add, xchg, max, min, and, or, xor).
Opcode: 8

Returns (old_value, new_token) tuple.
"""
function encode_AtomicRMWPtrOp!(cb::CodeBuilder,
                                 result_type::TypeId,
                                 token_type::TypeId,
                                 pointers::Value,
                                 arg::Value,
                                 mode::AtomicRMWMode;
                                 mask::Union{Value, Nothing}=nothing,
                                 token::Union{Value, Nothing}=nothing,
                                 memory_ordering::MemoryOrderingSemantics=MemoryAcqRel,
                                 memory_scope::MemoryScope=ScopeDevice)
    encode_varint!(cb.buf, Opcode.AtomicRMWPtrOp)
    # Result types (no length prefix, per Python reference)
    encode_varint!(cb.buf, result_type.id)
    encode_varint!(cb.buf, token_type.id)

    # Flags
    flags = 0
    if mask !== nothing
        flags |= 1
    end
    if token !== nothing
        flags |= 2
    end
    encode_varint!(cb.buf, flags)

    # Attributes
    encode_enum!(cb.buf, memory_ordering)
    encode_enum!(cb.buf, memory_scope)
    encode_enum!(cb.buf, mode)

    # Operands
    encode_operand!(cb.buf, pointers)
    encode_operand!(cb.buf, arg)
    encode_optional_operand!(cb.buf, mask)
    encode_optional_operand!(cb.buf, token)

    return new_op!(cb, 2)
end

#=============================================================================
 Additional math operations
=============================================================================#

"""
    encode_CeilOp!(cb, result_type, source) -> Value

Floating-point ceiling (round toward positive infinity).
Opcode: 13
"""
function encode_CeilOp!(cb::CodeBuilder, result_type::TypeId, source::Value)
    encode_varint!(cb.buf, Opcode.CeilOp)
    encode_typeid!(cb.buf, result_type)
    encode_operand!(cb.buf, source)
    return new_op!(cb)
end

"""
    encode_FloorOp!(cb, result_type, source) -> Value

Floating-point floor (round toward negative infinity).
Opcode: 39
"""
function encode_FloorOp!(cb::CodeBuilder, result_type::TypeId, source::Value)
    encode_varint!(cb.buf, Opcode.FloorOp)
    encode_typeid!(cb.buf, result_type)
    encode_operand!(cb.buf, source)
    return new_op!(cb)
end

"""
    encode_SinOp!(cb, result_type, source) -> Value

Element-wise sine.
Opcode: 98
"""
function encode_SinOp!(cb::CodeBuilder, result_type::TypeId, source::Value)
    encode_varint!(cb.buf, Opcode.SinOp)
    encode_typeid!(cb.buf, result_type)
    encode_operand!(cb.buf, source)
    return new_op!(cb)
end

"""
    encode_CosOp!(cb, result_type, source) -> Value

Element-wise cosine.
Opcode: 18
"""
function encode_CosOp!(cb::CodeBuilder, result_type::TypeId, source::Value)
    encode_varint!(cb.buf, Opcode.CosOp)
    encode_typeid!(cb.buf, result_type)
    encode_operand!(cb.buf, source)
    return new_op!(cb)
end

"""
    encode_TanOp!(cb, result_type, source) -> Value

Element-wise tangent.
Opcode: 105
"""
function encode_TanOp!(cb::CodeBuilder, result_type::TypeId, source::Value)
    encode_varint!(cb.buf, Opcode.TanOp)
    encode_typeid!(cb.buf, result_type)
    encode_operand!(cb.buf, source)
    return new_op!(cb)
end

"""
    encode_SinHOp!(cb, result_type, source) -> Value

Element-wise hyperbolic sine.
Opcode: 99
"""
function encode_SinHOp!(cb::CodeBuilder, result_type::TypeId, source::Value)
    encode_varint!(cb.buf, Opcode.SinHOp)
    encode_typeid!(cb.buf, result_type)
    encode_operand!(cb.buf, source)
    return new_op!(cb)
end

"""
    encode_CosHOp!(cb, result_type, source) -> Value

Element-wise hyperbolic cosine.
Opcode: 19
"""
function encode_CosHOp!(cb::CodeBuilder, result_type::TypeId, source::Value)
    encode_varint!(cb.buf, Opcode.CosHOp)
    encode_typeid!(cb.buf, result_type)
    encode_operand!(cb.buf, source)
    return new_op!(cb)
end

"""
    encode_TanHOp!(cb, result_type, source) -> Value

Element-wise hyperbolic tangent.
Opcode: 106
"""
function encode_TanHOp!(cb::CodeBuilder, result_type::TypeId, source::Value)
    encode_varint!(cb.buf, Opcode.TanHOp)
    encode_typeid!(cb.buf, result_type)
    encode_operand!(cb.buf, source)
    return new_op!(cb)
end

"""
    encode_FmaOp!(cb, result_type, a, b, c; rounding_mode, flush_to_zero) -> Value

Element-wise fused multiply-add: a * b + c.
Opcode: 40
"""
function encode_FmaOp!(cb::CodeBuilder, result_type::TypeId, a::Value, b::Value, c::Value;
                       rounding_mode::RoundingMode=RoundingNearestEven,
                       flush_to_zero::Bool=false)
    encode_varint!(cb.buf, Opcode.FmaOp)
    encode_typeid!(cb.buf, result_type)
    encode_varint!(cb.buf, flush_to_zero ? 1 : 0)
    encode_enum!(cb.buf, rounding_mode)
    encode_operand!(cb.buf, a)
    encode_operand!(cb.buf, b)
    encode_operand!(cb.buf, c)
    return new_op!(cb)
end

"""
    encode_RemFOp!(cb, result_type, lhs, rhs) -> Value

Floating-point remainder.
Opcode: 89
"""
function encode_RemFOp!(cb::CodeBuilder, result_type::TypeId, lhs::Value, rhs::Value)
    encode_varint!(cb.buf, Opcode.RemFOp)
    encode_typeid!(cb.buf, result_type)
    encode_operand!(cb.buf, lhs)
    encode_operand!(cb.buf, rhs)
    return new_op!(cb)
end

"""
    encode_AbsIOp!(cb, result_type, source) -> Value

Integer absolute value.
Opcode: 1
"""
function encode_AbsIOp!(cb::CodeBuilder, result_type::TypeId, source::Value)
    encode_varint!(cb.buf, Opcode.AbsIOp)
    encode_typeid!(cb.buf, result_type)
    encode_operand!(cb.buf, source)
    return new_op!(cb)
end

"""
    encode_MulhiIOp!(cb, result_type, lhs, rhs) -> Value

High bits of integer multiply (for extended precision arithmetic).
Opcode: 77
"""
function encode_MulhiIOp!(cb::CodeBuilder, result_type::TypeId, lhs::Value, rhs::Value)
    encode_varint!(cb.buf, Opcode.MulhiIOp)
    encode_typeid!(cb.buf, result_type)
    encode_operand!(cb.buf, lhs)
    encode_operand!(cb.buf, rhs)
    return new_op!(cb)
end
