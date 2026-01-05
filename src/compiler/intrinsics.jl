# Tile IR intrinsics
#
# Organized according to https://docs.nvidia.com/cuda/tile-ir/latest/sections/operations.html

module Intrinsics

using ..cuTile: Tile, TileArray, Constant, TensorView, PartitionView

end

emit_intrinsic!(ctx::CodegenContext, @nospecialize(func), args, @nospecialize(result_type)) = missing

include("intrinsics/core.jl")
include("intrinsics/conversions.jl")
include("intrinsics/memory.jl")
include("intrinsics/math.jl")
include("intrinsics/bitwise.jl")
include("intrinsics/atomics.jl")
include("intrinsics/views.jl")
include("intrinsics/misc.jl")

include("intrinsics/highlevel.jl")
