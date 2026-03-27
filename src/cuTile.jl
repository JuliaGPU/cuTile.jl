module cuTile

using IRStructurizer
using IRStructurizer: Block, ControlFlowOp, BlockArg,
                      YieldOp, ContinueOp, BreakOp, ConditionOp,
                      IfOp, ForOp, WhileOp, LoopOp, Undef

using Base: compilerbarrier, donotdelete
using Core: MethodInstance, CodeInfo, SSAValue, Argument, SlotNumber,
            ReturnNode, PiNode, QuoteNode, GlobalRef
using Core.Compiler
const CC = Core.Compiler

using CUDA_Tile_jll

using BFloat16s: BFloat16
using EnumX
public BFloat16

# Shared definitions
include("shapes.jl")

# Bytecode infrastructure
include("bytecode/basic.jl")
include("bytecode/types.jl")
include("bytecode/writer.jl")
include("bytecode/encodings.jl")

# Language definitions
include("language/types.jl")

# Compiler implementation
include("compiler/interface.jl")
include("compiler/codegen.jl")
include("compiler/intrinsics.jl")

# Language implementation
include("language/broadcast.jl")
include("language/overlays.jl")
include("language/arithmetic.jl")
include("language/math.jl")
include("language/operations.jl")
include("language/atomics.jl")

# Host-level abstractions
include("utils.jl")
include("tiled.jl")
include("broadcast.jl")
include("mapreduce.jl")

public launch, Tiled, ByTarget, @compiler_options, @.
launch(args...) = error("Please import CUDA.jl before using `cuTile.launch`.")

end # module cuTile
