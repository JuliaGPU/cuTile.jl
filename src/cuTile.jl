module cuTile

using IRStructurizer
using IRStructurizer: Block, ControlFlowOp, BlockArg,
                      YieldOp, ContinueOp, BreakOp, ConditionOp,
                      IfOp, ForOp, WhileOp, LoopOp

using Core: MethodInstance, CodeInfo, SSAValue, Argument, SlotNumber,
            ReturnNode, PiNode, QuoteNode, GlobalRef
using Core.Compiler
const CC = Core.Compiler

using CUDA_Tile_jll

# Language definition
include("language/types.jl")
include("language/broadcast.jl")
include("language/operations.jl")

# Bytecode infrastructure
include("bytecode/basic.jl")
include("bytecode/types.jl")
include("bytecode/writer.jl")
include("bytecode/encodings.jl")

# Compiler implementation
include("compiler/interpreter.jl")
include("compiler/target.jl")
include("compiler/codegen.jl")
include("compiler/intrinsics.jl")

public launch
launch() = error("Please import CUDA.jl before using `cuTile.launch`.")

end # module cuTile
