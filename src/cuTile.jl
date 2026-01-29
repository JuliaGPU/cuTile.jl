module cuTile

using IRStructurizer
using IRStructurizer: Block, ControlFlowOp, BlockArg,
                      YieldOp, ContinueOp, BreakOp, ConditionOp,
                      IfOp, ForOp, WhileOp, LoopOp

using Base: compilerbarrier, donotdelete
using Core: MethodInstance, CodeInfo, SSAValue, Argument, SlotNumber,
            ReturnNode, PiNode, QuoteNode, GlobalRef
using Core.Compiler
const CC = Core.Compiler

using CUDA_Tile_jll

using BFloat16s: BFloat16
public BFloat16

# Bytecode infrastructure
include("bytecode/basic.jl")
include("bytecode/types.jl")
include("bytecode/writer.jl")
include("bytecode/encodings.jl")

# Language definitions
include("language/types.jl")

# Compiler implementation
include("compiler/interpreter.jl")
include("compiler/target.jl")
include("compiler/codegen.jl")
include("compiler/intrinsics.jl")
include("compiler/reflection.jl")

# Language implementation
include("language/broadcast.jl")
include("language/overlays.jl")
include("language/arithmetic.jl")
include("language/math.jl")
include("language/operations.jl")
include("language/atomics.jl")

public launch
launch() = error("Please import CUDA.jl before using `cuTile.launch`.")

end # module cuTile
