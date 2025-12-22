module cuTile

using IRStructurizer
using IRStructurizer: Block, BlockArg, LocalSSAValue, Statement, PartialBlock, PartialControlFlowOp,
                      YieldOp, ContinueOp, BreakOp, ConditionOp

using Core: MethodInstance, CodeInfo, SSAValue, Argument, SlotNumber,
            ReturnNode, PiNode, QuoteNode, GlobalRef
using Core.Compiler
const CC = Core.Compiler

using Preferences

# Language definition
include("types.jl")
include("intrinsics.jl")

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
