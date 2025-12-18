module cuTile

using IRStructurizer
using IRStructurizer: Block, BlockArg, IfOp, ForOp, LoopOp, YieldOp,
                      ContinueOp, BreakOp, ControlFlowOp

using CUDA: CuModule, CuFunction, cudacall, device, capability

using Core: MethodInstance, CodeInfo, SSAValue, Argument, SlotNumber,
            ReturnNode, PiNode, QuoteNode, GlobalRef
using Core.Compiler
const CC = Core.Compiler

# Language definition
include("types.jl")
include("intrinsics.jl")
include("execution.jl")

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

end # module cuTile
