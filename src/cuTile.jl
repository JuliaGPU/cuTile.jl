module cuTile

using CUDA: CuModule, CuFunction, cudacall, device, capability
using Core: MethodInstance, CodeInfo, SSAValue, Argument, SlotNumber,
            GotoNode, GotoIfNot, ReturnNode, PhiNode, PiNode, QuoteNode, GlobalRef
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
include("compiler/ir.jl")
include("compiler/restructuring.jl")
include("compiler/interpreter.jl")
include("compiler/target.jl")
include("compiler/codegen.jl")
include("compiler/intrinsics.jl")

end # module cuTile
