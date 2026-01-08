module IRStructurizer

using Core: MethodInstance, CodeInfo, SSAValue, Argument, SlotNumber,
            GotoNode, GotoIfNot, ReturnNode, PhiNode, PiNode, QuoteNode, GlobalRef

# Import IRCode and compiler utilities from Core.Compiler
using Core.Compiler: IRCode, CFG, BasicBlock, InstructionStream, StmtRange,
                     construct_domtree, DomTree, dominates, bb_unreachable

# Import code_ircode from Base
using Base: code_ircode

# auxiliary data structures and analyses
include("graph.jl")
include("cfg.jl")

# control tree construction
include("control_tree.jl")

# structured IR definitions
include("ir.jl")
include("show.jl")

# control tree to structured IR
include("structure.jl")

# validation and public API
include("validation.jl")
include("interface.jl")

end
