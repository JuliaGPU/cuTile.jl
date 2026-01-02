module IRStructurizer

using Core: MethodInstance, CodeInfo, SSAValue, Argument, SlotNumber,
            GotoNode, GotoIfNot, ReturnNode, PhiNode, PiNode, QuoteNode, GlobalRef

# auxiliary data structures and analyses
include("graph.jl")
include("dominance.jl")

# control tree construction
include("blocks.jl")
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
