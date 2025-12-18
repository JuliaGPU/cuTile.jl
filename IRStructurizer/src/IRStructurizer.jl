module IRStructurizer

using Core: MethodInstance, CodeInfo, SSAValue, Argument, SlotNumber,
            GotoNode, GotoIfNot, ReturnNode, PhiNode, PiNode, QuoteNode, GlobalRef

# SPIRV.jl-style structural analysis
include("deltagraph.jl")
include("control_flow.jl")
include("structural_analysis.jl")

# IR definitions
include("ir.jl")

# Restructuring (uses structural analysis)
include("restructuring.jl")

include("validation.jl")
include("interface.jl")

end # module IRStructurizer
