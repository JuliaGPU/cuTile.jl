module IRStructurizer

using Core: MethodInstance, CodeInfo, SSAValue, Argument, SlotNumber,
            GotoNode, GotoIfNot, ReturnNode, PhiNode, PiNode, QuoteNode, GlobalRef

# graph-level analyses
include("graph.jl")
include("cfg.jl")
include("analysis.jl")

# structured IR definitions
include("ir.jl")

# IR-level patterning
include("block_cfg.jl")
include("patterns.jl")

# orchestration and public API
include("restructuring.jl")
include("validation.jl")
include("interface.jl")

end
