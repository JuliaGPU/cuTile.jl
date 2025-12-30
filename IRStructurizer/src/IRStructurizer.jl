module IRStructurizer

using Core: MethodInstance, CodeInfo, SSAValue, Argument, SlotNumber,
            GotoNode, GotoIfNot, ReturnNode, PhiNode, PiNode, QuoteNode, GlobalRef

# graph-level analyses
include("graph.jl")
include("cfg.jl")
include("analysis.jl")

# structured IR definitions
include("ir.jl")
include("show.jl")

# block-level CFG
include("block_cfg.jl")

# control tree to structured IR
include("structure.jl")

# pattern matching and loop upgrades
include("patterns.jl")

# validation and public API
include("validation.jl")
include("interface.jl")

end
