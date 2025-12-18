module IRStructurizer

using Core: MethodInstance, CodeInfo, SSAValue, Argument, SlotNumber,
            GotoNode, GotoIfNot, ReturnNode, PhiNode, PiNode, QuoteNode, GlobalRef

include("ir.jl")
include("restructuring.jl")
include("validation.jl")
include("interface.jl")

end # module IRStructurizer
