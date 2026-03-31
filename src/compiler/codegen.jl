# Codegen: Julia IR -> Tile IR bytecode

include("codegen/passes/rewrite.jl")
include("codegen/passes/canonicalize.jl")
include("codegen/passes/alias_analysis.jl")
include("codegen/passes/token_keys.jl")
include("codegen/passes/token_order.jl")
include("codegen/passes/dce.jl")
include("codegen/passes/pipeline.jl")
include("codegen/kernel.jl")
include("codegen/control_flow.jl")
include("codegen/statements.jl")
include("codegen/expressions.jl")
include("codegen/values.jl")
