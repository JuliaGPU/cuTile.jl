# Codegen: Julia IR -> Tile IR bytecode

include("codegen/passes/token_keys.jl")        # TokenKey, TokenRole, ACQUIRE_TOKEN_KEY
include("codegen/passes/rewrite.jl")            # @rewrite, rewrite_patterns! framework
include("codegen/passes/alias_analysis.jl")    # alias_analysis_pass!
include("codegen/passes/token_order.jl")       # token_order_pass!
include("codegen/passes/dce.jl")              # dce_pass!
include("codegen/passes/index_normalize.jl")  # index_lower_pass!
include("codegen/passes/pipeline.jl")          # run_passes!
include("codegen/kernel.jl")
include("codegen/control_flow.jl")
include("codegen/statements.jl")
include("codegen/expressions.jl")
include("codegen/values.jl")
