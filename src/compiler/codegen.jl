# Codegen: Julia IR -> Tile IR bytecode

include("codegen/utils.jl")
include("codegen/token_keys.jl")        # Defines TokenKey, TokenRole, ACQUIRE_TOKEN_KEY
include("codegen/alias_analysis.jl")    # Defines alias_analysis_pass!
include("codegen/token_order.jl")       # Defines get_alias_set, get_input_token!
include("codegen/kernel.jl")
include("codegen/control_flow.jl")
include("codegen/statements.jl")
include("codegen/expressions.jl")
include("codegen/values.jl")
