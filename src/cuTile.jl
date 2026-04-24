module cuTile

using IRStructurizer
using IRStructurizer: Block, ControlFlowOp, BlockArgument,
                      YieldOp, ContinueOp, BreakOp, ConditionOp,
                      IfOp, ForOp, WhileOp, LoopOp, Undef,
                      SourceLocation
import IRStructurizer: operands

using Base: compilerbarrier, donotdelete
using Core: MethodInstance, CodeInfo, SSAValue, Argument, SlotNumber,
            ReturnNode, PiNode, QuoteNode, GlobalRef
using Core.Compiler
const CC = Core.Compiler

using CUDA_Tile_jll

using BFloat16s: BFloat16
using EnumX
public BFloat16

import CompilerCaching
using CompilerCaching: CacheView, @setup_caching, method_instance, match_method_instance, typeinf!, results, get_source

# Shared definitions
include("shapes.jl")

# Bytecode infrastructure
include("bytecode/basic.jl")
include("bytecode/types.jl")
include("bytecode/writer.jl")
include("bytecode/encodings.jl")

# Language definitions
include("language/types.jl")

# Compiler implementation
include("compiler/interpreter.jl")
include("compiler/driver.jl")
include("compiler/reflection.jl")
include("compiler/utils.jl")
include("compiler/intrinsics.jl")
include("compiler/analysis/dataflow.jl")
include("compiler/analysis/alias.jl")
include("compiler/analysis/constant.jl")
include("compiler/transform/rewrite.jl")
include("compiler/transform/canonicalize.jl")
include("compiler/transform/control_flow.jl")
include("compiler/transform/token_keys.jl")
include("compiler/transform/token_order.jl")
include("compiler/transform/licm.jl")
include("compiler/transform/dce.jl")
include("compiler/transform/pipeline.jl")
include("compiler/codegen/debug.jl")
include("compiler/codegen/kernel.jl")
include("compiler/codegen/control_flow.jl")
include("compiler/codegen/statements.jl")
include("compiler/codegen/expressions.jl")
include("compiler/codegen/values.jl")

# Language implementation
include("language/broadcast.jl")
include("language/overlays.jl")
include("language/arithmetic.jl")
include("language/math.jl")
include("language/operations.jl")
include("language/atomics.jl")

# Host-level abstractions
include("utils.jl")
include("tiled.jl")
include("broadcast.jl")
include("mapreduce.jl")

public launch, Tiled, ByTarget, @compiler_options, @fpmode, @.
launch(args...) = error("Please import CUDA.jl before using `cuTile.launch`.")

end # module cuTile
