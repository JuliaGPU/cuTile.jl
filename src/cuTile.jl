module cuTile

using IRStructurizer
using IRStructurizer: Block, ControlFlowOp, BlockArgument,
                      YieldOp, ContinueOp, BreakOp, ConditionOp,
                      IfOp, ForOp, WhileOp, LoopOp, Undef,
                      SourceLocation
import IRStructurizer: operands, replace_uses!, insert_before!

using Base: compilerbarrier, donotdelete
using Core: MethodInstance, CodeInfo, SSAValue, Argument, SlotNumber,
            ReturnNode, PiNode, QuoteNode, GlobalRef
using Core.Compiler
const CC = Core.Compiler

using CUDA_Tile_jll

using CUDACore: CUDACore, CuArray

using BFloat16s: BFloat16
using EnumX
public BFloat16

import CompilerCaching
using CompilerCaching: CacheView, @setup_caching, method_instance, match_method_instance, typeinf!, results, lookup, get_source

# Shared definitions
include("shapes.jl")

# Bytecode infrastructure
include("bytecode/basic.jl")
include("bytecode/types.jl")
include("bytecode/writer.jl")
include("bytecode/encodings.jl")

# Language definitions
include("language/types.jl")
include("language/kernel_state.jl")

# Compiler implementation
include("compiler/interpreter.jl")
include("compiler/driver.jl")
include("compiler/reflection.jl")
include("compiler/utils.jl")
include("compiler/intrinsics.jl")
include("compiler/analysis/dataflow.jl")
include("compiler/analysis/alias.jl")
include("compiler/analysis/tilearray.jl")
include("compiler/analysis/constant.jl")
include("compiler/analysis/effects.jl")
include("compiler/analysis/divisibility.jl")
include("compiler/analysis/bounds.jl")
include("compiler/analysis/assume.jl")
include("compiler/transform/rewriter.jl")
include("compiler/transform/rewrite.jl")
include("compiler/transform/throws.jl")
include("compiler/transform/canonicalize.jl")
include("compiler/transform/control_flow.jl")
include("compiler/transform/token_keys.jl")
include("compiler/transform/token_order.jl")
include("compiler/transform/random.jl")
include("compiler/transform/cse.jl")
include("compiler/transform/no_wrap.jl")
include("compiler/transform/licm.jl")
include("compiler/transform/dce.jl")
include("compiler/transform/pipeline.jl")
include("compiler/codegen/debug.jl")
include("compiler/codegen/errors.jl")
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
include("language/random.jl")
include("language/atomics.jl")

# Host-level abstractions
include("utils.jl")
include("tiled.jl")
include("broadcast.jl")
include("mapreduce.jl")
include("cache.jl")
include("launch.jl")

public launch, TileBackend, DefaultBackend, Tiled, ByTarget,
       @compiler_options, @fpmode, @.,
       bytecode_version

# World age captured at __init__ time. The compilation pipeline
# (typeinf!, codegen, bytecode emission) is invoked in this world via
# `invoke_frozen` so that precompiled native code stays usable even after
# later-loaded packages would otherwise invalidate it. Default to typemax(UInt)
# so that during precompilation (before __init__ runs) `invoke_in_world` clamps
# to the current world and behaves normally.
const _initialization_world = Ref{UInt}(typemax(UInt))

"""
    invoke_frozen(f, args...; kwargs...)

Invoke `f(args...; kwargs...)` in the world captured at `__init__` time.
Lets precompiled native code for the cuTile compilation infrastructure
(`typeinf_local`, codegen, bytecode emission) stay live across method
insertions in later-loaded packages.
"""
function invoke_frozen(f, args...; kwargs...)
    @inline
    kwargs = merge(NamedTuple(), kwargs)
    if isempty(kwargs)
        return Base.invoke_in_world(_initialization_world[], f, args...)
    end
    return Base.invoke_in_world(_initialization_world[], Core.kwcall, kwargs, f, args...)
end

function __init__()
    _initialization_world[] = Base.get_world_counter()
    return
end

include("precompile.jl")

end # module cuTile
