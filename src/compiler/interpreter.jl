# Typed IR utilities for cuTile compilation
#
# Gets fully-typed SSA IR that we can translate to Tile IR bytecode.

using Core: MethodInstance, CodeInfo, SSAValue, Argument, GotoNode, GotoIfNot, ReturnNode
using Core.Compiler
const CC = Core.Compiler

"""
    get_typed_ir(f, argtypes; world=Base.get_world_counter(), optimize=true) -> (CodeInfo, return_type)

Get the typed SSA IR for a function with the given argument types.
If optimize=true (default), returns optimized IR with clean SSA form.
If optimize=false, returns IR that preserves all intermediate assignments.
"""
function get_typed_ir(@nospecialize(f), @nospecialize(argtypes);
                      world::UInt = Base.get_world_counter(),
                      optimize::Bool = true)
    sig = Base.signature_type(f, argtypes)
    results = Base.code_typed_by_type(sig; optimize, world)

    if isempty(results)
        error("Type inference failed for $f with argument types $argtypes")
    end

    ci, rettype = results[1]
    return ci, rettype
end

"""
    get_method_instance(f, argtypes; world=Base.get_world_counter()) -> MethodInstance

Get the MethodInstance for a function call.
"""
function get_method_instance(@nospecialize(f), @nospecialize(argtypes); world::UInt = Base.get_world_counter())
    tt = Base.signature_type(f, argtypes)
    match = Base._which(tt; world)
    return CC.specialize_method(match)
end
