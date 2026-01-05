public code_structured

# Typed IR utilities for cuTile compilation
#
# Gets fully-typed SSA IR that we can translate to Tile IR bytecode.

"""
    get_typed_ir(f, argtypes; world=Base.get_world_counter(), optimize=true, always_inline=true) -> (CodeInfo, return_type)

Get the typed SSA IR for a function with the given argument types.
If optimize=true (default), returns optimized IR with clean SSA form.
If optimize=false, returns IR that preserves all intermediate assignments.
If always_inline=true (default), forces all functions to be inlined.
"""
function get_typed_ir(@nospecialize(f), @nospecialize(argtypes);
                      world::UInt = Base.get_world_counter(),
                      optimize::Bool = true,
                      always_inline::Bool = true)
    sig = Base.signature_type(f, argtypes)

    if always_inline
        opt_params = CC.OptimizationParams(; inline_cost_threshold=typemax(Int))
        interp = CC.NativeInterpreter(world; opt_params)
        results = Base.code_typed_by_type(sig; optimize, world, interp)
    else
        results = Base.code_typed_by_type(sig; optimize, world)
    end

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
