@testset "Inference cache" begin
    choose_type(flag, x) = flag ? x + one(x) : Float32(x)
    world = Base.get_world_counter()
    cache = ct.CacheView{ct.CuTileResults}(gensym(:inference_cache), world)
    mi = ct.lookup_method_instance(choose_type, Tuple{Bool, Int32}; world)

    ci, generic = ct.ensure_compiled(cache, mi, nothing)
    @test ct.ensure_compiled(cache, mi, nothing) === (ci, generic)
    _, generic_rt = ct.get_inferred(cache, ci, mi)
    @test generic_rt == Union{Int32, Float32}

    specialized = ct.CuTileResults[]
    for (flag, expected_rt) in ((true, Int32), (false, Float32))
        argtypes = Any[ct.CC.Const(choose_type), ct.CC.Const(flag), Int32]
        specialized_ci, res = ct.ensure_compiled(cache, mi, argtypes)
        @test specialized_ci === ci
        @test res !== generic
        @test all(previous -> previous !== res, specialized)
        push!(specialized, res)

        # Source and return type must come from the same specialization.
        ir, rt = ct.get_inferred(cache, ci, mi; const_argtypes=argtypes)
        @test ir isa ct.CC.IRCode
        @test rt === expected_rt
        structured = ct.emit_structured!(cache, mi, ci, res; const_argtypes=argtypes)
        @test structured[2] === expected_rt

        # An equivalent argument vector must reuse the compiled results.
        @test ct.ensure_compiled(cache, mi, copy(argtypes)) === (ci, res)
        @test ct.emit_structured!(cache, mi, ci, res;
                                  const_argtypes=argtypes) === structured
    end
end
