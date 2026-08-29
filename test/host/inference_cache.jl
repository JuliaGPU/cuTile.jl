@testset "Inference cache" begin
    choose_type(flag, x) = flag ? x + one(x) : Float32(x)
    world = Base.get_world_counter()
    cache = ct.inference_cache(world)
    mi = ct.lookup_method_instance(choose_type, Tuple{Bool, Int32}; world)

    ci = ct.get_ci(cache, mi)
    @test ct.get_ci(cache, mi) === ci
    _, generic_rt = ct.get_inferred(cache, ci, mi)
    @test generic_rt == Union{Int32, Float32}

    # Attach codegen results before looking up specialized inference source.
    # Both result types live on this CI; lookup must use the inference view.
    generic_job = ct.tile_job(mi, world; sm_arch=v"10.0", bytecode_version=v"13.2")
    generic = ct.cached_results(generic_job)
    specialized = ct.CuTileResults[]
    for (flag, expected_rt) in ((true, Int32), (false, Float32))
        argtypes = Any[ct.CC.Const(choose_type), ct.CC.Const(flag), Int32]
        specialized_ci = ct.get_ci(cache, mi; const_argtypes=argtypes)
        entry = ct.specialization(cache, specialized_ci, argtypes)
        @test entry isa ct.CompilerCaching.SpecializedResult{Nothing}
        job = ct.tile_job(mi, world; sm_arch=v"10.0", bytecode_version=v"13.2",
                         const_argtypes=argtypes)
        res = ct.cached_results(job)
        @test specialized_ci === ci
        @test res !== generic
        @test all(previous -> previous !== res, specialized)
        push!(specialized, res)

        # Source and return type must come from the same specialization.
        ir, rt = ct.get_inferred(cache, ci, mi; const_argtypes=argtypes)
        @test ir isa ct.CC.IRCode
        @test rt === expected_rt
        structured = ct.emit_structured(ir, rt)
        @test structured[2] === expected_rt

        # An equivalent argument vector must reuse the compiled results.
        @test ct.get_ci(cache, mi; const_argtypes=copy(argtypes)) === ci
        @test ct.specialization(cache, ci, copy(argtypes)) === entry
        @test ct.cached_results(job) === res
    end
end
