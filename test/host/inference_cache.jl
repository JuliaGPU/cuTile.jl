@testset "Inference cache" begin
    choose_type(flag, x) = flag ? x + one(x) : Float32(x)
    world = Base.get_world_counter()
    cache = ct.inference_cache(world)
    mi = ct.lookup_method_instance(choose_type, Tuple{Bool, Int32}; world)

    ci = ct.infer(cache, mi)
    @test ci isa Core.CodeInstance && ci.owner === ct.TILE_CACHE_OWNER
    @test ct.infer(cache, mi) === ci
    _, generic_rt = ct.emit_julia(cache, mi)
    @test generic_rt == Union{Int32, Float32}

    # Every job of a kernel shares its inference result; results are stored per
    # config on the generic CodeInstance, or on the const-seeded entry.
    generic_job = ct.tile_job(mi, world; sm_arch=v"10.0", bytecode_version=v"13.2")
    @test ct.infer(generic_job) === ci
    generic = ct.cached_results(generic_job)
    @test ct.cached_results(generic_job) === generic
    hinted_job = ct.tile_job(mi, world; sm_arch=v"10.0", bytecode_version=v"13.2", opt_level=1)
    @test ct.infer(hinted_job) === ci
    @test ct.cached_results(hinted_job) !== generic

    specialized = ct.TileCompilerResults[]
    for (flag, expected_rt) in ((true, Int32), (false, Float32))
        const_argtypes = (ct.CC.Const(choose_type), ct.CC.Const(flag), Int32)
        job = ct.tile_job(mi, world; const_argtypes, sm_arch=v"10.0", bytecode_version=v"13.2")
        entry = ct.infer(job)
        @test entry isa ct.SpecializedResult{ct.JobResults}
        @test entry === ct.infer(cache, mi, collect(Any, const_argtypes))
        @test ct.infer(cache, mi) === ci

        ir, rt = ct.emit_julia(job)
        @test ir isa ct.CC.IRCode
        @test rt === expected_rt
        structured = ct.emit_structured(ir, rt)
        @test structured[2] === expected_rt

        res = ct.cached_results(job)
        @test res !== generic
        @test all(previous -> previous !== res, specialized)
        push!(specialized, res)

        # An equivalent job must reuse the inference result and the results slot.
        equivalent = ct.tile_job(mi, world; sm_arch=v"10.0", bytecode_version=v"13.2",
                                 const_argtypes=(ct.CC.Const(choose_type), ct.CC.Const(flag), Int32))
        @test equivalent === job
        @test ct.infer(equivalent) === entry
        @test ct.cached_results(equivalent) === res
    end
end

# Concatenation must not pass operands it can't type as tiles to generic
# collection code: when shapes are unknown (e.g. the non-constant inference of
# `rand(T, dims)`), those operands are `Any`, and the methods matching
# `maximum(ndims, ::Any)` drag host code into the kernel (#331). The probe's
# union stands in for `Any` with a host type whose `maximum` would be inferred
# if reached, regardless of how many other methods are loaded.
struct HostCatInput
    rank::Int
end
Base.maximum(::typeof(ndims), x::HostCatInput) = x.rank

@testset "Concatenation inference stays within tiles" begin
    cat_probe(x) = ct.cat_tiles(x[], 1)
    world = Base.get_world_counter()
    cache = ct.inference_cache(world)
    T = Union{Tuple{ct.Tile{Float32,Tuple{4}}}, HostCatInput}
    mi = ct.lookup_method_instance(cat_probe, Tuple{Base.RefValue{T}}; world)
    @test ct.inferred_rettype(ct.infer(cache, mi)) == ct.Tile{Float32,Tuple{4}}

    host = ct.lookup_method_instance(maximum, Tuple{typeof(ndims), HostCatInput}; world)
    @test get(cache, host, nothing) === nothing
    ct.infer(cache, host)
    @test get(cache, host, nothing) !== nothing
end
