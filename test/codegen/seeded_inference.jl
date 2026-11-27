# A kernel with `Constant` arguments is inferred with the constants known, and
# only so. Inferring it generically first typed its tiles as `Any`, letting
# inference wander into any method a dynamic call could match (#358), and
# stored the specialized code on a CodeInstance without its dependencies.

const SeededArray = ct.TileArray{Float32,1,Int32,ct.ArraySpec{1}(16, true)}

# A method no tile can reach: inferring it means a tile was typed `Any`.
struct SeededDecoy end
seeded_probe(x) = x
seeded_probe(::SeededDecoy) = nothing

function seeded_decoy_inferred()
    for mi in Base.specializations(only(methods(seeded_probe, Tuple{SeededDecoy})))
        isdefined(mi, :cache) || continue
        ci = mi.cache
        while true
            ci.owner === ct.TILE_CACHE_OWNER && return true
            isdefined(ci, :next) || break
            ci = ci.next
        end
    end
    return false
end

function seeded_decoy_kernel(a, c, n::Int)
    t = ct.load(a, ct.bid(1), (n,))
    ct.store(c, ct.bid(1), seeded_probe(t))
    return
end

@testset "constant arguments are not inferred generically" begin
    ct.code_typed(seeded_decoy_kernel, Tuple{SeededArray, SeededArray, ct.Constant{Int,32}})
    @test !seeded_decoy_inferred()
    world = Base.get_world_counter()
    mi = ct.kernel_method_instance(seeded_decoy_kernel,
                                   Tuple{SeededArray, SeededArray, Int}, world)
    @test !haskey(ct.inference_cache(world), mi)
end

# With four methods, generic inference of `seeded_scale(Val(n))` gives up
# without recording a dependency on the one the constant selects.
seeded_scale(::Val{8}) = 1f0
seeded_scale(::Val{16}) = 1f0
seeded_scale(::Val{32}) = 2f0
seeded_scale(::Val{64}) = 1f0

function seeded_scale_kernel(a, c, n::Int)
    t = ct.load(a, ct.bid(1), (n,))
    ct.store(c, ct.bid(1), t .* seeded_scale(Val(n)))
    return
end

@testset "constant arguments track their dependencies" begin
    tiled() = sprint(io -> ct.code_tiled(io, seeded_scale_kernel,
                                         Tuple{SeededArray, SeededArray, ct.Constant{Int,32}};
                                         sm_arch=v"10.0"))
    @test occursin("constant <f32: 2.000000e+00> : tile<32xf32>", Base.invokelatest(tiled))
    @eval seeded_scale(::Val{32}) = 3f0
    @test occursin("constant <f32: 3.000000e+00> : tile<32xf32>", Base.invokelatest(tiled))
end

@testset "source of constant arguments is recovered" begin
    tt = Tuple{SeededArray, SeededArray, ct.Constant{Int,16}}
    job = ct.tile_job(seeded_scale_kernel, tt; sm_arch=v"10.0")
    ci = ct.infer(job)
    results = ct.cached_results(job)

    # without the source a package image may drop, the constants are inferred again,
    # and the compilation results start over with that inference
    @atomic ci.inferred = nothing
    ir, _ = only(ct.code_typed(seeded_scale_kernel, tt))
    @test occursin("Tuple{16}", sprint(show, ir))
    @test ct.infer(job) !== ci
    @test ct.cached_results(job) !== results
    @test occursin("Tuple{16}", sprint(io -> ct.GPUCompiler.code_warntype(io, job)))
end
