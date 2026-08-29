using Serialization

@testset "toolchain versions" begin
    @test cuTile.parse_tileiras_version("tileiras V13.2.78\nBuild local") == v"13.2.78"
    @test_throws ArgumentError cuTile.parse_tileiras_version("future version format")
    @test_throws ArgumentError cuTile.parse_tileiras_version("")

    @test cuTile.select_bytecode_version(v"13.4", nothing) == v"13.4"
    @test cuTile.select_bytecode_version(v"13.4", v"13.2") == v"13.2"
    @test_throws ArgumentError cuTile.select_bytecode_version(v"13.3", v"13.4")
    @test_throws ArgumentError cuTile.select_bytecode_version(v"13.4", v"13.5")
    @test_throws ArgumentError cuTile.select_bytecode_version(v"13.5", nothing)
end

const CUBIN_CACHE_PRELUDE = quote
    using cuTile, Test
    import cuTile as ct

    const OC = cuTile.ObjCache
    const SM_ARCH = v"10.0"
    const OPT_LEVEL = 3

    spec = ct.ArraySpec{1}(16, true)
    const TT = Tuple{ct.TileArray{Float32,1,Int32,spec},
                     ct.TileArray{Float32,1,Int32,spec},
                     ct.TileArray{Float32,1,Int32,spec}}

    function cached_vadd(a, b, c)
        pid = ct.bid(1)
        tile_a = ct.load(a; index=pid, shape=(16,))
        tile_b = ct.load(b; index=pid, shape=(16,))
        ct.store(c; index=pid, tile=tile_a + tile_b)
        return
    end

    vadd_job() = ct.tile_job(cached_vadd, TT; sm_arch=SM_ARCH, opt_level=OPT_LEVEL)
    function compile_vadd()
        job = vadd_job()
        res = ct.compile_or_lookup(job)
        return (; cuda_bin=res.cuda_bin, tile_bc=ct.emit_bytecode(job).bytecode)
    end
    cubin_key(res) = OC.keyhash(ct.CUBIN_CACHE_SCHEMA,
                                ct.cubin_cache_fields(res.tile_bc,
                                                      SM_ARCH, OPT_LEVEL)...)
end

"""
Evaluate `expr` in a child Julia whose object cache is isolated at `objcache_path`.
Returns `(success, output)`.
"""
function run_objcache_child(expr::Expr, objcache_path::AbstractString; env=())
    project = dirname(Base.active_project())
    runner = joinpath(@__DIR__, "..", "objcache_child.jl")
    cmd = `$(Base.julia_cmd()) --startup-file=no --project=$project $runner`
    cmd = addenv(cmd, "JULIA_OBJCACHE_PATH" => objcache_path, env...)

    input = IOBuffer()
    serialize(input, (CUBIN_CACHE_PRELUDE, expr))
    seekstart(input)
    out = IOBuffer()
    proc = run(pipeline(ignorestatus(cmd); stdin=input, stdout=out, stderr=out))
    return success(proc), String(take!(out))
end

# Capture the final argument as Julia code, with `$` interpolation from the caller.
# The cache path comes first; any intervening arguments are environment key/value pairs.
macro with_objcache(objcache_path, args...)
    isempty(args) && error("@with_objcache requires a code block")
    expr = last(args)
    env = Expr(:tuple, map(esc, args[1:end-1])...)
    quote
        run_objcache_child($(esc(Expr(:quote, expr))), $(esc(objcache_path)); env=$env)
    end
end

function check_child(ok::Bool, out::AbstractString)
    ok || println(stderr, "--- child output ---\n", out, "\n--- end ---")
    @test ok
    return ok
end

digest(out) = match(r"CUBIN_SHA=([0-9a-f]+)", out)

# Cross-process persistence of the tileiras output. Process one compiles a kernel
# host-side (no CUDA context needed) and polls until the store has committed the CUBIN;
# process two must obtain the same CUBIN without invoking tileiras. Both run with the
# object cache isolated at a temporary JULIA_OBJCACHE_PATH.
@testset "cross-process CUBIN cache" begin
    mktempdir() do dir
        # Process one: compile, then wait for the asynchronous store commit before exiting.
        ok, out = @with_objcache dir "JULIA_OBJCACHE" => "1" begin
            @test OC.enabled()
            res = compile_vadd()
            @test res.cuda_bin isa Vector{UInt8} && !isempty(res.cuda_bin)
            k = cubin_key(res)
            t0 = time()
            while OC.get(ct.CUBIN_CACHE_NS, k) === nothing && time() - t0 < 10
                sleep(0.01)
            end
            @test OC.get(ct.CUBIN_CACHE_NS, k) == res.cuda_bin
            println("CUBIN_SHA=", bytes2hex(OC.keyhash(0, res.cuda_bin)))
        end
        check_child(ok, out)
        sha1 = digest(out)
        @test sha1 !== nothing
        expected_sha = sha1 === nothing ? "" : sha1[1]

        # Process two: tileiras must not run; the CUBIN comes from the store.
        ok, out = @with_objcache dir "JULIA_OBJCACHE" => "1" begin
            @test OC.enabled()
            ct.tileir_toolchain()  # discover the toolchain (runs `tileiras --version`) first
            @eval ct function run_tileiras(bytecode::Vector{UInt8},
                                           sm_arch::VersionNumber, opt_level::Int;
                                           remarks::Bool=false)
                error("tileiras invoked despite a warm object cache")
            end
            res = compile_vadd()
            @test res.cuda_bin isa Vector{UInt8} && !isempty(res.cuda_bin)
            @test bytes2hex(OC.keyhash(0, res.cuda_bin)) == $expected_sha
        end
        check_child(ok, out)
    end
end
