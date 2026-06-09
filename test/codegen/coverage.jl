@testset "device-code coverage" begin
    @inline function cov_only_scale(a, b)
        doubled = b * 2.0f0
        return a + doubled
    end

    function cov_kernel(a, b, c)
        pid = ct.bid(1)
        ta = ct.load(a, (pid,), (16,))
        tb = ct.load(b, (pid,), (16,))
        ct.store(c, (pid,), cov_only_scale(ta, tb))
        return
    end

    cov_spec = ct.ArraySpec{1}(16, true)
    cov_tt = Tuple{ct.TileArray{Float32,1,cov_spec}, ct.TileArray{Float32,1,cov_spec},
                ct.TileArray{Float32,1,cov_spec}}

    # Whether any line in `lo:hi` of `file` has a nonzero execution count in an lcov
    # tracefile.
    function lcov_any_covered(tracefile, file, lo, hi)
        in_block = false
        for l in eachline(tracefile)
            if startswith(l, "SF:")
                in_block = (l == "SF:" * file)
            elseif l == "end_of_record"
                in_block = false
            elseif in_block && startswith(l, "DA:")
                ln, cnt = parse.(Int, split(l[4:end], ","))
                lo <= ln <= hi && cnt >= 1 && return true
            end
        end
        return false
    end

    if Base.JLOptions().code_coverage == 0
        @test_skip "requires --code-coverage"
    else
        mktempdir() do dir
            # Emit Tile IR only — drives codegen (and the coverage marking) without
            # launching the kernel, so this needs no GPU execution of the body.
            ct.code_tiled(devnull, cov_kernel, cov_tt)

            # Flush coverage in-process and assert the device-only helper was marked.
            tracefile = joinpath(dir, "coverage.info")
            ccall(:jl_write_coverage_data, Cvoid, (Cstring,), tracefile)

            @test lcov_any_covered(tracefile, @__FILE__, m.line, m.line + 4)
        end
    end
end
