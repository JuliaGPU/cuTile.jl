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

    # Execution count recorded for an exact line, or `nothing` if not instrumented.
    function lcov_line_count(tracefile, file, line)
        in_block = false
        for l in eachline(tracefile)
            if startswith(l, "SF:")
                in_block = (l == "SF:" * file)
            elseif l == "end_of_record"
                in_block = false
            elseif in_block && startswith(l, "DA:")
                ln, cnt = parse.(Int, split(l[4:end], ","))
                ln == line && return cnt
            end
        end
        return nothing
    end

    if Base.JLOptions().code_coverage == 0
        @test_skip "requires --code-coverage"
    else
        mktempdir() do dir
            # Emit Tile IR only — drives codegen (and the coverage marking) without
            # launching the kernel, so this needs no GPU execution of the body.
            ct.code_tiled(devnull, cov_kernel, cov_tt)

            # Flush coverage in-process. Both the inlined device-only helper and the
            # kernel entry must show covered lines, even though neither ran on the host.
            tracefile = joinpath(dir, "coverage.info")
            ccall(:jl_write_coverage_data, Cvoid, (Cstring,), tracefile)

            for f in (cov_only_scale, cov_kernel)
                m = only(methods(f))
                @test lcov_any_covered(tracefile, string(m.file), m.line, m.line + 4)
            end

            # The kernel entry's definition (signature) line specifically must be
            # covered — the per-statement coverage effects only mark body lines, so
            # without the prologue visit the header shows as missed (red). Use the
            # compiled entry `cov_kernel`, not the `@inline`d `cov_only_scale` whose
            # signature line is intentionally not marked (Julia doesn't mark it either).
            m = only(methods(cov_kernel))
            @test lcov_line_count(tracefile, string(m.file), m.line) !== nothing
            @test something(lcov_line_count(tracefile, string(m.file), m.line), 0) >= 1
        end
    end
end
