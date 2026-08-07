spec1d = ct.ArraySpec{1}(16, true)
spec2d = ct.ArraySpec{2}(16, true)

@testset "@fpmode" begin

@testset "default (no @fpmode)" begin
    @test @filecheck begin
        @check_label "entry"
        code_tiled(Tuple{ct.TileArray{Float32,1,Int32,spec1d}}) do a
            pid = ct.bid(1)
            t = ct.load(a, pid, (16,))
            @check "addf"
            @check_not "rounding"
            @check_not "flush_to_zero"
            r = t + t
            ct.store(a, pid, r)
            return nothing
        end
    end
end

@testset "rounding_mode only" begin
    @test @filecheck begin
        @check_label "entry"
        code_tiled(Tuple{ct.TileArray{Float32,1,Int32,spec1d}}) do a
            pid = ct.bid(1)
            t = ct.load(a, pid, (16,))
            ct.@fpmode rounding_mode=ct.Rounding.Zero begin
                @check "addf"
                @check "rounding<zero>"
                @check_not "flush_to_zero"
                r = t + t
            end
            ct.store(a, pid, r)
            return nothing
        end
    end
end

@testset "flush_to_zero only" begin
    @test @filecheck begin
        @check_label "entry"
        code_tiled(Tuple{ct.TileArray{Float32,1,Int32,spec1d}}) do a
            pid = ct.bid(1)
            t = ct.load(a, pid, (16,))
            ct.@fpmode flush_to_zero=true begin
                @check "addf"
                @check "flush_to_zero"
                @check_not "rounding"
                r = t + t
            end
            ct.store(a, pid, r)
            return nothing
        end
    end
end

@testset "rounding_mode + flush_to_zero" begin
    @test @filecheck begin
        @check_label "entry"
        code_tiled(Tuple{ct.TileArray{Float32,1,Int32,spec1d}}) do a
            pid = ct.bid(1)
            t = ct.load(a, pid, (16,))
            ct.@fpmode rounding_mode=ct.Rounding.Zero flush_to_zero=true begin
                @check "addf"
                @check "rounding<zero>"
                @check "flush_to_zero"
                r = t + t
            end
            ct.store(a, pid, r)
            return nothing
        end
    end
end

@testset "scope: ops outside @fpmode are unaffected" begin
    @test @filecheck begin
        @check_label "entry"
        code_tiled(Tuple{ct.TileArray{Float32,1,Int32,spec1d}}) do a
            pid = ct.bid(1)
            t = ct.load(a, pid, (16,))
            # before: default
            @check "addf"
            @check_not "rounding"
            r1 = t + t
            ct.@fpmode rounding_mode=ct.Rounding.Zero flush_to_zero=true begin
                # inside: Zero + FTZ
                @check "addf"
                @check "rounding<zero>"
                @check "flush_to_zero"
                r2 = r1 + t
            end
            # after: default again
            @check "addf"
            @check_not "rounding"
            r3 = r2 + t
            ct.store(a, pid, r3)
            return nothing
        end
    end
end

@testset "nesting with inheritance" begin
    @test @filecheck begin
        @check_label "entry"
        code_tiled(Tuple{ct.TileArray{Float32,1,Int32,spec1d}}) do a
            pid = ct.bid(1)
            t = ct.load(a, pid, (16,))
            ct.@fpmode rounding_mode=ct.Rounding.Zero flush_to_zero=true begin
                # outer: Zero + FTZ
                @check "addf"
                @check "rounding<zero>"
                @check "flush_to_zero"
                r1 = t + t
                ct.@fpmode rounding_mode=ct.Rounding.NearestEven begin
                    # inner: NearestEven (overridden) + FTZ (inherited)
                    @check "addf"
                    @check_not "rounding"
                    @check "flush_to_zero"
                    r2 = r1 + t
                end
                # back to outer: Zero + FTZ
                @check "addf"
                @check "rounding<zero>"
                @check "flush_to_zero"
                r3 = r2 + t
            end
            ct.store(a, pid, r3)
            return nothing
        end
    end
end

@testset "subprogram propagation (reduction)" begin
    @test @filecheck begin
        @check_label "entry"
        code_tiled(Tuple{ct.TileArray{Float32,2,Int32,spec2d}}) do a
            pid = ct.bid(1)
            t = ct.load(a, pid, (4, 16))
            ct.@fpmode flush_to_zero=true begin
                @check "reduce"
                # The combiner body should inherit FTZ
                @check "addf"
                @check "flush_to_zero"
                s = sum(t; dims=2)
            end
            Base.donotdelete(s)
            return nothing
        end
    end
end

end
