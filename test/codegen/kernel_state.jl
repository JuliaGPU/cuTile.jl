# `Intrinsics.kernel_state()` plumbing tests
#
# Verifies that the implicit `KernelState` arg is destructured into a trailing
# kernel parameter and that field access flows through the standard
# destructured-arg `getfield` path.

@testset "kernel_state()" begin
    spec1d = ct.ArraySpec{1}(16, true)

    @testset "seed routes through trailing kernel param" begin
        @test @filecheck begin
            # Three user-arg params from the TileArray (ptr, size, stride),
            # then one trailing KernelState.seed UInt32 → four params total.
            # The seed value flows directly into the store via reshape — no
            # struct construction, no extra ops.
            @check "(%arg0: tile<ptr<i32>>"
            @check "%arg3: tile<i32>"
            @check "reshape %arg3"
            code_tiled(Tuple{ct.TileArray{UInt32,1,spec1d}}) do a
                pid = ct.bid(1)
                s = ct.Intrinsics.kernel_state()
                a[pid] = s.seed
                return
            end
        end
    end
end
