# `Intrinsics.kernel_state()` plumbing tests
#
# `KernelState` is currently empty (a ghost), so the intrinsic is a no-op at
# the IR level — `flatten_struct_params!` adds zero kernel params and the
# returned ghost value is DCE'd. These tests verify the wiring tolerates the
# empty case so that adding a field later (e.g. an RNG seed) needs no codegen
# changes — just a struct field declaration.

@testset "kernel_state()" begin
    spec1d = ct.ArraySpec{1}(16, true)

    @testset "ghost state adds no kernel params" begin
        @test @filecheck begin
            # `TileArray{Int32,1}` flattens to (ptr, size, stride) → 3 params.
            # KernelState is ghost so no trailing param is appended.
            @check "(%arg0: tile<ptr<i32>>, %arg1: tile<i32>, %arg2: tile<i32>)"
            code_tiled(Tuple{ct.TileArray{Int32,1,spec1d}}) do a
                pid = ct.bid(1)
                Base.donotdelete(ct.Intrinsics.kernel_state())
                a[pid] = pid
                return
            end
        end
    end
end
