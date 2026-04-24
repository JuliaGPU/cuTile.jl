# Unit tests for the divisibility analysis pass (pipeline.jl:run_passes!).
#
# These test the propagation rules in isolation (no full kernel compilation).
# Phase 3 will add integration tests that verify AssumeOp(DivBy) emission
# against Python parity.

using cuTile: divby_for_call, operand_divby, update_divby_gcd!, abs_divby,
              cap_divby, MAX_DIVBY, DivByResult

@testset "divby helpers" begin
    @testset "abs_divby" begin
        @test abs_divby(4) == 4
        @test abs_divby(-4) == 4
        @test abs_divby(16) == 16
        @test abs_divby(0) == MAX_DIVBY      # 0 is divisible by everything
        @test abs_divby(2^20) == MAX_DIVBY   # caps at MAX_DIVBY
    end

    @testset "cap_divby" begin
        @test cap_divby(Int128(4)) == 4
        @test cap_divby(Int128(MAX_DIVBY)) == MAX_DIVBY
        @test cap_divby(Int128(MAX_DIVBY + 1)) == MAX_DIVBY
        @test cap_divby(Int128(0)) == 1      # snap non-positive to 1
        @test cap_divby(Int128(-5)) == 1
    end

    @testset "operand_divby" begin
        divby = DivByResult()
        divby[Core.SSAValue(3)] = 16
        @test operand_divby(divby, Core.SSAValue(3)) == 16
        @test operand_divby(divby, Core.SSAValue(99)) == 1   # unknown → 1
        @test operand_divby(divby, 32) == 32                 # literal
        @test operand_divby(divby, -8) == 8                  # abs
        @test operand_divby(divby, QuoteNode(4)) == 4
        @test operand_divby(divby, "hello") == 1             # non-integer → 1
    end

    @testset "update_divby_gcd monotonicity" begin
        divby = DivByResult()
        dirty = Ref(false)
        ssa = Core.SSAValue(1)

        update_divby_gcd!(divby, ssa, 16, dirty)
        @test divby[ssa] == 16 && dirty[] == true

        dirty[] = false
        update_divby_gcd!(divby, ssa, 16, dirty)   # same value, no change
        @test divby[ssa] == 16 && dirty[] == false

        dirty[] = false
        update_divby_gcd!(divby, ssa, 12, dirty)   # gcd(16,12) = 4
        @test divby[ssa] == 4 && dirty[] == true

        dirty[] = false
        update_divby_gcd!(divby, ssa, 100, dirty)  # gcd(4,100) = 4
        @test divby[ssa] == 4 && dirty[] == false
    end
end

@testset "divby propagation rules" begin
    divby = DivByResult()
    divby[Core.SSAValue(1)] = 16     # 16-divisible
    divby[Core.SSAValue(2)] = 24     # 24-divisible
    divby[Core.SSAValue(3)] = 2      # 2-divisible

    @testset "addi / subi → gcd" begin
        @test divby_for_call(divby, ct.Intrinsics.addi,
                             [Core.SSAValue(1), Core.SSAValue(2)]) == gcd(16, 24)  # 8
        @test divby_for_call(divby, ct.Intrinsics.subi,
                             [Core.SSAValue(1), Core.SSAValue(3)]) == gcd(16, 2)   # 2
    end

    @testset "muli → product" begin
        @test divby_for_call(divby, ct.Intrinsics.muli,
                             [Core.SSAValue(1), Core.SSAValue(3)]) == 16 * 2        # 32
        # Cap at MAX_DIVBY for large products.
        divby[Core.SSAValue(10)] = 128
        divby[Core.SSAValue(11)] = 64
        @test divby_for_call(divby, ct.Intrinsics.muli,
                             [Core.SSAValue(10), Core.SSAValue(11)]) == MAX_DIVBY
    end

    @testset "negi / absi → pass-through" begin
        @test divby_for_call(divby, ct.Intrinsics.negi,
                             [Core.SSAValue(1)]) == 16
        @test divby_for_call(divby, ct.Intrinsics.absi,
                             [Core.SSAValue(1)]) == 16
    end

    @testset "broadcast / reshape → pass-through" begin
        @test divby_for_call(divby, ct.Intrinsics.broadcast,
                             [Core.SSAValue(2), (4, 4)]) == 24
        @test divby_for_call(divby, ct.Intrinsics.reshape,
                             [Core.SSAValue(2), (16,)]) == 24
    end

    @testset "literals in operands" begin
        # literal 4 + known SSA(3) with divby=2 → gcd(4, 2) = 2
        @test divby_for_call(divby, ct.Intrinsics.addi,
                             [4, Core.SSAValue(3)]) == 2
        # literal 8 * known SSA(1) with divby=16 → 8 * 16 = 128
        @test divby_for_call(divby, ct.Intrinsics.muli,
                             [8, Core.SSAValue(1)]) == 128
    end

    @testset "unknown func → divby = 1" begin
        @test divby_for_call(divby, println,
                             [Core.SSAValue(1)]) == 1
    end
end
