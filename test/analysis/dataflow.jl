# Unit tests for the generic forward sparse dataflow framework.
#
# These tests pin driver ergonomics and convergence behaviour using a toy
# analysis over plain Julia IR (via IRStructurizer's `code_structured`).
# Deep CFG-correctness tests live in `test/codegen/*.jl`, where real cuTile
# intrinsic IR exercises `IfOp` / `ForOp` / `WhileOp` paths with shape
# that we can assert against.

using Core: SSAValue, Argument, ReturnNode, PiNode, GlobalRef
using IRStructurizer
using IRStructurizer: code_structured, Block, BlockArgument, ForOp, LoopOp,
                      IfOp, YieldOp, ContinueOp, ControlFlowOp,
                      StructuredIRCode, blocks, instructions

using cuTile: ForwardAnalysis, DataflowResult, analyze, record!, has_value, LatticeAnchor
import cuTile: bottom, top, tmerge, transfer, max_iters, operand_value

# ── toy analysis: trivial 3-state constant tracking on Int scalars ───────

struct CToP end
const CTOP = CToP()

struct ConstInt <: ForwardAnalysis{Union{Nothing, Int, CToP}} end

bottom(::ConstInt) = nothing
top(::ConstInt) = CTOP
function tmerge(::ConstInt, a, b)
    a === nothing && return b
    b === nothing && return a
    a === CTOP && return CTOP
    b === CTOP && return CTOP
    a == b ? a : CTOP
end

function operand_value(::ConstInt, r::DataflowResult, @nospecialize(op))
    op isa Integer && return Int(op)
    op isa QuoteNode && op.value isa Integer && return Int(op.value)
    op isa LatticeAnchor && return r[op]
    CTOP
end

# Post-inference the IR contains Core.Intrinsics.add_int / sub_int rather
# than Base.:+ / Base.:- (which are inlined away), so hook those.
function transfer(a::ConstInt, r::DataflowResult, @nospecialize(func), ops,
                  ::Block, ::Any)
    if (func === Core.Intrinsics.add_int || func === Core.Intrinsics.sub_int) &&
       length(ops) == 2
        x = operand_value(a, r, ops[1]); y = operand_value(a, r, ops[2])
        if x isa Int && y isa Int
            return func === Core.Intrinsics.add_int ? x + y : x - y
        end
    end
    return CTOP
end


# ── driver basics ────────────────────────────────────────────────────────

@testset "analyze runs on straight-line IR" begin
    sci, _ = code_structured(Tuple{Int}) do n::Int
        return n + 5 + 3
    end |> only
    r = analyze(ConstInt(), sci)

    # init_arg seeds CTOP for the closure and the Int arg.
    @test r[Argument(2)] === CTOP
    # Every statement SSA in the entry block should have been visited.
    for inst in instructions(sci.entry)
        inst[:stmt] isa Expr || continue
        @test has_value(r, SSAValue(inst.ssa_idx))
    end
end

@testset "analyze runs on a loop-bearing IR (ForOp / WhileOp)" begin
    sci, _ = code_structured(Tuple{Int, Int}) do n::Int, acc::Int
        i = 0
        while i < n
            acc += 1
            i += 1
        end
        return acc
    end |> only
    r = analyze(ConstInt(), sci)

    # The loop's body args should have received a lattice value (at minimum
    # their init_values propagated in via propagate_loop_carried!).
    for inst in instructions(sci.entry)
        s = inst[:stmt]
        if s isa ForOp
            @test has_value(r, s.iv_arg)          # IV arg seeded to ⊤
            for arg in s.body.args
                arg === s.iv_arg && continue
                @test has_value(r, arg)
            end
            break
        end
    end
end


# Dummy CF op used to exercise the generic `transfer_cf!` fallback.
struct FakeCFOp <: ControlFlowOp
    region::Block
end
IRStructurizer.blocks(op::FakeCFOp) = (op.region,)

@testset "fallback transfer_cf! handles unknown ControlFlowOp subtypes" begin
    # Sub-block has no instructions we can assert on, but we can confirm the
    # outer op's SSA was recorded as `top` — the safe default.
    region = Block()
    region.terminator = ReturnNode(nothing)
    entry = Block()
    push!(entry, 1, FakeCFOp(region), Any)
    entry.terminator = ReturnNode(nothing)

    sci = StructuredIRCode(Any[Any], Any[], entry, 1)
    r = analyze(ConstInt(), sci)

    @test r[SSAValue(1)] === CTOP
end

@testset "nested LoopOp carries don't leak across loop scope" begin
    # Two nested LoopOps. The inner ContinueOp's values target the inner
    # loop; they must NOT be merged into the outer loop's body.args.
    #
    # Outer carry: init 100, outer ContinueOp 100  → a_out stays at 100.
    # Inner carry: init 200, inner ContinueOp 999  → a_in collapses to CTOP.
    # If the framework crossed the inner loop scope, 999 would land on a_out
    # too, dragging it to CTOP.
    a_out = BlockArgument(1, Int)
    a_in  = BlockArgument(2, Int)

    inner_body = Block()
    push!(inner_body.args, a_in)
    inner_body.terminator = ContinueOp(Any[999])
    inner_loop = LoopOp(inner_body, Any[200])

    outer_body = Block()
    push!(outer_body.args, a_out)
    push!(outer_body, 1, inner_loop, Nothing)
    outer_body.terminator = ContinueOp(Any[100])
    outer_loop = LoopOp(outer_body, Any[100])

    entry = Block()
    push!(entry, 2, outer_loop, Nothing)
    entry.terminator = ReturnNode(nothing)

    sci = StructuredIRCode(Any[Any], Any[], entry, 2)
    r = analyze(ConstInt(), sci)

    @test r[a_out] === 100      # outer carry preserved; inner didn't leak in
    @test r[a_in]  === CTOP     # inner carry: 200 ⊔ 999 → ⊤
end


# ── soundness: ⊥ must never act as best-info at a join ──────────────────

@testset "PiNode passes through; unmodeled statements record ⊤" begin
    entry = Block()
    push!(entry, 1, Expr(:call, Core.Intrinsics.add_int, 2, 3), Int)
    push!(entry, 2, PiNode(SSAValue(1), Int), Int)
    push!(entry, 3, GlobalRef(Base, :pi), Any)
    entry.terminator = ReturnNode(SSAValue(2))
    sci = StructuredIRCode(Any[Any], Any[], entry, 3)
    r = analyze(ConstInt(), sci)

    @test r[SSAValue(2)] === 5      # PiNode forwards its operand's fact
    @test r[SSAValue(3)] === CTOP   # no transfer rule — never lingers at ⊥
end

@testset "PiNode-fed loop carry does not keep the init value's fact" begin
    # Loop carry: init 16, ContinueOp yields a PiNode of an unanalysed
    # argument. The PiNode's SSA used to linger at ⊥ (walk! skipped
    # non-Expr statements), and ⊥ is the merge identity — so the carry
    # incorrectly kept "divisible by 16" across iterations that replace
    # it with an arbitrary value.
    carry = BlockArgument(1, Int)
    body = Block()
    push!(body.args, carry)
    push!(body, 1, PiNode(Argument(2), Int), Int)
    body.terminator = ContinueOp(Any[SSAValue(1)])
    loop = LoopOp(body, Any[16])
    entry = Block()
    push!(entry, 2, loop, Nothing)
    entry.terminator = ReturnNode(nothing)
    sci = StructuredIRCode(Any[Any, Int], Any[], entry, 2)

    r = cuTile.analyze_divisibility(sci)
    @test cuTile.div_by(r, carry) == 1
end

@testset "ForOp merges continues nested behind IfOp branches" begin
    # A ContinueOp may sit inside a nested IfOp branch rather than as
    # the direct body terminator. Its values must be merged into the
    # loop carries like the direct back-edge's.
    iv = BlockArgument(1, Int)
    carry = BlockArgument(2, Int)
    then_r = Block()
    then_r.terminator = ContinueOp(Any[999])
    else_r = Block()
    else_r.terminator = YieldOp(Any[])
    body = Block()
    push!(body.args, carry)
    push!(body, 1, IfOp(true, then_r, else_r), Nothing)
    body.terminator = ContinueOp(Any[carry])
    fop = ForOp(0, 10, 1, iv, body, Any[100])
    entry = Block()
    push!(entry, 2, fop, Nothing)
    entry.terminator = ReturnNode(nothing)
    sci = StructuredIRCode(Any[Any], Any[], entry, 2)

    r = analyze(ConstInt(), sci)
    @test r[carry] === CTOP     # 100 ⊔ 999 → ⊤, not 100
end


# ── user-written Intrinsics.assume ───────────────────────────────────────

@testset "assume(DivBy) refines through a QuoteNode predicate" begin
    # Pass-constructed assumes embed the predicate value directly; user-
    # written kernels arrive with it const-folded into a QuoteNode. Both
    # must refine.
    entry = Block()
    push!(entry, 1, Expr(:call, cuTile.Intrinsics.assume, Argument(2),
                         QuoteNode(cuTile.DivBy(16))), Int)
    entry.terminator = ReturnNode(SSAValue(1))
    sci = StructuredIRCode(Any[Any, Int], Any[], entry, 1)

    r = cuTile.analyze_divisibility(sci)
    @test cuTile.div_by(r, SSAValue(1)) == 16
end


# ── exti signedness (divisibility) ───────────────────────────────────────

@testset "exti preserves sound divisibility" begin
    mk_exti(scalar, T, s) = begin
        S = typeof(scalar)
        entry = Block()
        push!(entry, 1, Expr(:call, cuTile.Intrinsics.constant,
                             QuoteNode(()), scalar, S), S)
        push!(entry, 2, Expr(:call, cuTile.Intrinsics.exti,
                             SSAValue(1), T, QuoteNode(s)), T)
        entry.terminator = ReturnNode(SSAValue(2))
        sci = StructuredIRCode(Any[Any], Any[], entry, 2)
        cuTile.analyze_divisibility(sci)
    end

    # Sign-extension preserves the value, and so the divisor.
    r = mk_exti(Int8(-6), Int32, cuTile.Signedness.Signed)
    @test cuTile.div_by(r, SSAValue(2)) == 6

    # Zero-extension of Int8(-6) yields 250 = -6 + 2^8, which is not a
    # multiple of 6; only divisors of gcd(6, 2^8) = 2 survive.
    r = mk_exti(Int8(-6), Int32, cuTile.Signedness.Unsigned)
    @test cuTile.div_by(r, SSAValue(2)) == 2

    # Sign-extension of UInt8(250) produces -6, so the same reduction applies.
    r = mk_exti(UInt8(250), Int32, cuTile.Signedness.Signed)
    @test cuTile.div_by(r, SSAValue(2)) == 2

    # A differently signed destination also reinterprets the result.
    r = mk_exti(Int8(-6), UInt32, cuTile.Signedness.Signed)
    @test cuTile.div_by(r, SSAValue(2)) == 2

    # The reduction also works when 2^srcbits does not fit in Int.
    r = mk_exti(Int64(-6), Int128, cuTile.Signedness.Unsigned)
    @test cuTile.div_by(r, SSAValue(2)) == 2
end


@testset "bitcast preserves only sound analysis facts" begin
    function bitcast_sci(scalar, T; target=T)
        S = typeof(scalar)
        entry = Block()
        push!(entry, 1, Expr(:call, cuTile.Intrinsics.constant,
                             QuoteNode(()), scalar, S), S)
        push!(entry, 2, Expr(:call, cuTile.Intrinsics.bitcast,
                             SSAValue(1), target), T)
        entry.terminator = ReturnNode(SSAValue(2))
        StructuredIRCode(Any[Any], Any[], entry, 2)
    end

    # Reinterpretation preserves bits, not the scalar value tracked by
    # ConstantAnalysis.
    sci = bitcast_sci(Int32(-1), UInt32)
    constants = cuTile.analyze_constants(sci)
    @test cuTile.const_value(constants, SSAValue(2)) === nothing

    # A signedness change can add 2^width to the mathematical value. Only
    # power-of-two divisors survive that change.
    divisibility = cuTile.analyze_divisibility(bitcast_sci(Int32(-3), UInt32))
    @test cuTile.div_by(divisibility, SSAValue(2)) == 1
    divisibility = cuTile.analyze_divisibility(bitcast_sci(Int32(-4), UInt32))
    @test cuTile.div_by(divisibility, SSAValue(2)) == 4
    divisibility = cuTile.analyze_divisibility(
        bitcast_sci(Int32(-4), UInt32; target=GlobalRef(Core, :UInt32)))
    @test cuTile.div_by(divisibility, SSAValue(2)) == 4
    divisibility = cuTile.analyze_divisibility(bitcast_sci(Int64(-6), UInt64))
    @test cuTile.div_by(divisibility, SSAValue(2)) == 2

    # Same-signedness integer bitcasts retain the divisor; non-integer
    # bitcasts do not carry integer divisibility facts.
    divisibility = cuTile.analyze_divisibility(bitcast_sci(Int32(-6), Int32))
    @test cuTile.div_by(divisibility, SSAValue(2)) == 6
    divisibility = cuTile.analyze_divisibility(bitcast_sci(Int32(-4), Float32))
    @test cuTile.div_by(divisibility, SSAValue(2)) == 1
end


# ── convergence cap ──────────────────────────────────────────────────────

# A deliberately non-monotone analysis: its `tmerge` never stabilises, so the
# outer fixpoint loop must hit `max_iters` and error cleanly.
struct Oscillate <: ForwardAnalysis{Int} end
bottom(::Oscillate) = 0
top(::Oscillate) = typemax(Int)
tmerge(::Oscillate, a, b) = a == b ? a : a + 1      # non-monotone
transfer(::Oscillate, r, @nospecialize(func), ops, ::Block, ::Any) = 1
max_iters(::Oscillate) = 3

@testset "non-convergence is detected and errors" begin
    sci, _ = code_structured(Tuple{Int, Int}) do n::Int, a::Int
        acc = a
        i = 0
        while i < n
            acc += 1
            i += 1
        end
        return acc
    end |> only
    err = try
        analyze(Oscillate(), sci)
        nothing
    catch e
        e
    end
    @test err isa ErrorException
    # The diagnostic should name the analysis and surface a concrete
    # offending anchor to point the reader at what oscillated.
    @test occursin("Oscillate", err.msg)
    @test occursin("last changed anchor", err.msg)
end
