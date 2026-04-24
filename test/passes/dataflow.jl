# Unit tests for the generic forward sparse dataflow framework.
#
# These tests pin driver ergonomics and convergence behaviour using a toy
# analysis over plain Julia IR (via IRStructurizer's `code_structured`).
# Deep CFG-correctness tests live in `test/codegen/*.jl`, where real cuTile
# intrinsic IR exercises `IfOp` / `ForOp` / `WhileOp` paths with shape
# that we can assert against.

using Core: SSAValue, Argument, ReturnNode
using IRStructurizer
using IRStructurizer: code_structured, Block, BlockArgument, ForOp, LoopOp,
                      ContinueOp, ControlFlowOp, StructuredIRCode, blocks,
                      instructions, stmt

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
        stmt(inst) isa Expr || continue
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
        s = stmt(inst)
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
