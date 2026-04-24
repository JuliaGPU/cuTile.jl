# Unit tests for the generic forward sparse dataflow framework.
#
# These tests pin driver ergonomics and convergence behaviour using a toy
# analysis over plain Julia IR (via IRStructurizer's `code_structured`).
# Deep CFG-correctness tests live in `test/codegen/*.jl`, where real cuTile
# intrinsic IR exercises `IfOp` / `ForOp` / `WhileOp` paths with shape
# that we can assert against.

using Core: SSAValue, Argument
using IRStructurizer: code_structured, Block, ForOp, instructions, stmt

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
    @test_throws ErrorException analyze(Oscillate(), sci)
end
