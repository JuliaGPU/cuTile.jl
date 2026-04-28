# Codegen tests for `cse_pass!`: lightweight value-numbering on the
# StructuredIRCode. Mirrors LLVM's `EarlyCSE`: a recursive walk over
# the structured-control-flow tree maintains a per-scope hash table
# mapping `(func, return_type, operands)` to the canonical SSA, and
# replaces redundant computations with the canonical predecessor.

@testset "cse — TileView dedup on self-aliasing arg" begin
    # Three loads/stores on the same TileArray collapse to one
    # `make_tensor_view` and one `make_partition_view`. Without CSE,
    # each `ct.load`/`ct.store` would emit its own getfield+view chain.
    spec1d = ct.ArraySpec{1}(16, true, (4,), (16,))
    @test @filecheck begin
        @check_label "entry"
        code_tiled(Tuple{ct.TileArray{Float32,1,spec1d}}) do a
            t1 = ct.load(a, 1, (16,))
            t2 = ct.load(a, 1, (16,))
            ct.store(a, 1, t1 + t2)
            return
        end
        @check "make_tensor_view"
        @check_not "make_tensor_view"
    end
end

@testset "cse — distinct args don't dedup" begin
    spec1d = ct.ArraySpec{1}(16, true)
    @test @filecheck begin
        @check_label "entry"
        code_tiled(Tuple{ct.TileArray{Float32,1,spec1d},
                         ct.TileArray{Float32,1,spec1d}}) do a, b
            ct.store(b, 1, ct.load(a, 1, (16,)))
            return
        end
        @check "make_tensor_view"
        @check "make_tensor_view"
    end
end

@testset "cse — same-op different-result-type stay distinct" begin
    # Two `Intrinsics.broadcast(constant, shape)` calls with the same
    # operand list but different SCI return types (Tile{Int32} vs
    # Tile{Int64}) must not merge — CSE includes the return-type
    # annotation in the signature for exactly this reason.
    spec1d = ct.ArraySpec{1}(16, true)
    @test @filecheck begin
        @check_label "entry"
        code_tiled(Tuple{ct.TileArray{Int64,1,spec1d}}) do out
            a = ct.arange(16; dtype = Int64)
            b = ct.arange(16)  # default dtype = Int32
            # `a .- b` requires extending b to Int64 — no type collision
            # at codegen, but only if the two broadcast(1) ops stayed
            # distinct in the SCI.
            ct.store(out, 1, a .- b)
            return
        end
        @check "store"
    end
end

@testset "cse — parent-scope definitions visible in nested blocks" begin
    # A make_tensor_view in the entry block is reused by ct.store
    # *after* the if (which sees the entry's table), so the post-if
    # store doesn't emit a fresh chain. (CSE doesn't hoist redundant
    # definitions out of sibling branches — that's PRE, not value
    # numbering.)
    spec1d = ct.ArraySpec{1}(16, true)
    @test @filecheck begin
        @check_label "entry"
        code_tiled(Tuple{ct.TileArray{Float32,1,spec1d}, Bool}) do a, c
            t1 = ct.load(a, 1, (16,))     # entry: emits the canonical mtv
            if c
                Base.donotdelete(t1)
            end
            ct.store(a, 1, t1)            # reuses entry's mtv
            return
        end
        @check "make_tensor_view"
        @check_not "make_tensor_view"
    end
end

@testset "cse — redundancy entirely inside a nested block" begin
    # Both the canonical and the redundant make_tensor_view live inside
    # the if-branch. `replace_uses!` is called on the nested block, and
    # by SSA dominance every use of the redundant is confined to that
    # block's subtree — no walk up to the root needed.
    spec1d = ct.ArraySpec{1}(16, true)
    @test @filecheck begin
        @check_label "entry"
        code_tiled(Tuple{ct.TileArray{Float32,1,spec1d}, Bool}) do a, c
            if c
                ct.store(a, 1, ct.load(a, 1, (16,)))
                ct.store(a, 1, ct.load(a, 1, (16,)))
            end
            return
        end
        @check "make_tensor_view"
        @check_not "make_tensor_view"
    end
end

@testset "cse — sibling branches do not share table" begin
    # Identical expressions in then- and else- arms must stay distinct:
    # neither arm dominates the other, so additions in one branch must
    # not leak to its sibling.
    spec1d = ct.ArraySpec{1}(16, true)
    @test @filecheck begin
        @check_label "entry"
        code_tiled(Tuple{ct.TileArray{Float32,1,spec1d}, Bool}) do a, c
            if c
                ct.store(a, 1, ct.load(a, 1, (16,)))
            else
                ct.store(a, 1, ct.load(a, 1, (16,)))
            end
            return
        end
        @check "make_tensor_view"
        @check "make_tensor_view"
    end
end

@testset "cse — memory ops not deduplicated" begin
    # Two `ct.load(a, 1, …)` calls would have identical operand lists at
    # CSE time (token threading happens later, in `token_order_pass!`).
    # `is_pure_for_cse` bails at the `classify_memory_op == MEM_NONE` gate
    # — loads carry `IR_FLAG_EFFECT_FREE` (the load itself doesn't write)
    # but their value depends on memory state, not just operands.
    spec1d = ct.ArraySpec{1}(16, true)
    @test @filecheck begin
        @check_label "entry"
        code_tiled(Tuple{ct.TileArray{Float32,1,spec1d}}) do a
            t1 = ct.load(a, 1, (16,))
            t2 = ct.load(a, 1, (16,))
            ct.store(a, 1, t1 + t2)
            return
        end
        @check "load_view_tko"
        @check "load_view_tko"
    end
end
