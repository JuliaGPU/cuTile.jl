# Codegen tests for `token_order_pass!`, focused on the loop-parallel store
# optimization: a store whose index tuple contains a provably injective
# affine function of the induction variable — on an array that cannot alias
# itself internally — may consume the pre-loop token instead of a
# loop-carried one (iterations write disjoint memory, so no WAW ordering is
# needed). Every other store must keep iteration-ordering tokens, observable
# as `iter_values` token carries on the `for` and a `join_tokens` feeding
# the store.

spec1d = ct.ArraySpec{1}(16, true, (0,), (16,))
# Same layout facts, but distinct indices may map to the same memory
# (e.g. a zero stride) — parallel stores must be disabled.
spec1d_aliasing = ct.ArraySpec{1}(16, true, (0,), (16,), true)
AT = ct.TileArray{Float32, 1, spec1d}
AT_aliasing = ct.TileArray{Float32, 1, spec1d_aliasing}

@testset "token_order — identity IV store is loop-parallel" begin
    # `store(b, i, _)` lowers the index as `subi(iv, 1)` — injective, so the
    # store consumes the pre-loop token and the loop carries no tokens.
    @test @filecheck begin
        @check_label "entry"
        @check "[[TOK:%.+]] = make_token"
        @check_not "iter_values"
        @check "store_view_tko{{.*}}token = [[TOK]] :"
        code_tiled(Tuple{AT, AT, Int32}) do a, b, n
            for i in 1:n
                t = ct.load(a, i, (16,))
                ct.store(b, i, t + t)
            end
            return
        end
    end
end

@testset "token_order — injective affine IV store is loop-parallel" begin
    # Odd multiplication is injective even when fixed-width arithmetic wraps.
    @test @filecheck begin
        @check_label "entry"
        @check "[[TOK:%.+]] = make_token"
        @check_not "iter_values"
        @check "store_view_tko{{.*}}token = [[TOK]] :"
        code_tiled(Tuple{AT, AT, Int32}) do a, b, n
            for i in 1:n
                t = ct.load(a, i, (16,))
                ct.store(b, 3i, t + t)
            end
            return
        end
    end
end

@testset "token_order — even IV multiplier keeps token carry" begin
    # `muli` returns the low half of the product, so `2i` is not injective
    # over the full fixed-width integer domain.
    @test @filecheck begin
        @check_label "entry"
        @check "make_token"
        @check "iter_values"
        @check "[[JOIN:%.+]] = join_tokens"
        @check "store_view_tko{{.*}}token = [[JOIN]] :"
        code_tiled(Tuple{AT, AT, Int32}) do a, b, n
            for i in 1:n
                t = ct.load(a, i, (16,))
                ct.store(b, 2i, t + t)
            end
            return
        end
    end
end

@testset "token_order — non-injective IV store keeps token carry" begin
    # `(i + 1) ÷ 2` maps two consecutive iterations to the same tile: the
    # stores WAW-race unless the loop keeps carrying an ordering token.
    @test @filecheck begin
        @check_label "entry"
        @check "make_token"
        @check "iter_values"
        @check "[[JOIN:%.+]] = join_tokens"
        @check "store_view_tko{{.*}}token = [[JOIN]] :"
        code_tiled(Tuple{AT, AT, Int32}) do a, b, n
            for i in 1:n
                t = ct.load(a, i, (16,))
                ct.store(b, (i + 1) ÷ 2, t + t)
            end
            return
        end
    end
end

@testset "token_order — internally-aliasing array keeps token carry" begin
    # Even an identity-IV store may overlap across iterations when the
    # array itself can map distinct indices to the same memory.
    @test @filecheck begin
        @check_label "entry"
        @check "make_token"
        @check "iter_values"
        @check "[[JOIN:%.+]] = join_tokens"
        @check "store_view_tko{{.*}}token = [[JOIN]] :"
        code_tiled(Tuple{AT, AT_aliasing, Int32}) do a, b, n
            for i in 1:n
                t = ct.load(a, i, (16,))
                ct.store(b, i, t + t)
            end
            return
        end
    end
end

@testset "token_order — two stores on one array keep token carries" begin
    # Two stores per iteration on the same alias set are never parallel,
    # regardless of their indices.
    @test @filecheck begin
        @check_label "entry"
        @check "make_token"
        @check "iter_values"
        code_tiled(Tuple{AT, AT, Int32}) do a, b, n
            for i in 1:n
                t = ct.load(a, i, (16,))
                ct.store(b, i, t)
                ct.store(b, i, t + t)
            end
            return
        end
    end
end

@testset "layout_may_alias_internally" begin
    lmai = ct.layout_may_alias_internally
    @test !lmai((Int32(4),), (Int32(1),))                        # contiguous 1-D
    @test lmai((Int32(4),), (Int32(0),))                         # zero stride
    @test !lmai((Int32(4), Int32(4)), (Int32(1), Int32(4)))      # column-major
    @test !lmai((Int32(4), Int32(4)), (Int32(4), Int32(1)))      # row-major
    @test !lmai((Int32(4), Int32(4)), (Int32(1), Int32(8)))      # padded rows
    @test lmai((Int32(4), Int32(4)), (Int32(1), Int32(2)))       # overlapping
    @test lmai((Int32(4), Int32(4)), (Int32(1), Int32(1)))       # repeated stride
    @test !lmai((Int32(1), Int32(4)), (Int32(0), Int32(1)))      # extent-1 dim ignored
    @test !lmai((Int32(3), Int32(4)), (Int32(-1), Int32(4)))     # negative stride, disjoint
    @test lmai((Int32(3), Int32(4)), (Int32(-1), Int32(2)))      # negative stride, overlapping

    # compute_array_spec derives the flag from the runtime layout
    ptr = reinterpret(Ptr{Float32}, C_NULL + 128)
    dense = ct.compute_array_spec(ptr, (Int32(4), Int32(4)), (Int32(1), Int32(4)))
    @test !dense.may_alias_internally
    broadcasted = ct.compute_array_spec(ptr, (Int32(4), Int32(4)), (Int32(1), Int32(0)))
    @test broadcasted.may_alias_internally

    # hand-written specs default to non-aliasing, matching upstream cuTile
    @test !ct.ArraySpec{2}(16, true).may_alias_internally
    @test spec1d_aliasing.may_alias_internally
end
