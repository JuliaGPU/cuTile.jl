#=============================================================================
 Operations
=============================================================================#

@testset "Operations" begin

    #=========================================================================
     8.3 Core
    =========================================================================#
    @testset "Core" begin
        # TODO: cat - tile concatenation
        # TODO: extract - extract subtile
        # TODO: get_global - global variable access
        # TODO: global - global variable definition
        # TODO: mmai - integer matrix multiply-accumulate
        # TODO: offset - tile offset computation
        # TODO: pack - pack tiles
        # TODO: reshape - reshape tile dimensions
        # TODO: scan - parallel scan/prefix sum
        # TODO: unpack - unpack tiles

        @testset "broadcast" begin
            @test @filecheck begin
                check"CHECK-LABEL: entry"
                code_tiled(Tuple{Ptr{Float32}, Ptr{Float32}}) do a::Ptr{Float32}, b::Ptr{Float32}
                    pid = ct.bid(0)
                    tile = ct.load(a, pid, (1, 16))
                    check"CHECK: broadcast"
                    expanded = ct.broadcast_to(tile, (32, 16))
                    ct.store(b, pid, expanded)
                    return
                end
            end
        end

        @testset "cmpf" begin
            @test @filecheck begin
                check"CHECK-LABEL: entry"
                code_tiled(Tuple{Ptr{Float32}, Ptr{Float32}, Ptr{Float32}}) do a::Ptr{Float32}, b::Ptr{Float32}, c::Ptr{Float32}
                    pid = ct.bid(0)
                    tile_a = ct.load(a, pid, (16,))
                    tile_b = ct.load(b, pid, (16,))
                    check"CHECK: cmpf"
                    mask = tile_a .< tile_b
                    result = ct.where(mask, tile_a, tile_b)
                    ct.store(c, pid, result)
                    return
                end
            end
        end

        @testset "cmpi" begin
            @test @filecheck begin
                check"CHECK-LABEL: entry"
                code_tiled(Tuple{Ptr{Int32}, Ptr{Int32}, Ptr{Int32}}) do a::Ptr{Int32}, b::Ptr{Int32}, c::Ptr{Int32}
                    pid = ct.bid(0)
                    tile_a = ct.load(a, pid, (16,))
                    tile_b = ct.load(b, pid, (16,))
                    check"CHECK: cmpi"
                    mask = tile_a .< tile_b
                    result = ct.where(mask, tile_a, tile_b)
                    ct.store(c, pid, result)
                    return
                end
            end
        end

        @testset "constant" begin
            @test @filecheck begin
                check"CHECK-LABEL: entry"
                code_tiled(Tuple{Ptr{Float32}}) do a::Ptr{Float32}
                    pid = ct.bid(0)
                    check"CHECK: constant"
                    tile = ct.full((16,), 0.0f0, Float32)
                    ct.store(a, pid, tile)
                    return
                end
            end
        end

        @testset "get_num_tile_blocks" begin
            @test @filecheck begin
                check"CHECK-LABEL: entry"
                code_tiled(Tuple{Ptr{Float32}}) do a::Ptr{Float32}
                    check"CHECK: get_num_tile_blocks"
                    nb = ct.num_blocks(0)
                    return
                end
            end
        end

        @testset "get_tile_block_id" begin
            @test @filecheck begin
                check"CHECK-LABEL: entry"
                code_tiled(Tuple{Ptr{Float32}}) do a::Ptr{Float32}
                    check"CHECK: get_tile_block_id"
                    pid = ct.bid(0)
                    return
                end
            end
        end

        @testset "iota" begin
            @test @filecheck begin
                check"CHECK-LABEL: entry"
                code_tiled(Tuple{Ptr{Int32}}) do a::Ptr{Int32}
                    pid = ct.bid(0)
                    check"CHECK: iota"
                    tile = ct.arange((16,), Int32)
                    ct.store(a, pid, tile)
                    return
                end
            end
        end

        @testset "mmaf" begin
            @test @filecheck begin
                check"CHECK-LABEL: entry"
                code_tiled(Tuple{Ptr{Float32}, Ptr{Float32}, Ptr{Float32}}) do a::Ptr{Float32}, b::Ptr{Float32}, c::Ptr{Float32}
                    bidx = ct.bid(0)
                    bidy = ct.bid(1)
                    tile_a = ct.load(a, bidx, (32, 16))
                    tile_b = ct.load(b, bidy, (16, 32))
                    acc = ct.full((32, 32), 0.0f0, Float32)
                    check"CHECK: mma"
                    result = ct.mma(tile_a, tile_b, acc)
                    ct.store(c, (bidx, bidy), result)
                    return
                end
            end
        end

        @testset "permute" begin
            @test @filecheck begin
                check"CHECK-LABEL: entry"
                code_tiled(Tuple{Ptr{Float32}, Ptr{Float32}}) do a::Ptr{Float32}, b::Ptr{Float32}
                    bidx = ct.bid(0)
                    bidy = ct.bid(1)
                    tile = ct.load(a, (bidx, bidy), (32, 64))
                    check"CHECK: permute"
                    transposed = ct.transpose(tile)
                    ct.store(b, (bidy, bidx), transposed)
                    return
                end
            end
        end

        @testset "reduce" begin
            @test @filecheck begin
                check"CHECK-LABEL: entry"
                code_tiled(Tuple{Ptr{Float32}, Ptr{Float32}}) do a::Ptr{Float32}, b::Ptr{Float32}
                    pid = ct.bid(0)
                    tile = ct.load(a, pid, (4, 16))
                    check"CHECK: reduce"
                    check"CHECK: addf"
                    sums = ct.reduce_sum(tile, 1)
                    ct.store(b, pid, sums)
                    return
                end
            end

            @test @filecheck begin
                check"CHECK-LABEL: entry"
                code_tiled(Tuple{Ptr{Float32}, Ptr{Float32}}) do a::Ptr{Float32}, b::Ptr{Float32}
                    pid = ct.bid(0)
                    tile = ct.load(a, pid, (4, 16))
                    check"CHECK: reduce"
                    check"CHECK: maxf"
                    maxes = ct.reduce_max(tile, 1)
                    ct.store(b, pid, maxes)
                    return
                end
            end
        end

        @testset "select" begin
            @test @filecheck begin
                check"CHECK-LABEL: entry"
                code_tiled(Tuple{Ptr{Float32}, Ptr{Float32}, Ptr{Float32}}) do a::Ptr{Float32}, b::Ptr{Float32}, c::Ptr{Float32}
                    pid = ct.bid(0)
                    tile_a = ct.load(a, pid, (16,))
                    tile_b = ct.load(b, pid, (16,))
                    mask = tile_a .> tile_b
                    check"CHECK: select"
                    result = ct.where(mask, tile_a, tile_b)
                    ct.store(c, pid, result)
                    return
                end
            end
        end
    end

    #=========================================================================
     8.4 Conversions
    =========================================================================#
    @testset "Conversions" begin
        # TODO: bitcast - reinterpret bits as different type
        # TODO: exti - sign/zero extend integer
        # TODO: ftoi - float to integer
        # TODO: itof - integer to float
        # TODO: int_to_ptr - integer to pointer
        # TODO: ptr_to_int - pointer to integer
        # TODO: ptr_to_ptr - pointer cast
        # TODO: trunci - truncate integer

        @testset "ftof" begin
            # Float32 -> Float16
            @test @filecheck begin
                check"CHECK-LABEL: entry"
                code_tiled(Tuple{Ptr{Float32}, Ptr{Float16}}) do a::Ptr{Float32}, b::Ptr{Float16}
                    pid = ct.bid(0)
                    tile = ct.load(a, pid, (16,))
                    check"CHECK: ftof"
                    converted = ct.astype(tile, Float16)
                    ct.store(b, pid, converted)
                    return
                end
            end

            # Float32 -> TFloat32
            @test @filecheck begin
                check"CHECK-LABEL: entry"
                code_tiled(Tuple{Ptr{Float32}, Ptr{Float32}}) do a::Ptr{Float32}, b::Ptr{Float32}
                    pid = ct.bid(0)
                    tile = ct.load(a, pid, (16,))
                    check"CHECK: ftof"
                    converted = convert(ct.Tile{ct.TFloat32}, tile)
                    ct.store(b, pid, ct.astype(converted, Float32))
                    return
                end
            end
        end
    end

    #=========================================================================
     8.5 Control Flow
    =========================================================================#
    @testset "Control Flow" begin
        # TODO: assert - runtime assertion
        # TODO: if - conditional branching (explicit test)

        @testset "for" begin
            @test @filecheck begin
                check"CHECK-LABEL: entry"
                code_tiled(Tuple{Ptr{Float32}, Ptr{Float32}, Int32}) do a::Ptr{Float32}, b::Ptr{Float32}, n::Int32
                    pid = ct.bid(0)
                    acc = ct.full((16,), 0.0f0, Float32)
                    check"CHECK: for"
                    k = Int32(0)
                    while k < n
                        tile = ct.load(a, pid * n + k, (16,))
                        acc = acc + tile
                        k += Int32(1)
                    end
                    ct.store(b, pid, acc)
                    return
                end
            end
        end

        @testset "loop" begin
            spec = ct.ArraySpec{1}(16, true)
            @test @filecheck begin
                check"CHECK-LABEL: entry"
                code_tiled(Tuple{ct.TileArray{Int32,1,spec}, ct.TileArray{Float32,1,spec}}) do locks, data
                    bid = ct.bid(0)
                    check"CHECK: loop"
                    # Spinloop - unbounded iteration
                    while ct.atomic_cas(locks, bid, Int32(0), Int32(1);
                                       memory_order=ct.MemoryOrder.Acquire) == Int32(1)
                    end
                    tile = ct.load(data, bid, (16,))
                    ct.store(data, bid, tile)
                    ct.atomic_xchg(locks, bid, Int32(0);
                                  memory_order=ct.MemoryOrder.Release)
                    return
                end
            end
        end

        @testset "return" begin
            @test @filecheck begin
                check"CHECK-LABEL: entry"
                code_tiled(Tuple{Ptr{Float32}}) do a::Ptr{Float32}
                    check"CHECK: return"
                    return
                end
            end
        end
    end

    #=========================================================================
     8.6 Memory
    =========================================================================#
    @testset "Memory" begin
        # TODO: join_tokens - join multiple memory tokens
        # TODO: load_ptr_tko - direct pointer load (vs view-based)
        # TODO: store_ptr_tko - direct pointer store (vs view-based)

        @testset "Ptr load/store" begin
            # 1D
            @test @filecheck begin
                check"CHECK-LABEL: entry"
                code_tiled(Tuple{Ptr{Float32}, Ptr{Float32}}) do a::Ptr{Float32}, b::Ptr{Float32}
                    pid = ct.bid(0)
                    check"CHECK: load_view_tko"
                    tile = ct.load(a, pid, (16,))
                    check"CHECK: store_view_tko"
                    ct.store(b, pid, tile)
                    return
                end
            end

            # 2D
            @test @filecheck begin
                check"CHECK-LABEL: entry"
                code_tiled(Tuple{Ptr{Float32}, Ptr{Float32}}) do a::Ptr{Float32}, b::Ptr{Float32}
                    bidx = ct.bid(0)
                    bidy = ct.bid(1)
                    check"CHECK: load_view_tko"
                    tile = ct.load(a, (bidx, bidy), (32, 32))
                    check"CHECK: store_view_tko"
                    ct.store(b, (bidx, bidy), tile)
                    return
                end
            end
        end

        @testset "make_token" begin
            @test @filecheck begin
                check"CHECK-LABEL: entry"
                code_tiled(Tuple{Ptr{Float32}}) do a::Ptr{Float32}
                    check"CHECK: make_token"
                    return
                end
            end
        end
    end

    #=========================================================================
     8.7 Floating Point
    =========================================================================#
    @testset "Floating Point" begin
        # TODO: absf - absolute value
        # TODO: atan2 - two-argument arctangent
        # TODO: ceil - ceiling
        # TODO: cos - cosine
        # TODO: cosh - hyperbolic cosine
        # TODO: exp - exponential
        # TODO: exp2 - base-2 exponential
        # TODO: floor - floor
        # TODO: fma - fused multiply-add
        # TODO: log - natural logarithm
        # TODO: log2 - base-2 logarithm
        # TODO: maxf - element-wise maximum
        # TODO: minf - element-wise minimum
        # TODO: negf - negation
        # TODO: pow - power
        # TODO: remf - floating-point remainder
        # TODO: rsqrt - reciprocal square root
        # TODO: sin - sine
        # TODO: sinh - hyperbolic sine
        # TODO: tan - tangent
        # TODO: tanh - hyperbolic tangent

        @testset "addf" begin
            @test @filecheck begin
                check"CHECK-LABEL: entry"
                code_tiled(Tuple{Ptr{Float32}, Ptr{Float32}, Ptr{Float32}}) do a::Ptr{Float32}, b::Ptr{Float32}, c::Ptr{Float32}
                    pid = ct.bid(0)
                    tile_a = ct.load(a, pid, (16,))
                    tile_b = ct.load(b, pid, (16,))
                    check"CHECK: addf"
                    result = tile_a + tile_b
                    ct.store(c, pid, result)
                    return
                end
            end
        end

        @testset "subf" begin
            @test @filecheck begin
                check"CHECK-LABEL: entry"
                code_tiled(Tuple{Ptr{Float32}, Ptr{Float32}, Ptr{Float32}}) do a::Ptr{Float32}, b::Ptr{Float32}, c::Ptr{Float32}
                    pid = ct.bid(0)
                    tile_a = ct.load(a, pid, (16,))
                    tile_b = ct.load(b, pid, (16,))
                    check"CHECK: subf"
                    result = tile_a - tile_b
                    ct.store(c, pid, result)
                    return
                end
            end
        end

        @testset "mulf" begin
            @test @filecheck begin
                check"CHECK-LABEL: entry"
                code_tiled(Tuple{Ptr{Float32}, Ptr{Float32}, Ptr{Float32}}) do a::Ptr{Float32}, b::Ptr{Float32}, c::Ptr{Float32}
                    pid = ct.bid(0)
                    tile_a = ct.load(a, pid, (16,))
                    tile_b = ct.load(b, pid, (16,))
                    check"CHECK: mulf"
                    result = tile_a * tile_b
                    ct.store(c, pid, result)
                    return
                end
            end
        end

        @testset "divf" begin
            @test @filecheck begin
                check"CHECK-LABEL: entry"
                code_tiled(Tuple{Ptr{Float32}, Ptr{Float32}, Ptr{Float32}}) do a::Ptr{Float32}, b::Ptr{Float32}, c::Ptr{Float32}
                    pid = ct.bid(0)
                    tile_a = ct.load(a, pid, (16,))
                    tile_b = ct.load(b, pid, (16,))
                    check"CHECK: divf"
                    result = tile_a / tile_b
                    ct.store(c, pid, result)
                    return
                end
            end
        end

        @testset "sqrt" begin
            @test @filecheck begin
                check"CHECK-LABEL: entry"
                code_tiled(Tuple{Ptr{Float32}, Ptr{Float32}}) do a::Ptr{Float32}, b::Ptr{Float32}
                    pid = ct.bid(0)
                    tile = ct.load(a, pid, (16,))
                    check"CHECK: sqrt"
                    result = sqrt(tile)
                    ct.store(b, pid, result)
                    return
                end
            end
        end

        @testset "scalar broadcast" begin
            # tile + scalar
            @test @filecheck begin
                check"CHECK-LABEL: entry"
                code_tiled(Tuple{Ptr{Float32}, Ptr{Float32}}) do a::Ptr{Float32}, b::Ptr{Float32}
                    pid = ct.bid(0)
                    tile = ct.load(a, pid, (16,))
                    check"CHECK: broadcast"
                    check"CHECK: addf"
                    result = tile .+ 1.0f0
                    ct.store(b, pid, result)
                    return
                end
            end

            # scalar - tile
            @test @filecheck begin
                check"CHECK-LABEL: entry"
                code_tiled(Tuple{Ptr{Float32}, Ptr{Float32}}) do a::Ptr{Float32}, b::Ptr{Float32}
                    pid = ct.bid(0)
                    tile = ct.load(a, pid, (16,))
                    check"CHECK: broadcast"
                    check"CHECK: subf"
                    result = 1.0f0 .- tile
                    ct.store(b, pid, result)
                    return
                end
            end

            # tile * scalar
            @test @filecheck begin
                check"CHECK-LABEL: entry"
                code_tiled(Tuple{Ptr{Float32}, Ptr{Float32}}) do a::Ptr{Float32}, b::Ptr{Float32}
                    pid = ct.bid(0)
                    tile = ct.load(a, pid, (16,))
                    check"CHECK: broadcast"
                    check"CHECK: mulf"
                    result = tile .* 2.0f0
                    ct.store(b, pid, result)
                    return
                end
            end

            # tile / scalar
            @test @filecheck begin
                check"CHECK-LABEL: entry"
                code_tiled(Tuple{Ptr{Float32}, Ptr{Float32}}) do a::Ptr{Float32}, b::Ptr{Float32}
                    pid = ct.bid(0)
                    tile = ct.load(a, pid, (16,))
                    check"CHECK: broadcast"
                    check"CHECK: divf"
                    result = tile ./ 2.0f0
                    ct.store(b, pid, result)
                    return
                end
            end
        end
    end

    #=========================================================================
     8.8 Integer
    =========================================================================#
    @testset "Integer" begin
        # TODO: absi - absolute value
        # TODO: maxi - element-wise maximum
        # TODO: muli - multiplication (explicit test)
        # TODO: mulhii - high bits of multiplication
        # TODO: negi - negation
        # TODO: ori - bitwise OR
        # TODO: shli - shift left
        # TODO: shri - shift right
        # TODO: subi - subtraction (explicit test)
        # TODO: xori - bitwise XOR

        @testset "addi" begin
            @test @filecheck begin
                check"CHECK-LABEL: entry"
                code_tiled(Tuple{Ptr{Int32}, Ptr{Int32}}) do a::Ptr{Int32}, b::Ptr{Int32}
                    pid = ct.bid(0)
                    tile = ct.load(a, pid, (16,))
                    check"CHECK: addi"
                    result = tile + tile
                    ct.store(b, pid, result)
                    return
                end
            end
        end

        @testset "divi" begin
            @test @filecheck begin
                check"CHECK-LABEL: entry"
                code_tiled(Tuple{Ptr{Float32}, Int32}) do a::Ptr{Float32}, N::Int32
                    check"CHECK: divi"
                    result = ct.floordiv(N, Int32(16))
                    return
                end
            end
        end

        @testset "mini" begin
            @test @filecheck begin
                check"CHECK-LABEL: entry"
                code_tiled(Tuple{Ptr{Float32}, Int32}) do a::Ptr{Float32}, N::Int32
                    check"CHECK: mini"
                    result = min(N, Int32(256))
                    return
                end
            end
        end

        @testset "remi" begin
            @test @filecheck begin
                check"CHECK-LABEL: entry"
                code_tiled(Tuple{Ptr{Float32}, Int32}) do a::Ptr{Float32}, N::Int32
                    check"CHECK: remi"
                    result = rem(N, Int32(16))
                    return
                end
            end
        end
    end

    #=========================================================================
     8.9 Bitwise
    =========================================================================#
    # TODO: andi - bitwise AND

    #=========================================================================
     8.10 Atomics
    =========================================================================#
    @testset "Atomics" begin
        @testset "atomic_cas_tko" begin
            spec = ct.ArraySpec{1}(16, true)
            @test @filecheck begin
                check"CHECK-LABEL: entry"
                code_tiled(Tuple{ct.TileArray{Int32,1,spec}}) do locks
                    bid = ct.bid(0)
                    check"CHECK: atomic_cas_tko"
                    old = ct.atomic_cas(locks, bid, Int32(0), Int32(1);
                                        memory_order=ct.MemoryOrder.Acquire)
                    return
                end
            end
        end

        @testset "atomic_rmw_tko" begin
            spec = ct.ArraySpec{1}(16, true)
            # xchg
            @test @filecheck begin
                check"CHECK-LABEL: entry"
                code_tiled(Tuple{ct.TileArray{Int32,1,spec}}) do locks
                    bid = ct.bid(0)
                    check"CHECK: atomic_rmw_tko"
                    ct.atomic_xchg(locks, bid, Int32(0);
                                  memory_order=ct.MemoryOrder.Release)
                    return
                end
            end

            # add
            spec_f32 = ct.ArraySpec{1}(16, true)
            @test @filecheck begin
                check"CHECK-LABEL: entry"
                code_tiled(Tuple{ct.TileArray{Float32,1,spec_f32}}) do counters
                    bid = ct.bid(0)
                    check"CHECK: atomic_rmw_tko"
                    ct.atomic_add(counters, bid, 1.0f0)
                    return
                end
            end
        end
    end

    #=========================================================================
     8.11 Views
    =========================================================================#
    @testset "Views" begin
        # TODO: get_index_space_shape - get partition index space shape
        # TODO: get_tensor_shape - get tensor shape

        @testset "load_view_tko / store_view_tko" begin
            spec = ct.ArraySpec{1}(16, true)
            # 1D
            @test @filecheck begin
                check"CHECK-LABEL: entry"
                code_tiled(Tuple{ct.TileArray{Float32,1,spec}, ct.TileArray{Float32,1,spec}}) do a, b
                    pid = ct.bid(0)
                    check"CHECK: load_view_tko"
                    tile = ct.load(a, pid, (16,))
                    check"CHECK: store_view_tko"
                    ct.store(b, pid, tile)
                    return
                end
            end

            # 2D
            spec2d = ct.ArraySpec{2}(16, true)
            @test @filecheck begin
                check"CHECK-LABEL: entry"
                code_tiled(Tuple{ct.TileArray{Float32,2,spec2d}, ct.TileArray{Float32,2,spec2d}}) do a, b
                    bidx = ct.bid(0)
                    bidy = ct.bid(1)
                    check"CHECK: load_view_tko"
                    tile = ct.load(a, (bidx, bidy), (32, 32))
                    check"CHECK: store_view_tko"
                    ct.store(b, (bidx, bidy), tile)
                    return
                end
            end

            # with padding_mode
            @test @filecheck begin
                check"CHECK-LABEL: entry"
                code_tiled(Tuple{ct.TileArray{Float32,1,spec}, ct.TileArray{Float32,1,spec}}) do a, b
                    pid = ct.bid(0)
                    check"CHECK: load_view_tko"
                    tile = ct.load(a, pid, (16,); padding_mode=ct.PaddingMode.Zero)
                    ct.store(b, pid, tile)
                    return
                end
            end
        end

        @testset "num_tiles helper" begin
            spec = ct.ArraySpec{2}(16, true)
            @test @filecheck begin
                check"CHECK-LABEL: entry"
                code_tiled(Tuple{ct.TileArray{Float32,2,spec}}) do a
                    check"CHECK: addi"
                    check"CHECK: divi"
                    num = ct.num_tiles(a, 0, (32, 32))
                    return
                end
            end
        end
    end

    #=========================================================================
     8.12 Miscellaneous
    =========================================================================#
    # TODO: assume - optimization hint
    # TODO: print_tko - debug printing
end

#=============================================================================
 Type Support
=============================================================================#

@testset "Type Support" begin
    @testset "Float32" begin
        @test @filecheck begin
            check"CHECK-LABEL: entry"
            code_tiled(Tuple{Ptr{Float32}, Ptr{Float32}}) do a::Ptr{Float32}, b::Ptr{Float32}
                pid = ct.bid(0)
                tile = ct.load(a, pid, (16,))
                ct.store(b, pid, tile)
                return
            end
        end
    end

    @testset "Float64" begin
        @test @filecheck begin
            check"CHECK-LABEL: entry"
            code_tiled(Tuple{Ptr{Float64}, Ptr{Float64}}) do a::Ptr{Float64}, b::Ptr{Float64}
                pid = ct.bid(0)
                tile = ct.load(a, pid, (16,))
                check"CHECK: addf"
                result = tile + tile
                ct.store(b, pid, result)
                return
            end
        end
    end

    @testset "Float16" begin
        @test @filecheck begin
            check"CHECK-LABEL: entry"
            code_tiled(Tuple{Ptr{Float16}, Ptr{Float16}}) do a::Ptr{Float16}, b::Ptr{Float16}
                pid = ct.bid(0)
                tile = ct.load(a, pid, (16,))
                check"CHECK: addf"
                result = tile + tile
                ct.store(b, pid, result)
                return
            end
        end
    end

    @testset "Int32" begin
        @test @filecheck begin
            check"CHECK-LABEL: entry"
            code_tiled(Tuple{Ptr{Int32}, Ptr{Int32}}) do a::Ptr{Int32}, b::Ptr{Int32}
                pid = ct.bid(0)
                tile = ct.load(a, pid, (16,))
                check"CHECK: addi"
                result = tile + tile
                ct.store(b, pid, result)
                return
            end
        end
    end
end

#=============================================================================
 Integration Tests
=============================================================================#

@testset "Integration" begin
    @testset "vector add kernel" begin
        spec = ct.ArraySpec{1}(16, true)
        @test @filecheck begin
            check"CHECK-LABEL: entry"
            code_tiled(Tuple{ct.TileArray{Float32,1,spec}, ct.TileArray{Float32,1,spec}, ct.TileArray{Float32,1,spec}, ct.Constant{Int,16}}) do a, b, c, tile
                check"CHECK: get_tile_block_id"
                bid = ct.bid(0)
                check"CHECK: load_view_tko"
                a_tile = ct.load(a, bid, (tile[],))
                check"CHECK: load_view_tko"
                b_tile = ct.load(b, bid, (tile[],))
                check"CHECK: addf"
                result = a_tile + b_tile
                check"CHECK: store_view_tko"
                ct.store(c, bid, result)
                check"CHECK: return"
                return
            end
        end
    end

    @testset "transpose kernel" begin
        spec = ct.ArraySpec{2}(16, true)
        @test @filecheck begin
            check"CHECK-LABEL: entry"
            code_tiled(Tuple{ct.TileArray{Float32,2,spec}, ct.TileArray{Float32,2,spec}, ct.Constant{Int,32}, ct.Constant{Int,32}}) do x, y, tm, tn
                check"CHECK: get_tile_block_id"
                bidx = ct.bid(0)
                bidy = ct.bid(1)
                check"CHECK: load_view_tko"
                input_tile = ct.load(x, (bidx, bidy), (tm[], tn[]))
                check"CHECK: permute"
                transposed_tile = ct.transpose(input_tile)
                check"CHECK: store_view_tko"
                ct.store(y, (bidy, bidx), transposed_tile)
                check"CHECK: return"
                return
            end
        end
    end

    @testset "matmul reduction loop" begin
        spec = ct.ArraySpec{2}(16, true)
        @test @filecheck begin
            check"CHECK-LABEL: entry"
            code_tiled(Tuple{ct.TileArray{Float32,2,spec}, ct.TileArray{Float32,2,spec}, ct.TileArray{Float32,2,spec}, ct.Constant{Int,32}, ct.Constant{Int,32}, ct.Constant{Int,16}}) do A, B, C, tm, tn, tk
                bid = ct.bid(0)
                num_k = ct.num_tiles(A, 1, (tm[], tk[]))
                check"CHECK: broadcast"
                acc = ct.full((tm[], tn[]), zero(Float32), Float32)
                check"CHECK: for"
                k = Int32(0)
                while k < num_k
                    check"CHECK: load_view_tko"
                    a = ct.load(A, (bid, k), (tm[], tk[]); padding_mode=ct.PaddingMode.Zero)
                    check"CHECK: load_view_tko"
                    b = ct.load(B, (k, bid), (tk[], tn[]); padding_mode=ct.PaddingMode.Zero)
                    check"CHECK: mma"
                    acc = ct.mma(a, b, acc)
                    k += Int32(1)
                end
                check"CHECK: store_view_tko"
                ct.store(C, (bid, bid), acc)
                return
            end
        end
    end

    @testset "layernorm forward pattern (multiple sequential for loops)" begin
        # This test captures the pattern from layer_norm_fwd:
        # Multiple sequential for loops (mean accumulation, then output pass)
        spec = ct.ArraySpec{2}(16, true)
        spec1d = ct.ArraySpec{1}(16, true)
        @test @filecheck begin
            check"CHECK-LABEL: entry"
            code_tiled(Tuple{ct.TileArray{Float32,2,spec}, ct.TileArray{Float32,2,spec},
                           ct.TileArray{Float32,1,spec1d}, ct.Constant{Int,16}}) do X, Y, Sum, TILE_N
                bid_m = ct.bid(0)
                num_tiles = ct.num_tiles(X, 1, (1, TILE_N[]))

                # First for loop: compute sum
                check"CHECK: for"
                acc = ct.full((1, TILE_N[]), 0.0f0, Float32)
                j = Int32(0)
                while j < num_tiles
                    tx = ct.load(X, (bid_m, j), (1, TILE_N[]); padding_mode=ct.PaddingMode.Zero)
                    acc = acc .+ tx
                    j += Int32(1)
                end
                check"CHECK: reduce"
                sum_val = ct.reduce_sum(acc, 1)
                ct.store(Sum, bid_m, sum_val)

                # Second for loop: scale output by sum
                check"CHECK: for"
                j = Int32(0)
                while j < num_tiles
                    tx = ct.load(X, (bid_m, j), (1, TILE_N[]); padding_mode=ct.PaddingMode.Zero)
                    ty = tx .* sum_val
                    ct.store(Y, (bid_m, j), ty)
                    j += Int32(1)
                end
                return
            end
        end
    end

    @testset "layernorm backward pattern (atomic spinloop)" begin
        # This test captures the actual pattern from layer_norm_bwd_dx_partial_dwdb:
        # A for loop iterating over tiles, with a spinloop inside for atomic accumulation
        spec = ct.ArraySpec{1}(16, true)
        spec2d = ct.ArraySpec{2}(16, true)
        @test @filecheck begin
            check"CHECK-LABEL: entry"
            code_tiled(Tuple{ct.TileArray{Float32,2,spec2d}, ct.TileArray{Float32,2,spec2d},
                           ct.TileArray{Int32,1,spec}, Int32, ct.Constant{Int,16}}) do DW, Partial, Locks, group_bid, TILE_N
                bid = ct.bid(0)
                num_tiles = ct.num_tiles(DW, 1, (1, TILE_N[]))

                check"CHECK: for"
                j = Int32(0)
                while j < num_tiles
                    # Load and compute partial result
                    partial = ct.load(Partial, (bid, j), (1, TILE_N[]); padding_mode=ct.PaddingMode.Zero)

                    check"CHECK: loop"
                    # Acquire spinlock (nested inside for loop)
                    while ct.atomic_cas(Locks, group_bid, Int32(0), Int32(1);
                                       memory_order=ct.MemoryOrder.Acquire) == Int32(1)
                        # spin
                    end

                    # Critical section: accumulate
                    check"CHECK: load_view_tko"
                    acc = ct.load(DW, (group_bid, j), (1, TILE_N[]); padding_mode=ct.PaddingMode.Zero)
                    check"CHECK: addf"
                    acc = acc .+ partial
                    check"CHECK: store_view_tko"
                    ct.store(DW, (group_bid, j), acc)

                    # Release spinlock
                    check"CHECK: atomic_rmw_tko"
                    ct.atomic_xchg(Locks, group_bid, Int32(0);
                                  memory_order=ct.MemoryOrder.Release)

                    j += Int32(1)
                end
                return
            end
        end
    end

    @testset "nested spinloop uses correct loop index (regression test)" begin
        # This test catches a bug where nested while loops inside for loops
        # shadow the for loop's induction variable, causing incorrect indexing.
        # The bug: store uses (group_bid, group_bid) instead of (group_bid, loopIdx)
        spec = ct.ArraySpec{2}(16, true)
        spec1d = ct.ArraySpec{1}(16, true)
        @test @filecheck begin
            check"CHECK-LABEL: entry"
            check"CHECK: for %loopIdx in"
            check"CHECK: loop iter_values"
            # The store MUST use loopIdx for the column index, not the spinloop result
            # First index can be any value (direct outer ref or iter_arg), second must be loopIdx
            check"CHECK: store_view_tko{{.*}}[%{{[^,]+}}, %loopIdx]"
            code_tiled(Tuple{ct.TileArray{Float32,2,spec}, ct.TileArray{Int32,1,spec1d},
                           Int32, ct.Constant{Int,4}, ct.Constant{Int,4}}) do DB, Locks, num_iters, GROUP_SIZE_M, TILE_N
                bid_m = ct.bid(0)
                group_bid_m = bid_m % Int32(GROUP_SIZE_M[])

                j = Int32(0)
                while j < num_iters
                    # Nested spinloop - this must not shadow loopIdx
                    while ct.atomic_cas(Locks, group_bid_m, Int32(0), Int32(1);
                                       memory_order=ct.MemoryOrder.Acquire) == Int32(1)
                    end

                    val = ct.full((1, TILE_N[]), 1.0f0, Float32)
                    ct.store(DB, (group_bid_m, j), val)

                    ct.atomic_xchg(Locks, group_bid_m, Int32(0);
                                  memory_order=ct.MemoryOrder.Release)

                    j += Int32(1)
                end
                return
            end
        end
    end

    @testset "nested spinloop captures correct outer variable (regression test)" begin
        # This test catches a bug where nested while loops inside for loops
        # capture the for loop's induction variable instead of the correct outer variable.
        # The bug: spinloop uses loopIdx for atomic_cas instead of group_bid_m, causing hangs.
        #
        # The inner loop should capture %iterArg0 (group_bid_m), NOT %loopIdx.
        # Bug produces: loop iter_values(%arg9 = %loopIdx, ...)
        # Correct:      loop iter_values(%arg9 = %iterArg0, ...)
        spec = ct.ArraySpec{2}(16, true)
        spec1d = ct.ArraySpec{1}(16, true)
        @test begin
            tir = code_tiled(Tuple{ct.TileArray{Float32,2,spec}, ct.TileArray{Int32,1,spec1d},
                               Int32, ct.Constant{Int,4}, ct.Constant{Int,4}}) do DB, Locks, num_iters, GROUP_SIZE_M, TILE_N
                bid_m = ct.bid(0)
                group_bid_m = bid_m % Int32(GROUP_SIZE_M[])

                j = Int32(0)
                while j < num_iters
                    # Spinloop should use group_bid_m for the lock, not j
                    while ct.atomic_cas(Locks, group_bid_m, Int32(0), Int32(1);
                                       memory_order=ct.MemoryOrder.Acquire) == Int32(1)
                    end

                    val = ct.full((1, TILE_N[]), 1.0f0, Float32)
                    ct.store(DB, (group_bid_m, j), val)

                    ct.atomic_xchg(Locks, group_bid_m, Int32(0);
                                  memory_order=ct.MemoryOrder.Release)

                    j += Int32(1)
                end
                return
            end
            tir_str = string(tir)
            # The inner loop must NOT capture %loopIdx - it should capture %iterArg0
            # Bug: "loop iter_values(%arg9 = %loopIdx"
            !occursin(r"loop iter_values\([^)]*= %loopIdx", tir_str)
        end
    end

    @testset "Multiple loop results (regression test)" begin
        # Regression test: A while loop with multiple iter_args must generate
        # different result indices (%for#0, %for#1, etc.) for each result.
        # Previously, all loop results resolved to %for#0, causing incorrect code.
        TILE_M = 32
        TILE_N = 1024

        # Use ArraySpec with shape_div_by to match real CuArray behavior
        spec2d = ct.ArraySpec{2}(128, true, (0, 4), (32, 32))
        spec1d = ct.ArraySpec{1}(128, true, (0,), (32,))

        @test @filecheck begin
            check"CHECK-LABEL: entry"
            # The for loop should have multiple results
            check"CHECK: for %loopIdx in"
            # We should see both %for#0 and %for#1 used (not the same one twice)
            check"CHECK: reduce %for#1"
            check"CHECK: reduce %for#0"
            code_tiled(Tuple{ct.TileArray{Float32, 2, spec2d}, ct.TileArray{Float32, 2, spec2d},
                           ct.TileArray{Float32, 1, spec1d}, ct.TileArray{Float32, 1, spec1d},
                           ct.Constant{Int, TILE_M}, ct.Constant{Int, TILE_N}}) do DW, DB, FINAL_DW, FINAL_DB, _TILE_M, _TILE_N
                bid_n = ct.bid(0)
                num_tiles = ct.num_tiles(DW, 0, (_TILE_M[], _TILE_N[]))

                dw = ct.zeros((_TILE_M[], _TILE_N[]), Float32)
                db = ct.zeros((_TILE_M[], _TILE_N[]), Float32)
                i = Int32(0)
                while i < num_tiles
                    dw = dw .+ ct.load(DW, (i, bid_n), (_TILE_M[], _TILE_N[]); padding_mode=ct.PaddingMode.Zero)
                    db = db .+ ct.load(DB, (i, bid_n), (_TILE_M[], _TILE_N[]); padding_mode=ct.PaddingMode.Zero)
                    i += Int32(1)
                end

                sum_dw = ct.reduce_sum(dw, 0)
                sum_db = ct.reduce_sum(db, 0)

                ct.store(FINAL_DW, bid_n, sum_dw)
                ct.store(FINAL_DB, bid_n, sum_db)
                return
            end
        end
    end
end
