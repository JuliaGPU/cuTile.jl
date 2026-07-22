import IRStructurizer

const bounds_spec = ct.ArraySpec{1}(128, true)
const BoundsArray = ct.TileArray{Float32, 1, bounds_spec}

function bounds_context(a::BoundsArray, b::BoundsArray)
    check_bounds = Base.@_boundscheck
    indices = ct.arange((128,))
    tile = ct.gather(a, indices; check_bounds)
    ct.scatter(b, indices, tile; check_bounds)
    return
end

function bounds_literal(a::BoundsArray, b::BoundsArray)
    indices = ct.arange((128,))
    tile = ct.gather(a, indices; check_bounds=true)
    ct.scatter(b, indices, tile; check_bounds=true)
    return
end

function bounds_ternary(a::BoundsArray, b::BoundsArray)
    check_bounds = Base.@_boundscheck
    indices = ct.arange((16,))
    mask = check_bounds ? indices .<= Int32(size(a, 1)) : nothing
    tile = ct.gather(a, indices; mask)
    ct.scatter(b, indices, tile)
    return
end

function bounds_loop(a::BoundsArray, b::BoundsArray, n::Int32)
    check_bounds = Base.@_boundscheck
    acc = ct.broadcast_to(ct.Tile(0.0f0), (1,))
    for i in Int32(1):n
        indices = (i - Int32(1)) * Int32(128) .+ ct.arange((128,))
        tile = ct.gather(a, indices; check_bounds)
        acc += sum(tile; dims=1)
    end
    ct.store(b, 1, acc)
    return
end

function bounds_loop_literal(a::BoundsArray, b::BoundsArray, n::Int32)
    acc = ct.broadcast_to(ct.Tile(0.0f0), (1,))
    for i in Int32(1):n
        indices = (i - Int32(1)) * Int32(128) .+ ct.arange((128,))
        tile = ct.gather(a, indices; check_bounds=true)
        acc += sum(tile; dims=1)
    end
    ct.store(b, 1, acc)
    return
end

function structured_if_count(@nospecialize(f), @nospecialize(argtypes))
    sci = only(ct.code_structured(f, argtypes)).first
    return count(inst -> inst[:stmt] isa IRStructurizer.IfOp,
                 Iterators.flatten(IRStructurizer.instructions(block)
                                   for block in IRStructurizer.eachblock(sci)))
end

function test_bytecode(@nospecialize(f), @nospecialize(argtypes))
    stripped, const_argtypes = ct.process_const_argtypes(f, argtypes)
    world = Base.get_world_counter()
    mi = ct.lookup_method_instance(f, stripped; world)
    cache = ct.CacheView{ct.CuTileResults}(:cuTile, world)
    ir, rettype = ct.emit_julia(cache, mi; const_argtypes)
    sci, rettype, kernel_meta = ct.emit_structured(ir, rettype)
    empty!(sci.line_map)
    opts = ct.CGOpts((sm_arch=nothing, opt_level=nothing, num_ctas=nothing,
                      occupancy=nothing, num_worker_warps=nothing,
                      bytecode_version=ct.bytecode_version()))
    return ct.emit_tile(sci, rettype, kernel_meta;
                        name="boundscheck_test", opts, cache, const_argtypes)
end

if Base.JLOptions().check_bounds == 2
    # These oracles compare context-dependent checks with checked literals.
    # The --check-bounds=no policy is covered by the mode-aware resolver test.
    @testset "bounds-check branch integration" begin
        @test_skip false
    end
else
    @testset "bounds-check branch integration" begin
        @test @filecheck begin
            @check_not " if %"
            @check_count 1 "load_ptr_tko"
            @check_count 1 "store_ptr_tko"
            @check_not " if %"
            code_tiled(bounds_context, Tuple{BoundsArray, BoundsArray})
        end

        @test test_bytecode(bounds_context, Tuple{BoundsArray, BoundsArray}) ==
              test_bytecode(bounds_literal, Tuple{BoundsArray, BoundsArray})
        @test structured_if_count(bounds_ternary, Tuple{BoundsArray, BoundsArray}) == 0
        loop_types = Tuple{BoundsArray, BoundsArray, Int32}
        @test test_bytecode(bounds_loop, loop_types) ==
              test_bytecode(bounds_loop_literal, loop_types)

        range_types = Tuple{ct.TileArray{Int32, 1, ct.ArraySpec{1}(1, true)}, Int32, Int32}
        checked_kernel = function(out, n, i)
            value = (Int32(1):n)[i]
            ct.store(out, 1, ct.Tile(value))
            return
        end
        unchecked_kernel = function(out, n, i)
            value = @inbounds (Int32(1):n)[i]
            ct.store(out, 1, ct.Tile(value))
            return
        end
        checked = sprint(io -> code_tiled(io, checked_kernel, range_types))
        unchecked = sprint(io -> code_tiled(io, unchecked_kernel, range_types))
        check_bounds = Base.JLOptions().check_bounds
        if check_bounds == 1
            @test occursin("assert", checked)
            @test occursin("assert", unchecked)
        else
            @test occursin("assert", checked)
            @test !occursin("assert", unchecked)
            @test !occursin(" if %", unchecked)
        end
    end
end
