using PrecompileTools: @setup_workload, @compile_workload

# Load REPL so that interactive use doesn't invalidate cuTile.jl (REPL.jl's
# `OptimizationParams(::REPLInterpreter)` and other AbsInt methods cause
# invalidation of cuTileInterpreter versions)
import REPL

@setup_workload begin
    function vadd_1d(a::TileArray{T,1}, b::TileArray{T,1},
                     c::TileArray{T,1}, tile) where {T}
        bid = cuTile.bid(1)
        a_tile = cuTile.load(a; index=bid, shape=(tile,))
        b_tile = cuTile.load(b; index=bid, shape=(tile,))
        cuTile.store(c; index=bid, tile=a_tile + b_tile)
        return
    end

    function vadd_2d(a::TileArray{T,2}, b::TileArray{T,2},
                     c::TileArray{T,2}, tx, ty) where {T}
        bx = cuTile.bid(1); by = cuTile.bid(2)
        a_tile = cuTile.load(a; index=(bx, by), shape=(tx, ty))
        b_tile = cuTile.load(b; index=(bx, by), shape=(tx, ty))
        cuTile.store(c; index=(bx, by), tile=a_tile + b_tile)
        return
    end

    function vadd_gather(a::TileArray{T,1}, b::TileArray{T,1},
                         c::TileArray{T,1}, tile) where {T}
        bid = cuTile.bid(1)
        offsets = cuTile.arange(tile)
        base = cuTile.Tile((bid - Int32(1)) * Int32(tile))
        indices = cuTile.broadcast_to(base, (tile,)) .+ offsets
        a_tile = cuTile.gather(a, indices)
        b_tile = cuTile.gather(b, indices)
        cuTile.scatter(c, indices, a_tile + b_tile)
        return
    end

    # Drive the host-side `compile` so the precompile workload reaches
    # `tileiras` → CUBIN without needing a CUDA context. The runtime path
    # tacks `link` on top to load the CUBIN onto the GPU.
    function precompile_kernel(@nospecialize(f), @nospecialize(tt))
        argtypes, const_argtypes = unwrap_argtypes(f, tt)
        bv = bytecode_version()
        for sm_arch in [v"8.0", v"8.6", v"8.7", v"8.9",
                        v"10.0", v"11.0", v"12.0", v"12.1"]
            key = TileCacheKey(sm_arch, bv, nothing, nothing, nothing)
            compile(f, argtypes, const_argtypes, key)
        end
        return
    end

    @compile_workload begin
        precompile_kernel(identity, Tuple{Nothing})

        # 1D vec_add: load/add/store across float types.
        spec1d = ArraySpec{1}(16, true)
        for T in (Float32, Float16)
            tt = Tuple{TileArray{T, 1, spec1d},
                       TileArray{T, 1, spec1d},
                       TileArray{T, 1, spec1d},
                       Constant{Int, 1024}}
            precompile_kernel(vadd_1d, tt)
        end

        # 2D vec_add: multi-dim block IDs and shapes.
        spec2d = ArraySpec{2}(16, true)
        let tt = Tuple{TileArray{Float32, 2, spec2d},
                       TileArray{Float32, 2, spec2d},
                       TileArray{Float32, 2, spec2d},
                       Constant{Int, 32}, Constant{Int, 32}}
            precompile_kernel(vadd_2d, tt)
        end

        # Gather/scatter path: arange, broadcast_to, gather, scatter, and
        # the contiguous_gather assume infrastructure.
        let tt = Tuple{TileArray{Float32, 1, spec1d},
                       TileArray{Float32, 1, spec1d},
                       TileArray{Float32, 1, spec1d},
                       Constant{Int, 1024}}
            precompile_kernel(vadd_gather, tt)
        end
    end
end
