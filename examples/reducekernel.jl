using Test
using CUDA
using cuTile
import cuTile as ct

elType = UInt16
function reduceKernel(a::ct.TileArray{elType,1}, b::ct.TileArray{elType,1}, tileSz::ct.Constant{Int})
    bid = ct.bid(1)
    tile = ct.load(a, bid, (tileSz[],))
    result = ct.reduce_min(tile, Val(1))
    ct.store(b, bid, result)
    return nothing
end

sz = 32
N = 2^15
a = CUDA.rand(elType, N)
b = CUDA.zeros(elType, cld(N, sz))
CUDA.@sync ct.launch(reduceKernel, cld(length(a), sz), a, b, ct.Constant(sz))
res = Array(b)
