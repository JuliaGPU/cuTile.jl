# EXCLUDE FROM TESTING
# MWE: StepRange `for` loop now works with steprange_last overlay
#
# The overlay in overlays.jl replaces Base.steprange_last with a GPU-safe
# version using unsigned arithmetic, eliminating ArgumentError, overflow_case,
# and checked_srem_int from the IR.
#
# Run: julia --project examples/mwe.jl

import cuTile as ct

spec = ct.ArraySpec{1}(16, true)

# UnitRange: always worked
function unit_range_for(data::ct.TileArray{Float32,1}, out::ct.TileArray{Float32,1}, n::Int32)
    pid = ct.bid(1)
    acc = zeros(Float32, (16,))
    for i in Int32(1):n
        acc = acc .+ ct.load(data, i, (16,))
    end
    ct.store(out, pid, acc)
    return
end

println("UnitRange for loop (Int32(1):n):")
ct.code_tiled(devnull, unit_range_for,
              Tuple{ct.TileArray{Float32,1,spec}, ct.TileArray{Float32,1,spec}, Int32})
println("  PASS ✓")

# StepRange: now works with overlay
function step_range_for(data::ct.TileArray{Float32,1}, out::ct.TileArray{Float32,1}, n::Int32)
    pid = ct.bid(1)
    acc = zeros(Float32, (16,))
    for i in Int32(1):Int32(2):n
        acc = acc .+ ct.load(data, i, (16,))
    end
    ct.store(out, pid, acc)
    return
end

println("\nStepRange for loop (Int32(1):Int32(2):n):")
ct.code_tiled(devnull, step_range_for,
              Tuple{ct.TileArray{Float32,1,spec}, ct.TileArray{Float32,1,spec}, Int32})
println("  PASS ✓")

# Decrementing StepRange
function decr_for(data::ct.TileArray{Float32,1}, out::ct.TileArray{Float32,1}, n::Int32)
    pid = ct.bid(1)
    acc = zeros(Float32, (16,))
    for i in n:Int32(-1):Int32(1)
        acc = acc .+ ct.load(data, i, (16,))
    end
    ct.store(out, pid, acc)
    return
end

println("\nDecrementing for loop (n:Int32(-1):Int32(1)):")
ct.code_tiled(devnull, decr_for,
              Tuple{ct.TileArray{Float32,1,spec}, ct.TileArray{Float32,1,spec}, Int32})
println("  PASS ✓")
