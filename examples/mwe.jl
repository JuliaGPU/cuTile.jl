# EXCLUDE FROM TESTING
# MWE: StepRange `for` loop fails in cuTile codegen
#
# `for i in start:stop` (UnitRange) works fine.
# `for i in start:step:stop` (StepRange) fails because Julia's StepRange
# construction pulls in ArgumentError, overflow_case, checked_srem_int.
#
# Run: julia --project examples/mwe.jl

import cuTile as ct

spec = ct.ArraySpec{1}(16, true)

# This works: UnitRange (unit step)
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
try
    ct.code_tiled(devnull, unit_range_for,
                  Tuple{ct.TileArray{Float32,1,spec}, ct.TileArray{Float32,1,spec}, Int32})
    println("  PASS ✓")
catch e
    println("  FAIL: ", sprint(showerror, e))
end

# This fails: StepRange (non-unit step)
function step_range_for(data::ct.TileArray{Float32,1}, out::ct.TileArray{Float32,1},
                        start::Int32, step::Int32, stop::Int32)
    pid = ct.bid(1)
    acc = zeros(Float32, (16,))
    for i in start:step:stop
        acc = acc .+ ct.load(data, i, (16,))
    end
    ct.store(out, pid, acc)
    return
end

println("\nStepRange for loop (start:step:stop):")
try
    ct.code_tiled(devnull, step_range_for,
                  Tuple{ct.TileArray{Float32,1,spec}, ct.TileArray{Float32,1,spec}, Int32, Int32, Int32})
    println("  PASS ✓")
catch e
    println("  FAIL ✗ — ", split(sprint(showerror, e), '\n')[1])
end

# Workaround: use while loop for non-unit step
function step_while(data::ct.TileArray{Float32,1}, out::ct.TileArray{Float32,1},
                    start::Int32, step::Int32, stop::Int32)
    pid = ct.bid(1)
    acc = zeros(Float32, (16,))
    i = start
    while i <= stop
        acc = acc .+ ct.load(data, i, (16,))
        i += step
    end
    ct.store(out, pid, acc)
    return
end

println("\nWhile-loop workaround (start:step:stop):")
try
    ct.code_tiled(devnull, step_while,
                  Tuple{ct.TileArray{Float32,1,spec}, ct.TileArray{Float32,1,spec}, Int32, Int32, Int32})
    println("  PASS ✓")
catch e
    println("  FAIL: ", sprint(showerror, e))
end
