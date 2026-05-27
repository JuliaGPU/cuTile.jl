module Experimental

using ..cuTile
using ..cuTile: cuTileconvert, default_sm_arch, temporary_cufunction,
                _SCOPED_INF_CACHE

using CUDACore: CUDACore

using Base.ScopedValues: with
import Core.Compiler as CC
using Random

# Builds a fresh inference cache compatible with the running Julia version.
# Used to wrap an autotune pass in `with(_SCOPED_INF_CACHE => ...)` so all the
# per-config const-seeded inference calls share results instead of paying
# the slow paths (e.g. `ct.load(..., order=...)`) once per config.
@inline _fresh_inf_cache() = @static if isdefined(CC, :InferenceCache)
    CC.InferenceCache()
else
    Vector{CC.InferenceResult}()
end

include("search_space.jl")
include("autotune.jl")
include("autotune_macro.jl")

end
