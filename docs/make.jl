using Documenter
using cuTile

# CUDA.jl provides the `@cuda backend=cuTile` launch path, while DLFP8Types and Microfloats
# back cuTile's weak-dependency extensions. Load them so their docstrings resolve.
using CUDA, DLFP8Types, Microfloats

function main()
    ci = get(ENV, "CI", "") == "true"

    makedocs(
        sitename = "cuTile.jl",
        authors = "Tim Besard",
        format = Documenter.HTML(
            # Use clean URLs on CI
            prettyurls = ci
        ),
        modules = [cuTile],
        pages = [
            "Home" => "index.md",
            "Installation" => "installation.md",
            "Manual" => [
                "man/programming_model.md",
                "man/types.md",
                "man/kernels.md",
                "man/memory.md",
                "man/operations.md",
                "man/atomics.md",
                "man/memory_model.md",
                "man/random.md",
                "man/host.md",
                "man/performance.md",
                "man/compatibility.md",
                "man/debugging.md",
                "man/comparison.md",
            ],
            "API reference" => [
                "lib/essentials.md",
                "lib/kernels.md",
                "lib/memory.md",
                "lib/operations.md",
                "lib/atomics.md",
                "lib/random.md",
                "lib/host.md",
                "lib/reflection.md",
            ],
        ],
        # Most examples are kernels that need a GPU to run; doctests are enabled
        # per-page as their output is made reproducible.
        doctest = false,
        warnonly = [:missing_docs],
    )

    if ci
        deploydocs(
            repo = "github.com/JuliaGPU/cuTile.jl.git",
            push_preview = true
        )
    end
end

isinteractive() || main()
