module FileCheck

import LLVM_jll

export filecheck, @filecheck
export @check, @check_label, @check_next, @check_same
export @check_not, @check_dag, @check_empty, @check_count

global filecheck_path::String
function __init__()
    global filecheck_path = joinpath(LLVM_jll.artifact_dir, "tools", "FileCheck")
end

function filecheck_exe(; adjust_PATH::Bool=true, adjust_LIBPATH::Bool=true)
    env = Base.invokelatest(
        LLVM_jll.JLLWrappers.adjust_ENV!,
        copy(ENV),
        LLVM_jll.PATH[],
        LLVM_jll.LIBPATH[],
        adjust_PATH,
        adjust_LIBPATH
    )

    return Cmd(Cmd([filecheck_path]); env)
end

function filecheck(f, input)
    # FileCheck assumes that the input is available as a file
    mktemp() do path, input_io
        write(input_io, input)
        close(input_io)

        # get the output of `f` and print it into a temporary buffer
        output_io = IOBuffer()
        print(output_io, f(input))
        println(output_io)

        # determine some useful prefixes for FileCheck
        prefixes = ["CHECK",
                    "JULIA$(VERSION.major)_$(VERSION.minor)",
                    "LLVM$(Base.libllvm_version.major)"]
        # TODO: add CUDA version prefix and target architecture?

        # now pass the collected output to FileCheck
        seekstart(output_io)
        filecheck_io = Pipe()
        cmd = ```$(filecheck_exe())
                 --color
                 --allow-unused-prefixes
                 --check-prefixes $(join(prefixes, ','))
                 $path```
        proc = run(pipeline(ignorestatus(cmd); stdin=output_io, stdout=filecheck_io, stderr=filecheck_io); wait=false)
        close(filecheck_io.in)

        # collect the output of FileCheck
        reader = Threads.@spawn String(read(filecheck_io))
        Base.wait(proc)
        log = strip(fetch(reader))

        # error out if FileCheck did not succeed.
        # otherwise, return true so that `@test @filecheck` works as expected.
        if !success(proc)
            error(log)
        end
        return true
    end
end

# collect checks used in the @filecheck block by piggybacking on macro expansion
const checks = String[]

macro check(str)
    push!(checks, "CHECK: $str")
    nothing
end

macro check_label(str)
    push!(checks, "CHECK-LABEL: $str")
    nothing
end

macro check_next(str)
    push!(checks, "CHECK-NEXT: $str")
    nothing
end

macro check_same(str)
    push!(checks, "CHECK-SAME: $str")
    nothing
end

macro check_not(str)
    push!(checks, "CHECK-NOT: $str")
    nothing
end

macro check_dag(str)
    push!(checks, "CHECK-DAG: $str")
    nothing
end

macro check_empty(str)
    push!(checks, "CHECK-EMPTY: $str")
    nothing
end

macro check_count(n, str)
    push!(checks, "CHECK-COUNT-$n: $str")
    nothing
end

macro filecheck(ex)
    ex = Base.macroexpand(__module__, ex)
    if isempty(checks)
        error("No checks provided within the @filecheck macro block")
    end
    check_str = join(checks, "\n")
    empty!(checks)

    esc(quote
        filecheck($check_str) do _
            $ex
        end
    end)
end

end # module FileCheck
