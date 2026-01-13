export code_tiled, @code_tiled

"""
    emit_tileir(f, argtypes; name, sm_arch, num_ctas, occupancy) -> Vector{UInt8}

Compile a Julia function to Tile IR bytecode.
"""
function emit_tileir(@nospecialize(f), @nospecialize(argtypes);
                     name::Union{String, Nothing} = nothing,
                     sm_arch::Union{String, Nothing} = nothing,
                     num_ctas::Union{Int, Nothing} = nothing,
                     occupancy::Union{Int, Nothing} = nothing)
    target = TileTarget(f, argtypes)
    kernel_name = name === nothing ? string(target.mi.def.name) : name

    if compile_hook[] !== nothing
        compile_hook[](f, argtypes; name=name)
    end

    buf = write_bytecode!(1) do writer, func_buf
        emit_kernel!(writer, func_buf, target; name=kernel_name, sm_arch,
                     num_ctas, occupancy)
    end

    return buf
end

function disassemble_tileir(bytecode::Vector{UInt8})::String
    mktempdir() do dir
        input_path = joinpath(dir, "kernel.tile")
        output_path = joinpath(dir, "kernel.disasm")
        write(input_path, bytecode)
        read(`$(cuda_tile_translate()) --cudatilebc-to-mlir $input_path`, String)
    end
end

"""
    code_tiled(f, argtypes; name, sm_arch, num_ctas, occupancy) -> String

Return the CUDA Tile IR for a Julia function as a textual MLIR representation.
Analogous to `code_typed` or `code_structured`.
"""
function code_tiled(@nospecialize(f), @nospecialize(argtypes);
                   kwargs...)
    bytecode = emit_tileir(f, argtypes; kwargs...)
    disassemble_tileir(bytecode)
end

# compilation hooking: taken from GPUCompiler.jl
const compile_hook = Ref{Union{Nothing,Function}}(nothing)
function emit_hooked_compilation(inner_hook, ex...)
    user_code = ex[end]
    user_kwargs = ex[1:end-1]
    quote
        # we only want to invoke the hook once for every compilation
        seen = Set()
        function outer_hook(f, tt; kwargs...)
            key = (f, tt)
            if !in(key, seen)
                # the user hook might invoke the compiler again, so disable the hook
                old_hook = $compile_hook[]
                try
                    $compile_hook[] = nothing
                    $inner_hook(f, tt; kwargs..., $(map(esc, user_kwargs)...))
                finally
                    $compile_hook[] = old_hook
                end
                push!(seen, key)
            end
        end

        # now invoke the user code with this hook in place
        try
            $compile_hook[] = outer_hook
            $(esc(user_code))
        finally
            $compile_hook[] = nothing
        end

        if isempty(seen)
            error("no kernels executed while evaluating the given expression")
        end

        nothing
    end
end

macro code_tiled(ex...)
    function hook(f, tt; io::IO=stdout, kwargs...)
        println(io, "// $f($(join(map(string, tt.parameters), ", ")))")
        println(io)
        println(io, code_tiled(f, tt; kwargs...))
    end
    emit_hooked_compilation(hook, ex...)
end
