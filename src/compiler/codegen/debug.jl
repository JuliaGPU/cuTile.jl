# Debug info emission: converts IRStructurizer SourceLocation stacks to Tile IR debug attrs

"""
    DebugInfoEmitter

Converts Julia source location information (from IRStructurizer's `source_location`)
into Tile IR debug attributes (DIFile, DICompileUnit, DISubprogram, DILoc, CallSite).
"""
struct DebugInfoEmitter
    debug_attrs::DebugAttrTable
    subprogram_cache::Dict{Any, DebugAttrId}
    file_cache::Dict{Tuple{String,String}, DebugAttrId}
    compile_unit_cache::Dict{DebugAttrId, DebugAttrId}
end

function DebugInfoEmitter(debug_attrs::DebugAttrTable)
    DebugInfoEmitter(
        debug_attrs,
        Dict{Any, DebugAttrId}(),
        Dict{Tuple{String,String}, DebugAttrId}(),
        Dict{DebugAttrId, DebugAttrId}()
    )
end

function get_file!(emitter::DebugInfoEmitter, filepath::Symbol)
    s = string(filepath)
    name = basename(s)
    dir = dirname(s)
    key = (name, dir)
    get!(emitter.file_cache, key) do
        file!(emitter.debug_attrs, name, dir)
    end
end

function get_compile_unit!(emitter::DebugInfoEmitter, file_attr::DebugAttrId)
    get!(emitter.compile_unit_cache, file_attr) do
        compile_unit!(emitter.debug_attrs, file_attr)
    end
end

"""
    get_subprogram!(emitter, loc; linkage_name) -> DebugAttrId

Get or create a DISubprogram for a source location. On Julia 1.12+, `loc.method` is a
`Method`/`MethodInstance` with full metadata. On Julia 1.11, it's a `Symbol` and we
use `loc.file`/`loc.line` instead.
"""
function get_subprogram!(emitter::DebugInfoEmitter, loc::SourceLocation;
                         linkage_name::Union{String, Nothing}=nothing)
    method = loc.method
    # Use (method, linkage_name) as cache key to distinguish kernel entry from inlined copies
    cache_key = linkage_name !== nothing ? (method, linkage_name) : method
    get!(emitter.subprogram_cache, cache_key) do
        if method isa Method
            file_attr = get_file!(emitter, method.file)
            cu_attr = get_compile_unit!(emitter, file_attr)
            ln = @something linkage_name sanitize_name(string(method.name))
            subprogram!(emitter.debug_attrs, file_attr, Int(method.line),
                        string(method.name), ln, cu_attr, Int(method.line))
        elseif method isa MethodInstance
            get_subprogram!(emitter, SourceLocation(method.def, loc.file, loc.line);
                            linkage_name)
        else
            # Symbol (Julia 1.11) or unknown — use file/line from SourceLocation
            name = string(method)
            file_attr = get_file!(emitter, loc.file)
            cu_attr = get_compile_unit!(emitter, file_attr)
            ln = @something linkage_name sanitize_name(name)
            subprogram!(emitter.debug_attrs, file_attr, Int(loc.line),
                        name, ln, cu_attr, Int(loc.line))
        end
    end
end

"""
    make_diloc(emitter, loc::SourceLocation) -> DebugAttrId

Create a DILoc for a single source location entry, scoped to its method's subprogram.
"""
function make_diloc(emitter::DebugInfoEmitter, loc::SourceLocation;
                    linkage_name::Union{String, Nothing}=nothing)
    sp = get_subprogram!(emitter, loc; linkage_name)
    loc!(emitter.debug_attrs, sp, string(loc.file), Int(loc.line), 0)
end

"""
    resolve_debug_attr!(emitter, sci, ssa_idx) -> DebugAttrId

Resolve the debug attribute for a statement at `ssa_idx`. Returns `DebugAttrId(0)`
if no debug info is available.

The inlining stack from `source_location(sci, ssa_idx)` is converted to a chain of
DILoc and CallSite attributes: innermost location wrapped by successive CallSite entries.
"""
function resolve_debug_attr!(emitter::DebugInfoEmitter, sci::StructuredIRCode,
                             ssa_idx::Int;
                             linkage_name::Union{String, Nothing}=nothing)
    stack = source_location(sci, ssa_idx)
    isempty(stack) && return DebugAttrId(0)

    # Build from innermost out
    attr = make_diloc(emitter, stack[end])
    for i in length(stack)-1:-1:1
        # Outermost frame gets the kernel's linkage_name
        ln = i == 1 ? linkage_name : nothing
        caller = make_diloc(emitter, stack[i]; linkage_name=ln)
        attr = call_site!(emitter.debug_attrs, attr, caller)
    end

    # Single entry (no inlining): use the kernel's linkage_name
    if length(stack) == 1 && linkage_name !== nothing
        attr = make_diloc(emitter, stack[1]; linkage_name)
    end

    return attr
end

"""
    make_func_debug_attr(emitter, sci; linkage_name) -> DebugAttrId

Create a function-level debug attribute (DILoc scoped to DISubprogram) for a kernel.
"""
function make_func_debug_attr(emitter::DebugInfoEmitter, sci::StructuredIRCode;
                              linkage_name::String)
    # Find the kernel function from the first instruction with debug info
    for inst in instructions(sci.entry)
        stack = source_location(sci, inst)
        isempty(stack) && continue
        outer = stack[1]
        sp = get_subprogram!(emitter, outer; linkage_name)
        m = outer.method
        m isa MethodInstance && (m = m.def)
        file = m isa Method ? m.file : outer.file
        line = m isa Method ? Int(m.line) : Int(outer.line)
        return loc!(emitter.debug_attrs, sp, string(file), line, 0)
    end
    return DebugAttrId(0)
end
