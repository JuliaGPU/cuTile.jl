# Deferred codegen diagnostics
#
# Codegen does not abort on the first unsupported construct. Each eager
# `throw(IRError(...))` is caught at the per-statement boundary (`emit_block!`),
# recorded with the offending statement's kernel-side inlining stack, and
# replaced by a poison placeholder so emission continues and collects every
# problem. `report_errors!` raises the aggregate at the end of `emit_kernel!`.
#
# This is the same trick Julia uses for statically-known errors: inference emits
# them into the IR (a `Union{}`-typed call followed by `unreachable`) instead of
# throwing during abstract interpretation, which would be unsound under
# speculative passes (a tile shape that is non-`Const` early on may become
# `Const` after constant propagation). Codegen is where "the path was actually
# reached" holds: it runs once on the final IR, where the path is live and
# concrete.

"""
    record_error!(ctx, msg)

Record a deferred codegen diagnostic for the statement currently being emitted
(`ctx.current_ssa_idx`), capturing its kernel-side inlining stack. Returns
`nothing`.
"""
function record_error!(ctx::CGCtx, msg::AbstractString)
    stack = source_location(ctx.sci, ctx.current_ssa_idx)
    push!(ctx.errors, CodegenError(String(msg), stack))
    return nothing
end

"""
    poison_type!(ctx) -> TypeId

A valid placeholder Tile IR type (0-D `i32` tile) for a statement whose real
result type could not be determined. The bytecode is discarded once any error
is recorded, so the placeholder only needs to keep the Julia-side encoder
structurally happy while emission continues.
"""
poison_type!(ctx::CGCtx) = tile_type_for_julia!(ctx, Tile{Int32, Tuple{}})

"""
    poison_value(ctx) -> CGVal

A real (0-D `i32` constant) placeholder value stored for a poisoned SSA result,
so downstream consumers that read it get a structurally valid `Value` rather
than crashing on a missing one. The SSA index is separately tracked in
`ctx.poisoned` so consumers can recognise the read as poison.

Carries no `constant` (unlike a normal `emit_value!` of a literal): a poison
result must not masquerade as a compile-time constant, or consumers that read
`tv.constant` would propagate a bogus value (e.g. feed it as a count to a
type-specialised helper).
"""
function poison_value(ctx::CGCtx)
    tid = poison_type!(ctx)
    v = encode_ConstantOp!(ctx.cb, tid, constant_to_bytes(Int32(0), Int32))
    CGVal(v, tid, Tile{Int32, Tuple{}}, RowMajorShape(()), nothing, nothing, nothing)
end

"""
    result_type_or_poison!(ctx, T; context) -> (TypeId, ok::Bool)

Lower a control-flow / block result type `T` to a Tile IR type. If `T` has no
representation (typically `Any`/`Union{}` left when inference could not pin a
tile down), record a diagnostic and return a poison type with `ok = false`, so
the caller can keep emitting the region (surfacing the real cause inside it) and
mark its result poison. The message stays neutral; the precise cause is reported
by the offending op below, which sits at a deeper inlining frame so
`report_errors!` orders it first.
"""
function result_type_or_poison!(ctx::CGCtx, @nospecialize(T); context::AbstractString="result")
    tid = tile_type_for_julia!(ctx, T; throw_error=false)
    tid === nothing || return (tid, true)
    record_error!(ctx, "$context has no Tile IR representation (inferred `$(CC.widenconst(T))`)")
    return (poison_type!(ctx), false)
end

"""
    collect_result_types!(ctx, types; context) -> (Vector{TypeId}, poisoned::Bool)

Lower each Julia type in `types` to a Tile IR result type via
`result_type_or_poison!`, returning the `TypeId`s and whether any was poisoned.
Shared by the control-flow emitters (if/for/while/loop) for their carried
result types; the caller marks its result SSA poison when `poisoned` is true.
"""
function collect_result_types!(ctx::CGCtx, types; context::AbstractString="result")
    result_types = TypeId[]
    poisoned = false
    for T in types
        tid, ok = result_type_or_poison!(ctx, T; context)
        ok || (poisoned = true)
        push!(result_types, tid)
    end
    return result_types, poisoned
end

"""
    CodegenErrors <: Exception

Aggregates every deferred [`CodegenError`](@ref) from one kernel compilation.
Its `showerror` renders each as a separate `Reason: …` block with a Julia-style
kernel-side backtrace (via `Base.show_backtrace`), mirroring GPUCompiler's
`InvalidIRError`.
"""
struct CodegenErrors <: Exception
    errors::Vector{CodegenError}
end

# Convert a `source_location` stack (outermost→innermost) into Base stack
# frames (innermost-first, as backtraces read). Passing the `MethodInstance`
# as `linfo` lets `Base.show_backtrace` render the specialised signature and
# `@ Module file:line` line exactly like a native Julia backtrace.
function codegen_frames(stack::Vector{SourceLocation})
    frames = Base.StackTraces.StackFrame[]
    for loc in Iterators.reverse(stack)
        m = loc.method
        local name, linfo
        if m isa Core.MethodInstance
            name = m.def isa Method ? m.def.name : Symbol(m.def)
            linfo = m
        elseif m isa Method
            name, linfo = m.name, m
        else
            name, linfo = Symbol(m), nothing
        end
        push!(frames, Base.StackTraces.StackFrame(name, loc.file, Int(loc.line),
                                                  linfo, false, true, UInt64(0)))
    end
    return frames
end

function Base.showerror(io::IO, err::CodegenErrors)
    n = length(err.errors)
    print(io, "CodegenErrors: kernel compilation produced ", n,
          n == 1 ? " error:" : " errors:")
    for e in err.errors
        printstyled(io, "\nReason: ", e.msg; color=:red)
        Base.show_backtrace(io, codegen_frames(e.stack))
        println(io)
    end
    return
end

# Sort key placing the most deeply-inlined (root-cause) errors first, with a
# deterministic tie-break on the innermost frame's location and the message.
function error_sort_key(e::CodegenError)
    if isempty(e.stack)
        return (0, "", 0, e.msg)
    end
    inner = e.stack[end]   # source_location is outermost→innermost
    return (-length(e.stack), string(inner.file), Int(inner.line), e.msg)
end

"""
    report_errors!(ctx)

If any deferred diagnostics were recorded during emission, raise them together
as a single [`CodegenErrors`](@ref) (each reported separately, GPUCompiler-
style). No-op otherwise.
"""
function report_errors!(ctx::CGCtx)
    isempty(ctx.errors) && return nothing

    # Dedup identical (message + stack) entries, e.g. the same non-const `fill`
    # reached from both arms of an `if`. Key on message + full stack.
    dedup_key(e) = string(e.msg, '\0',
                          join((string(loc.method, '|', loc.file, '|', loc.line)
                                for loc in e.stack), ';'))
    deduped = unique(dedup_key, ctx.errors)

    # Order by source location: deepest inlining frame first. The offending op
    # is the most deeply-inlined; a control-flow op that merely observed the
    # resulting `Any` sits at a shallower frame, so root causes sort ahead of
    # their symptoms. Tie-break on the innermost frame's file/line, then the
    # message, for a fully deterministic order.
    sort!(deduped; by=error_sort_key)

    throw(CodegenErrors(deduped))
end
