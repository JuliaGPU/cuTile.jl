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
mark its result poison. When the region body then records errors of its own,
the caller drops this carry diagnostic again via [`prune_derived_errors!`](@ref):
it was just the symptom.
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
    prune_derived_errors!(ctx, range)

Delete the carry-type diagnostics recorded at `range` in `ctx.errors` if any
further error was recorded after them: region emission then surfaced a root
cause, and the non-representable carry type is just its symptom. Control-flow
emitters bracket `collect_result_types!` with `length(ctx.errors)` marks and
call this after building their regions.
"""
function prune_derived_errors!(ctx::CGCtx, range::UnitRange{Int})
    isempty(range) && return nothing
    length(ctx.errors) > last(range) && deleteat!(ctx.errors, range)
    return nothing
end

"""
    reads_poison(ctx, stmt) -> Bool

Statically check whether any operand of `stmt` is a poisoned SSA value. The
dynamic `touched_poison` flag only catches operands the emitter actually read
(via `emit_value!`) before failing; emitters that throw early — e.g. rejecting
a statement on its type before touching arguments — would otherwise report a
cascade as a fresh root cause.
"""
function reads_poison(ctx::CGCtx, @nospecialize(stmt))
    check(@nospecialize(v)) = v isa SSAValue && v.id in ctx.poisoned
    if stmt isa Expr
        return any(check, stmt.args)
    elseif stmt isa IfOp
        return check(stmt.condition)
    elseif stmt isa ForOp
        return check(stmt.lower) || check(stmt.upper) || check(stmt.step) ||
               any(check, stmt.init_values)
    elseif stmt isa Union{WhileOp, LoopOp}
        return any(check, stmt.init_values)
    elseif stmt isa PiNode
        return check(stmt.val)
    elseif stmt isa TokenResultNode
        # References its memory op by raw SSA index; a poisoned (failed) memory
        # op registered no result token, so the extraction failure is a cascade.
        return stmt.mem_op_ssa in ctx.poisoned
    elseif stmt isa JoinTokensNode
        return any(check, stmt.tokens)
    else
        return check(stmt)   # alias statements: the stmt IS the operand
    end
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

"""
    InternalCompilerError <: Exception

A non-`IRError` exception escaped statement emission: a bug in cuTile.jl
itself, not a kernel diagnostic. The per-statement boundary in `emit_block!`
wraps it with the offending statement's kernel-side stack and aborts
compilation immediately (the codegen context may be inconsistent). Thrown
from inside the `catch`, so the original exception and its backtrace render
below it via Julia's exception-cause chain.
"""
struct InternalCompilerError <: Exception
    stack::Vector{SourceLocation}
end

function Base.showerror(io::IO, err::InternalCompilerError)
    print(io, "InternalCompilerError: unexpected exception during Tile IR code generation")
    if !isempty(err.stack)
        print(io, " while emitting:")
        Base.show_backtrace(io, codegen_frames(err.stack))
    end
    print(io, "\n\nThis is a bug in cuTile.jl, not in your kernel. Please file an issue at\n",
              "https://github.com/JuliaGPU/cuTile.jl/issues, including the kernel source\n",
              "and the cause below.")
    return
end

"""
    report_errors!(ctx)

If any deferred diagnostics were recorded during emission, raise them together
as a single [`CodegenErrors`](@ref) (each reported separately, GPUCompiler-
style). No-op otherwise.

Errors are reported in recording order, which is emission order and hence the
kernel's program order: an error on line 3 prints before one on line 4.
The same message reached through several inlining paths of the same kernel
statement (e.g. multiple `fill` calls inside one `randn`) is reported once:
entries are deduplicated on the message plus the outermost frame, i.e. one
report per kernel line per problem, Clang-style.
"""
function report_errors!(ctx::CGCtx)
    isempty(ctx.errors) && return nothing

    dedup_key(e) = (e.msg,
                    isempty(e.stack) ? nothing : (e.stack[1].file, e.stack[1].line))
    deduped = unique(dedup_key, ctx.errors)

    throw(CodegenErrors(deduped))
end
