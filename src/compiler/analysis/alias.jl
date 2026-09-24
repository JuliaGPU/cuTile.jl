# Alias Analysis Pass
#
# Forward dataflow over StructuredIRCode that determines which SSA values may
# point into the same allocation. Each array argument, and each array inside a
# tuple or struct argument (`FieldRoot`), starts in its own alias set; the
# analysis propagates those sets through getfield, pointer arithmetic, view
# constructors, and pointer passthroughs. This is only sound for arrays that do
# not overlap, so the launch checks them (`overlap_labels`) and arrays that
# overlap share one set (`AliasGroup`). Raw pointers cannot be checked and are
# in the universe set.
#
# Unknown operations conservatively produce ALIAS_UNIVERSE (may alias anything).
#
# Consumers go through the public query API (`alias_class`, `alias_classes`,
# `aliases`) which hides the underlying lattice; in particular, the
# "unvisited → may alias anything" policy is encoded once, in `alias_class`.
# The dataflow framework is an implementation detail.

#=============================================================================
 Lattice
=============================================================================#

"""
    AliasElement

3-state lattice for alias analysis (internal):

    nothing         — ⊥: not yet analysed
    Set{Any}        — concrete root tags this anchor may point into
    ALIAS_UNIVERSE  — ⊤: may alias anything
"""
const AliasElement = Union{Nothing, AliasSet}

"""
    AliasAnalysis

Forward sparse dataflow analysis whose lattice element is `AliasElement`.
Concrete `Set{Any}`s carry root alias tags (`Argument(i)`, `FieldRoot` or
`AliasGroup`); `ALIAS_UNIVERSE` is the top. Join is set union;
`ALIAS_UNIVERSE ∪ x = ALIAS_UNIVERSE`.

The framework handles block walking, fixpoint iteration, and structured-
control-flow merges — this file only supplies the per-op transfer rules.
"""
struct AliasAnalysis <: ForwardAnalysis{AliasElement}
    # The group tag of each `(argument index, field path)` array that the
    # launch found overlapping another.
    groups::Dict{Tuple{Int, Tuple{Vararg{Int}}}, Any}
end

"""
    AliasGroup(id)

Root alias tag shared by the kernel arrays in launch-time overlap group `id`
(see `AliasGroups`), so accesses through any of them stay ordered.
"""
struct AliasGroup
    id::Int
end

function AliasAnalysis(alias_groups::AliasGroups=())
    groups = Dict{Tuple{Int, Tuple{Vararg{Int}}}, Any}()
    for (id, group) in enumerate(alias_groups), leaf in group
        groups[leaf] = AliasGroup(id)
    end
    return AliasAnalysis(groups)
end

"""
    FieldRoot(n, path)

Root alias tag for the value at field `path` (field indices, outermost first)
of kernel argument `n`. Distinct paths get distinct tags: arrays inside one
tuple or struct argument are assumed not to overlap, as separate array
arguments are. `Argument(n)` is the tag of the argument itself.
"""
struct FieldRoot
    n::Int
    path::Tuple{Vararg{Int}}
end

# The tag of the value at `path` in argument `n`: its overlap group's, if any.
root_tag(a::AliasAnalysis, n::Int, path::Tuple{Vararg{Int}}) =
    get(a.groups, (n, path), isempty(path) ? Argument(n) : FieldRoot(n, path))

# The tag of field `i` of a value tagged `tag`. Group tags only name arrays,
# whose fields are handled by the caller, so they have no field tags.
field_root(a::AliasAnalysis, tag::Argument, i::Int) = root_tag(a, tag.n, (i,))
field_root(a::AliasAnalysis, tag::FieldRoot, i::Int) = root_tag(a, tag.n, (tag.path..., i))
field_root(::AliasAnalysis, @nospecialize(_), ::Int) = nothing

bottom(::AliasAnalysis) = nothing
top(::AliasAnalysis) = ALIAS_UNIVERSE

tmerge(::AliasAnalysis, ::Nothing, ::Nothing) = nothing
tmerge(::AliasAnalysis, ::Nothing, b::AliasSet) = b
tmerge(::AliasAnalysis, a::AliasSet, ::Nothing) = a
tmerge(::AliasAnalysis, a::AliasSet, b::AliasSet) = union(a, b)

function init_arg(a::AliasAnalysis, i::Int, @nospecialize(argtype))
    T = CC.widenconst(argtype)
    # Only arrays get alias sets of their own, since only arrays are checked
    # for overlap at launch; a raw pointer may point anywhere. The launch does
    # not check arrays captured by the kernel function (argument 1).
    i > 1 && carries_arrays(T) && return Set{Any}([root_tag(a, i, ())])
    contains_pointers(T) && return ALIAS_UNIVERSE
    return nothing
end

function transfer(a::AliasAnalysis, r::DataflowResult, @nospecialize(func),
                  ops, block::Block, ::Any)
    # getfield: a TileArray's `ptr` keeps the array's tag; a field of a tuple
    # or struct that holds arrays gets a tag of its own. Anything else,
    # including a raw pointer field, is UNIVERSE.
    if func === getfield && length(ops) >= 2
        parent = operand_value(a, r, ops[1])
        parent isa Set || return ALIAS_UNIVERSE
        field = ops[2] isa QuoteNode ? ops[2].value : ops[2]
        T = value_type(block, ops[1])
        if T === nothing || T <: TileArray
            return field === :ptr ? parent : ALIAS_UNIVERSE
        end
        i = field_index(T, field)
        i === nothing && return ALIAS_UNIVERSE
        carries_arrays(fieldtype(T, i)) || return ALIAS_UNIVERSE
        tags = Set{Any}()
        for tag in parent
            ftag = field_root(a, tag, i)
            ftag === nothing && return ALIAS_UNIVERSE
            push!(tags, ftag)
        end
        return tags
    end

    # Pointer arithmetic: propagate from the pointer operand (first operand
    # whose alias set is concrete).
    if func === Base.:+ || func === Base.:-
        for arg in ops
            av = operand_value(a, r, arg)
            av isa Set && return av
        end
        return ALIAS_UNIVERSE
    end

    # View constructors and pointer passthroughs: propagate from the source
    # operand. `make_tensor_view(::Type{T}, ptr, sizes, strides)` — alias source
    # is the ptr (operand 2). Tile-view constructors take their TensorView as
    # operand 1.
    if is_view_constructor(func) || is_pointer_passthrough(func)
        src_idx = func === Intrinsics.make_tensor_view ? 2 : 1
        length(ops) >= src_idx && return operand_value(a, r, ops[src_idx])
        return ALIAS_UNIVERSE
    end

    ALIAS_UNIVERSE
end


# Helper functions

contains_pointers(T) = T <: Ptr || T <: TileArray || (T <: Tile && eltype(T) <: Ptr)

# Whether `T` is an array or a tuple or struct holding some, which is what the
# launch checks for overlap (see `array_leaves`).
carries_arrays(@nospecialize(T)) =
    T <: TileArray || (!is_ghost_type(T) && !isprimitivetype(T) && isstructtype(T) &&
                       any(carries_arrays, fieldtypes(T)))

# Index of a constant `getfield` field (a name or an integer) in `T`, or
# `nothing` when it cannot be resolved statically.
function field_index(@nospecialize(T), @nospecialize(field))
    T isa DataType && isconcretetype(T) || return nothing
    if field isa Symbol
        i = Base.fieldindex(T, field, false)
        return i == 0 ? nothing : i
    elseif field isa Integer
        return 1 <= field <= fieldcount(T) ? Int(field) : nothing
    end
    return nothing
end

"""
    root_type(root, argtypes) -> Type or nothing

Type of the kernel-argument value a root alias tag names.
"""
function root_type(root::Argument, argtypes::Vector{Any})
    checkbounds(Bool, argtypes, root.n) || return nothing
    return CC.widenconst(argtypes[root.n])
end
function root_type(root::FieldRoot, argtypes::Vector{Any})
    T = root_type(Argument(root.n), argtypes)
    for i in root.path
        T isa DataType && 1 <= i <= fieldcount(T) || return nothing
        T = fieldtype(T, i)
    end
    return T
end
root_type(@nospecialize(_), ::Vector{Any}) = nothing

"""
    is_view_constructor(func) -> Bool

Check if a resolved function is a tensor/partition view constructor.
These propagate alias identity from their first operand.
"""
function is_view_constructor(func)
    return func === Intrinsics.make_tensor_view ||
        func === Intrinsics.make_partition_view ||
        func === Intrinsics.make_strided_view ||
        func === Intrinsics.make_gather_scatter_view
end

function is_pointer_passthrough(func)
    return func === Intrinsics.offset ||
        func === Core.Intrinsics.bitcast
end


#=============================================================================
 Public query API
=============================================================================#

"""
    AliasInfo

Result of running alias analysis. Consumers query it via `alias_class`,
`alias_classes`, and `aliases`; the underlying lattice representation is an
implementation detail.
"""
const AliasInfo = DataflowResult{AliasAnalysis, AliasElement}

"""
    analyze_aliases(sci::StructuredIRCode; alias_groups=()) -> AliasInfo

Run forward alias analysis on `sci`. Arrays listed together in `alias_groups`
share an alias set.
"""
analyze_aliases(sci::StructuredIRCode; alias_groups::AliasGroups=()) =
    analyze(AliasAnalysis(alias_groups), sci)::AliasInfo

"""
    alias_class(info::AliasInfo, op) -> AliasSet

Alias set associated with `op`. Operands the analysis didn't reach (and
non-anchor operands like literals) collapse to `ALIAS_UNIVERSE` — the
"may alias anything" default policy is encoded here, once.
"""
function alias_class(info::AliasInfo, @nospecialize(op))
    op isa SSAValue || op isa Argument || op isa SlotNumber || return ALIAS_UNIVERSE
    return something(info[op], ALIAS_UNIVERSE)
end

"""
    alias_classes(info::AliasInfo)

Iterator over the distinct alias sets the analysis recorded. Used by
consumers that need to enumerate alias equivalence classes (e.g. to seed
per-class state).
"""
alias_classes(info::AliasInfo) = values(info)

"""
    AliasResult

Result of a binary alias query (`aliases`). `MustAlias` / `PartialAlias` are
not produced — the analysis isn't flow-precise enough — so the lattice is
just `NoAlias` / `MayAlias`.
"""
@enum AliasResult NoAlias MayAlias

"""
    aliases(a::AliasSet, b::AliasSet) -> AliasResult

Binary alias query. `MayAlias` if either set is `ALIAS_UNIVERSE` or the two
sets share a root tag; `NoAlias` otherwise.
"""
function aliases(a::AliasSet, b::AliasSet)
    (a isa AliasUniverse || b isa AliasUniverse) && return MayAlias
    isempty(intersect(a, b)) ? NoAlias : MayAlias
end
