# Broadcast Macro for cuTile
#
# Transforms Julia broadcast expressions into fused cuTile kernels.
# `@fuse c .= BFloat16.(a) .+ b` generates a kernel that loads tiles,
# evaluates the expression via TileStyle broadcast, and stores the result.

public @fuse

const _CUTILE_MAX_NDIMS = 8  # only used as a sanity check, not a hard limit

#=============================================================================
 AST Utilities
=============================================================================#

# Bottom-up expression rewriter (no MacroTools dependency)
_postwalk(f, x) = f(x)
_postwalk(f, ex::Expr) = f(Expr(ex.head, map(x -> _postwalk(f, x), ex.args)...))

# Rewrite Type.(args...) to convert.(Type, args...) in broadcast expressions.
# cuTile can't compile broadcasted(::DataType, ...) because typeof(Type) = DataType is abstract.
# The convert.() form works because convert is a concrete function and the Type argument
# goes through TypeRef via the broadcastable overlay.
# Heuristic: symbols starting with uppercase are types (covers Float16, BFloat16, Int32, etc.)
# Returns (rewritten_expr, Vector{Symbol} of type symbols).
function _rewrite_type_broadcasts(ex)
    type_syms = Symbol[]
    result = _postwalk(ex) do node
        if node isa Expr && node.head === :. && length(node.args) == 2 &&
           node.args[2] isa Expr && node.args[2].head === :tuple
            f = node.args[1]
            if f isa Symbol && isuppercase(first(string(f)))
                f in type_syms || push!(type_syms, f)
                # T.(args...) → convert.(T, args...)
                return Expr(:., :convert, Expr(:tuple, f, node.args[2].args...))
            end
        end
        node
    end
    return result, type_syms
end

# Collect all symbols in value position from a broadcast expression.
# Skips function-position symbols (args[1] of :call and :. with :tuple).
function _collect_leaves(ex)
    leaves = Symbol[]
    _collect_leaves!(leaves, ex)
    return unique!(leaves)
end

_collect_leaves!(leaves, ex::Symbol) = push!(leaves, ex)
_collect_leaves!(leaves, _) = nothing  # Numbers, QuoteNodes, etc.

function _collect_leaves!(leaves, ex::Expr)
    if ex.head === :call
        # args[1] is the function (e.g. :.+), skip it
        for i in 2:length(ex.args)
            _collect_leaves!(leaves, ex.args[i])
        end
    elseif ex.head === :. && length(ex.args) == 2 &&
           ex.args[2] isa Expr && ex.args[2].head === :tuple
        # f.(args...) syntax — args[1] is the function (e.g. :BFloat16), skip it
        for a in ex.args[2].args
            _collect_leaves!(leaves, a)
        end
    else
        for a in ex.args
            _collect_leaves!(leaves, a)
        end
    end
end

#=============================================================================
 Kernel Code Generator
=============================================================================#

# Generate the kernel body Expr for a specific ndims N.
# Called both from the macro (multi-method path) and from @generated compile time.
function _gen_kernel_body(N::Int, ct::Module, dest::Symbol,
                          operands::Vector{Symbol}, leaves::Vector{Symbol},
                          type_syms::Vector{Symbol}, rhs_expr,
                          op_singleton_params::Dict{Symbol,Symbol},
                          type_param_vars::Dict{Symbol,Symbol})
    tiles_param = Symbol("_tiles")
    grids_param = Symbol("_grids")
    bid_vars = [gensym("bid_$d") for d in 1:N]

    leaf_tile_vars = Dict{Symbol, Symbol}()
    for s in leaves
        leaf_tile_vars[s] = gensym(string(s))
    end

    body = Expr[]

    # 1. Compute N-dimensional block indices from CUDA grid (up to 3 dims).
    if N <= 3
        for d in 1:N
            push!(body, :($(bid_vars[d]) = $ct.bid($d)))
        end
    else
        push!(body, :($(bid_vars[1]) = $ct.bid(1)))
        push!(body, :($(bid_vars[2]) = $ct.bid(2)))
        rem_var = gensym("rem")
        push!(body, :($rem_var = $ct.bid(3) - Int32(1)))
        for d in 3:N
            gp_idx = d - 2
            if d < N
                push!(body, :($(bid_vars[d]) = rem($rem_var, Int32($grids_param[$gp_idx])) + Int32(1)))
                push!(body, :($rem_var = fld($rem_var, Int32($grids_param[$gp_idx]))))
            else
                push!(body, :($(bid_vars[d]) = $rem_var + Int32(1)))
            end
        end
    end

    # 2. Load each leaf
    for s in leaves
        tv = leaf_tile_vars[s]
        if s === dest && !(s in operands)
            idx = N == 1 ? bid_vars[1] : Expr(:tuple, bid_vars...)
            shape = Expr(:tuple, [:($tiles_param[$d]) for d in 1:N]...)
            push!(body, :($tv = $ct.load($s, $idx, $shape)))
        elseif s in operands
            sp = op_singleton_params[s]
            adj_bids = [:($(sp)[$d] ? Int32(1) : $(bid_vars[d])) for d in 1:N]
            adj_tiles = [:($(sp)[$d] ? 1 : $tiles_param[$d]) for d in 1:N]
            idx = N == 1 ? adj_bids[1] : Expr(:tuple, adj_bids...)
            shape = Expr(:tuple, adj_tiles...)
            push!(body, :($tv = if $s isa $ct.TileArray
                $ct.load($s, $idx, $shape)
            else
                $s
            end))
        end
    end

    # 3. Rewrite RHS expression
    rewritten = _postwalk(rhs_expr) do node
        if node isa Symbol
            haskey(leaf_tile_vars, node) && return leaf_tile_vars[node]
            haskey(type_param_vars, node) && return type_param_vars[node]
        end
        node
    end

    # 4. Evaluate, convert to dest eltype, and store
    result_var = gensym("result")
    push!(body, :($result_var = $rewritten))
    push!(body, :($result_var = convert($ct.Tile{eltype($dest)}, $result_var)))
    store_idx = N == 1 ? bid_vars[1] : Expr(:tuple, bid_vars...)
    push!(body, :($ct.store($dest, $store_idx, $result_var)))
    push!(body, :(return))

    return Expr(:block, body...)
end

# Generate a full kernel method definition for ndims N (used in multi-method fallback).
function _gen_kernel(N::Int, ct::Module, kernel_name, dest::Symbol,
                     operands::Vector{Symbol}, leaves::Vector{Symbol},
                     type_syms::Vector{Symbol}, rhs_expr)
    op_singleton_params = Dict{Symbol,Symbol}(op => gensym("$(op)_s") for op in operands)
    type_param_vars = Dict{Symbol,Symbol}(ts => gensym("type_$(ts)") for ts in type_syms)

    dest_type = Expr(:curly, :($ct.TileArray), Expr(:(<:), :Any), N)
    params = Any[:($(dest)::$(dest_type))]
    for op in operands; push!(params, op); end
    push!(params, Symbol("_tiles"))
    push!(params, Symbol("_grids"))
    for op in operands; push!(params, op_singleton_params[op]); end
    for ts in type_syms; push!(params, type_param_vars[ts]); end

    body = _gen_kernel_body(N, ct, dest, operands, leaves, type_syms, rhs_expr,
                            op_singleton_params, type_param_vars)

    return Expr(:function, Expr(:call, kernel_name, params...), body)
end

#=============================================================================
 Launch Wrapper Generator
=============================================================================#

function _gen_launch(ct::Module, launch_name, dest::Symbol,
                     operands::Vector{Symbol}, type_syms::Vector{Symbol},
                     kernel_name, tile_size_expr)
    all_args = Any[dest; operands; type_syms]

    arg_vars = Dict{Symbol, Symbol}()
    singleton_vars = Dict{Symbol, Symbol}()
    for op in operands
        arg_vars[op] = gensym("$(op)_arg")
        singleton_vars[op] = gensym("$(op)_s")
    end

    N_var = gensym("N")
    ts_var = gensym("ts")
    grid_var = gensym("grid")
    dest_ta = gensym("dest_ta")

    body = Any[]

    push!(body, :($N_var = ndims($dest)))
    push!(body, :($N_var <= $_CUTILE_MAX_NDIMS ||
        error("@fuse: ndims ", $N_var, " exceeds maximum supported (", $_CUTILE_MAX_NDIMS, ")")))
    # Expand scalar tile_size to N-tuple: first min(N,2) dims get tile_size, rest get 1.
    push!(body, :($ts_var = if $tile_size_expr isa Tuple
        ntuple(i -> i <= length($tile_size_expr) ? $tile_size_expr[i] : 1, $N_var)
    else
        ntuple(i -> i <= min($N_var, 2) ? $tile_size_expr : 1, $N_var)
    end))
    push!(body, :($grid_var = ntuple(i -> cld(size($dest, i), $ts_var[i]), $N_var)))

    # Build CUDA grid: up to 3 dims. For N>3, flatten dims 3:N into dim 3.
    launch_grid = gensym("launch_grid")
    push!(body, :($launch_grid = if $N_var <= 3
        $grid_var
    else
        ($grid_var[1], $grid_var[2], prod($grid_var[i] for i in 3:$N_var))
    end))

    push!(body, :($dest_ta = $ct.TileArray($dest)))

    for op in operands
        av = arg_vars[op]
        push!(body, :($av = $op isa Number ? $op : $ct.TileArray($op)))
    end

    for op in operands
        push!(body, :(if !($op isa Number) && ndims($op) != $N_var
            error("@fuse: all array operands must have the same ndims as destination (",
                  $N_var, "), got ", ndims($op), " for operand")
        end))
    end

    for op in operands
        sv = singleton_vars[op]
        push!(body, :($sv = $op isa Number ?
            ntuple(_ -> true, $N_var) :
            ntuple(d -> isone(size($op, d)), $N_var)))
    end

    # Build the launch call — always pass grid overflow (empty tuple for N≤3)
    launch_args = Any[dest_ta]
    for op in operands
        push!(launch_args, arg_vars[op])
    end

    singleton_constants = [:(($ct.Constant)($(singleton_vars[op]))) for op in operands]
    # Use Constant{Type{T}, T}() instead of Constant(T) — typeof(T) is DataType (abstract),
    # but Type{T} is concrete and passes isdispatchtuple.
    type_constants = [:(($ct.Constant){Type{$ts}, $ts}()) for ts in type_syms]

    grid_const_var = gensym("grid_const")
    push!(body, :($grid_const_var = $N_var > 3 ? $grid_var[3:end] : ()))

    push!(body, :($ct.launch(
        $kernel_name, $launch_grid,
        $(launch_args...),
        ($ct.Constant)($ts_var),
        ($ct.Constant)($grid_const_var),
        $(singleton_constants...),
        $(type_constants...))))

    return Expr(:function,
        Expr(:call, launch_name, all_args..., Expr(:kw, :tile_size, tile_size_expr)),
        Expr(:block, body...))
end

#=============================================================================
 @fuse Macro
=============================================================================#

# Shared implementation for @fuse and @.
function _fuse_impl(ex, tile_size)
    if !(ex isa Expr && ex.head === :.=)
        error("@fuse: expected `dest .= expr`, got: $ex")
    end
    dest = ex.args[1]
    rhs = ex.args[2]
    dest isa Symbol || error("@fuse: destination must be a symbol, got: $dest")

    rhs, type_syms = _rewrite_type_broadcasts(rhs)

    leaves = filter(s -> !(s in type_syms), _collect_leaves(rhs))
    isempty(leaves) && error("@fuse: expression must contain at least one array operand")

    operands = filter(!=(dest), leaves)

    ct = @__MODULE__

    # Generate kernel methods (one per ndims, dispatched on dest::TileArray{<:Any, N})
    kernel_name = gensym("fuse_kernel")
    kernel_defs = [_gen_kernel(N, ct, kernel_name, dest, operands, leaves, type_syms, rhs)
                   for N in 1:_CUTILE_MAX_NDIMS]

    # Generate launch wrapper
    launch_name = gensym("fuse_launch")
    launch_def = _gen_launch(ct, launch_name, dest, operands, type_syms,
                             kernel_name, tile_size)

    all_args = Any[dest; operands; type_syms]
    result = Expr(:block,
        kernel_defs...,
        launch_def,
        Expr(:call, launch_name, all_args...))

    return esc(result)
end

function _parse_fuse_options(args)
    tile_size = 64
    ex = args[end]
    for a in args[1:end-1]
        if a isa Expr && a.head === :(=) && a.args[1] === :tile_size
            tile_size = a.args[2]
        else
            error("@fuse: unknown option `$a`. Usage: @fuse [tile_size=N] dest .= expr")
        end
    end
    return tile_size, ex
end

"""
    @fuse [tile_size=N] dest .= expr

Generate a fused cuTile kernel from a broadcast expression.
"""
macro fuse(args...)
    tile_size, ex = _parse_fuse_options(args)
    _fuse_impl(ex, tile_size)
end

"""
    @. dest = expr

Like `Base.@.` but generates a fused cuTile kernel.
`ct.@. c = BFloat16(a) + b` is equivalent to `ct.@fuse c .= BFloat16.(a) .+ b`.
"""
macro __dot__(ex)
    _fuse_impl(Base.Broadcast.__dot__(ex), 64)
end
