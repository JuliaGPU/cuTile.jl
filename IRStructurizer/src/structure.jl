# Phase 1: Control Tree to Structured IR
#
# Converts a ControlTree (from graph contraction) to structured IR with PartialBlock,
# Statement, and PartialControlFlowOp objects. All loops become :loop in this phase.
# Pattern matching (:for/:while) and substitutions happen in later phases.

using AbstractTrees: PreOrderDFS

#=============================================================================
 Phase 1 Helpers
=============================================================================#

"""
    get_loop_blocks(tree::ControlTree, blocks::Vector{BlockInfo}) -> Set{Int}

Get all block indices contained in a loop control tree.
"""
function get_loop_blocks(tree::ControlTree, blocks::Vector{BlockInfo})
    loop_blocks = Set{Int}()
    for subtree in PreOrderDFS(tree)
        idx = node_index(subtree)
        if 1 <= idx <= length(blocks)
            push!(loop_blocks, idx)
        end
    end
    return loop_blocks
end

"""
    convert_phi_value(val) -> IRValue

Convert a phi node value to an IRValue.
"""
function convert_phi_value(val)
    if val isa SSAValue
        return val
    elseif val isa Argument
        return val
    elseif val isa Integer
        return val
    elseif val isa QuoteNode
        return val.value
    else
        return 0  # Fallback
    end
end

#=============================================================================
 Control Tree to Structured IR
=============================================================================#

"""
    control_tree_to_structured_ir(ctree::ControlTree, code::CodeInfo, blocks::Vector{BlockInfo}) -> PartialBlock

Convert a control tree to structured IR entry block.
All loops become PartialControlFlowOp(:loop, ...) (no pattern matching yet, no substitutions).
"""
function control_tree_to_structured_ir(ctree::ControlTree, code::CodeInfo, blocks::Vector{BlockInfo})
    return tree_to_block(ctree, code, blocks)
end

"""
    tree_to_block(tree::ControlTree, code::CodeInfo, blocks::Vector{BlockInfo}) -> PartialBlock

Convert a control tree node to a PartialBlock. Creates Statement objects with raw expressions (no substitutions).
"""
function tree_to_block(tree::ControlTree, code::CodeInfo, blocks::Vector{BlockInfo})
    idx = node_index(tree)
    rtype = region_type(tree)
    block = PartialBlock()

    if rtype == REGION_BLOCK
        handle_block_region!(block, tree, code, blocks)
    elseif rtype == REGION_IF_THEN_ELSE
        handle_if_then_else!(block, tree, code, blocks)
    elseif rtype == REGION_IF_THEN
        handle_if_then!(block, tree, code, blocks)
    elseif rtype == REGION_TERMINATION
        handle_termination!(block, tree, code, blocks)
    elseif rtype == REGION_WHILE_LOOP || rtype == REGION_NATURAL_LOOP
        handle_loop!(block, tree, code, blocks)
    elseif rtype == REGION_SELF_LOOP
        handle_self_loop!(block, tree, code, blocks)
    elseif rtype == REGION_PROPER
        handle_proper_region!(block, tree, code, blocks)
    elseif rtype == REGION_SWITCH
        handle_switch!(block, tree, code, blocks)
    else
        # Fallback: collect statements
        handle_block_region!(block, tree, code, blocks)
    end

    # Set terminator if not already set
    set_block_terminator!(block, code, blocks)

    return block
end

#=============================================================================
 Region Handlers
=============================================================================#

"""
    handle_block_region!(block::PartialBlock, tree::ControlTree, code::CodeInfo, blocks::Vector{BlockInfo})

Handle REGION_BLOCK - a linear sequence of blocks.
"""
function handle_block_region!(block::PartialBlock, tree::ControlTree, code::CodeInfo, blocks::Vector{BlockInfo})
    if isempty(children(tree))
        # Leaf node - collect statements from the block
        idx = node_index(tree)
        if 1 <= idx <= length(blocks)
            collect_block_statements!(block, blocks[idx], code)
        end
    else
        # Non-leaf - process children in order
        for child in children(tree)
            child_rtype = region_type(child)
            if child_rtype == REGION_BLOCK
                handle_block_region!(block, child, code, blocks)
            else
                # Nested control flow - create appropriate op
                handle_nested_region!(block, child, code, blocks)
            end
        end
    end
end

"""
    handle_nested_region!(block::PartialBlock, tree::ControlTree, code::CodeInfo, blocks::Vector{BlockInfo})

Handle a nested control flow region.
"""
function handle_nested_region!(block::PartialBlock, tree::ControlTree, code::CodeInfo, blocks::Vector{BlockInfo})
    rtype = region_type(tree)

    if rtype == REGION_IF_THEN_ELSE
        handle_if_then_else!(block, tree, code, blocks)
    elseif rtype == REGION_IF_THEN
        handle_if_then!(block, tree, code, blocks)
    elseif rtype == REGION_TERMINATION
        handle_termination!(block, tree, code, blocks)
    elseif rtype == REGION_WHILE_LOOP || rtype == REGION_NATURAL_LOOP
        handle_loop!(block, tree, code, blocks)
    elseif rtype == REGION_SELF_LOOP
        handle_self_loop!(block, tree, code, blocks)
    elseif rtype == REGION_PROPER
        handle_proper_region!(block, tree, code, blocks)
    elseif rtype == REGION_SWITCH
        handle_switch!(block, tree, code, blocks)
    else
        handle_block_region!(block, tree, code, blocks)
    end
end

"""
    handle_if_then_else!(block::PartialBlock, tree::ControlTree, code::CodeInfo, blocks::Vector{BlockInfo})

Handle REGION_IF_THEN_ELSE.
"""
function handle_if_then_else!(block::PartialBlock, tree::ControlTree, code::CodeInfo, blocks::Vector{BlockInfo})
    tree_children = children(tree)
    length(tree_children) >= 3 || return handle_block_region!(block, tree, code, blocks)

    # First child is the condition block
    cond_tree = tree_children[1]
    cond_idx = node_index(cond_tree)

    # Collect condition block statements and find condition
    if 1 <= cond_idx <= length(blocks)
        cond_block = blocks[cond_idx]
        for si in cond_block.range
            stmt = code.code[si]
            if !(stmt isa GotoNode || stmt isa GotoIfNot || stmt isa ReturnNode || stmt isa PhiNode)
                push!(block.body, Statement(si, stmt, code.ssavaluetypes[si]))
            end
        end
    end

    cond_value = find_condition_value(cond_idx, code, blocks)

    # Then and else blocks
    then_tree = tree_children[2]
    else_tree = tree_children[3]

    then_blk = tree_to_block(then_tree, code, blocks)
    else_blk = tree_to_block(else_tree, code, blocks)

    # Create PartialControlFlowOp(:if, ...) - no outer capture yet, Phase 2 will handle it
    if_op = PartialControlFlowOp(:if, Dict{Symbol,Any}(:then => then_blk, :else => else_blk);
                                  operands=(condition=cond_value,))
    push!(block.body, if_op)
end

"""
    handle_if_then!(block::PartialBlock, tree::ControlTree, code::CodeInfo, blocks::Vector{BlockInfo})

Handle REGION_IF_THEN.
"""
function handle_if_then!(block::PartialBlock, tree::ControlTree, code::CodeInfo, blocks::Vector{BlockInfo})
    tree_children = children(tree)
    length(tree_children) >= 2 || return handle_block_region!(block, tree, code, blocks)

    # First child is the condition block
    cond_tree = tree_children[1]
    cond_idx = node_index(cond_tree)

    # Collect condition block statements
    if 1 <= cond_idx <= length(blocks)
        cond_block = blocks[cond_idx]
        for si in cond_block.range
            stmt = code.code[si]
            if !(stmt isa GotoNode || stmt isa GotoIfNot || stmt isa ReturnNode || stmt isa PhiNode)
                push!(block.body, Statement(si, stmt, code.ssavaluetypes[si]))
            end
        end
    end

    cond_value = find_condition_value(cond_idx, code, blocks)

    # Then block
    then_tree = tree_children[2]
    then_blk = tree_to_block(then_tree, code, blocks)

    # Empty else block
    else_blk = PartialBlock()

    # Create PartialControlFlowOp(:if, ...) - no outer capture yet, Phase 2 will handle it
    if_op = PartialControlFlowOp(:if, Dict{Symbol,Any}(:then => then_blk, :else => else_blk);
                                  operands=(condition=cond_value,))
    push!(block.body, if_op)
end

"""
    handle_termination!(block::PartialBlock, tree::ControlTree, code::CodeInfo, blocks::Vector{BlockInfo})

Handle REGION_TERMINATION - branches where some paths terminate.
"""
function handle_termination!(block::PartialBlock, tree::ControlTree, code::CodeInfo, blocks::Vector{BlockInfo})
    tree_children = children(tree)
    isempty(tree_children) && return handle_block_region!(block, tree, code, blocks)

    # First child is the condition block
    cond_tree = tree_children[1]
    cond_idx = node_index(cond_tree)

    # Collect condition block statements
    if 1 <= cond_idx <= length(blocks)
        cond_block = blocks[cond_idx]
        for si in cond_block.range
            stmt = code.code[si]
            if !(stmt isa GotoNode || stmt isa GotoIfNot || stmt isa ReturnNode || stmt isa PhiNode)
                push!(block.body, Statement(si, stmt, code.ssavaluetypes[si]))
            end
        end
    end

    cond_value = find_condition_value(cond_idx, code, blocks)

    # Build then and else blocks from remaining children
    if length(tree_children) >= 3
        then_tree = tree_children[2]
        else_tree = tree_children[3]
        then_blk = tree_to_block(then_tree, code, blocks)
        else_blk = tree_to_block(else_tree, code, blocks)
        if_op = PartialControlFlowOp(:if, Dict{Symbol,Any}(:then => then_blk, :else => else_blk);
                                      operands=(condition=cond_value,))
        push!(block.body, if_op)
    elseif length(tree_children) == 2
        then_tree = tree_children[2]
        then_blk = tree_to_block(then_tree, code, blocks)
        else_blk = PartialBlock()
        if_op = PartialControlFlowOp(:if, Dict{Symbol,Any}(:then => then_blk, :else => else_blk);
                                      operands=(condition=cond_value,))
        push!(block.body, if_op)
    end
end

"""
    handle_loop!(block::PartialBlock, tree::ControlTree, code::CodeInfo, blocks::Vector{BlockInfo})

Handle REGION_WHILE_LOOP and REGION_NATURAL_LOOP.
Phase 1: Always creates PartialControlFlowOp(:loop, ...) with metadata. Pattern matching happens in Phase 3.
"""
function handle_loop!(block::PartialBlock, tree::ControlTree, code::CodeInfo, blocks::Vector{BlockInfo})
    loop_op = build_loop_op_phase1(tree, code, blocks)
    push!(block.body, loop_op)
end

"""
    handle_self_loop!(block::PartialBlock, tree::ControlTree, code::CodeInfo, blocks::Vector{BlockInfo})

Handle REGION_SELF_LOOP.
"""
function handle_self_loop!(block::PartialBlock, tree::ControlTree, code::CodeInfo, blocks::Vector{BlockInfo})
    idx = node_index(tree)

    body_blk = PartialBlock()

    if 1 <= idx <= length(blocks)
        collect_block_statements!(body_blk, blocks[idx], code)
    end

    loop_op = PartialControlFlowOp(:loop, Dict{Symbol,Any}(:body => body_blk))
    push!(block.body, loop_op)
end

"""
    handle_proper_region!(block::PartialBlock, tree::ControlTree, code::CodeInfo, blocks::Vector{BlockInfo})

Handle REGION_PROPER - acyclic region not matching other patterns.
"""
function handle_proper_region!(block::PartialBlock, tree::ControlTree, code::CodeInfo, blocks::Vector{BlockInfo})
    # Process as a sequence of blocks
    handle_block_region!(block, tree, code, blocks)
end

"""
    handle_switch!(block::PartialBlock, tree::ControlTree, code::CodeInfo, blocks::Vector{BlockInfo})

Handle REGION_SWITCH.
"""
function handle_switch!(block::PartialBlock, tree::ControlTree, code::CodeInfo, blocks::Vector{BlockInfo})
    # For now, handle as a nested if-else chain
    # TODO: Implement proper switch handling if needed
    handle_block_region!(block, tree, code, blocks)
end

#=============================================================================
 Statement Collection Helpers
=============================================================================#

"""
    collect_block_statements!(block::PartialBlock, info::BlockInfo, code::CodeInfo)

Collect statements from a BlockInfo into a PartialBlock, excluding control flow.
Creates Statement objects with raw expressions (no substitutions).
"""
function collect_block_statements!(block::PartialBlock, info::BlockInfo, code::CodeInfo)
    stmts = code.code
    types = code.ssavaluetypes
    for si in info.range
        stmt = stmts[si]
        if stmt isa ReturnNode
            block.terminator = stmt
        elseif !(stmt isa GotoNode || stmt isa GotoIfNot || stmt isa PhiNode)
            push!(block.body, Statement(si, stmt, types[si]))
        end
    end
end

"""
    find_condition_value(block_idx::Int, code::CodeInfo, blocks::Vector{BlockInfo}) -> IRValue

Find the condition value for a GotoIfNot in the given block.
"""
function find_condition_value(block_idx::Int, code::CodeInfo, blocks::Vector{BlockInfo})
    block_idx < 1 || block_idx > length(blocks) && return SSAValue(1)

    block = blocks[block_idx]
    for si in block.range
        stmt = code.code[si]
        if stmt isa GotoIfNot
            cond = stmt.cond
            if cond isa SSAValue || cond isa SlotNumber || cond isa Argument
                return cond
            else
                return SSAValue(max(1, si - 1))
            end
        end
    end

    return SSAValue(max(1, first(block.range)))
end

"""
    set_block_terminator!(block::PartialBlock, code::CodeInfo, blocks::Vector{BlockInfo})

Set the block terminator based on statements.
"""
function set_block_terminator!(block::PartialBlock, code::CodeInfo, blocks::Vector{BlockInfo})
    block.terminator !== nothing && return

    # Find the last statement index in body
    last_idx = nothing
    for item in reverse(block.body)
        if item isa Statement
            last_idx = item.idx
            break
        end
    end
    if last_idx !== nothing && last_idx < length(code.code)
        next_stmt = code.code[last_idx + 1]
        if next_stmt isa ReturnNode
            block.terminator = next_stmt
        end
    end
end

#=============================================================================
 Loop Construction
=============================================================================#

"""
    build_loop_op_phase1(tree::ControlTree, code::CodeInfo, blocks::Vector{BlockInfo}) -> PartialControlFlowOp

Build a PartialControlFlowOp(:loop, ...) for Phase 1. Pure structure building - no BlockArgs or substitutions.
BlockArg creation and SSA→BlockArg substitution happens in Phase 2 (apply_block_args!).
"""
function build_loop_op_phase1(tree::ControlTree, code::CodeInfo, blocks::Vector{BlockInfo})
    stmts = code.code
    types = code.ssavaluetypes
    header_idx = node_index(tree)
    loop_blocks = get_loop_blocks(tree, blocks)

    @assert 1 <= header_idx <= length(blocks) "Invalid header_idx from control tree: $header_idx"
    header_block = blocks[header_idx]
    stmt_to_blk = stmt_to_block_map(blocks, length(stmts))

    # Find phi nodes in header - these become loop-carried values and results
    init_values = IRValue[]      # Entry values for each phi
    carried_values = IRValue[]   # Loop-back values for each phi (SSAValues)
    result_vars = SSAValue[]     # SSA indices of phi nodes

    for si in header_block.range
        stmt = stmts[si]
        if stmt isa PhiNode
            push!(result_vars, SSAValue(si))
            phi = stmt

            entry_val = nothing
            carried_val = nothing

            for (edge_idx, _) in enumerate(phi.edges)
                if isassigned(phi.values, edge_idx)
                    val = phi.values[edge_idx]

                    if val isa SSAValue
                        val_stmt = val.id
                        if val_stmt > 0 && val_stmt <= length(stmts)
                            val_block = stmt_to_blk[val_stmt]
                            if val_block ∈ loop_blocks
                                carried_val = val
                            else
                                entry_val = convert_phi_value(val)
                            end
                        else
                            entry_val = convert_phi_value(val)
                        end
                    else
                        entry_val = convert_phi_value(val)
                    end
                end
            end

            entry_val !== nothing && push!(init_values, entry_val)
            carried_val !== nothing && push!(carried_values, carried_val)
        end
    end

    # Build loop body block (no BlockArgs yet - Phase 2 will add them)
    body = PartialBlock()
    # body.args stays empty - Phase 2 will populate it

    # Find the condition for loop exit
    condition = nothing
    for si in header_block.range
        stmt = stmts[si]
        if stmt isa GotoIfNot
            condition = stmt.cond
            break
        end
    end

    # Collect header statements (excluding phi nodes and control flow)
    for si in header_block.range
        stmt = stmts[si]
        if !(stmt isa PhiNode || stmt isa GotoNode || stmt isa GotoIfNot || stmt isa ReturnNode)
            push!(body.body, Statement(si, stmt, types[si]))
        end
    end

    # Create the conditional structure inside the loop body
    if condition !== nothing
        cond_value = convert_phi_value(condition)

        then_blk = PartialBlock()

        # Process loop body blocks (excluding header)
        for child in children(tree)
            child_idx = node_index(child)
            if child_idx != header_idx
                handle_block_region!(then_blk, child, code, blocks)
            end
        end
        # ContinueOp with raw carried_values (SSAValues) - Phase 2 will substitute
        then_blk.terminator = ContinueOp(copy(carried_values))

        else_blk = PartialBlock()
        # BreakOp with result_vars (SSAValues) - Phase 2 will substitute to BlockArgs
        else_blk.terminator = BreakOp(IRValue[rv for rv in result_vars])

        if_op = PartialControlFlowOp(:if, Dict{Symbol,Any}(:then => then_blk, :else => else_blk);
                                      operands=(condition=cond_value,))
        push!(body.body, if_op)
    else
        # No condition - process children directly
        for child in children(tree)
            child_idx = node_index(child)
            if child_idx != header_idx
                handle_block_region!(body, child, code, blocks)
            end
        end
        # ContinueOp with raw carried_values (SSAValues) - Phase 2 will substitute
        body.terminator = ContinueOp(copy(carried_values))
    end

    # Create PartialControlFlowOp(:loop, ...) - no outer capture yet, Phase 2 will handle it
    return PartialControlFlowOp(:loop, Dict{Symbol,Any}(:body => body);
                                 init_values=init_values, result_vars=result_vars)
end

"""
    collect_defined_ssas!(defined::Set{Int}, block::PartialBlock)

Collect all SSA indices defined by statements in the block (recursively).
Also includes result_vars from loops (phi nodes define SSAValues).
"""
function collect_defined_ssas!(defined::Set{Int}, block::PartialBlock)
    for item in block.body
        if item isa Statement
            push!(defined, item.idx)
        elseif item isa PartialControlFlowOp
            # result_vars define SSAValues
            for rv in item.result_vars
                push!(defined, rv.id)
            end
            # Recurse into all regions
            for (_, region) in item.regions
                collect_defined_ssas!(defined, region)
            end
        end
    end
end

#=============================================================================
 Phase 2: Apply Block Arguments
=============================================================================#

"""
    apply_block_args!(block::PartialBlock, types, defined::Set{Int}=Set{Int}(), parent_subs::Substitutions=Substitutions())

Single pass that creates BlockArgs and substitutes SSAValue references.

Phase 2 of structurization - called after control_tree_to_structured_ir.
For each :loop op: creates BlockArgs for phi nodes (result_vars) and outer captures.
For each :if op: creates BlockArgs for outer captures.
Substitutes SSAValue → BlockArg references throughout.

The parent_subs parameter carries substitutions from outer scopes, so nested
control flow ops can convert SSAValues to the correct BlockArgs.
"""
function apply_block_args!(block::PartialBlock, types, defined::Set{Int}=Set{Int}(), parent_subs::Substitutions=Substitutions())
    # Track what's defined at this level
    defined = copy(defined)
    for item in block.body
        if item isa Statement
            push!(defined, item.idx)
        end
    end

    # Process each control flow op
    for item in block.body
        if item isa PartialControlFlowOp
            if item.head == :loop
                process_loop_block_args!(item, types, defined, parent_subs)
            elseif item.head == :if
                process_if_block_args!(item, types, defined, parent_subs)
            end
        end
    end
end

"""
    process_loop_block_args!(loop::PartialControlFlowOp, types, parent_defined::Set{Int}, parent_subs::Substitutions)

Create BlockArgs for a :loop op and substitute SSAValue references.

1. Create BlockArgs for phi nodes (from result_vars)
2. Collect outer refs (SSAValues not defined in loop or as phi)
3. Create BlockArgs for outer captures
4. Apply parent substitutions to init_values (so nested loops get parent's BlockArgs)
5. Substitute all SSAValue → BlockArg in body
6. Recurse into nested blocks
"""
function process_loop_block_args!(loop::PartialControlFlowOp, types, parent_defined::Set{Int}, parent_subs::Substitutions)
    @assert loop.head == :loop
    body = loop.regions[:body]::PartialBlock
    subs = Substitutions()

    # 1. Create BlockArgs for phi nodes (from result_vars)
    for (i, result_var) in enumerate(loop.result_vars)
        phi_type = types[result_var.id]
        new_arg = BlockArg(i, phi_type)
        push!(body.args, new_arg)
        subs[result_var.id] = new_arg
    end

    # 2. Collect outer refs (SSAValues not defined in loop body or as phi)
    loop_defined = Set{Int}(rv.id for rv in loop.result_vars)
    collect_defined_ssas!(loop_defined, body)
    outer_refs = collect_outer_refs(body, loop_defined; recursive=true)

    # 3. Create BlockArgs for outer captures
    n_existing = length(body.args)
    for (i, ref) in enumerate(outer_refs)
        ref_type = types[ref.id]
        new_arg = BlockArg(n_existing + i, ref_type)
        push!(body.args, new_arg)
        push!(loop.init_values, ref)
        subs[ref.id] = new_arg
    end

    # 4. Apply parent substitutions to init_values
    # This converts SSAValues to parent's BlockArgs for nested control flow
    for (j, v) in enumerate(loop.init_values)
        loop.init_values[j] = substitute_ssa(v, parent_subs)
    end

    # 5. Substitute all SSAValue → BlockArg in body (shallow - don't recurse into nested ops)
    substitute_block_shallow!(body, subs)

    # 6. Recurse into nested blocks, passing merged substitutions
    # Merge parent subs with this loop's subs so nested ops can access both
    merged_subs = merge(parent_subs, subs)
    nested_defined = Set{Int}(rv.id for rv in loop.result_vars)
    collect_defined_ssas!(nested_defined, body)
    apply_block_args!(body, types, nested_defined, merged_subs)
end

"""
    process_if_block_args!(if_op::PartialControlFlowOp, types, parent_defined::Set{Int}, parent_subs::Substitutions)

Create BlockArgs for an :if op and substitute SSAValue references.

1. Collect outer refs from both blocks
2. Create matching BlockArgs in both blocks
3. Apply parent substitutions to init_values
4. Substitute SSAValue → BlockArg in both blocks
5. Recurse into nested blocks
"""
function process_if_block_args!(if_op::PartialControlFlowOp, types, parent_defined::Set{Int}, parent_subs::Substitutions)
    @assert if_op.head == :if
    then_blk = if_op.regions[:then]::PartialBlock
    else_blk = if_op.regions[:else]::PartialBlock

    # Build combined defined set from parent + what's defined in each branch
    then_defined = copy(parent_defined)
    else_defined = copy(parent_defined)
    collect_defined_ssas!(then_defined, then_blk)
    collect_defined_ssas!(else_defined, else_blk)

    # 1. Collect outer refs from both blocks, filtering out refs already captured by parents.
    # Use recursive=true to find all refs, but skip refs that are in parent_subs since those
    # have already been captured by an enclosing scope.
    outer_refs = SSAValue[]
    seen = Set{Int}()
    for ref in collect_outer_refs(then_blk, then_defined; recursive=true)
        if ref.id ∉ seen && !haskey(parent_subs, ref.id)
            push!(outer_refs, ref)
            push!(seen, ref.id)
        end
    end
    for ref in collect_outer_refs(else_blk, else_defined; recursive=true)
        if ref.id ∉ seen && !haskey(parent_subs, ref.id)
            push!(outer_refs, ref)
            push!(seen, ref.id)
        end
    end

    subs = Substitutions()
    if !isempty(outer_refs)
        # 2. Create matching BlockArgs in both blocks
        for (i, ref) in enumerate(outer_refs)
            ref_type = types[ref.id]
            new_arg = BlockArg(i, ref_type)
            push!(then_blk.args, new_arg)
            push!(else_blk.args, new_arg)
            push!(if_op.init_values, ref)
            subs[ref.id] = new_arg
        end

        # 3. Apply parent substitutions to init_values
        for (j, v) in enumerate(if_op.init_values)
            if_op.init_values[j] = substitute_ssa(v, parent_subs)
        end

        # 4. Substitute SSAValue → BlockArg in both blocks (shallow - don't recurse into nested ops)
        substitute_block_shallow!(then_blk, subs)
        substitute_block_shallow!(else_blk, subs)
    end

    # 5. Recurse into nested blocks, passing merged substitutions
    merged_subs = merge(parent_subs, subs)
    apply_block_args!(then_blk, types, then_defined, merged_subs)
    apply_block_args!(else_blk, types, else_defined, merged_subs)
end
