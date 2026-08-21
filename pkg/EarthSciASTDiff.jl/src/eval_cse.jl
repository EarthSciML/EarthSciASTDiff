# Evaluation-model CSE — stage 2 of the coefficient-size control roadmap item.
#
# Template CSE at emission (cse.jl) shrinks the SERIALIZED block; this pass
# shrinks the derived EVALUATION model `prepare_jacobian` compiles. The same
# hash-consed extraction runs over the band coefficients, but instead of
# `expression_templates` each repeated subtree becomes an OBSERVED ARRAY
# variable of the derived document, defined by a `makearray` over the union
# of its use regions, and every occurrence becomes an `index` read of it.
# The tree-walk's factored array-observed buffers (§2b-f of
# EarthSciAST/src/tree_walk.jl) then evaluate each shared subtree ONCE per
# cell per RHS call into a buffer all band fields gather — so both the
# derived-document size (the compile-time lever: parsing + `_Node` IR over
# PPM-scale coefficients dominates `prepare_jacobian`) and the per-call
# duplicate work drop together.
#
# Free row-index names in a hoisted fragment become the observed array's
# axes (sorted for determinism); a fragment with no free index names hoists
# to a scalar observed (a per-call prelude slot). The pass is conservative:
# any inconsistency — a template whose free index names are not bound by
# every band that (transitively) uses it — abandons hoisting entirely and
# the caller falls back to the plain (inline-coefficient) document.

# Every `apply_expression_template` reference name in `e`.
function _template_refs(e::ASTExpr, acc::Set{String} = Set{String}())
    e isa OpExpr || return acc
    e.op == "apply_expression_template" && e.name !== nothing && push!(acc, e.name)
    map_children(c -> (_template_refs(c, acc); c), e)
    return acc
end

# Replace template references by their reads (name → read expression).
function _deref(e::ASTExpr, readmap::Dict{String,ASTExpr})::ASTExpr
    e isa OpExpr || return e
    e.op == "apply_expression_template" && e.name !== nothing &&
        haskey(readmap, e.name) && return readmap[e.name]
    return map_children(x -> _deref(x, readmap), e)
end

"""
    hoist_observed(entries, sv::SysView; min_nodes = 12) -> NamedTuple | nothing

Extract repeated coefficient subtrees (≥ `min_nodes` AST nodes, ≥ 2
occurrences; [`cse_templates`](@ref)) and package each as an observed
variable of the derived evaluation document. Returns `nothing` when there is
nothing to hoist (or when a template's index names cannot be mapped
consistently — the caller then emits the plain document). Otherwise:

  - `obs::OrderedDict{String,Any}`   — variable name → serialized variable dict
  - `axes::OrderedDict{String,Int}`  — index-set additions (axis name → size)
  - `coefs::Vector{ASTExpr}`         — the entries' coefficients, rewritten to
                                       read the hoisted observeds
"""
function hoist_observed(entries::Vector{JacEntry}, sv::SysView; min_nodes::Int = 12)
    coefs = ASTExpr[en.band.coef for en in entries]
    # Hoisted names must be fresh in the derived model's variable namespace.
    pfx = "Jcse"
    while any(startswith(k, pfx) for k in keys(sv.variables))
        pfx *= "x"
    end
    templates, rewritten = cse_templates(coefs; min_nodes = min_nodes, prefix = pfx)
    isempty(templates) && return nothing
    tnames = collect(keys(templates))

    # Per entry: index name → (lo, hi) box — row dims plus any free
    # contracted column dims (their names are just as free in the
    # coefficient, and the derived J_k field binds them the same way).
    boxof = [begin
                 d = Dict{String,Tuple{Int,Int}}(
                     String(r) => en.band.rows[k]
                     for (k, r) in enumerate(en.band.ridx) if r isa String)
                 for (n, lo, hi) in en.band.contracted
                     d[n] = (lo, hi)
                 end
                 d
             end
             for en in entries]
    namesk = [Set{String}(keys(b)) for b in boxof]
    allnames = isempty(namesk) ? Set{String}() : union(namesk...)

    # Which entries evaluate each template, transitively through the
    # reference DAG (a template read inside another template's body runs
    # wherever the outer one does).
    refs_body = Dict{String,Set{String}}(n => _template_refs(templates[n]) for n in tnames)
    uses = Dict{String,Set{Int}}(n => Set{Int}() for n in tnames)
    for (k, e) in enumerate(rewritten), n in _template_refs(e)
        push!(uses[n], k)
    end
    changed = true
    while changed
        changed = false
        for n in tnames, n2 in refs_body[n]
            before = length(uses[n2])
            union!(uses[n2], uses[n])
            length(uses[n2]) != before && (changed = true)
        end
    end
    live = [n for n in tnames if !isempty(uses[n])]

    # Free row-index names per template (transitive; sorted = the axis order).
    S = Dict{String,Vector{String}}()
    function s_of(n)
        haskey(S, n) && return S[n]
        s = Set{String}(intersect(free_variables(templates[n]), allnames))
        for n2 in refs_body[n]
            union!(s, Set(s_of(n2)))
        end
        return (S[n] = sort!(collect(s)))
    end
    foreach(s_of, live)
    # Consistency: every user band must bind every axis name. (Holds by
    # construction — a fragment's free index names are free in each using
    # coefficient, hence in that band's ridx — unless an index name collides
    # with a variable name; then hoisting is silently abandoned.)
    for n in live, k in uses[n]
        issubset(Set(S[n]), namesk[k]) || return nothing
    end

    # Use boxes (deduplicated, ascending entry order) and per-axis extents.
    boxes = Dict{String,Vector{Vector{Tuple{Int,Int}}}}()
    for n in live
        bs = Vector{Vector{Tuple{Int,Int}}}(); seen = Set{String}()
        for k in sort!(collect(uses[n]))
            box = [boxof[k][x] for x in S[n]]
            key = string(box)
            key in seen || (push!(seen, key); push!(bs, box))
        end
        boxes[n] = bs
    end

    # Axis index sets: one per distinct extent, reusing an existing set of
    # exactly that size when the name is free.
    axes = OrderedDict{String,Int}()
    existing = Dict{String,Int}()
    for (nm, s) in sv.index_sets
        s.size === nothing || (existing[String(nm)] = s.size)
    end
    function axis_for(sz::Int)
        nm = "jcax$(sz)"
        while haskey(existing, nm) && existing[nm] != sz
            nm *= "x"
        end
        haskey(existing, nm) || (axes[nm] = sz; existing[nm] = sz)
        return nm
    end

    readmap = Dict{String,ASTExpr}(
        n => isempty(S[n]) ? VarExpr(n) :
             op("index", VarExpr(n), (VarExpr(x) for x in S[n])...)
        for n in live)

    obs = OrderedDict{String,Any}()
    # esm 1.0.0 moved a definition out of the variable and into the model's
    # equations, so a hoisted observed now comes back in TWO pieces: the
    # declaration in `obs` and its defining RHS in `obs_eqs`, which the caller
    # (emit.jl) pushes onto the document's `equations`. Same order as `obs`.
    obs_eqs = Pair{String,Any}[]
    for n in live
        body = _deref(templates[n], readmap)
        if isempty(S[n])
            obs[n] = OrderedDict{String,Any}("type" => "unknown")
            push!(obs_eqs, n => _ser_expr(body))
            continue
        end
        exts = [maximum(b[j][2] for b in boxes[n]) for j in eachindex(S[n])]
        regions = Any[Any[Any[1, e] for e in exts]]
        values = Any[0.0]
        for b in boxes[n]
            push!(regions, Any[Any[lo, hi] for (lo, hi) in b])
            push!(values, OrderedDict{String,Any}(
                "op" => "aggregate", "args" => Any[],
                "output_idx" => Any[S[n]...],
                "ranges" => OrderedDict{String,Any}(
                    x => Any[b[j][1], b[j][2]] for (j, x) in enumerate(S[n])),
                "expr" => _ser_expr(body)))
        end
        obs[n] = OrderedDict{String,Any}(
            "type" => "unknown",
            "shape" => Any[axis_for(e) for e in exts])
        push!(obs_eqs, n => OrderedDict{String,Any}(
            "op" => "makearray", "args" => Any[],
            "regions" => regions, "values" => values))
    end
    return (obs = obs, obs_eqs = obs_eqs, axes = axes,
            coefs = ASTExpr[_deref(e, readmap) for e in rewritten])
end
