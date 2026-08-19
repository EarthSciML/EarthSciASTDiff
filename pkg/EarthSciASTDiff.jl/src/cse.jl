# Hash-consed common-subexpression extraction over emitted band coefficients
# — the template half of the coefficient-size control roadmap item.
#
# Repeated coefficient subtrees (across bands, and repeated occurrences
# within one band) are hoisted into zero-parameter, match-less
# `expression_templates` entries carried INSIDE the `jacobians` block
# (esm-jacobian-spec.md §4); each occurrence is replaced by an
# `apply_expression_template` reference. This reuses the format's single
# structural-substitution mechanism (esm-spec §9.6) rather than inventing a
# new sharing construct: a zero-param template is a fixed named fragment,
# references may form a DAG (template body referencing an earlier template),
# and expansion is pure syntactic substitution — so
# [`expand_templates`](@ref) reproduces the original trees exactly.
#
# Free index names inside a hoisted fragment (band row indices, aggregate
# output indices) stay free in the template body and re-bind at the
# reference site after expansion — deliberate capture, since extraction is
# exact subtree factoring, not hygienic abstraction.

# ── hash-consing ─────────────────────────────────────────────────────────────
#
# Bottom-up interning: each structurally distinct subtree gets an integer id.
# A node's key is its serialized form with every child replaced by a
# placeholder carrying the child's id — O(node) per node instead of the
# quadratic full-subtree serialization.

struct _Intern
    ids::Dict{String,Int}    # shell key → id
    reps::Vector{ASTExpr}    # representative subtree per id
    nnodes::Vector{Int}      # full subtree node count per id
    counts::Vector{Int}      # total occurrences over everything interned
end
_Intern() = _Intern(Dict{String,Int}(), ASTExpr[], Int[], Int[])

_placeholder(id::Int) = VarExpr("__cse⟦$id⟧")

# (shell key, node count) of `e` given already-interned children.
function _shell_key(e::ASTExpr, childids::Vector{Int}, it::_Intern)
    if e isa OpExpr
        i = Ref(0)
        shell = map_children(_ -> (i[] += 1; _placeholder(childids[i[]])), e)
        nn = 1
        for cid in childids
            nn += it.nnodes[cid]
        end
        return stable_json(EarthSciAST.serialize_expression(shell)), nn
    end
    return stable_json(EarthSciAST.serialize_expression(e)), 1
end

function _intern!(it::_Intern, e::ASTExpr)::Int
    childids = Int[]
    e isa OpExpr && map_children(c -> (push!(childids, _intern!(it, c)); c), e)
    key, nn = _shell_key(e, childids, it)
    id = get!(it.ids, key) do
        push!(it.reps, e); push!(it.nnodes, nn); push!(it.counts, 0)
        length(it.reps)
    end
    it.counts[id] += 1
    return id
end

# Rewrite: replace every subtree whose id is `target` by `repl`, recomputing
# original ids bottom-up against the SAME (read-only) table. Returns
# (rewritten node, ORIGINAL id of `e`).
function _replace_id(e::ASTExpr, it::_Intern, target::Int, repl::ASTExpr)
    childids = Int[]; newchildren = ASTExpr[]
    if e isa OpExpr
        map_children(e) do c
            nc, cid = _replace_id(c, it, target, repl)
            push!(childids, cid); push!(newchildren, nc)
            c
        end
    end
    key, _ = _shell_key(e, childids, it)
    id = it.ids[key]
    id == target && return (repl, id)
    if e isa OpExpr && !isempty(newchildren)
        i = Ref(0)
        e = map_children(_ -> (i[] += 1; newchildren[i[]]), e)
    end
    return (e, id)
end

# ── extraction ───────────────────────────────────────────────────────────────

"""
    cse_templates(exprs; min_nodes = 12, prefix = "jt")
        -> (templates::OrderedDict{String,ASTExpr}, rewritten::Vector{ASTExpr})

Greedy largest-first extraction of repeated subtrees over `exprs`. Each
round interns the whole working set (input expressions plus the bodies of
templates already extracted), picks the largest subtree that occurs at least
twice and spans at least `min_nodes` AST nodes, names it `"\$(prefix)\$k"`,
and replaces every occurrence — in the inputs AND in earlier template
bodies — with an `apply_expression_template` reference. Terminates when no
candidate remains; the result is a reference DAG (acyclic by strict
containment). Deterministic: ties between equally-sized candidates break by
interning (first-encounter) order over the given expression order.
"""
function cse_templates(exprs::Vector{ASTExpr}; min_nodes::Int = 12,
                       prefix::String = "jt")
    work = ASTExpr[exprs...]
    nin = length(work)
    names = String[]
    while true
        it = _Intern()
        for e in work
            _intern!(it, e)
        end
        best = 0
        for id in 1:length(it.reps)
            it.counts[id] >= 2 || continue
            it.nnodes[id] >= min_nodes || continue
            r = it.reps[id]
            r isa OpExpr || continue
            r.op == "apply_expression_template" && continue
            (best == 0 || it.nnodes[id] > it.nnodes[best]) && (best = id)
        end
        best == 0 && break
        name = "$(prefix)$(length(names) + 1)"
        push!(names, name)
        repl = OpExpr("apply_expression_template", ASTExpr[]; name = name)
        body = it.reps[best]
        work = ASTExpr[_replace_id(e, it, best, repl)[1] for e in work]
        push!(work, body)   # the body joins the pool: later rounds may factor it
    end
    templates = OrderedDict{String,ASTExpr}(
        n => work[nin + k] for (k, n) in enumerate(names))
    return templates, work[1:nin]
end

"""
    expand_templates(e::ASTExpr, registry) -> ASTExpr

Inverse of [`cse_templates`](@ref): substitute every zero-parameter
`apply_expression_template` reference by its body from `registry`
(name → body `ASTExpr`), recursively (the registry is a DAG). Throws
`ArgumentError` on a reference to a name the registry does not contain.
"""
function expand_templates(e::ASTExpr, registry::AbstractDict{String,ASTExpr})::ASTExpr
    e isa OpExpr || return e
    if e.op == "apply_expression_template"
        e.name !== nothing && haskey(registry, e.name) || throw(ArgumentError(
            "apply_expression_template reference `$(e.name)` not in the block's " *
            "expression_templates"))
        return expand_templates(registry[e.name], registry)
    end
    return map_children(x -> expand_templates(x, registry), e)
end
