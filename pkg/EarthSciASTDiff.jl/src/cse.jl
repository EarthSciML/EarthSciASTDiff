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
# Free index names inside a hoisted fragment (band row indices, faq
# output indices) stay free in the template body and re-bind at the
# reference site after expansion — deliberate capture, since extraction is
# exact subtree factoring, not hygienic abstraction.

# ── interning ────────────────────────────────────────────────────────────────
#
# One bottom-up pass gives every structurally distinct subtree an integer id
# (the structural ids of hashcons.jl: a node is keyed by its operator and its
# children's ids, or by its shell when it has fields beyond `op`/`args`), in
# first-encounter post-order over the inputs, children in `map_children`
# order. So a child's id is always smaller than its parent's. Nodes are also
# memoized by identity: a subtree the inputs share in memory (band
# coefficients are interned, hashcons.jl) is walked once, however many
# parents reach it.

struct _CSEIndex
    ix::StructIndex
    reps::Vector{ASTExpr}           # representative subtree per id
    kids::Vector{Vector{Int}}       # child ids per id, in `map_children` order
    nnodes::Vector{Int}             # full subtree node count per id
    memo::IdDict{OpExpr, Int}       # node identity → id
end
_CSEIndex() = _CSEIndex(StructIndex(), ASTExpr[], Vector{Int}[], Int[], IdDict{OpExpr, Int}())

function _cse_id!(t::_CSEIndex, e::ASTExpr)::Int
    if e isa OpExpr
        cached = get(t.memo, e, 0)
        cached == 0 || return cached
        kids = Int[]
        map_children(c -> (push!(kids, _cse_id!(t, c)); c), e)
        id = _shell_id!(t.ix, e, kids)
        if id > length(t.reps)
            nn = 1
            for k in kids
                nn += t.nnodes[k]
            end
            push!(t.reps, e); push!(t.kids, kids); push!(t.nnodes, nn)
        end
        t.memo[e] = id
        return id
    end
    id = sid(t.ix, e)
    id > length(t.reps) && (push!(t.reps, e); push!(t.kids, Int[]); push!(t.nnodes, 1))
    return id
end

# ── extraction ───────────────────────────────────────────────────────────────

"""
    cse_templates(exprs; min_nodes = 12, prefix = "jt")
        -> (templates::OrderedDict{String,ASTExpr}, rewritten::Vector{ASTExpr})

Extract repeated subtrees of `exprs` by the rule of esm-jacobian-spec §4.1.
Every structurally distinct subtree is visited once, largest first (ties in
first-encounter post-order over the inputs). Its count is its number of
occurrences that remain once every larger extracted subtree is replaced by a
reference and kept once as a template body. A subtree that spans at least
`min_nodes` AST nodes and still counts at least 2 is extracted, named
`"\$(prefix)\$k"` in visiting order, and every occurrence (in the inputs and
in larger template bodies) is replaced by an `apply_expression_template`
reference. The result is a reference DAG, acyclic by strict containment, and
a body only references templates named after it. Θ(distinct subtrees ·
log) after one walk of the inputs (a subtree shared in memory is walked
once).
"""
function cse_templates(
        exprs::Vector{ASTExpr}; min_nodes::Int = 12,
        prefix::String = "jt"
    )
    t = _CSEIndex()
    roots = Int[_cse_id!(t, e) for e in exprs]
    n = length(t.reps)
    # Largest first; equal sizes cannot contain one another, so the tie order
    # only decides names, and first encounter (id order) settles it.
    order = sortperm(t.nnodes; rev = true, alg = Base.Sort.DEFAULT_STABLE)
    uses = zeros(Int, n)
    for r in roots
        uses[r] += 1
    end
    name = Vector{String}(undef, n)
    extracted = falses(n)
    names = String[]
    for id in order
        t.nnodes[id] == 1 && break            # leaves from here on
        r = t.reps[id]
        if uses[id] >= 2 && t.nnodes[id] >= min_nodes &&
                r isa OpExpr && r.op != "apply_expression_template"
            extracted[id] = true
            push!(names, "$(prefix)$(length(names) + 1)")
            name[id] = names[end]
        end
        # An extracted subtree's body is kept once; any other is inlined
        # wherever it occurs.
        w = extracted[id] ? 1 : uses[id]
        for k in t.kids[id]
            uses[k] += w
        end
    end
    isempty(names) && return OrderedDict{String, ASTExpr}(), ASTExpr[exprs...]
    # Rewrite bottom-up in id order (children first): `body[id]` is the
    # subtree with its children rewritten, `out[id]` what a parent holds.
    body = Vector{ASTExpr}(undef, n)
    out = Vector{ASTExpr}(undef, n)
    for id in 1:n
        r = t.reps[id]
        b = if isempty(t.kids[id])
            r
        else
            ks = t.kids[id]; i = Ref(0)
            map_children(_ -> (i[] += 1; out[ks[i[]]]), r)
        end
        body[id] = b
        out[id] = extracted[id] ?
            OpExpr("apply_expression_template", ASTExpr[]; name = name[id]) : b
    end
    templates = OrderedDict{String, ASTExpr}()
    for id in order
        extracted[id] && (templates[name[id]] = body[id])
    end
    return templates, ASTExpr[out[r] for r in roots]
end

"""
    expand_templates(e::ASTExpr, registry) -> ASTExpr

Inverse of [`cse_templates`](@ref): substitute every zero-parameter
`apply_expression_template` reference by its body from `registry`
(name → body `ASTExpr`), recursively (the registry is a DAG). Throws
`ArgumentError` on a reference to a name the registry does not contain.
"""
function expand_templates(e::ASTExpr, registry::AbstractDict{String, ASTExpr})::ASTExpr
    e isa OpExpr || return e
    if e.op == "apply_expression_template"
        e.name !== nothing && haskey(registry, e.name) || throw(
            ArgumentError(
                "apply_expression_template reference `$(e.name)` not in the block's " *
                    "expression_templates"
            )
        )
        return expand_templates(registry[e.name], registry)
    end
    return map_children(x -> expand_templates(x, registry), e)
end
