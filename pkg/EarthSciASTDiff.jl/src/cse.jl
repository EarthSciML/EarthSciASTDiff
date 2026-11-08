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
# Every structurally distinct subtree gets a dense CSE id (1, 2, …) in
# first-encounter post-order over the inputs, a node's CHILD SLOTS in the
# order esm-jacobian-spec §4.1 fixes (`_cse_children`), so a child's id is
# always smaller than its parent's and every binding numbers alike. Equality
# is the structural id of hashcons.jl ([`sid`](@ref), whose identity memo
# also means a subtree the inputs share in memory is keyed once); the CSE id
# is a dense renumbering of the structural ids in that walk order. A subtree
# already numbered is not walked again, so the walk visits each distinct
# subtree once.

struct _CSEIndex
    ix::StructIndex
    dense::Vector{Int}              # structural id → CSE id (0: not yet)
    reps::Vector{ASTExpr}           # representative subtree per CSE id
    kids::Vector{Vector{Int}}       # child CSE ids per CSE id, in slot order
    nnodes::Vector{Int}             # full subtree node count per CSE id
end
_CSEIndex() = _CSEIndex(StructIndex(), Int[], ASTExpr[], Vector{Int}[], Int[])

# The field `s` of `e`, or `nothing` when this EarthSciAST's `OpExpr` has no
# such field (`hasfield` on the type folds to a constant).
_cse_field(e::OpExpr, s::Symbol) = hasfield(OpExpr, s) ? getfield(e, s) : nothing

"""
    _cse_children(e::OpExpr) -> Vector{ASTExpr}

The child slots of `e` in the order of esm-jacobian-spec §4.1: `args`,
`lower`, `upper`, `expr`, `filter`, `values`, `axes` by ascending key, `key`,
`bindings` by ascending key. Expressions inside `ranges` are part of the node,
not slots. Throws if `map_children` reaches a child this list misses (a field
a newer EarthSciAST added), rather than silently numbering it differently
from the other bindings.
"""
function _cse_children(e::OpExpr)::Vector{ASTExpr}
    cs = Vector{ASTExpr}(e.args)
    for f in (:lower, :upper, :expr_body, :filter)
        c = _cse_field(e, f)
        c === nothing || push!(cs, c)
    end
    v = _cse_field(e, :values)
    v === nothing || append!(cs, v)
    ax = _cse_field(e, :table_axes)
    if ax !== nothing
        for name in sort!(collect(keys(ax)))
            push!(cs, ax[name])
        end
    end
    kx = _cse_field(e, :key)
    kx === nothing || push!(cs, kx)
    b = _cse_field(e, :bindings)
    if b !== nothing
        for name in sort!(collect(keys(b)))
            push!(cs, b[name])
        end
    end
    nnative = length(_native_children(e))
    rg = _cse_field(e, :ranges)
    nrange = 0
    if rg !== nothing
        for r in values(rg)
            r isa AbstractVector && (nrange += count(x -> x isa ASTExpr, r))
        end
    end
    nnative == length(cs) + nrange || throw(
        ArgumentError(
            "cse_templates: `$(e.op)` node has an expression child outside the " *
                "child slots of esm-jacobian-spec §4.1"
        )
    )
    return cs
end

function _cse_fresh!(t::_CSEIndex, s::Int, e::ASTExpr, kids::Vector{Int})::Int
    nn = 1
    for k in kids
        nn += t.nnodes[k]
    end
    push!(t.reps, e); push!(t.kids, kids); push!(t.nnodes, nn)
    length(t.dense) < s && append!(t.dense, zeros(Int, s - length(t.dense)))
    t.dense[s] = length(t.reps)
    return length(t.reps)
end

_cse_dense(t::_CSEIndex, s::Int) = s <= length(t.dense) ? t.dense[s] : 0

# CSE id of `e`, numbering every subtree not seen before. Iterative: a deep
# chain costs no stack.
function _cse_id!(t::_CSEIndex, e::ASTExpr)::Int
    s = sid(t.ix, e)                        # keys the whole tree (iterative)
    d = _cse_dense(t, s)
    d == 0 || return d
    e isa OpExpr || return _cse_fresh!(t, s, e, Int[])
    # frame: (node, its structural id, child slots, child CSE ids so far)
    stack = Tuple{OpExpr, Int, Vector{ASTExpr}, Vector{Int}}[(e, s, _cse_children(e), Int[])]
    id = 0
    while !isempty(stack)
        n, sn, cs, kids = stack[end]
        if length(kids) < length(cs)
            c = cs[length(kids) + 1]
            sc = sid(t.ix, c)               # memo hit / leaf lookup
            dc = _cse_dense(t, sc)
            if dc != 0
                push!(kids, dc)
            elseif c isa OpExpr
                push!(stack, (c, sc, _cse_children(c), Int[]))
            else
                push!(kids, _cse_fresh!(t, sc, c, Int[]))
            end
            continue
        end
        pop!(stack)
        id = _cse_fresh!(t, sn, n, kids)
        isempty(stack) || push!(stack[end][4], id)
    end
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
        # Nothing from here on is large enough to extract, and a count only
        # decides extraction, so the rest need no counts.
        t.nnodes[id] < min_nodes && break
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
        b = isempty(t.kids[id]) ? r : _cse_rebuild(r, t.kids[id], out)
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

# `r` with each child slot replaced by `out` of its CSE id. `map_children`
# visits fields in its own order, so a child is matched to its slot by
# identity; a child that is no slot (inside `ranges`) is kept. Identity-
# preserving: an unchanged node is returned as is.
function _cse_rebuild(r::OpExpr, ks::Vector{Int}, out::Vector{ASTExpr})::ASTExpr
    cs = _cse_children(r)
    slot = IdDict{Any, Int}()
    for (j, c) in enumerate(cs)
        get!(slot, c, j)
    end
    return map_children(c -> (j = get(slot, c, 0); j == 0 ? c : out[ks[j]]), r)
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
