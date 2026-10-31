# Derivatives through a faq node's WINDOW and REDUCTION (esm-jacobian-spec.md
# §3.5): filtered faq nodes, the `max` / `min` / `*` reductions beside `+`,
# and reads whose index is a value (a level search used as a subscript).
#
# A filter is a gate whose Dirac term is dropped, exactly like an `ifelse`
# condition (§3.2), so the derivative of a filtered reduction is a `+` over
# the same window of each admitted term's derivative, weighted by what the
# reduction does with that term:
#
#   +        1
#   max/min  the indicator of the FIRST extremal term in ascending order
#            (the left fold's tie rule), built from two max/min faq nodes
#   *        the product of every OTHER admitted term (never a division)
#
# An index-only filter conjunct is decided exactly — by splitting the band's
# box at its affine breakpoints, and by the band's `where` guard for what is
# left (a triangular window); a value-dependent conjunct stays in the
# coefficient as `ifelse(φ, d, 0)` and its pattern is the whole window.

# The ⊕ of faq node `e` ("+", "*", "max" or "min"); a `semiring` supersedes
# `reduce`, and its ⊗ is already written out in the body. A join or a
# value-invention node has no rule. `E` is the caller's error type.
function _reduce_op(e::OpExpr, E = BandError)::String
    e.join === nothing || throw(E("joined faq nodes are not differentiated"))
    (e.distinct === true || e.key !== nothing) && throw(
        E("value-invention faq nodes (`distinct`, `key`) are not differentiated")
    )
    if e.semiring === nothing
        r = something(e.reduce, "+")
        r in ("+", "*", "max", "min") && return r
    elseif e.semiring == "sum_product"
        return "+"
    elseif e.semiring in ("max_product", "max_sum")
        return "max"
    elseif e.semiring == "min_sum"
        return "min"
    end
    throw(E("$(_reduction(e)) has no derivative rule"))
end

_conjuncts(g::ASTExpr) = g isa OpExpr && g.op == "and" ?
    ASTExpr[c for x in g.args for c in _conjuncts(x)] : ASTExpr[g]
_conj(xs) = isempty(xs) ? nothing : length(xs) == 1 ? xs[1] : op("and", xs...)
_conj(a::Union{Nothing, ASTExpr}, b::Union{Nothing, ASTExpr}) =
    a === nothing ? b : b === nothing ? a : _conj(vcat(_conjuncts(a), _conjuncts(b)))

# Index-only: reads no name but the given index names (literals and `const`
# gathers carry none), so it can be decided at assembly, point by point.
_index_only(g::ASTExpr, names) = issubset(free_variables(g), names)

# `base`, then `suffix` appended until clear of `taken`, which it joins.
function _fresh!(taken::Set{String}, base::String, suffix::String)::String
    f = base
    while f in taken
        f *= suffix
    end
    push!(taken, f)
    return f
end

"""
    _window_weight(red, e, cn, d, E) -> ASTExpr

The derivative `d` of one term of the reduction `red` of faq node `e` over
its contracted names `cn`, weighted by what the reduction does with that
term (see the file header). The max/min indicator and the exclusion product
read the node's whole window — its own ranges and filter — through copies of
the body with the contracted names renamed apart (`<name>_p`, `_p` repeated
until clear of every name in `e`), so no node they add binds a name the
term's own contraction binds. A one-point window (no contracted name)
weighs 1. A max/min over more than one contracted name has no single
ascending order, so it is refused.
"""
function _window_weight(red::String, e::OpExpr, cn::Vector{String}, d::ASTExpr, E)
    (red == "+" || isempty(cn) || iszero_lit(d)) && return d
    rng = something(e.ranges, Dict{String, Any}())
    red == "*" || length(cn) == 1 || throw(
        E(
            "a `$red` reduction over $(length(cn)) contracted indices " *
                "($(join(cn, ", "))) has no first-extremum order"
        )
    )
    taken = _all_names!(Set{String}(), e)
    ren = Dict{String, ASTExpr}(n => VarExpr(_fresh!(taken, n * "_p", "_p")) for n in cn)
    b′ = _subst_free(e.expr_body, ren)
    φ′ = e.filter === nothing ? nothing : _subst_free(e.filter, ren)
    r′ = Dict{String, Any}(ren[n].name => rng[n] for n in cn)
    if red == "*"
        ne = ASTExpr[op("!=", ren[n], VarExpr(n)) for n in cn]
        others = OpExpr(
            "faq", ASTExpr[]; output_idx = Any[], reduce = "*", ranges = r′,
            filter = _conj(φ′, length(ne) == 1 ? ne[1] : op("or", ne...)),
            expr_body = b′
        )
        return mul(d, others)
    end
    n = cn[1]
    m = OpExpr(
        "faq", ASTExpr[]; output_idx = Any[], reduce = red, ranges = r′,
        filter = φ′, expr_body = b′
    )
    m_before = OpExpr(
        "faq", ASTExpr[]; output_idx = Any[], reduce = red, ranges = r′,
        filter = _conj(φ′, op("<", ren[n], VarExpr(n))), expr_body = b′
    )
    ge, gt = red == "max" ? (">=", ">") : ("<=", "<")
    first_ext = op("and", op(ge, e.expr_body, m), op(gt, e.expr_body, m_before))
    return op("ifelse", first_ext, d, lit(0))
end

const _LIT_CMP = Dict{String, Function}(
    "<" => <, "<=" => <=, ">" => >, ">=" => >=, "==" => ==, "!=" => !=,
)

# The value of a literal, or of `+` / `-` / `*` over literals; else `nothing`.
function _lit_value(e::ASTExpr)
    (e isa IntExpr || e isa NumExpr) && return e.value
    (e isa OpExpr && e.op in ("+", "-", "*") && !isempty(e.args)) || return nothing
    xs = Any[]
    for x in e.args
        t = _lit_value(x)
        t === nothing && return nothing
        push!(xs, t)
    end
    e.op == "+" && return sum(xs)
    e.op == "*" && return prod(xs)
    return length(xs) == 1 ? -xs[1] : length(xs) == 2 ? xs[1] - xs[2] : nothing
end

# Truth of a decided predicate: `true` / `false`, or the residual expression.
function _bool_fold(g::ASTExpr)
    (g isa IntExpr || g isa NumExpr) && return g.value != 0
    g isa OpExpr || return g
    if haskey(_LIT_CMP, g.op) && length(g.args) == 2
        # A comparison of literals — what pinning one-value names leaves of
        # `k != i` (`_decide` folds only the affine `<`/`<=`/`>`/`>=`).
        a = _lit_value(g.args[1]); b = _lit_value(g.args[2])
        (a === nothing || b === nothing) || return _LIT_CMP[g.op](a, b)
    end
    if g.op == "and" || g.op == "or"
        absorb = g.op == "or"                     # the value that decides it
        rest = ASTExpr[]
        for x in g.args
            t = _bool_fold(x)
            t === absorb && return absorb
            t === !absorb && continue
            push!(rest, t)
        end
        isempty(rest) && return !absorb
        return length(rest) == 1 ? rest[1] : op(g.op, rest...)
    elseif g.op == "not" && length(g.args) == 1
        t = _bool_fold(g.args[1])
        return t isa Bool ? !t : op("not", t)
    elseif g.op == "true"
        return true
    elseif g.op == "false"
        return false
    end
    return g
end

"""
    _split_guard(g, names, box) -> Vector{(box′, guard′)}

Decide the index-only predicate `g` over the box `names[k] ∈ box[k]` as far
as the box can: pin one-value names, split the box at the predicate's affine
breakpoints (the region clipper's), and decide each cell — dropped where `g`
is false throughout, unguarded (`nothing`) where it is true throughout, and
guarded by the undecided residual otherwise (a window no box can express,
such as `gk < gke`). Cells are in the clipper's ascending order.
"""
function _split_guard(g::ASTExpr, names::Vector{String}, box::Vector{Tuple{Int, Int}})
    pins = Dict{String, ASTExpr}(
        n => lit(lo) for (n, (lo, hi)) in zip(names, box) if lo == hi
    )
    isempty(pins) || (g = substitute(g, pins))
    out = Tuple{Vector{Tuple{Int, Int}}, Union{Nothing, ASTExpr}}[]
    for (cell, g′) in clip_regions(g, names, box)
        t = _bool_fold(
            _decide(
                g′, Dict{String, Tuple{Int, Int}}(
                    n => cell[k] for (k, n) in enumerate(names)
                )
            )
        )
        t === false && continue
        push!(out, (cell, t === true ? nothing : t))
    end
    return out
end

"""
    _one_hot(cidx, coef, contracted, ridx, v, ctx) -> (cidx, coef, contracted)

A column index that reads a value — `x[κ(u) + d]` with `κ` a level search —
is integer-valued, hence piecewise constant (the §3.2 `floor` rule): its own
derivative is 0, and the read is the one-hot sum `Σ_k [k == κ(u)+d] x[k]`.
Each such dimension becomes a contracted name (`_oh<dim>`, `_j` appended
until clear) over the static extent of that axis of `v`, and the selection
moves into the coefficient. The pattern is the structural union: every cell
of the axis. A column reading only the band's own index names is unchanged.
"""
function _one_hot(cidx::Vector{ASTExpr}, coef::ASTExpr, contracted, ridx, v::String, ctx::Ctx)
    bound = Set{String}(r for r in ridx if r isa String)
    for (n, _, _) in contracted
        push!(bound, n)
    end
    hot = [k for (k, c) in enumerate(cidx) if !_index_only(c, bound)]
    isempty(hot) && return cidx, coef, contracted
    shape = get(ctx.shapes, v, Int[])
    taken = _all_names!(copy(bound), coef)
    foreach(c -> _all_names!(taken, c), cidx)
    cidx = copy(cidx); contracted = collect(Tuple{String, Int, Int}, contracted)
    for k in hot
        k <= length(shape) || throw(
            BandError(
                "column dimension $k of `$v` reads a value, but `$v` has no static " *
                    "extent along it"
            )
        )
        f = _fresh!(taken, "_oh$k", "_j")
        coef = op("ifelse", op("==", VarExpr(f), cidx[k]), coef, lit(0))
        cidx[k] = VarExpr(f)
        push!(contracted, (f, 1, shape[k]))
    end
    return cidx, coef, contracted
end
