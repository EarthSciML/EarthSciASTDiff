# Scalar derivative rules — one rule per evaluable-core operator.
#
# This table is the draft conformance surface of esm-jacobian-spec.md §3:
# every rule here must match that spec section, and every language binding
# must implement the same table. Policy decisions (documented there, tested
# in scalar_rules_test.jl):
#
#   * `ifelse(c, a, b)`      → `ifelse(c, da, db)`; the condition carries no
#                              derivative (its Dirac term is dropped).
#   * `max`/`min` (n-ary)    → left-fold branch selection via `ifelse`; the
#                              a.e. derivative, ties resolve to the earlier
#                              argument. `max`/`min` are the spec's canonical
#                              limiter encoding, so this is the rule flux
#                              limiters differentiate through.
#   * `abs`                  → `ifelse(x < 0, -1, 1) · dx` (subgradient at 0).
#   * comparisons, logicals, `sign`/`floor`/`ceil` → 0. Comparisons return
#     1.0/0.0 VALUES and may appear multiplicatively; still 0.
#   * `interp.linear`        → slope of the active segment (0 outside the
#                              axis — extrapolate-flat), emitted as a segment
#                              `ifelse` over slopes precomputed from the
#                              literal `const` tables — no registry addition.
#   * `datetime.*`, `interp.searchsorted` → 0 (piecewise-constant / integer,
#                              matching the a.e.-zero contract in
#                              EarthSciAST/src/registered_functions.jl).
#   * rewrite-target ops (spatial `D`, `grad`, `table_lookup`, …) → error:
#     they have no evaluator and must be lowered before differentiation,
#     mirroring the tree-walk's `unlowered_operator` gate.

const ZERO_DERIV_OPS = Set([
    "<", "<=", ">", ">=", "==", "!=", "and", "or", "not",
    "sign", "floor", "ceil", "pi", "π", "e", "true", "false", "const", "enum",
    "Pre",
])

struct DerivativeRuleError <: Exception
    msg::String
end
Base.showerror(io::IO, e::DerivativeRuleError) = print(io, "DerivativeRuleError: ", e.msg)

"""
    dscalar(e::ASTExpr, s::Site) -> ASTExpr

Derivative of the scalar expression `e` with respect to site `s`, as a new
ESM expression. Structural rules only; a subtree in which `s` does not occur
differentiates to a literal zero without descent. Throws
[`DerivativeRuleError`](@ref) on an op with no rule (rewrite targets,
semiring aggregates in scalar position, unknown closed functions).
"""
dscalar(e::ASTExpr, s::Site)::ASTExpr = dscalar(e, s, IdDict{OpExpr,Bool}())

# `occ` is the occurrence memo threaded through ONE derivative scan (see
# `occurs`, expr_helpers.jl). The short-circuit below asks "does the site
# occur under here?" at every node on the way down, and the answer is a
# property of the node alone — so without a shared memo the scan re-walks
# each node's whole subtree once per ancestor, Θ(n·depth). On a flattened
# document, where one D-equation's RHS holds the operator stencil of every
# grid cell, that quadratic is the difference between seconds and never.
# Memoized it is Θ(n) per site, and the derivative walk itself is Θ(n).
function dscalar(e::ASTExpr, s::Site, occ::IdDict{OpExpr,Bool})::ASTExpr
    is_site(e, s) && return lit(1)
    e isa OpExpr || return lit(0)          # literal, or a different variable
    occurs(e, s, occ) || return lit(0)     # structural short-circuit
    o = e.op; a = e.args
    d(x) = dscalar(x, s, occ)
    if o == "+"
        return add((d(x) for x in a)...)
    elseif o == "-"
        return length(a) == 1 ? neg(d(a[1])) : add(d(a[1]), neg(d(a[2])))
    elseif o == "neg"
        return neg(d(a[1]))
    elseif o == "*"
        terms = ASTExpr[]
        for (i, x) in enumerate(a)
            dx = d(x); iszero_lit(dx) && continue
            push!(terms, mul(dx, (a[j] for j in eachindex(a) if j != i)...))
        end
        return add(terms...)
    elseif o == "/"
        n, m = a[1], a[2]; dn, dm = d(n), d(m)
        t1 = iszero_lit(dn) ? lit(0) : op("/", dn, m)
        t2 = iszero_lit(dm) ? lit(0) : neg(op("/", mul(n, dm), op("^", m, lit(2))))
        return add(t1, t2)
    elseif o == "^" || o == "pow"
        b, x = a[1], a[2]; db, dx = d(b), d(x)
        t1 = iszero_lit(db) ? lit(0) : mul(x, op("^", b, add(x, lit(-1))), db)
        t2 = iszero_lit(dx) ? lit(0) : mul(op("^", b, x), op("log", b), dx)
        return add(t1, t2)
    elseif o == "exp";   return mul(e, d(a[1]))
    elseif o == "log";   return op("/", d(a[1]), a[1])
    elseif o == "log10"; return op("/", d(a[1]), mul(a[1], op("log", lit(10))))
    elseif o == "sqrt";  return op("/", d(a[1]), mul(lit(2), e))
    elseif o == "sin";   return mul(op("cos", a[1]), d(a[1]))
    elseif o == "cos";   return neg(mul(op("sin", a[1]), d(a[1])))
    elseif o == "tan";   return mul(add(lit(1), op("^", e, lit(2))), d(a[1]))
    elseif o == "tanh";  return mul(add(lit(1), neg(op("^", e, lit(2)))), d(a[1]))
    elseif o == "sinh";  return mul(op("cosh", a[1]), d(a[1]))
    elseif o == "cosh";  return mul(op("sinh", a[1]), d(a[1]))
    elseif o == "asin";  return op("/", d(a[1]), op("sqrt", add(lit(1), neg(op("^", a[1], lit(2))))))
    elseif o == "acos";  return neg(op("/", d(a[1]), op("sqrt", add(lit(1), neg(op("^", a[1], lit(2)))))))
    elseif o == "asinh"; return op("/", d(a[1]), op("sqrt", add(op("^", a[1], lit(2)), lit(1))))
    elseif o == "acosh"; return op("/", d(a[1]), op("sqrt", add(op("^", a[1], lit(2)), lit(-1))))
    elseif o == "atanh"; return op("/", d(a[1]), add(lit(1), neg(op("^", a[1], lit(2)))))
    elseif o == "atan" && length(a) == 1
        return op("/", d(a[1]), add(lit(1), op("^", a[1], lit(2))))
    elseif o == "atan" || o == "atan2"   # atan(y, x)
        y, x = a[1], a[2]
        den = add(op("^", x, lit(2)), op("^", y, lit(2)))
        return op("/", add(mul(x, d(y)), neg(mul(y, d(x)))), den)
    elseif o == "abs"
        return mul(op("ifelse", op("<", a[1], lit(0)), lit(-1), lit(1)), d(a[1]))
    elseif o == "ifelse"
        return op("ifelse", a[1], d(a[2]), d(a[3]))
    elseif o == "max" || o == "min"
        # a.e. derivative of the n-ary extremum: select the argext branch.
        # Left fold: max(x1, x2, x3) = max(max(x1, x2), x3); ties keep the
        # earlier argument's derivative (`>=` / `<=`).
        acc, dacc = a[1], d(a[1])
        cmp = o == "max" ? ">=" : "<="
        for x in a[2:end]
            dacc = op("ifelse", op(cmp, acc, x), dacc, d(x))
            acc = op(o, acc, x)
        end
        return dacc
    elseif o in ZERO_DERIV_OPS
        return lit(0)
    elseif o == "fn"
        return dfn(e, s, occ)
    elseif o == "index"
        # A site-bearing `index` that is not itself the site: index of a
        # non-variable array expression, or index expressions that depend on
        # the site (neither occurs in a lowered model RHS).
        throw(DerivativeRuleError(
            "`index` of a non-variable array expression (or a site-dependent " *
            "index) has no scalar rule: $(skey(e))"))
    else
        throw(DerivativeRuleError("no derivative rule for op `$o` " *
            "(rewrite-target ops must be lowered before differentiation)"))
    end
end

# The numeric vector of a literal `const` table node, or nothing.
function _const_vector(e::ASTExpr)
    e isa OpExpr && e.op == "const" || return nothing
    v = e.value
    v isa AbstractVector && all(x -> x isa Real, v) || return nothing
    return Float64[Float64(x) for x in v]
end

# The numeric matrix of a literal 2-D `const` table node (`value[i][j]` →
# `T[i, j]`), or nothing.
function _const_matrix(e::ASTExpr)
    e isa OpExpr && e.op == "const" || return nothing
    v = e.value
    v isa AbstractVector || return nothing
    rows = Vector{Vector{Float64}}()
    for r in v
        (r isa AbstractVector && all(x -> x isa Real, r)) || return nothing
        push!(rows, Float64[Float64(x) for x in r])
    end
    (isempty(rows) || any(length(r) != length(rows[1]) for r in rows)) &&
        return nothing
    return permutedims(reduce(hcat, rows))
end

# ∂(bilinear blend)/∂(the axis of `xv`), as a nested segment `ifelse`:
# outer chain locates the x-cell `i` (flat-zero outside `xv` — the clamp
# makes the blend constant in x there); each leaf is an inner chain over the
# y-cells blending the two x-slope rows with the CLAMPED y-weight
# (`(y − y_j)/Δy_j` per cell, the edge rows outside `yv`). The knot
# convention matches spec §9.2 step 2 (a query exactly on an interior knot
# selects the cell whose weight is 0). ∂/∂y is this function on the
# transposed table with the axes and query expressions swapped.
function _bilinear_partial(T::Matrix{Float64}, xv::Vector{Float64},
                           yv::Vector{Float64}, xe::ASTExpr, ye::ASTExpr)::ASTExpr
    nx, ny = length(xv), length(yv)
    out = lit(0)                              # x ≥ axis_x[nx]: flat in x
    for i in (nx-1):-1:1
        sl(j) = (T[i+1, j] - T[i, j]) / (xv[i+1] - xv[i])
        leaf = lit(sl(ny))                    # y ≥ axis_y[ny]: top edge row
        for j in (ny-1):-1:1
            wy = op("/", add(ye, lit(-yv[j])), lit(yv[j+1] - yv[j]))
            e = add(lit(sl(j)), mul(wy, add(lit(sl(j + 1)), neg(lit(sl(j))))))
            leaf = op("ifelse", op("<", ye, lit(yv[j+1])), e, leaf)
        end
        leaf = op("ifelse", op("<", ye, lit(yv[1])), lit(sl(1)), leaf)  # bottom edge
        out = op("ifelse", op("<", xe, lit(xv[i+1])), leaf, out)
    end
    return op("ifelse", op("<", xe, lit(xv[1])), lit(0), out)
end

# Closed-function derivative table (esm-spec §9.2: exactly 12 names, all
# decided here — an unknown name is a spec violation, not a fallthrough).
dfn(e::OpExpr, s::Site)::ASTExpr = dfn(e, s, IdDict{OpExpr,Bool}())

function dfn(e::OpExpr, s::Site, occ::IdDict{OpExpr,Bool})::ASTExpr
    name = e.name
    a = e.args
    if startswith(name, "datetime.") || name == "interp.searchsorted"
        return lit(0)
    elseif name == "interp.linear"
        # interp.linear(table, axis, x) — spec §9.2: TABLE first, AXIS second,
        # extrapolate-flat boundaries, and both arrays MUST be literal
        # `const` nodes (interp_table_not_const otherwise). d/dx is the
        # active segment's slope, 0 outside the axis (flat extrapolation),
        # emitted as a nested segment `ifelse` over slopes precomputed here
        # (esm-jacobian-spec.md §3.4).
        table, axis, x = a[1], a[2], a[3]
        dx = dscalar(x, s, occ); iszero_lit(dx) && return lit(0)
        yv = _const_vector(table); xv = _const_vector(axis)
        (yv === nothing || xv === nothing) && throw(DerivativeRuleError(
            "interp.linear table/axis must be literal `const` arrays " *
            "(spec §9.2 interp_table_not_const)"))
        n = length(xv)
        (n == length(yv) && n >= 2) || throw(DerivativeRuleError(
            "interp.linear table/axis length mismatch or < 2 knots"))
        slopes = [(yv[k+1] - yv[k]) / (xv[k+1] - xv[k]) for k in 1:(n-1)]
        slope = lit(0)                                # x ≥ axis[N]: flat
        for k in (n-1):-1:1
            slope = op("ifelse", op("<", x, lit(xv[k+1])), lit(slopes[k]), slope)
        end
        slope = op("ifelse", op("<", x, lit(xv[1])), lit(0), slope)  # flat below
        return mul(slope, dx)
    elseif name == "interp.bilinear"
        # interp.bilinear(table, axis_x, axis_y, x, y) — spec §9.2: row-major
        # literal table (`table[i][j]` at `(axis_x[i], axis_y[j])`), per-axis
        # clamp (extrapolate-flat). On cell (i, j), ∂/∂x is the x-slope of
        # the y-blend — constant in x, linear in the CLAMPED y-weight — and
        # symmetrically for ∂/∂y. Each partial is a nested segment `ifelse`
        # over the partial's own axis whose leaves are inner chains over the
        # other axis (esm-jacobian-spec.md §3.4); constants precomputed here.
        table, ax, ay, x, y = a[1], a[2], a[3], a[4], a[5]
        dx = dscalar(x, s, occ); dy = dscalar(y, s, occ)
        iszero_lit(dx) && iszero_lit(dy) && return lit(0)
        T = _const_matrix(table); xv = _const_vector(ax); yv = _const_vector(ay)
        (T === nothing || xv === nothing || yv === nothing) && throw(DerivativeRuleError(
            "interp.bilinear table/axes must be literal `const` arrays " *
            "(spec §9.2 interp_table_not_const)"))
        nx, ny = length(xv), length(yv)
        (size(T) == (nx, ny) && nx >= 2 && ny >= 2) || throw(DerivativeRuleError(
            "interp.bilinear table/axis shape mismatch or < 2 knots"))
        terms = ASTExpr[]
        iszero_lit(dx) || push!(terms, mul(_bilinear_partial(T, xv, yv, x, y), dx))
        iszero_lit(dy) ||
            push!(terms, mul(_bilinear_partial(permutedims(T), yv, xv, y, x), dy))
        return add(terms...)
    else
        throw(DerivativeRuleError("unknown closed function `$name`"))
    end
end
