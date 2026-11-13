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
#   * `Pre(a)`               → `d[a]`: in a right-hand side `Pre` is the
#                              identity, so its derivative is its operand's.
#                              `inline_observed` lowers it away before
#                              differentiation; this rule covers an
#                              expression that reaches `dscalar` unlowered.
#   * `index(A, i…)` that is not the site itself → 0 when `A` is a
#                              literal array (it reads no variable): the
#                              subscripts are integer-valued, so the read is
#                              piecewise-constant in them. This is the rule
#                              the esm:std interpolation and calendar
#                              templates differentiate through (esm 2.0.0
#                              removed the `fn` closed functions they
#                              replace); the explicit `x` in a template's
#                              blend carries the slope.
#   * rewrite-target ops (spatial `D`, `grad`, …) → error: they have no
#     evaluator and must be lowered before differentiation, mirroring the
#     tree-walk's `unlowered_operator` gate. The ops esm 2.0.0 removed
#     (`fn`, `table_lookup`) never load, and have no rule either.

const ZERO_DERIV_OPS = Set(
    [
        "<", "<=", ">", ">=", "==", "!=", "and", "or", "not",
        "sign", "floor", "ceil", "pi", "π", "e", "true", "false", "const", "enum",
    ]
)

"""
    DerivativeRuleError(msg)

A scalar operator has no derivative rule (spec §3).
"""
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
joined or value-invention `faq` nodes in scalar position, a dynamic read
of the site's own variable).
"""
dscalar(e::ASTExpr, s::Site)::ASTExpr = dscalar(e, s, _SiteScan(s))

# A faq node's reduction, for messages (`_reduce_op` decides it, reductions.jl).
_reduction(e::OpExpr)::String = e.semiring === nothing ? "reduce=`$(e.reduce)`" :
    "semiring=`$(e.semiring)`"

# A scalar `faq` sum Σ_k body — the form a CONTRACTING observed inlines to
# (inline.jl `_beta_faq`). The sum is linear, so its derivative is the sum of
# the body's, split by what the site's column reads:
#   * no bound index (`w` in Σ_k w·u[k]): the same sum over the body
#     derivative, a closed scalar;
#   * a bound index (`u[i,k]` in Σ_k u[i,k]): each value of k reads a
#     DIFFERENT column, so that index is not summed here but left FREE, and
#     the band emitter turns it into a contracted column dimension (§4,
#     `_lift_nested` in bands.jl). Bound indices the column does not read
#     are still summed.
# A `max`, `min` or `*` reduction differentiates the same way once each
# term's derivative is weighted by what the reduction does with it
# (`_window_weight`, reductions.jl), and the result is a `+` sum. A filter
# keeps its window — the remaining sum's filter, or an `ifelse` once every
# bound index is left free — and its own Dirac term is dropped. A join or a
# value-invention node has no rule (spec §3.5).
function _dsum(e::OpExpr, s::Site, occ::_Occ)::ASTExpr
    red = _reduce_op(e, DerivativeRuleError)
    db = dscalar(e.expr_body, s, occ)
    iszero_lit(db) && return lit(0)
    ranges = e.ranges === nothing ? Dict{String, Any}() : e.ranges
    db = _window_weight(red, e, sort!(String.(collect(keys(ranges)))), db, DerivativeRuleError)
    # A rebuilt `+` node: the reduction is now a plain sum of weighted terms.
    plus(kw...) = red == "+" ? EarthSciAST.reconstruct(e; expr_body = db, kw...) :
        EarthSciAST.reconstruct(e; expr_body = db, reduce = "+", semiring = nothing, kw...)
    read = Set{String}()
    if s.expr isa OpExpr
        for x in s.expr.args[2:end]
            union!(read, free_variables(x))
        end
    end
    summed = Dict{String, Any}(n => r for (n, r) in ranges if !(n in read))
    length(summed) == length(ranges) && return plus()
    isempty(summed) && return e.filter === nothing ? db : op("ifelse", e.filter, db, lit(0))
    return plus(ranges = summed)
end

# `occ` is the occurrence memo threaded through ONE derivative scan (see
# `occurs`, expr_helpers.jl). The short-circuit below asks "does the site
# occur under here?" at every node on the way down, and the answer is a
# property of the node alone — so without a shared memo the scan re-walks
# each node's whole subtree once per ancestor, Θ(n·depth). On the 4×10⁵-node
# RHS a lowered transport operator produces, that quadratic was half of
# `jacobian_bands`' entire runtime (measured). Memoized it is Θ(n) per site,
# and the derivative walk itself is Θ(n).
function dscalar(e::ASTExpr, s::Site, occ::_Occ)::ASTExpr
    _is_site(e, s, occ) && return lit(1)
    e isa OpExpr || return lit(0)          # literal, or a different variable
    occurs(e, s, occ) || return lit(0)     # structural short-circuit
    occ isa _SiteScan || return _dscalar_op(e, s, occ)
    # A node reached through several parents is differentiated once (see
    # `_SiteScan`).
    done = get(occ.d, e, nothing)
    done === nothing || return done
    d = occ.d[e] = _dscalar_op(e, s, occ)
    _keep!(occ.ix, e, d)
    return d
end

function _dscalar_op(e::OpExpr, s::Site, occ::_Occ)::ASTExpr
    o = e.op; a = e.args
    d(x) = dscalar(x, s, occ)
    if o == "+"
        return add((d(x) for x in a)...)
    elseif o == "-"
        return length(a) == 1 ? neg(d(a[1])) : add(d(a[1]), neg(d(a[2])))
    elseif o == "neg"
        return neg(d(a[1]))
    elseif o == "Pre"
        return d(a[1])
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
    elseif o == "exp"
        return mul(e, d(a[1]))
    elseif o == "log"
        return op("/", d(a[1]), a[1])
    elseif o == "log10"
        return op("/", d(a[1]), mul(a[1], op("log", lit(10))))
    elseif o == "sqrt"
        return op("/", d(a[1]), mul(lit(2), e))
    elseif o == "sin"
        return mul(op("cos", a[1]), d(a[1]))
    elseif o == "cos"
        return neg(mul(op("sin", a[1]), d(a[1])))
    elseif o == "tan"
        return mul(add(lit(1), op("^", e, lit(2))), d(a[1]))
    elseif o == "tanh"
        return mul(add(lit(1), neg(op("^", e, lit(2)))), d(a[1]))
    elseif o == "sinh"
        return mul(op("cosh", a[1]), d(a[1]))
    elseif o == "cosh"
        return mul(op("sinh", a[1]), d(a[1]))
    elseif o == "asin"
        return op("/", d(a[1]), op("sqrt", add(lit(1), neg(op("^", a[1], lit(2))))))
    elseif o == "acos"
        return neg(op("/", d(a[1]), op("sqrt", add(lit(1), neg(op("^", a[1], lit(2)))))))
    elseif o == "asinh"
        return op("/", d(a[1]), op("sqrt", add(op("^", a[1], lit(2)), lit(1))))
    elseif o == "acosh"
        return op("/", d(a[1]), op("sqrt", add(op("^", a[1], lit(2)), lit(-1))))
    elseif o == "atanh"
        return op("/", d(a[1]), add(lit(1), neg(op("^", a[1], lit(2)))))
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
    elseif o == "faq" && isempty(something(e.output_idx, Any[]))
        return _dsum(e, s, occ)
    elseif o in ZERO_DERIV_OPS
        return lit(0)
    elseif o == "index"
        # A site-bearing `index` that is not itself the site. Its subscripts
        # are integer-valued, so a read of a literal array (an expanded
        # template's `const` table or axis) is piecewise-constant in them: 0.
        # A read of a variable (which may be the site's own, or a recurrence
        # array that depends on it) at a site-dependent subscript, or of an
        # array expression that reads variables, has no rule.
        _index_is_literal_read(e) && return lit(0)
        throw(
            DerivativeRuleError(
                "`index` of a non-variable array expression (or a site-dependent " *
                    "index) has no scalar rule: $(skey(e))"
            )
        )
    else
        throw(
            DerivativeRuleError(
                "no derivative rule for op `$o` " *
                    "(rewrite-target ops must be lowered before differentiation)"
            )
        )
    end
end

"""
    drule_check(e, s, occ) -> Bool

The rule check of `dscalar(e, s, occ)` without the derivative, for the
structure-only mode (esm-jacobian-spec.md §2.1): throws exactly the
[`DerivativeRuleError`](@ref) `dscalar` would, and returns `false` exactly
when `dscalar` would return a literal 0, so a check the full mode skips on a
zero derivative (`_window_weight`'s) is skipped here too. It visits the
nodes `dscalar` visits and builds no derivative.
"""
drule_check(e::ASTExpr, s::Site, occ::_Occ)::Bool =
    _drule_check(e, s, occ, IdDict{OpExpr, Bool}())

function _drule_check(e::ASTExpr, s::Site, occ::_Occ, memo::IdDict{OpExpr, Bool})::Bool
    _is_site(e, s, occ) && return true
    e isa OpExpr || return false
    occurs(e, s, occ) || return false
    done = get(memo, e, nothing)
    done === nothing || return done
    return memo[e] = _drule_check_op(e, s, occ, memo)
end

# `_dscalar_op`, branch for branch: every child it differentiates is checked
# (`map`, never a short-circuit, so no rule error is missed), and the result
# follows the literal-zero folding of `add`, `mul` and `neg` there — an
# `op(…)` node it builds is never a literal 0.
function _drule_check_op(e::OpExpr, s::Site, occ::_Occ, memo)::Bool
    o = e.op; a = e.args
    d(x) = _drule_check(x, s, occ, memo)
    if o in ("+", "-", "neg", "/")
        return any(map(d, a))
    elseif o == "*"
        nz = map(d, a)
        return any(
            nz[i] && !any(j -> j != i && iszero_lit(a[j]), eachindex(a))
                for i in eachindex(a)
        )
    elseif o == "^" || o == "pow"
        db, dx = d(a[1]), d(a[2])
        return (db && !iszero_lit(a[2])) || dx
    elseif o in ("exp", "sin", "cos", "tan", "tanh", "sinh", "cosh", "abs")
        return d(a[1])
    elseif o in ("log", "log10", "sqrt", "asin", "acos", "asinh", "acosh", "atanh", "atan", "atan2")
        foreach(d, a)
        return true
    elseif o == "ifelse"
        d(a[2]); d(a[3])
        return true
    elseif o == "max" || o == "min"
        nz = map(d, a)
        return length(a) == 1 ? nz[1] : true
    elseif o == "faq" && isempty(something(e.output_idx, Any[]))
        red = _reduce_op(e, DerivativeRuleError)
        d(e.expr_body) || return false
        ranges = e.ranges === nothing ? Dict{String, Any}() : e.ranges
        _check_window(red, sort!(String.(collect(keys(ranges)))), DerivativeRuleError)
        return true
    elseif o in ZERO_DERIV_OPS
        return false
    end
    # `index`, and every op without a rule: `_dscalar_op` throws.
    return !iszero_lit(_dscalar_op(e, s, occ))
end

# Does the `index` node `e` read a literal array (one that reads no
# variable)? Its derivative is then 0 (spec §3.1).
_index_is_literal_read(e::OpExpr)::Bool =
    !isempty(e.args) && e.args[1] isa OpExpr && isempty(free_variables(e.args[1]))
