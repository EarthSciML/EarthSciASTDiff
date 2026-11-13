"""Scalar derivative rules — one rule per evaluable-core operator.

Must match esm-jacobian-spec.md §3 row-for-row (the same table the Julia
reference implements in ``scalar_rules.jl``); the shared band-list goldens
pin the two implementations against each other.
"""

from __future__ import annotations

from dataclasses import replace

from earthsci_ast.esm_types import Expr, ExprNode
from earthsci_ast.expression import free_variables

from .expr_helpers import Site, add, is_site, is_zero, mul, neg, occurs, occurs_memo, op
from .reductions import check_window, reduce_op, window_weight

ZERO_DERIV_OPS = {
    "<",
    "<=",
    ">",
    ">=",
    "==",
    "!=",
    "and",
    "or",
    "not",
    "sign",
    "floor",
    "ceil",
    "pi",
    "π",
    "e",
    "true",
    "false",
    "const",
    "enum",
}


class DerivativeRuleError(Exception):
    """Raised when a scalar operator has no derivative rule (spec §3)."""


def dscalar(e: Expr, s: Site) -> Expr:
    """Derivative of the scalar expression ``e`` w.r.t. site ``s``."""
    if is_site(e, s):
        return 1
    if not isinstance(e, ExprNode):
        return 0  # literal, or a different variable
    if not occurs(e, s):
        return 0  # structural short-circuit
    o, a = e.op, e.args

    def d(x: Expr) -> Expr:
        return dscalar(x, s)

    if o == "+":
        return add(*(d(x) for x in a))
    if o == "-":
        return neg(d(a[0])) if len(a) == 1 else add(d(a[0]), neg(d(a[1])))
    if o == "neg":
        return neg(d(a[0]))
    if o == "Pre":
        # the identity in a right-hand side (scalar_rules.jl): d[a]
        return d(a[0])
    if o == "*":
        terms = []
        for i, x in enumerate(a):
            dx = d(x)
            if is_zero(dx):
                continue
            terms.append(mul(dx, *(a[j] for j in range(len(a)) if j != i)))
        return add(*terms)
    if o == "/":
        n, m = a[0], a[1]
        dn, dm = d(n), d(m)
        t1 = 0 if is_zero(dn) else op("/", dn, m)
        t2 = 0 if is_zero(dm) else neg(op("/", mul(n, dm), op("^", m, 2)))
        return add(t1, t2)
    if o in ("^", "pow"):
        b, x = a[0], a[1]
        db, dx = d(b), d(x)
        t1 = 0 if is_zero(db) else mul(x, op("^", b, add(x, -1)), db)
        t2 = 0 if is_zero(dx) else mul(op("^", b, x), op("log", b), dx)
        return add(t1, t2)
    if o == "exp":
        return mul(e, d(a[0]))
    if o == "log":
        return op("/", d(a[0]), a[0])
    if o == "log10":
        return op("/", d(a[0]), mul(a[0], op("log", 10)))
    if o == "sqrt":
        return op("/", d(a[0]), mul(2, e))
    if o == "sin":
        return mul(op("cos", a[0]), d(a[0]))
    if o == "cos":
        return neg(mul(op("sin", a[0]), d(a[0])))
    if o == "tan":
        return mul(add(1, op("^", e, 2)), d(a[0]))
    if o == "tanh":
        return mul(add(1, neg(op("^", e, 2))), d(a[0]))
    if o == "sinh":
        return mul(op("cosh", a[0]), d(a[0]))
    if o == "cosh":
        return mul(op("sinh", a[0]), d(a[0]))
    if o == "asin":
        return op("/", d(a[0]), op("sqrt", add(1, neg(op("^", a[0], 2)))))
    if o == "acos":
        return neg(op("/", d(a[0]), op("sqrt", add(1, neg(op("^", a[0], 2))))))
    if o == "asinh":
        return op("/", d(a[0]), op("sqrt", add(op("^", a[0], 2), 1)))
    if o == "acosh":
        return op("/", d(a[0]), op("sqrt", add(op("^", a[0], 2), -1)))
    if o == "atanh":
        return op("/", d(a[0]), add(1, neg(op("^", a[0], 2))))
    if o == "atan" and len(a) == 1:
        return op("/", d(a[0]), add(1, op("^", a[0], 2)))
    if o in ("atan", "atan2"):  # atan(y, x)
        y, x = a[0], a[1]
        den = add(op("^", x, 2), op("^", y, 2))
        return op("/", add(mul(x, d(y)), neg(mul(y, d(x)))), den)
    if o == "abs":
        return mul(op("ifelse", op("<", a[0], 0), -1, 1), d(a[0]))
    if o == "ifelse":
        return op("ifelse", a[0], d(a[1]), d(a[2]))
    if o in ("max", "min"):
        # Left-fold branch selection; ties keep the earlier argument.
        acc, dacc = a[0], d(a[0])
        cmp = ">=" if o == "max" else "<="
        for x in a[1:]:
            dacc = op("ifelse", op(cmp, acc, x), dacc, d(x))
            acc = op(o, acc, x)
        return dacc
    if o == "faq" and not (e.output_idx or []):
        return _dsum(e, s)
    if o in ZERO_DERIV_OPS:
        return 0
    if o == "index":
        # A site-bearing `index` that is not itself the site. Its subscripts
        # are integer-valued, so a read of a literal array (an expanded
        # template's `const` table or axis) is piecewise-constant in them: 0.
        # A read of a variable (which may be the site's own, or a recurrence
        # array that depends on it) at a site-dependent subscript, or of an
        # array expression that reads variables, has no rule.
        if _index_is_literal_read(e):
            return 0
        raise DerivativeRuleError(
            "`index` of a non-variable array expression (or a site-dependent "
            "index) has no scalar rule"
        )
    raise DerivativeRuleError(
        f"no derivative rule for op `{o}` "
        "(rewrite-target ops must be lowered before differentiation)"
    )


_NZ_UNARY = ("exp", "sin", "cos", "tan", "tanh", "sinh", "cosh", "abs")
_DIV_OPS = ("log", "log10", "sqrt", "asin", "acos", "asinh", "acosh", "atanh", "atan", "atan2")


def drule_check(e: Expr, s: Site) -> bool:
    """The rule check of ``dscalar(e, s)`` without the derivative, for the
    structure-only mode (spec §2.1; scalar_rules.jl ``drule_check``): raises
    exactly the ``DerivativeRuleError`` ``dscalar`` would, and returns False
    exactly when ``dscalar`` would return a literal 0, so a check the full
    mode skips on a zero derivative (``window_weight``'s) is skipped here
    too. It visits the nodes ``dscalar`` visits and builds no derivative."""
    return _drule_check(e, s, {}, {})


def _drule_check(e: Expr, s: Site, memo: dict, occ: dict) -> bool:
    # `occ` memoizes the occurrence short-circuit for the whole scan (as the
    # Julia reference's `_SiteScan` does), so asking it at every node on the
    # way down costs Θ(n) per site, not Θ(n·depth).
    if is_site(e, s):
        return True
    if not isinstance(e, ExprNode) or not occurs_memo(e, s, occ):
        return False
    k = id(e)
    if k not in memo:
        memo[k] = (_drule_check_op(e, s, memo, occ), e)  # keep e alive: ids stay unique
    return memo[k][0]


def _drule_check_op(e: ExprNode, s: Site, memo: dict, occ: dict) -> bool:
    # ``dscalar``, branch for branch: every child it differentiates is checked
    # (a list, never a short-circuit, so no rule error is missed), and the
    # result follows the literal-zero folding of ``add``, ``mul`` and ``neg``
    # there — an ``op(...)`` node it builds is never a literal 0.
    o, a = e.op, e.args

    def d(x: Expr) -> bool:
        return _drule_check(x, s, memo, occ)

    if o in ("+", "-", "neg", "/"):
        return any([d(x) for x in a])
    if o == "*":
        nz = [d(x) for x in a]
        return any(
            nz[i] and not any(is_zero(a[j]) for j in range(len(a)) if j != i) for i in range(len(a))
        )
    if o in ("^", "pow"):
        db, dx = d(a[0]), d(a[1])
        return (db and not is_zero(a[1])) or dx
    if o in _NZ_UNARY:
        return d(a[0])
    if o in _DIV_OPS:
        for x in a:
            d(x)
        return True
    if o == "ifelse":
        d(a[1])
        d(a[2])
        return True
    if o in ("max", "min"):
        nz = [d(x) for x in a]
        return nz[0] if len(a) == 1 else True
    if o == "faq" and not (e.output_idx or []):
        red = reduce_op(e, DerivativeRuleError)
        if not d(e.expr):
            return False
        check_window(red, sorted(str(n) for n in (e.ranges or {})), DerivativeRuleError)
        return True
    if o in ZERO_DERIV_OPS:
        return False
    # ``index``, and every op without a rule: ``dscalar`` raises.
    return not is_zero(dscalar(e, s))


def reduction(e: ExprNode) -> str:
    """A faq node's reduction, for messages (``reduce_op`` decides it,
    reductions.py)."""
    return f"reduce=`{e.reduce}`" if e.semiring is None else f"semiring=`{e.semiring}`"


def _dsum(e: ExprNode, s: Site) -> Expr:
    """A scalar faq reduction (scalar_rules.jl ``_dsum``): the ``+`` sum of
    the body's derivatives, each weighted by what the reduction does with its
    term (``window_weight``, reductions.py). Bound indices the site's column
    does not read are summed; those it does read (``u[i,k]`` under Σ_k) are
    left FREE, for the band emitter to turn into contracted column dimensions
    (bands.py ``_lift_nested``). A filter keeps its window — the remaining
    sum's filter, or an ``ifelse`` once every bound index is left free — and
    its own Dirac term is dropped. A join or a value-invention node has no
    rule (spec §3.5)."""
    red = reduce_op(e, DerivativeRuleError)
    db = dscalar(e.expr, s)
    if is_zero(db):
        return 0
    ranges = e.ranges or {}
    db = window_weight(red, e, sorted(str(n) for n in ranges), db, DerivativeRuleError)

    def plus(**kw):
        # A rebuilt `+` node: the reduction is now a plain sum of weighted terms.
        if red == "+":
            return replace(e, expr=db, **kw)
        return replace(e, expr=db, reduce="+", semiring=None, **kw)

    read: set[str] = set()
    if isinstance(s.expr, ExprNode):
        for x in s.expr.args[1:]:
            read |= set(free_variables(x))
    summed = {n: r for n, r in ranges.items() if n not in read}
    if len(summed) == len(ranges):
        return plus()
    if not summed:
        return db if e.filter is None else op("ifelse", e.filter, db, 0)
    return plus(ranges=summed)


def _index_is_literal_read(e: ExprNode) -> bool:
    """Does the ``index`` node ``e`` read a literal array (one that reads no
    variable)? Its derivative is then 0 (spec §3.2)."""
    return bool(e.args) and isinstance(e.args[0], ExprNode) and not free_variables(e.args[0])
