"""Scalar derivative rules — one rule per evaluable-core operator.

Must match esm-jacobian-spec.md §3 row-for-row (the same table the Julia
reference implements in ``scalar_rules.jl``); the shared band-list goldens
pin the two implementations against each other.
"""
from __future__ import annotations

from earthsci_ast.esm_types import Expr, ExprNode

from .expr_helpers import Site, add, is_site, is_zero, mul, neg, occurs, op

ZERO_DERIV_OPS = {
    "<", "<=", ">", ">=", "==", "!=", "and", "or", "not",
    "sign", "floor", "ceil", "pi", "π", "e", "true", "false", "const", "enum",
    "Pre",
}


class DerivativeRuleError(Exception):
    pass


def dscalar(e: Expr, s: Site) -> Expr:
    """Derivative of the scalar expression ``e`` w.r.t. site ``s``."""
    if is_site(e, s):
        return 1
    if not isinstance(e, ExprNode):
        return 0                     # literal, or a different variable
    if not occurs(e, s):
        return 0                     # structural short-circuit
    o, a = e.op, e.args

    def d(x: Expr) -> Expr:
        return dscalar(x, s)

    if o == "+":
        return add(*(d(x) for x in a))
    if o == "-":
        return neg(d(a[0])) if len(a) == 1 else add(d(a[0]), neg(d(a[1])))
    if o == "neg":
        return neg(d(a[0]))
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
    if o in ("atan", "atan2"):       # atan(y, x)
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
    if o in ZERO_DERIV_OPS:
        return 0
    if o == "fn":
        return dfn(e, s)
    if o == "index":
        raise DerivativeRuleError(
            "`index` of a non-variable array expression (or a site-dependent "
            "index) has no scalar rule")
    raise DerivativeRuleError(
        f"no derivative rule for op `{o}` "
        "(rewrite-target ops must be lowered before differentiation)")


def _const_vector(e: Expr) -> list[float] | None:
    if not (isinstance(e, ExprNode) and e.op == "const"):
        return None
    v = e.value
    if not (isinstance(v, list) and
            all(isinstance(x, (int, float)) and not isinstance(x, bool)
                for x in v)):
        return None
    return [float(x) for x in v]


def _const_matrix(e: Expr) -> list[list[float]] | None:
    """value[i][j] → T[i][j] (row-major, spec §9.2)."""
    if not (isinstance(e, ExprNode) and e.op == "const"):
        return None
    v = e.value
    if not isinstance(v, list) or not v:
        return None
    rows = []
    for r in v:
        if not (isinstance(r, list) and
                all(isinstance(x, (int, float)) and not isinstance(x, bool)
                    for x in r)):
            return None
        rows.append([float(x) for x in r])
    if any(len(r) != len(rows[0]) for r in rows):
        return None
    return rows


def _bilinear_partial(T: list[list[float]], xv: list[float], yv: list[float],
                      xe: Expr, ye: Expr) -> Expr:
    """∂(bilinear blend)/∂(the axis of ``xv``) — see scalar_rules.jl."""
    nx, ny = len(xv), len(yv)
    out: Expr = 0                            # x ≥ axis_x[nx]: flat in x
    for i in range(nx - 2, -1, -1):          # 0-based cell i ↔ knots i, i+1

        def sl(j: int) -> float:
            return (T[i + 1][j] - T[i][j]) / (xv[i + 1] - xv[i])

        leaf: Expr = sl(ny - 1)              # y ≥ axis_y[ny]: top edge row
        for j in range(ny - 2, -1, -1):
            wy = op("/", add(ye, -yv[j]), yv[j + 1] - yv[j])
            ex = add(sl(j), mul(wy, add(sl(j + 1), neg(sl(j)))))
            leaf = op("ifelse", op("<", ye, yv[j + 1]), ex, leaf)
        leaf = op("ifelse", op("<", ye, yv[0]), sl(0), leaf)   # bottom edge
        out = op("ifelse", op("<", xe, xv[i + 1]), leaf, out)
    return op("ifelse", op("<", xe, xv[0]), 0, out)


def dfn(e: ExprNode, s: Site) -> Expr:
    """Closed-function derivative table (esm-spec §9.2)."""
    name = e.name
    a = e.args
    if name is None:
        raise DerivativeRuleError("`fn` node without a name")
    if name.startswith("datetime.") or name == "interp.searchsorted":
        return 0
    if name == "interp.linear":
        table, axis, x = a[0], a[1], a[2]
        dx = dscalar(x, s)
        if is_zero(dx):
            return 0
        yv, xv = _const_vector(table), _const_vector(axis)
        if yv is None or xv is None:
            raise DerivativeRuleError(
                "interp.linear table/axis must be literal `const` arrays "
                "(spec §9.2 interp_table_not_const)")
        n = len(xv)
        if n != len(yv) or n < 2:
            raise DerivativeRuleError(
                "interp.linear table/axis length mismatch or < 2 knots")
        slopes = [(yv[k + 1] - yv[k]) / (xv[k + 1] - xv[k])
                  for k in range(n - 1)]
        slope: Expr = 0                        # x ≥ axis[N]: flat
        for k in range(n - 2, -1, -1):
            slope = op("ifelse", op("<", x, xv[k + 1]), slopes[k], slope)
        slope = op("ifelse", op("<", x, xv[0]), 0, slope)      # flat below
        return mul(slope, dx)
    if name == "interp.bilinear":
        table, ax, ay, x, y = a[0], a[1], a[2], a[3], a[4]
        dx, dy = dscalar(x, s), dscalar(y, s)
        if is_zero(dx) and is_zero(dy):
            return 0
        T = _const_matrix(table)
        xv, yv = _const_vector(ax), _const_vector(ay)
        if T is None or xv is None or yv is None:
            raise DerivativeRuleError(
                "interp.bilinear table/axes must be literal `const` arrays "
                "(spec §9.2 interp_table_not_const)")
        nx, ny = len(xv), len(yv)
        if len(T) != nx or any(len(r) != ny for r in T) or nx < 2 or ny < 2:
            raise DerivativeRuleError(
                "interp.bilinear table/axis shape mismatch or < 2 knots")
        terms = []
        if not is_zero(dx):
            terms.append(mul(_bilinear_partial(T, xv, yv, x, y), dx))
        if not is_zero(dy):
            Tt = [[T[i][j] for i in range(nx)] for j in range(ny)]
            terms.append(mul(_bilinear_partial(Tt, yv, xv, y, x), dy))
        return add(*terms)
    raise DerivativeRuleError(f"unknown closed function `{name}`")
