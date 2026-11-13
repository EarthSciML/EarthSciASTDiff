"""Per-op derivative rules against central finite differences of the
package's own evaluator (mirrors the Julia scalar_rules_test.jl)."""

import pytest
from earthsci_ast.numpy_interpreter import evaluate
from earthsci_ast.parse import _parse_expression as ex
from earthsci_astdiff import DerivativeRuleError, Site, dscalar, inline_observed
from earthsci_astdiff.expr_helpers import op

H = 1e-6


def numderiv(expr, x, h=H):
    return (evaluate(expr, {"x": x + h}) - evaluate(expr, {"x": x - h})) / (2 * h)


def symderiv(expr, x):
    return evaluate(dscalar(expr, Site.of("x")), {"x": x})


CASES = [
    ({"op": "+", "args": ["x", "x", 2.0]}, 0.7),
    ({"op": "-", "args": ["x"]}, 0.7),
    ({"op": "-", "args": [3.0, "x"]}, 0.7),
    ({"op": "*", "args": ["x", "x", "x"]}, 0.9),
    ({"op": "/", "args": [1.0, "x"]}, 0.8),
    ({"op": "/", "args": ["x", {"op": "+", "args": ["x", 2.0]}]}, 0.6),
    ({"op": "^", "args": ["x", 3]}, 1.3),
    ({"op": "^", "args": [2.0, "x"]}, 1.1),
    ({"op": "exp", "args": ["x"]}, 0.4),
    ({"op": "log", "args": ["x"]}, 1.7),
    ({"op": "log10", "args": ["x"]}, 1.7),
    ({"op": "sqrt", "args": ["x"]}, 2.3),
    ({"op": "sin", "args": ["x"]}, 0.9),
    ({"op": "cos", "args": ["x"]}, 0.9),
    ({"op": "tan", "args": ["x"]}, 0.5),
    ({"op": "tanh", "args": ["x"]}, 0.5),
    ({"op": "sinh", "args": ["x"]}, 0.5),
    ({"op": "cosh", "args": ["x"]}, 0.5),
    ({"op": "asin", "args": ["x"]}, 0.4),
    ({"op": "acos", "args": ["x"]}, 0.4),
    ({"op": "atan", "args": ["x"]}, 0.8),
    ({"op": "abs", "args": ["x"]}, -0.7),
    (
        {
            "op": "ifelse",
            "args": [
                {"op": "<", "args": ["x", 1.0]},
                {"op": "*", "args": [2.0, "x"]},
                {"op": "^", "args": ["x", 2]},
            ],
        },
        0.4,
    ),
    (
        {
            "op": "max",
            "args": [{"op": "*", "args": [2.0, "x"]}, {"op": "*", "args": [3.0, "x"]}, 1.0],
        },
        0.8,
    ),
    ({"op": "min", "args": [{"op": "*", "args": [2.0, "x"]}, {"op": "exp", "args": ["x"]}]}, 0.3),
]


@pytest.mark.parametrize("data,x", CASES)
def test_rule_vs_fd(data, x):
    e = ex(data)
    assert symderiv(e, x) == pytest.approx(numderiv(e, x), abs=1e-6)


def _pre(a):
    return {"op": "Pre", "args": [a]}


# `Pre` composed with the product, chain, power and extremum rules (spec §3.2).
# earthsci-ast's evaluator has no rule for `Pre`, so the expression and its
# derivative are evaluated after `inline_observed` lowers `Pre(a)` to `a`, the
# step that precedes differentiation in the band calculus; the derivative
# itself is taken of the UNLOWERED expression, so the rule is what is tested.
PRE_CASES = [
    ({"op": "*", "args": [_pre("x"), "x"]}, 0.7),
    ({"op": "exp", "args": [_pre("x")]}, 0.7),
    ({"op": "^", "args": [_pre("x"), 3]}, 1.3),
    ({"op": "*", "args": [{"op": "sin", "args": [_pre("x")]}, "x"]}, 0.9),
    (
        {
            "op": "max",
            "args": [{"op": "*", "args": [2.0, _pre("x")]}, {"op": "*", "args": ["x", "x"]}],
        },
        0.6,
    ),
    (
        {
            "op": "max",
            "args": [{"op": "*", "args": [2.0, _pre("x")]}, {"op": "*", "args": ["x", "x"]}],
        },
        2.5,
    ),
    ({"op": "/", "args": [_pre(_pre("x")), {"op": "+", "args": ["x", 2.0]}]}, 0.6),
]


@pytest.mark.parametrize("data,x", PRE_CASES)
def test_pre_rule_vs_fd(data, x):
    e = ex(data)
    lowered = inline_observed(e, {})
    assert "Pre" not in repr(lowered)
    d = inline_observed(dscalar(e, Site.of("x")), {})
    assert evaluate(d, {"x": x}) == pytest.approx(numderiv(lowered, x), abs=1e-6)


def test_structural_zero_and_policies():
    # a different variable is a structural zero without descent
    assert dscalar(ex({"op": "*", "args": ["y", 3.0]}), Site.of("x")) == 0
    # comparisons are zero-derivative
    assert dscalar(ex({"op": "<", "args": ["x", 1.0]}), Site.of("x")) == 0
    # `index` of a literal array is piecewise-constant in its integer
    # subscripts: derivative 0 (spec §3.2)
    tbl = ex({"op": "const", "args": [], "value": [1.0, 3.0, 2.0]})
    fx = op("floor", "x")
    assert dscalar(op("index", tbl, fx), Site.of("x")) == 0
    # ... but a site-dependent read of a variable has no rule (another
    # variable may depend on the site: a recurrence array is never inlined),
    # nor a read of an array expression that reads variables
    with pytest.raises(DerivativeRuleError):
        dscalar(op("index", "q", fx), Site.of("x"))
    ui = Site.of(op("index", "u", 1))
    with pytest.raises(DerivativeRuleError):
        dscalar(op("index", "u", op("floor", op("index", "u", 1))), ui)
    with pytest.raises(DerivativeRuleError):
        dscalar(op("index", op("+", tbl, "y"), fx), Site.of("x"))
    # unknown ops fail loudly
    with pytest.raises(DerivativeRuleError):
        dscalar(ex({"op": "grad", "args": ["x"]}), Site.of("x"))
    # `Pre(x)` differentiates as `x` (spec §3.2): the identity in a right-hand
    # side, as EarthSciAST evaluates it, not a constant
    assert dscalar(ex({"op": "Pre", "args": ["x"]}), Site.of("x")) == 1
    pre3 = ex({"op": "*", "args": [{"op": "Pre", "args": ["x"]}, 3.0]})
    assert evaluate(dscalar(pre3, Site.of("x")), {"x": 0.7}) == pytest.approx(3.0)


# The esm:std templates (esm 2.0.0) that replaced the `fn` closed functions
# differentiate through their expansion by the core rules plus the `index`
# rule: a 2.0.0 document of one model `M` whose state `w` (and `v`) reads
# `rhs`, checked against central differences of the source and the expected
# slopes (the reference's scalar_rules_test.jl).
def _std_doc(rhs, imports, v=False):
    import earthsci_ast as ea

    variables = {"w": {"type": "unknown", "default": 0.4}}
    eqs = [{"lhs": {"op": "D", "args": ["w"], "wrt": "t"}, "rhs": rhs}]
    if v:
        variables["v"] = {"type": "unknown", "default": 7.0}
        eqs.append({"lhs": {"op": "D", "args": ["v"], "wrt": "t"}, "rhs": 0.0})
    return ea.load_document(
        {
            "esm": "2.0.0",
            "metadata": {"name": "StdTemplates"},
            "models": {
                "M": {
                    "expression_template_imports": [
                        {"ref": f"esm:std/{f}.esm", "only": [n]} for f, n in imports
                    ],
                    "variables": variables,
                    "equations": eqs,
                }
            },
        }
    )


def _cnst(v):
    return {"op": "const", "args": [], "value": v}


def _apply(name, b):
    return {"op": "apply_expression_template", "args": [], "name": name, "bindings": b}


def test_std_bilinear_both_partials():
    from oracle import check_against_fd

    file = _std_doc(
        _apply(
            "bilinear",
            {
                "table": _cnst([[1.0, 2.0, 4.0], [3.0, 5.0, 9.0], [4.0, 7.0, 13.0]]),
                "axis_x": _cnst([0.0, 1.0, 2.0]),
                "axis_y": _cnst([0.0, 10.0, 30.0]),
                "x": "w",
                "y": "v",
                "nx": 3,
                "ny": 3,
            },
        ),
        [("interp", "bilinear")],
        v=True,
    )
    # every cell, each clamp region and the corners: both states feed the
    # same lookup, so the Jacobian row carries BOTH partials
    for u in (
        [0.4, 7.0],
        [1.5, 7.0],
        [0.4, 21.0],
        [1.5, 21.0],
        [-1.0, 12.0],
        [3.0, 12.0],
        [0.7, -5.0],
        [0.7, 45.0],
        [-1.0, -5.0],
        [3.0, 45.0],
    ):
        check_against_fd(file, u, tol=1e-6)


def test_std_linear_segment_slope():
    from docvals import doc_jacobian

    file = _std_doc(
        _apply(
            "linear",
            {"table": _cnst([0.0, 10.0, 12.0]), "axis": _cnst([0.0, 1.0, 2.0]), "x": "w", "n": 3},
        ),
        [("interp", "linear")],
    )
    # inside each segment, and flat outside the axis
    for x, slope in ((0.4, 10.0), (1.6, 2.0), (2.5, 0.0), (-0.5, 0.0)):
        J = doc_jacobian(file, "M", states={"w": x}).J
        assert J.get(("w", "w"), 0.0) == pytest.approx(slope, abs=1e-12)


def test_std_calendar_hour_flat_julian_day_not():
    from docvals import doc_jacobian
    from earthsci_astdiff import jacobian_bands

    # hour(w) is piecewise-constant: no entry; julian_day(3w) has slope
    # 3/86400 (it carries the fractional time of day)
    hour = _std_doc({"op": "*", "args": [_apply("hour", {"t": "w"}), 2.0]}, [("calendar", "hour")])
    assert jacobian_bands(hour, "M") == []
    jd = _std_doc(
        _apply("julian_day", {"t": {"op": "*", "args": [3, "w"]}}), [("calendar", "julian_day")]
    )
    for x in (0.4, 1.0e9 + 0.3):
        J = doc_jacobian(jd, "M", states={"w": x}).J
        assert J[("w", "w")] == pytest.approx(3 / 86400, rel=1e-12)
