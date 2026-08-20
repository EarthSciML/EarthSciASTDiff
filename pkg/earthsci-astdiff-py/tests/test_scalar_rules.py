"""Per-op derivative rules against central finite differences of the
package's own evaluator (mirrors the Julia scalar_rules_test.jl)."""
import math

import pytest
from earthsci_ast.numpy_interpreter import evaluate
from earthsci_ast.parse import _parse_expression as ex

from earthsci_astdiff import DerivativeRuleError, Site, dscalar
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
    ({"op": "ifelse", "args": [{"op": "<", "args": ["x", 1.0]},
                               {"op": "*", "args": [2.0, "x"]},
                               {"op": "^", "args": ["x", 2]}]}, 0.4),
    ({"op": "max", "args": [{"op": "*", "args": [2.0, "x"]},
                            {"op": "*", "args": [3.0, "x"]}, 1.0]}, 0.8),
    ({"op": "min", "args": [{"op": "*", "args": [2.0, "x"]},
                            {"op": "exp", "args": ["x"]}]}, 0.3),
]


@pytest.mark.parametrize("data,x", CASES)
def test_rule_vs_fd(data, x):
    e = ex(data)
    assert symderiv(e, x) == pytest.approx(numderiv(e, x), abs=1e-6)


def test_structural_zero_and_policies():
    # a different variable is a structural zero without descent
    assert dscalar(ex({"op": "*", "args": ["y", 3.0]}), Site.of("x")) == 0
    # comparisons and datetime.* are zero-derivative
    assert dscalar(ex({"op": "<", "args": ["x", 1.0]}), Site.of("x")) == 0
    assert dscalar(ex({"op": "fn", "name": "datetime.year", "args": ["x"]}),
                   Site.of("x")) == 0
    # unknown ops fail loudly
    with pytest.raises(DerivativeRuleError):
        dscalar(ex({"op": "grad", "args": ["x"]}), Site.of("x"))


def test_interp_linear_slope():
    table = ex({"op": "const", "args": [], "value": [0.0, 10.0, 12.0]})
    axis = ex({"op": "const", "args": [], "value": [0.0, 1.0, 2.0]})
    e = op("fn", table, axis, "x", name="interp.linear")
    d = dscalar(e, Site.of("x"))
    assert evaluate(d, {"x": 0.4}) == pytest.approx(10.0)
    assert evaluate(d, {"x": 1.6}) == pytest.approx(2.0)
    assert evaluate(d, {"x": -0.5}) == 0.0
    assert evaluate(d, {"x": 2.5}) == 0.0
    with pytest.raises(DerivativeRuleError):
        dscalar(op("fn", "tbl", "ax", "x", name="interp.linear"), Site.of("x"))


def test_interp_bilinear_partials():
    tbl = ex({"op": "const", "args": [],
              "value": [[1.0, 2.0, 4.0], [3.0, 5.0, 9.0], [4.0, 7.0, 13.0]]})
    axx = ex({"op": "const", "args": [], "value": [0.0, 1.0, 2.0]})
    axy = ex({"op": "const", "args": [], "value": [0.0, 10.0, 30.0]})
    e = op("fn", tbl, axx, axy, "x", "y", name="interp.bilinear")
    dx = dscalar(e, Site.of("x"))
    dy = dscalar(e, Site.of("y"))

    def f(x, y):
        return evaluate(e, {"x": x, "y": y})

    h = 1e-5
    pts = [(0.4, 7.0), (1.5, 7.0), (0.4, 21.0), (1.5, 21.0),
           (-1.0, 12.0), (3.0, 12.0), (0.7, -5.0), (0.7, 45.0),
           (-1.0, -5.0), (3.0, 45.0)]
    for x, y in pts:
        assert evaluate(dx, {"x": x, "y": y}) == pytest.approx(
            (f(x + h, y) - f(x - h, y)) / (2 * h), abs=1e-8)
        assert evaluate(dy, {"x": x, "y": y}) == pytest.approx(
            (f(x, y + h) - f(x, y - h)) / (2 * h), abs=1e-8)
