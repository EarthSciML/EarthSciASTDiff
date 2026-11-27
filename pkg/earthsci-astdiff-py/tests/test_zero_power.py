"""The power rule at a zero base with a zero exponent (spec §3, ``^``): in a
product over ions of ``m^stoich`` with a parameter exponent, an absent ion
(``m = 0``) whose exponent is 0 made ``x·b^(x−1)`` = ``0·0^(−1)`` NaN, where
the derivative of ``b^0`` is 0. The shared golden
(tests/goldens/zero_power_product.points.json) holds the exact derivatives
at the singular point and at a point away from it, where they match central
differences; the Julia and Rust bindings check it too."""

import json
import math

import earthsci_ast as ea
import pytest
from docvals import doc_jacobian
from earthsci_ast.esm_types import ExprNode

from earthsci_astdiff import Site, skey
from earthsci_astdiff.expr_helpers import is_zero
from earthsci_astdiff.scalar_rules import dscalar


def test_the_shared_points(golden_dir, valid_dir):
    g = json.loads((golden_dir / "zero_power_product.points.json").read_text())
    file = ea.load_path(str(valid_dir / "zero_power_product.esm"))
    for pt in g["points"]:
        got = doc_jacobian(
            file, next(iter(file.models)), wrt="states", states=pt["u"], params=g["params"], t=g["t"]
        ).J
        assert all(math.isfinite(v) for v in got.values()), got
        for r, c, v in pt["entries"]:
            gv = got.get((r, c), 0.0)
            if v == 0:
                assert gv == 0, (r, c)
            else:
                assert gv == pytest.approx(v, rel=1e-12), (r, c)


def test_the_rule():
    """A literal exponent 0 gives 0, another literal the usual form, and an
    exponent that is not a literal the guarded form."""
    b = Site.of("b")

    def d(x):
        return dscalar(ExprNode(op="^", args=["b", x]), b)

    assert is_zero(d(0))
    assert "ifelse" not in skey(d(3))
    assert "ifelse" in skey(d("x"))
