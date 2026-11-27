"""A guard holds over a composed derivative (spec §3.6, named steps): the
derivative of ``g = ifelse(s > 0, 10^lg, 1)`` through the observed
``lg = x/s`` is composed as ``(∂g/∂lg)·_du_lg`` with
``∂g/∂lg = ifelse(s > 0, …, 0)``, and where ``s = 0`` the inner derivative
``_du_lg`` is NaN; the product is written ``ifelse(s > 0, …·_du_lg, 0)``, so
the guard keeps the derivative finite, as MOSAIC's MTEM sulfate-poor log
gamma needs. The shared golden (tests/goldens/guarded_quotient.points.json)
holds the exact derivatives with the guard holding and open; the Julia and
Rust bindings check it too."""

import json
import math

import earthsci_ast as ea
import pytest
from docvals import doc_jacobian


def test_the_shared_points(golden_dir, valid_dir):
    g = json.loads((golden_dir / "guarded_quotient.points.json").read_text())
    file = ea.load_path(str(valid_dir / "guarded_quotient.esm"))
    for pt in g["points"]:
        got = doc_jacobian(
            file, next(iter(file.models)), wrt="states", states=pt["u"], params=pt["params"], t=g["t"]
        ).J
        assert all(math.isfinite(v) for v in got.values()), got
        for r, c, v in pt["entries"]:
            gv = got.get((r, c), 0.0)
            if v == 0:
                assert gv == 0, (r, c)
            else:
                assert gv == pytest.approx(v, rel=1e-12), (r, c)
