"""The Jacobian EarthSciAST evaluates from this binding's ``JvpEval`` product
document (tests/docvals.py) against the Julia reference's value goldens
(tests/goldens/*.jvals.json): the stored (row, col) key set — the structural
pattern — must match EXACTLY and every value to 1e-10 relative (bit-equality
across runtimes is unrealistic — evaluation order differs)."""

import json

import earthsci_ast as ea
import pytest
from docvals import doc_jacobian

NAMES = (
    "bd_chem",
    "adv_interior",
    "contracted_ops",
    "flux_form_adv",
    "minmod_adv",
    "contracting_observed",
    "tgrad_photolysis",
    "forcing_bracket",
    "filtered_scan",
    "semiring_reduce",
)


# (fixture, axis, golden suffix) beyond the default states-axis goldens; the
# array parameters / forcing buffers take their document defaults.
AXIS_VALUES = (
    ("tgrad_photolysis", "time", ""),
    ("forcing_bracket", "forcing", ""),
    ("forcing_bracket", "forcing", ".w0"),
    ("forcing_bracket", "parameters", ""),
    ("forcing_bracket", "parameters", ".w0"),
)


@pytest.mark.parametrize("name,wrt,sfx", [(n, "states", "") for n in NAMES] + list(AXIS_VALUES))
def test_values_golden(name, wrt, sfx, golden_dir, valid_dir):
    tag = name if wrt == "states" else f"{name}.{wrt}{sfx}"
    g = json.loads((golden_dir / f"{tag}.jvals.json").read_text())
    file = ea.load_path(str(valid_dir / f"{name}.esm"))
    got = doc_jacobian(
        file, next(iter(file.models)), wrt=wrt, states=g["u"], params=g["params"], t=g["t"]
    ).J
    want = {(r, c): v for r, c, v in g["entries"]}
    assert set(got) == set(want)
    for k, v in want.items():
        assert got[k] == pytest.approx(v, rel=1e-10, abs=1e-13), k
