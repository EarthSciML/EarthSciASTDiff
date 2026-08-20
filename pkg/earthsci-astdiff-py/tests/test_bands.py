"""Band-calculus structure tests over the shared fixtures (fixture-level
mirror of the Julia clip/contracted testsets)."""
import earthsci_ast as ea
import pytest

from earthsci_astdiff import (BandError, jacobian_bands, jacobian_pattern,
                              skey, stable_json)
from earthsci_astdiff.expr_helpers import ser_expr


def test_flux_form_guard_free(valid_dir):
    file = ea.load(str(valid_dir / "flux_form_adv.esm"))
    entries = jacobian_bands(file, "FluxFormAdv")
    for en in entries:
        assert "ifelse" not in stable_json(ser_expr(en.band.coef))
    # diagonal every row + sub-diagonal rows 2..8, no inflow-row ghost
    pairs, _ = jacobian_pattern(file, "FluxFormAdv")
    assert ("u[1]", "u[0]") not in pairs
    n = 8
    assert len(pairs) == 2 * n - 1


def test_contracted_shapes(valid_dir):
    file = ea.load(str(valid_dir / "contracted_ops.esm"))
    entries = jacobian_bands(file, "ContractedOps")
    yu = [e for e in entries if e.u == "y" and e.v == "u"]
    gath = [e for e in yu if e.band.contracted]
    diag = [e for e in yu if not e.band.contracted]
    assert len(gath) == 1 and gath[0].band.contracted == [("j", 1, 6)]
    assert len(diag) == 1 and '"aggregate"' in skey(diag[0].band.coef)
    zu = [e for e in entries if e.u == "z" and e.v == "u"]
    assert len(zu) == 1 and '"const"' in skey(zu[0].band.cidx[0])
    # conn duplicate neighbor accumulates into ONE pattern position
    pairs, _ = jacobian_pattern(file, "ContractedOps")
    assert ("z[3]", "u[5]") in pairs


def test_guards(valid_dir):
    doc = {
        "esm": "0.8.0",
        "metadata": {"name": "G"},
        "index_sets": {"x": {"kind": "interval", "size": 3}},
        "models": {"G": {
            "variables": {"u": {"type": "state", "shape": ["x"],
                                "default": 0.5}},
            "equations": [{
                "lhs": {"op": "D", "args": ["u"], "wrt": "t"},
                "rhs": {"op": "aggregate", "args": [], "output_idx": ["i"],
                        "reduce": "min",
                        "ranges": {"i": [1, 3], "j": [1, 3]},
                        "expr": {"op": "index", "args": ["u", "j"]}}}]}}}
    file = ea.load(doc)
    with pytest.raises(BandError):
        jacobian_bands(file, "G")


def test_wrt_parameters_and_time(valid_dir):
    file = ea.load(str(valid_dir / "flux_form_adv.esm"))
    entries = jacobian_bands(file, "FluxFormAdv", wrt="parameters")
    assert {e.v for e in entries} <= {"uin", "c", "dx"}
    pairs, _ = jacobian_pattern(file, "FluxFormAdv", wrt="parameters")
    assert ("u[1]", "uin") in pairs
    assert not any(r != "u[1]" and c == "uin" for r, c in pairs)
    # autonomous: no time bands
    assert jacobian_bands(file, "FluxFormAdv", wrt="time") == []
