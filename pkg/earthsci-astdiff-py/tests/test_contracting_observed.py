"""A CONTRACTING observed (issue #6; port of the reference's
test/contracting_observed_test.jl): ``s[i] = Σ_k u[i,k]²`` read inside a faq
that binds ``k`` too. Inlined as the bare body, the sum is dropped and the
leftover ``k`` is captured — the column sum collapses to the local cell, with
no error. The read must keep the sum, and the band calculus must follow the
contraction through it.
"""
import earthsci_ast as ea
import pytest

from earthsci_astdiff import assemble_jacobian, jacobian_bands
from earthsci_astdiff.bands import BandError
from earthsci_astdiff.scalar_rules import DerivativeRuleError

NX, NZ = 2, 3
U = [[0.1 * (i + NX * k + 1) for k in range(NZ)] for i in range(NX)]  # U[i][k]
S = [sum(U[i][k] ** 2 for k in range(NZ)) for i in range(NX)]


def _idx(v, *i):
    return {"op": "index", "args": [v, *i]}


def _sq(x):
    return {"op": "^", "args": [x, 2]}


def _neg(x):
    return {"op": "-", "args": [x]}


def _sdef(**kw):
    return {"op": "faq", "args": [], "output_idx": ["i"],
            "ranges": {"i": [1, NX], "k": [1, NZ]}, "reduce": "+",
            "expr": _sq(_idx("u", "i", "k")), **kw}


def _doc(body, oidx, s=None):
    return ea.load_document({
        "esm": "1.1.0",
        "metadata": {"name": "ContractObs"},
        "index_sets": {"x": {"kind": "interval", "size": NX},
                       "z": {"kind": "interval", "size": NZ}},
        "models": {"ContractObs": {
            "variables": {
                "u": {"type": "unknown", "shape": ["x", "z"], "default": 0.5},
                "s": {"type": "unknown", "shape": ["x"]}},
            "equations": [
                {"lhs": "s", "rhs": s or _sdef()},
                {"lhs": {"op": "D", "args": ["u"], "wrt": "t"},
                 "rhs": {"op": "faq", "args": [], "output_idx": list(oidx),
                         "ranges": {oidx[0]: [1, NX], oidx[1]: [1, NZ]},
                         "expr": body(oidx)}}]}}})


# (body, ∂ D(u)[i,k] / ∂ u[i,k'] for 0-based i, k, k')
CASES = {
    "-s[i]": (lambda o: _neg(_idx("s", o[0])),
              lambda i, k, kp: -2 * U[i][kp]),
    "-s[i]^2": (lambda o: _neg(_sq(_idx("s", o[0]))),
                lambda i, k, kp: -2 * S[i] * 2 * U[i][kp]),
    "-s[i]*u[i,k]": (
        lambda o: _neg({"op": "*", "args": [_idx("s", o[0]), _idx("u", *o)]}),
        lambda i, k, kp: -2 * U[i][kp] * U[i][k] - (S[i] if k == kp else 0)),
}


@pytest.mark.parametrize("oidx", [("i", "k"), ("i", "j")])
@pytest.mark.parametrize("case", list(CASES))
def test_contracting_observed_keeps_its_sum(case, oidx):
    body, want = CASES[case]
    j = assemble_jacobian(_doc(body, oidx), states={"u": U})
    expect = {(f"u[{i + 1},{k + 1}]", f"u[{i + 1},{kp + 1}]"): want(i, k, kp)
              for i in range(NX) for k in range(NZ) for kp in range(NZ)}
    assert set(j) == set(expect), "pattern: one dense block per column of u"
    for key, val in expect.items():
        assert j[key] == pytest.approx(val, abs=1e-13), key


def test_filtered_contracting_observed_is_refused():
    # Refused, as the same filter written inline is — never dropped.
    s = _sdef(filter={"op": "<", "args": ["k", 3]})
    file = _doc(CASES["-s[i]"][0], ("i", "k"), s=s)
    with pytest.raises((BandError, DerivativeRuleError)):
        jacobian_bands(file)
