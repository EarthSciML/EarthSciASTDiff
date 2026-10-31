"""Inline pruning (``inlinable_observed``): an observed that cannot move with
the differentiation target is not inlined (port of the reference's
test/prune_test.jl). The hazard is a SILENT ZERO, so values are checked
against the analytic Jacobian on both axes (EarthSciAST evaluating the
``JvpEval`` document, tests/docvals.py): ``src = 2k`` is state-independent
(pruned for ``wrt="states"``) but parameter-dependent (MUST be kept for
``wrt="parameters"``).

    D(u)[i] = -(src·u[i]) + src   ⇒   ∂/∂u[i] = -2k,   ∂/∂k = 2 − 2u[i]
"""

import earthsci_ast as ea
import pytest
from docvals import doc_jacobian
from earthsci_astdiff import sysview
from earthsci_astdiff.system import inlinable_observed, observed_defs

K = 3.0


def _probe():
    def idx(v, i):
        return {"op": "index", "args": [v, i]}

    body = {
        "op": "+",
        "args": [{"op": "-", "args": [{"op": "*", "args": ["src", idx("u", "i")]}]}, "src"],
    }
    return ea.load_document(
        {
            "esm": "1.1.0",
            "metadata": {"name": "PruneProbe"},
            "index_sets": {"x": {"kind": "interval", "size": 4}},
            "models": {
                "PruneProbe": {
                    "variables": {
                        "u": {"type": "unknown", "shape": ["x"], "default": 0.5},
                        "k": {"type": "parameter", "default": K},
                        "src": {"type": "unknown"},
                    },
                    "equations": [
                        {"lhs": "src", "rhs": {"op": "*", "args": ["k", 2]}},
                        {
                            "lhs": {"op": "D", "args": ["u"], "wrt": "t"},
                            "rhs": {
                                "op": "faq",
                                "args": [],
                                "output_idx": ["i"],
                                "ranges": {"i": [1, 4]},
                                "expr": body,
                            },
                        },
                    ],
                }
            },
        }
    )


def test_pruning_decision_is_per_axis():
    sv = sysview(_probe())
    obs = observed_defs(sv)
    assert "src" in obs
    assert "src" not in inlinable_observed(obs, sv.equations, {"u"})
    assert "src" in inlinable_observed(obs, sv.equations, {"k"})


def test_pruned_observed_values_on_both_axes():
    file = _probe()
    u = [0.3, 0.7, 1.1, 1.9]
    states = {f"u[{i}]": x for i, x in enumerate(u, start=1)}
    ju = doc_jacobian(file, states=states, params={"k": K}).J
    assert len(ju) == 4
    for i in range(1, 5):
        # A coefficient that dropped the pruned observed would read -1; one
        # that zeroed it, 0.
        assert ju[(f"u[{i}]", f"u[{i}]")] == pytest.approx(-2 * K, abs=1e-14)
    jp = doc_jacobian(file, wrt="parameters", states=states, params={"k": K}).J
    assert len(jp) == 4, "the k column must not be pruned to zero"
    for i, ui in enumerate(u, start=1):
        assert jp[(f"u[{i}]", "k")] == pytest.approx(2 - 2 * ui, abs=1e-14)
