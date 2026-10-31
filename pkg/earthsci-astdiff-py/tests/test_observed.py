"""Observed classification on the inputs ``observed_defs`` must not miss —
each a SILENT failure if it does (port of the reference's
test/observed_test.jl for the two shared cases).

1. A ``FlattenedSystem``'s variables carry their DERIVED role (``state`` /
   ``observed`` / ``species`` / ``parameter``), never ``unknown``; matching
   only ``unknown`` finds no states and returns an empty band list.
2. An ARRAYED observed written the indexed way (``y[i] ~ f(i)``) is an
   observed, not a solver slot: it must be inlined, or the chain rule through
   it is lost and ``y`` becomes a spurious column target.

    y[i] = k·u[i]²,  D(u)[i] = -y[i]   ⇒   ∂/∂u[i] = -2k·u[i]
"""

import earthsci_ast as ea
import pytest
from bandvals import assemble_jacobian
from earthsci_astdiff import jacobian_bands, sysview
from earthsci_astdiff.system import observed_defs, view_states

K = 3.0
U = [0.3, 0.7, 1.1, 1.9]


def _idx(v, i):
    return {"op": "index", "args": [v, i]}


def _faq(body):
    return {"op": "faq", "args": [], "output_idx": ["i"], "ranges": {"i": [1, 4]}, "expr": body}


def _doc(ylhs):
    return ea.load_document(
        {
            "esm": "1.1.0",
            "metadata": {"name": "ArrObs"},
            "index_sets": {"x": {"kind": "interval", "size": 4}},
            "models": {
                "ArrObs": {
                    "variables": {
                        "u": {"type": "unknown", "shape": ["x"], "default": 0.5},
                        "y": {"type": "unknown", "shape": ["x"]},
                        "k": {"type": "parameter", "default": K},
                    },
                    "equations": [
                        {
                            "lhs": ylhs,
                            "rhs": {
                                "op": "*",
                                "args": ["k", {"op": "^", "args": [_idx("u", "i"), 2]}],
                            },
                        },
                        {
                            "lhs": {"op": "D", "args": ["u"], "wrt": "t"},
                            "rhs": _faq({"op": "-", "args": [_idx("y", "i")]}),
                        },
                    ],
                }
            },
        }
    )


def test_indexed_observed_is_inlined():
    file = _doc(_faq(_idx("y", "i")))
    sv = sysview(file)
    assert "y" in observed_defs(sv)
    assert view_states(sv) == ["u"], "y must not be a solver slot"
    assert {e.v for e in jacobian_bands(file)} == {"u"}
    j = assemble_jacobian(file, states={"u": U}, params={"k": K})
    assert len(j) == 4
    for i, ui in enumerate(U, start=1):
        assert j[(f"u[{i}]", f"u[{i}]")] == pytest.approx(-2 * K * ui, abs=1e-14)


def test_partial_indexed_definition_is_rejected():
    # Pinned (`y[1] ~ …`): an observed, but not rewritable to a whole-array
    # definition — loud, not a silent state slot.
    file = _doc(_idx("y", 1))
    with pytest.raises(ValueError, match="partial definitions"):
        observed_defs(sysview(file))


def test_flattened_system_has_states():
    # Scalar so the flattener keeps it simple: y = k·u², D(u) = -y.
    file = ea.load_document(
        {
            "esm": "1.1.0",
            "metadata": {"name": "Flat"},
            "models": {
                "Flat": {
                    "variables": {
                        "u": {"type": "unknown", "default": 0.5},
                        "y": {"type": "unknown"},
                        "k": {"type": "parameter", "default": K},
                    },
                    "equations": [
                        {
                            "lhs": "y",
                            "rhs": {"op": "*", "args": ["k", {"op": "^", "args": ["u", 2]}]},
                        },
                        {
                            "lhs": {"op": "D", "args": ["u"], "wrt": "t"},
                            "rhs": {"op": "-", "args": ["y"]},
                        },
                    ],
                }
            },
        }
    )
    flat = ea.flatten(file)
    sv = sysview(flat)
    states = view_states(sv)
    assert len(states) == 1, f"no solver slots in a flattened view: {states}"
    (u,) = states
    entries = jacobian_bands(sv)
    assert entries, "EMPTY band list for a flattened system (the silent zero)"
    j = assemble_jacobian(
        sv, states={u: 0.7}, params={n: K for n, v in sv.variables.items() if v.type == "parameter"}
    )
    assert j[(u, u)] == pytest.approx(-2 * K * 0.7, abs=1e-14)


def test_element_derivative_lhs_is_refused():
    # `D(index(u, 1)) = …` is no form lhs_state knows: skipping it would lose
    # its rows from the Jacobian.
    from earthsci_astdiff.bands import BandError

    file = ea.load_document(
        {
            "esm": "1.1.0",
            "metadata": {"name": "Elem"},
            "index_sets": {"x": {"kind": "interval", "size": 3}},
            "models": {
                "Elem": {
                    "variables": {"u": {"type": "unknown", "shape": ["x"], "default": 1.0}},
                    "equations": [
                        {
                            "lhs": {"op": "D", "wrt": "t", "args": [_idx("u", 1)]},
                            "rhs": {"op": "-", "args": [_idx("u", 1)]},
                        }
                    ],
                }
            },
        }
    )
    with pytest.raises(BandError, match="unsupported derivative LHS"):
        jacobian_bands(file)
