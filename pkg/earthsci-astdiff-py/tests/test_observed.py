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
from docvals import doc_jacobian
from earthsci_astdiff import jacobian_bands, jacobian_pattern, serialize_band, stable_json, sysview
from earthsci_astdiff.system import observed_defs, view_states

K = 3.0
U = [0.3, 0.7, 1.1, 1.9]


def _cells(u):
    return {f"u[{i}]": x for i, x in enumerate(u, start=1)}


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
    j = doc_jacobian(file, states=_cells(U), params={"k": K}).J
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
    assert jacobian_pattern(sv)[0] == [(u, u)]
    # The flattened view's bands are the model's own, namespaced. Python
    # EarthSciAST refuses to load a derived document with dotted names
    # ("reference 'Flat.u' to undefined system"), so the value is the model's,
    # evaluated from its own `JvpEval` document.
    bands = stable_json([serialize_band(en) for en in entries]).replace("Flat.", "")
    assert bands == stable_json([serialize_band(en) for en in jacobian_bands(file)])
    j = doc_jacobian(file, states={"u": 0.7}, params={"k": K}).J
    assert j[("u", "u")] == pytest.approx(-2 * K * 0.7, abs=1e-14)


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


def test_full_rank_faq_value_over_a_pinned_makearray_region():
    # Port of the reference's inline_test.jl. A makearray region VALUE may be
    # a full-rank faq (naming every region dim, the pinned one included: a
    # 3-D field's boundary face) as well as a reduced-rank one (esm-spec
    # §4.3.2); the inliner accepted only the reduced-rank shape. `F` is
    # state-dependent so pruning cannot make this vacuous.
    #
    #   F = makearray(0 over [1,3]x[1,2]; 2·u[i,j] over the pinned row i = 2)
    #   D(u)[i,j] = -u[i,j] + F[i,j]   ⇒   ∂/∂u = -1, and +1 on row 2
    def u2(i, j):
        return {"op": "index", "args": ["u", i, j]}

    face = {
        "op": "faq",
        "args": [],
        "output_idx": ["i", "j"],
        "ranges": {"i": [2, 2], "j": [1, 2]},
        "expr": {"op": "*", "args": [2.0, u2("i", "j")]},
    }
    fdef = {
        "op": "makearray",
        "args": [],
        "regions": [[[1, 3], [1, 2]], [[2, 2], [1, 2]]],
        "values": [0.0, face],
    }
    rhs = {
        "op": "faq",
        "args": [],
        "output_idx": ["i", "j"],
        "ranges": {"i": [1, 3], "j": [1, 2]},
        "expr": {
            "op": "+",
            "args": [{"op": "-", "args": [u2("i", "j")]}, {"op": "index", "args": ["F", "i", "j"]}],
        },
    }
    file = ea.load_document(
        {
            "esm": "1.1.0",
            "metadata": {"name": "PinnedFace"},
            "index_sets": {
                "x": {"kind": "interval", "size": 3},
                "y": {"kind": "interval", "size": 2},
            },
            "models": {
                "PinnedFace": {
                    "variables": {
                        "u": {"type": "unknown", "shape": ["x", "y"], "default": 0.5},
                        "F": {"type": "unknown", "shape": ["x", "y"]},
                    },
                    "equations": [
                        {"lhs": "F", "rhs": fdef},
                        {"lhs": {"op": "D", "args": ["u"], "wrt": "t"}, "rhs": rhs},
                    ],
                }
            },
        }
    )
    u = [0.3, 0.7, 1.1, 1.9, 2.3, 2.9]
    cells = {f"u[{i},{j}]": u[2 * (i - 1) + (j - 1)] for i in (1, 2, 3) for j in (1, 2)}
    j = doc_jacobian(file, states=cells).J
    for c in cells:
        want = 1.0 if c.startswith("u[2,") else -1.0
        assert j.get((c, c), 0.0) == pytest.approx(want, abs=1e-14), c
    assert sum(1 for v in j.values() if v != 0.0) == 6, j
