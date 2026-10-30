"""Forcing-buffer and array-parameter columns (spec §4, §4.3) — port of the
reference's test/forcing_test.jl where the binding has the surface. The
byte/value goldens themselves are checked in test_goldens.py,
test_blocks.py and test_values.py."""

import earthsci_ast as ea
import numpy as np
import pytest
from earthsci_astdiff import (
    BandError,
    assemble_jacobian,
    column_names,
    forcing_names,
    jacobian_bands,
    jacobian_columns,
    jacobian_pattern,
    sysview,
    vjp,
    vjp_forcing,
)


@pytest.fixture(scope="module")
def bracket(valid_dir):
    return ea.load_path(str(valid_dir / "forcing_bracket.esm"))


def _ix(v, *i):
    return {"op": "index", "args": [v, *i]}


def _op(o, *a):
    return {"op": o, "args": list(a)}


def _bracket_doc(body, extra=None):
    """One model around `F` (shape [2, 5]) and a state `c` (shape [4]) with
    the given `D(c)` body over `i`; `extra` adds variables."""
    variables = {
        "c": {"type": "unknown", "shape": ["x"], "default": 0.5},
        "F": {"type": "parameter", "shape": ["rec", "xf"], "default": 1.0},
    }
    variables.update(extra or {})
    return ea.load_document(
        {
            "esm": "1.1.0",
            "metadata": {"name": "B"},
            "index_sets": {
                "x": {"kind": "interval", "size": 4},
                "xf": {"kind": "interval", "size": 5},
                "rec": {"kind": "interval", "size": 2},
            },
            "models": {
                "B": {
                    "variables": variables,
                    "equations": [
                        {
                            "lhs": {"op": "D", "args": ["c"], "wrt": "t"},
                            "rhs": {
                                "op": "faq",
                                "args": [],
                                "output_idx": ["i"],
                                "ranges": {"i": {"from": "x"}},
                                "expr": body,
                            },
                        }
                    ],
                }
            },
        }
    )


def test_forcing_set(bracket):
    sv = sysview(bracket, "Bracket")
    # inferred: the array parameters with a `data` update; E (no update) and
    # the scalars stay on the parameters axis
    assert forcing_names(sv) == ["F", "G"]
    assert jacobian_columns(sv, wrt="forcing") == [("F", [2, 5]), ("G", [2, 3])]
    assert jacobian_columns(sv, wrt="parameters") == [
        ("E", [4]),
        ("dt", []),
        ("k", []),
        ("t_ref", []),
    ]
    # an explicit set (a planner that builds the providers itself)
    assert forcing_names(sv, ["G"]) == ["G"]
    assert [n for n, _ in jacobian_columns(sv, wrt="parameters", forcing=["G"])] == [
        "E",
        "F",
        "dt",
        "k",
        "t_ref",
    ]
    # canonical ordinal: variables in order, each one's cells column-major
    assert column_names(jacobian_columns(sv, wrt="forcing"))[:3] == ["F[1,1]", "F[2,1]", "F[1,2]"]
    # a name that resolves to no buffer throws, never an empty axis
    with pytest.raises(ValueError):
        forcing_names(sv, ["F", "nope"])
    with pytest.raises(ValueError):
        jacobian_bands(sv, wrt="forcing", forcing=["c"])


def test_bracket_columns(bracket):
    from earthsci_astdiff.expr_helpers import ser_expr

    fb = jacobian_bands(bracket, "Bracket", wrt="forcing")
    cols = {(en.u, en.v, str(ser_expr(en.band.cidx[0]))) for en in fb}
    assert {("c", "F", "1"), ("c", "F", "2"), ("S", "F", "1")} <= cols
    assert any(en.v == "G" and en.band.contracted for en in fb)


def test_array_parameter_columns_are_real(bracket):
    got = assemble_jacobian(bracket, "Bracket", wrt="parameters", t=0.37)
    for i in range(1, 5):
        assert got[(f"c[{i}]", f"E[{i}]")] == 1.0


def test_refused_rather_than_silently_wrong():
    # a column index that reads a parameter cannot be placed at build time
    off = {"off": {"type": "parameter", "default": 1}}
    bad = _bracket_doc(_op("*", _ix("c", "i"), _ix("F", 1, _op("+", "i", "off"))), off)
    with pytest.raises(BandError):
        jacobian_bands(bad, "B", wrt="forcing", forcing=["F"])
    # a literal offset is closed
    ok = _bracket_doc(_op("*", _ix("c", "i"), _ix("F", 2, _op("+", "i", 1))))
    assert len(jacobian_bands(ok, "B", wrt="forcing", forcing=["F"])) == 1
    pairs, _ = jacobian_pattern(ok, "B", wrt="forcing", forcing=["F"])
    assert len(pairs) == 4
    # a column outside the buffer's extent is an error, not a ghost skip
    past = _bracket_doc(_op("*", _ix("c", "i"), _ix("F", 1, _op("+", "i", 2))))
    assert len(jacobian_bands(past, "B", wrt="forcing", forcing=["F"])) == 1
    with pytest.raises(BandError):
        jacobian_pattern(past, "B", wrt="forcing", forcing=["F"])
    with pytest.raises(BandError):
        assemble_jacobian(past, "B", wrt="forcing", forcing=["F"])
    # the same read on the parameters axis (F is not data-fed here)
    with pytest.raises(BandError):
        jacobian_pattern(past, "B", wrt="parameters")
    # while a state stencil reading past the domain stays a ghost skip
    ghost = _bracket_doc(_op("*", -1, _ix("c", _op("+", "i", 1))))
    assert len(assemble_jacobian(ghost, "B")) == 3
    # reads forcing, yields no band: refused, never an empty block
    flr = _bracket_doc(_op("*", _ix("c", "i"), _op("floor", _ix("F", 1, "i"))))
    with pytest.raises(BandError):
        jacobian_bands(flr, "B", wrt="forcing", forcing=["F"])
    # a model that does not read the forcing set at all has an empty block
    none = _bracket_doc(_op("*", -1, _ix("c", "i")))
    assert jacobian_bands(none, "B", wrt="forcing", forcing=["F"]) == []


def test_vjp_is_the_transposed_scatter(bracket):
    sv = sysview(bracket, "Bracket")
    rows = ["S", "c[1]", "c[2]", "c[3]", "c[4]"]
    rng = np.random.default_rng(7)
    lam = dict(zip(rows, rng.standard_normal(len(rows))))
    for wrt in ("forcing", "parameters"):
        cnames = column_names(jacobian_columns(sv, wrt=wrt))
        J = assemble_jacobian(sv, wrt=wrt, t=0.37)
        want = np.zeros(len(cnames))
        for (r, c), v in J.items():
            want[cnames.index(c)] += v * lam[r]
        g = np.zeros(len(cnames))
        vjp(g, sv, lam, wrt=wrt, t=0.37)
        np.testing.assert_allclose(g, want, atol=1e-13)
        vjp(g, sv, lam, wrt=wrt, t=0.37)  # accumulates
        np.testing.assert_allclose(g, 2 * want, atol=1e-13)
    # buffer-shaped accumulators, indexed by cell
    cnames = column_names(jacobian_columns(sv, wrt="forcing"))
    flat = vjp_forcing(np.zeros(len(cnames)), sv, lam, t=0.37)
    acc = {"F": np.zeros((2, 5)), "G": np.zeros((2, 3))}
    vjp_forcing(acc, sv, lam, t=0.37)
    assert acc["F"][1, 2] == pytest.approx(flat[cnames.index("F[2,3]")])
    np.testing.assert_allclose(
        np.concatenate([acc["F"].ravel(order="F"), acc["G"].ravel(order="F")]), flat, atol=1e-13
    )
    with pytest.raises(ValueError):
        vjp_forcing({"F": np.zeros((2, 5))}, sv, lam, t=0.37)
    with pytest.raises(ValueError):
        vjp_forcing(np.zeros(3), sv, lam, t=0.37)
