"""Forcing-buffer and array-parameter columns (spec §4, §4.3) — port of the
reference's test/forcing_test.jl where the binding has the surface. The
byte/value goldens themselves are checked in test_goldens.py,
test_blocks.py and test_values.py. Values come from EarthSciAST evaluating the
product documents (tests/docvals.py)."""

import earthsci_ast as ea
import numpy as np
import pytest
from docvals import doc_jacobian, doc_vjp
from earthsci_astdiff import (
    BandError,
    column_names,
    forcing_names,
    jacobian_bands,
    jacobian_block,
    jacobian_columns,
    jacobian_pattern,
    jvp_document,
    sysview,
    vjp_document,
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
    res = doc_jacobian(bracket, "Bracket", wrt="parameters", t=0.37)
    assert res.colnames == column_names(
        jacobian_columns(sysview(bracket, "Bracket"), wrt="parameters")
    )
    for i in range(1, 5):
        assert res.J[(f"c[{i}]", f"E[{i}]")] == 1.0


def test_refused_rather_than_silently_wrong():
    # a column index that reads a parameter is integer-valued, so piecewise
    # constant: its dimension becomes one-hot over the buffer's extent, every
    # point of which is a placeable column (spec §3.5)
    off = {"off": {"type": "parameter", "default": 1}}
    hot = _bracket_doc(_op("*", _ix("c", "i"), _ix("F", 1, _op("+", "i", "off"))), off)
    (en,) = jacobian_bands(hot, "B", wrt="forcing", forcing=["F"])
    assert [n for n, _, _ in en.band.contracted] == ["_oh2"]
    assert en.band.cidx[1] == "_oh2"
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
        jvp_document(past, "B", wrt="forcing", forcing=["F"])
    with pytest.raises(BandError):
        vjp_document(past, "B", wrt="forcing", forcing=["F"])
    # a model that does not read the forcing set at all has an empty block
    none = _bracket_doc(_op("*", -1, _ix("c", "i")))
    assert jacobian_bands(none, "B", wrt="forcing", forcing=["F"]) == []


def test_parameter_read_past_extent_is_a_ghost_read():
    # the same read on the parameters axis (F is not data-fed here) is a
    # ghost read, as a state stencil reading past the domain is: i = 4 reads
    # F[1,6]
    past = _bracket_doc(_op("*", _ix("c", "i"), _ix("F", 1, _op("+", "i", 2))))
    pairs, _ = jacobian_pattern(past, "B", wrt="parameters")
    assert pairs == [(f"c[{i}]", f"F[1,{i + 2}]") for i in range(1, 4)]
    # the documents drop the ghost point (a nonzero off the pattern raises)
    assert len(doc_jacobian(past, "B", wrt="parameters").J) == 3
    ghost = _bracket_doc(_op("*", -1, _ix("c", _op("+", "i", 1))))
    assert len(doc_jacobian(ghost, "B").J) == 3


def test_identically_zero_derivative_is_an_empty_block():
    # forcing read only where the derivative is 0 (inside a piecewise-
    # constant op, or as an `ifelse` condition) yields no band and no error
    flr = _bracket_doc(_op("*", _ix("c", "i"), _op("floor", _ix("F", 1, "i"))))
    assert jacobian_bands(flr, "B", wrt="forcing", forcing=["F"]) == []
    cond = _bracket_doc(
        _op("ifelse", _op(">", _ix("F", 1, "i"), 0), _op("*", -1, _ix("c", "i")), 0)
    )
    assert jacobian_bands(cond, "B", wrt="forcing", forcing=["F"]) == []
    states, forcing = jacobian_block(cond, "B", wrt=["states", "forcing"], forcing=["F"])
    assert len(states["entries"]) == 1 and forcing["entries"] == []


def test_one_forcing_name_as_a_str():
    ok = _bracket_doc(_op("*", _ix("c", "i"), _ix("F", 2, _op("+", "i", 1))))
    sv = sysview(ok, "B")
    assert forcing_names(sv, "F") == ["F"]
    assert jacobian_block(sv, wrt="forcing", forcing="F") == jacobian_block(
        sv, wrt="forcing", forcing=["F"]
    )


def test_vjp_is_the_transposed_product(bracket):
    # Jᵀλ from one VjpEval call against J from the JvpEval columns
    sv = sysview(bracket, "Bracket")
    rows = ["S", "c[1]", "c[2]", "c[3]", "c[4]"]
    rng = np.random.default_rng(7)
    lam = dict(zip(rows, rng.standard_normal(len(rows)), strict=True))
    for wrt in ("forcing", "parameters"):
        cnames = column_names(jacobian_columns(sv, wrt=wrt))
        for t in (0.0, 0.37):
            res = doc_jacobian(sv, wrt=wrt, t=t)
            assert res.rownames == rows and res.colnames == cnames
            want = dict.fromkeys(cnames, 0.0)
            for (r, c), v in res.J.items():
                want[c] += v * lam[r]
            g = doc_vjp(sv, lam, wrt=wrt, t=t)
            assert list(g) == cnames
            np.testing.assert_allclose(list(g.values()), list(want.values()), atol=1e-13)
