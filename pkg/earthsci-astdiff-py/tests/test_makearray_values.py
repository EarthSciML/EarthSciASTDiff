"""``makearray`` region values read by position (esm-spec §4.3.2; port of
the Rust binding's tests/makearray_values.rs). A value fills its region by
position: its element at its own index ``origin + p`` lands on region index
``lo + p``, where ``origin`` is a faq value's range start (1 for any other
array). A value may be FULL-RANK (spanning every region dim, a pinned one
included) or REDUCED-RANK (the non-singleton dims only), and may be a faq
node, a makearray, or an array variable. Each case is read through an
observed (the inliner's ``index``-of-``makearray`` lowering) and as the
right-hand side itself (``_bands_makearray``), against the Jacobian worked
out by hand (the Rust test checks the same cases against central
differences of earthsci-ast's own right-hand side).

On a 3×2 state ``u``: ``F = makearray(0 over [1,3]×[1,2]; value over
region)``, and ``D(u) = -u + F`` (observed) or ``D(u) = F``.
"""

import earthsci_ast as ea
import pytest
from docvals import doc_jacobian

U = [[0.3, 0.7], [1.1, 1.9], [2.3, 2.9]]


def u2(i, j):
    return {"op": "index", "args": ["u", i, j]}


def twice(oidx, ranges, a, b):
    return {
        "op": "faq",
        "args": [],
        "output_idx": oidx,
        "ranges": ranges,
        "expr": {"op": "*", "args": [2.0, u2(a, b)]},
    }


def _doc(observed, region, value, extra_vars, extra_eqs):
    fdef = {
        "op": "makearray",
        "args": [],
        "regions": [[[1, 3], [1, 2]], region],
        "values": [0.0, value],
    }
    variables = {"u": {"type": "unknown", "shape": ["x", "y"], "default": 0.5}}
    if observed:
        variables["F"] = {"type": "unknown", "shape": ["x", "y"]}
        rhs = {
            "op": "faq",
            "args": [],
            "output_idx": ["i", "j"],
            "ranges": {"i": [1, 3], "j": [1, 2]},
            "expr": {
                "op": "+",
                "args": [
                    {"op": "-", "args": [u2("i", "j")]},
                    {"op": "index", "args": ["F", "i", "j"]},
                ],
            },
        }
        eqs = [
            {"lhs": "F", "rhs": fdef},
            {"lhs": {"op": "D", "args": ["u"], "wrt": "t"}, "rhs": rhs},
        ]
    else:
        eqs = [{"lhs": {"op": "D", "args": ["u"], "wrt": "t"}, "rhs": fdef}]
    variables.update(extra_vars)
    return ea.load_document(
        {
            "esm": "1.1.0",
            "metadata": {"name": "S"},
            "index_sets": {
                "x": {"kind": "interval", "size": 3},
                "y": {"kind": "interval", "size": 2},
                "one": {"kind": "interval", "size": 1},
                "two": {"kind": "interval", "size": 2},
            },
            "models": {"S": {"variables": variables, "equations": eqs + extra_eqs}},
        }
    )


def _check(region, value, reads, extra_vars=None, extra_eqs=None):
    """``reads``: (row, column) cells with ∂ = 2 from the value."""
    for observed in (True, False):
        file = _doc(observed, region, value, extra_vars or {}, extra_eqs or [])
        cells = {f"u[{i},{j}]": U[i - 1][j - 1] for i in (1, 2, 3) for j in (1, 2)}
        want: dict = {}
        if observed:
            for c in cells:
                want[(c, c)] = -1.0
        for r, c in reads:
            want[(r, c)] = want.get((r, c), 0.0) + 2.0
        got = {k: v for k, v in doc_jacobian(file, states=cells).J.items() if v != 0.0}
        want = {k: v for k, v in want.items() if v != 0.0}
        assert set(got) == set(want), (observed, got)
        for k, v in want.items():
            assert got[k] == pytest.approx(v, abs=1e-12), (observed, k)


@pytest.mark.parametrize("r", [1, 2, 3])
def test_full_rank_faq_value_over_a_shifted_range(r):
    # Region row 2, value range i ∈ [r,r]: row 2 reads u[r,j], not u[2,j].
    _check(
        [[2, 2], [1, 2]],
        twice(["i", "j"], {"i": [r, r], "j": [1, 2]}, "i", "j"),
        [(f"u[2,{j}]", f"u[{r},{j}]") for j in (1, 2)],
    )


def test_reduced_rank_faq_value_over_a_shifted_region():
    # Region rows [2,3] of column 2, value k ∈ [1,2]: row i reads u[i−1,1].
    _check(
        [[2, 3], [2, 2]],
        twice(["k"], {"k": [1, 2]}, "k", 1),
        [("u[2,2]", "u[1,1]"), ("u[3,2]", "u[2,1]")],
    )


def test_array_variable_values():
    # Full-rank G (one×y) over the pinned row 2, reduced-rank H (two) over
    # rows [2,3] of column 2: both were read at the free rank alone.
    _check(
        [[2, 2], [1, 2]],
        "G",
        [("u[2,1]", "u[1,1]"), ("u[2,2]", "u[1,2]")],
        {"G": {"type": "unknown", "shape": ["one", "y"]}},
        [{"lhs": "G", "rhs": twice(["i", "j"], {"i": [1, 1], "j": [1, 2]}, "i", "j")}],
    )
    _check(
        [[2, 3], [2, 2]],
        "H",
        [("u[2,2]", "u[1,1]"), ("u[3,2]", "u[2,1]")],
        {"H": {"type": "unknown", "shape": ["two"]}},
        [{"lhs": "H", "rhs": twice(["k"], {"k": [1, 2]}, "k", 1)}],
    )


def test_full_rank_makearray_value():
    inner = {
        "op": "makearray",
        "args": [],
        "regions": [[[1, 1], [1, 2]]],
        "values": [twice(["j"], {"j": [1, 2]}, 1, "j")],
    }
    _check([[2, 2], [1, 2]], inner, [("u[2,1]", "u[1,1]"), ("u[2,2]", "u[1,2]")])
