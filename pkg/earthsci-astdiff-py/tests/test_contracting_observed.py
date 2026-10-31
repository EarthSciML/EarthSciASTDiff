"""A CONTRACTING observed (issue #6; port of the reference's
test/contracting_observed_test.jl): ``s[i] = Σ_k u[i,k]²`` read inside a faq
that binds ``k`` too. Inlined as the bare body, the sum is dropped and the
leftover ``k`` is captured — the column sum collapses to the local cell, with
no error. The read must keep the sum, and the band calculus must follow the
contraction through it.
"""

import earthsci_ast as ea
import pytest
from earthsci_ast.esm_types import ExprNode
from earthsci_ast.expr_walk import map_children
from earthsci_astdiff import assemble_jacobian, inline_observed
from earthsci_astdiff.expr_helpers import skey
from earthsci_astdiff.system import lhs_state, observed_defs, sysview
from oracle import check_against_fd

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
    return {
        "op": "faq",
        "args": [],
        "output_idx": ["i"],
        "ranges": {"i": [1, NX], "k": [1, NZ]},
        "reduce": "+",
        "expr": _sq(_idx("u", "i", "k")),
        **kw,
    }


def _mul(*a):
    return {"op": "*", "args": list(a)}


def _ksum(body):
    return {"op": "faq", "args": [], "output_idx": [], "ranges": {"k": [1, NZ]}, "expr": body}


def _doc(body, oidx=("i", "k"), s=None, variables=None, equations=()):
    return ea.load_document(
        {
            "esm": "1.1.0",
            "metadata": {"name": "ContractObs"},
            "index_sets": {
                "x": {"kind": "interval", "size": NX},
                "z": {"kind": "interval", "size": NZ},
            },
            "models": {
                "ContractObs": {
                    "variables": {
                        "u": {"type": "unknown", "shape": ["x", "z"], "default": 0.5},
                        "s": {"type": "unknown", "shape": ["x"]},
                        **(variables or {}),
                    },
                    "equations": [
                        {"lhs": "s", "rhs": s or _sdef()},
                        *equations,
                        {
                            "lhs": {"op": "D", "args": ["u"], "wrt": "t"},
                            "rhs": {
                                "op": "faq",
                                "args": [],
                                "output_idx": list(oidx),
                                "ranges": {oidx[0]: [1, NX], oidx[1]: [1, NZ]},
                                "expr": body(oidx),
                            },
                        },
                    ],
                }
            },
        }
    )


# (body, ∂ D(u)[i,k] / ∂ u[i,k'] for 0-based i, k, k')
CASES = {
    "-s[i]": (lambda o: _neg(_idx("s", o[0])), lambda i, k, kp: -2 * U[i][kp]),
    "-s[i]^2": (lambda o: _neg(_sq(_idx("s", o[0]))), lambda i, k, kp: -2 * S[i] * 2 * U[i][kp]),
    "-s[i]*u[i,k]": (
        lambda o: _neg({"op": "*", "args": [_idx("s", o[0]), _idx("u", *o)]}),
        lambda i, k, kp: -2 * U[i][kp] * U[i][k] - (S[i] if k == kp else 0),
    ),
}


@pytest.mark.parametrize("oidx", [("i", "k"), ("i", "j")])
@pytest.mark.parametrize("case", list(CASES))
def test_contracting_observed_keeps_its_sum(case, oidx):
    body, want = CASES[case]
    j = assemble_jacobian(_doc(body, oidx), states={"u": U})
    expect = {
        (f"u[{i + 1},{k + 1}]", f"u[{i + 1},{kp + 1}]"): want(i, k, kp)
        for i in range(NX)
        for k in range(NZ)
        for kp in range(NZ)
    }
    assert set(j) == set(expect), "pattern: one dense block per column of u"
    for key, val in expect.items():
        assert j[key] == pytest.approx(val, abs=1e-13), key


def _sq_s(o):
    return _neg(_sq(_idx("s", o[0])))


def test_filtered_contracting_observed_keeps_its_window():
    # s[i] = Σ_{k<3} u[i,k]², D(u)[i,k] = -s[i]²: the filter keeps its window
    # (spec §3.5) and the lifted column takes the guard — never dropped.
    s = _sdef(filter={"op": "<", "args": ["k", 3]})
    file = _doc(_sq_s, s=s)
    sf = [sum(U[i][k] ** 2 for k in range(2)) for i in range(NX)]
    _check(
        assemble_jacobian(file, states={"u": U}),
        _expect(lambda i, k, a, b: -2 * sf[i] * 2 * U[i][b] if a == i and b < 2 else 0),
    )
    x = [U[i][k] for i in range(NX) for k in range(NZ)]
    check_against_fd(file, x)


def test_semiring_max_contracting_observed():
    # `semiring` supersedes `reduce`: a max_product node with no `reduce` is
    # a max, not a sum — s[i] = max_k u[i,k], here at k = NZ.
    s = _sdef(semiring="max_product")
    del s["reduce"]
    file = _doc(_sq_s, s=s)
    smax = [max(U[i][k] ** 2 for k in range(NZ)) for i in range(NX)]
    j = assemble_jacobian(file, states={"u": U})
    want = _expect(lambda i, k, a, b: -2 * smax[i] * 2 * U[i][b] if a == i and b == NZ - 1 else 0)
    for key, val in want.items():
        assert j[key] == pytest.approx(val, abs=1e-13), key
    assert all(v == 0 for key, v in j.items() if key not in want)
    x = [U[i][k] for i in range(NX) for k in range(NZ)]
    check_against_fd(file, x)


C = {"c": {"type": "unknown"}}


def _expect(f):
    """{(row, col): value} over D(u)[i,k] for a closed-form ∂/∂u[a,b]
    ``f(i, k, a, b)`` (0-based), nonzeros only."""
    out = {}
    for i in range(NX):
        for k in range(NZ):
            for a in range(NX):
                for b in range(NZ):
                    v = f(i, k, a, b)
                    if v != 0:
                        out[(f"u[{i + 1},{k + 1}]", f"u[{a + 1},{b + 1}]")] = v
    return out


def _check(j, expect):
    assert set(j) == set(expect)
    for key, val in expect.items():
        assert j[key] == pytest.approx(val, abs=1e-13), key


def test_scalar_sum_observed_bare_read():
    # c = Σ_k u[1,k], D(u)[i,k] = -c·u[i,k]
    c = sum(U[0])
    file = _doc(
        lambda o: _neg(_mul("c", _idx("u", *o))),
        variables=C,
        equations=[{"lhs": "c", "rhs": _ksum(_idx("u", 1, "k"))}],
    )
    _check(
        assemble_jacobian(file, states={"u": U}),
        _expect(lambda i, k, a, b: -(c if (a, b) == (i, k) else 0) - (U[i][k] if a == 0 else 0)),
    )


def test_observed_with_a_sum_below_its_root():
    # c = 2·Σ_k u[2,k]²: spliced as is, then renamed apart from the reader's k
    c = 2 * sum(x**2 for x in U[1])
    file = _doc(
        lambda o: _neg(_mul("c", _idx("u", *o))),
        variables=C,
        equations=[{"lhs": "c", "rhs": _mul(2, _ksum(_sq(_idx("u", 2, "k"))))}],
    )
    _check(
        assemble_jacobian(file, states={"u": U}),
        _expect(
            lambda i, k, a, b: (
                -(c if (a, b) == (i, k) else 0) - (U[i][k] * 4 * U[1][b] if a == 1 else 0)
            )
        ),
    )


def test_hand_written_shadowing_sum():
    # -u[i,k]·Σ_k u[i,k]²: the same site outside and inside a sum that
    # shadows the reader's k
    file = _doc(lambda o: _neg(_mul(_idx("u", *o), _ksum(_sq(_idx("u", "i", "k"))))))
    _check(
        assemble_jacobian(file, states={"u": U}),
        _expect(
            lambda i, k, a, b: 0 if a != i else -(S[i] if b == k else 0) - U[i][k] * 2 * U[i][b]
        ),
    )


def test_index_named_like_an_observed():
    # An index an enclosing faq binds is that index, not a read of the
    # observed of the same name.
    file = _doc(
        lambda o: _neg(_idx("u", *o)),
        variables={"k": {"type": "unknown"}},
        equations=[{"lhs": "k", "rhs": _ksum(_idx("u", 1, "k"))}],
    )
    _check(
        assemble_jacobian(file, states={"u": U}),
        _expect(lambda i, k, a, b: -1 if (a, b) == (i, k) else 0),
    )


def _sums(e, acc):
    if isinstance(e, ExprNode):
        if e.op == "faq" and not (e.output_idx or []):
            acc.append(e)
        map_children(e, lambda x: (_sums(x, acc), x)[1])
    return acc


def _inlined(file):
    sv = sysview(file)
    obs = observed_defs(sv)
    (eq,) = [e for e in sv.equations if lhs_state(e) is not None]
    return eq, obs


def test_repeated_reads_inline_to_one_tree():
    # Two reads of one contracting observed stay structurally equal, so
    # template CSE and hoisting still share the sum.
    eq, obs = _inlined(
        _doc(lambda o: {"op": "-", "args": [_neg(_idx("s", o[0])), _sq(_idx("s", o[0]))]})
    )
    sums = _sums(inline_observed(eq.rhs, obs), [])
    assert len(sums) == 2
    assert skey(sums[0]) == skey(sums[1])
    assert "k" not in sums[0].ranges  # renamed apart from the reader's k


def test_scalar_sum_without_output_idx():
    # The typed AST allows output_idx None (the schema requires the key).
    from dataclasses import replace

    eq, obs = _inlined(
        _doc(
            lambda o: _neg(_mul("c", _idx("u", *o))),
            variables=C,
            equations=[{"lhs": "c", "rhs": _ksum(_idx("u", 1, "k"))}],
        )
    )
    obs2 = dict(obs, c=replace(obs["c"], output_idx=None))
    a, b = inline_observed(eq.rhs, obs), inline_observed(eq.rhs, obs2)
    assert len(_sums(a, [])) == len(_sums(b, [])) == 1
