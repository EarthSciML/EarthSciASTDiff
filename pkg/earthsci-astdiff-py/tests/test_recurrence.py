"""Causal recurrences (spec §4.5, §9.7), mirroring the Rust binding's
tests/recurrence.rs. The ``recurrence_sweep`` goldens (written by the Rust
binding, tests/write_goldens.rs) are checked against numbers no band of this
package produced: central differences of the SOURCE right-hand side,
evaluated by EarthSciAST, which executes the recurrences itself. The three
derivations of ``J`` — the materialized bands through the transfer
recurrence (the evaluation document), the tangent sweeps (``JvpEval``) and
the adjoint sweeps (``VjpEval``) — agree to rounding, and the second-order
contraction with central differences of ``Σ μᵢᵀ J wᵢ``."""

from __future__ import annotations

import json
import math

import earthsci_ast as ea
import pytest
from docvals import _Doc, _doc_cell, doc_hvp, doc_jacobian, doc_vjp, state_cells
from earthsci_astdiff import (
    BandError,
    ctx_of,
    jacobian_bands,
    jacobian_block,
    jvp_document,
    recurrences,
    sysview,
)
from earthsci_astdiff.emit import _cellname, _evaluation_document, foreach_point
from earthsci_astdiff.system import observed_defs

NAME = "recurrence_sweep"


def _load(valid_dir, name=NAME):
    return ea.load_path(str(valid_dir / f"{name}.esm"))


def _source(valid_dir, name=NAME):
    sv = sysview(_load(valid_dir, name))
    raw = json.loads((valid_dir / f"{name}.esm").read_text())
    return _Doc(raw, next(iter(raw["models"])), state_cells(sv)), state_cells(sv)


def _jvals(golden_dir, tag):
    g = json.loads((golden_dir / f"{tag}.jvals.json").read_text())
    return g, {(r, c): v for r, c, v in g["entries"]}


def test_jacobian_matches_central_differences_of_the_source(valid_dir, golden_dir):
    g, want = _jvals(golden_dir, NAME)
    src, cells = _source(valid_dir)
    h = 1e-6
    for col in cells:
        up = src.call({**g["u"], col: g["u"][col] + h}, g["params"], g["t"], {})
        dn = src.call({**g["u"], col: g["u"][col] - h}, g["params"], g["t"], {})
        for row in cells:
            fd = (up(row) - dn(row)) / (2 * h)
            if (row, col) in want:
                assert want[(row, col)] == pytest.approx(fd, rel=1e-7, abs=1e-7), (row, col)
            else:
                assert abs(fd) < 1e-9, (row, col)


def test_parameter_jacobian_matches_central_differences_of_the_source(valid_dir, golden_dir):
    g, want = _jvals(golden_dir, f"{NAME}.parameters")
    src, cells = _source(valid_dir)
    h = 1e-6
    eps = g["params"]["eps"]
    up = src.call(g["u"], {**g["params"], "eps": eps + h}, g["t"], {})
    dn = src.call(g["u"], {**g["params"], "eps": eps - h}, g["t"], {})
    for row in cells:
        fd = (up(row) - dn(row)) / (2 * h)
        if (row, "eps") in want:
            assert want[(row, "eps")] == pytest.approx(fd, rel=1e-7, abs=1e-7), row
        else:
            assert abs(fd) < 1e-9, row


def _materialized(valid_dir, wrt, g):
    # J assembled from the evaluation document's band fields (the
    # materialized bands through the transfer recurrence, spec §4.5)
    sv = sysview(_load(valid_dir))
    ctx = ctx_of(sv)
    entries = jacobian_bands(sv, wrt=wrt)
    doc, names = _evaluation_document(sv, entries)
    d = _Doc(doc, "JacobianEval", state_cells(sv))
    out = d.call(g["u"], g["params"], g["t"], {})
    J: dict = {}
    for en, nm in zip(entries, names, strict=True):
        shape = ctx.shapes.get(en.v) or []
        for cell, cc, _, cidx in foreach_point(en.band):
            if any(not (1 <= c <= n) for c, n in zip(cidx, shape, strict=False)):
                continue
            key = (_cellname(en.u, cell), _cellname(en.v, cidx))
            J[key] = J.get(key, 0.0) + out(_cellname(nm, [*cell, *cc]))
    return J


@pytest.mark.parametrize("wrt,tag", [("states", NAME), ("parameters", f"{NAME}.parameters")])
def test_the_three_derivations_agree(wrt, tag, valid_dir, golden_dir):
    g, _ = _jvals(golden_dir, tag)
    file = _load(valid_dir)
    sweep = doc_jacobian(file, wrt=wrt, states=g["u"], params=g["params"], t=g["t"]).J
    mat = _materialized(valid_dir, wrt, g)
    assert set(sweep) == set(mat)
    for k, v in sweep.items():
        assert mat[k] == pytest.approx(v, rel=1e-12, abs=1e-14), k
    # ⟨J v, λ⟩ = ⟨v, Jᵀ λ⟩, the tangent sweeps against the adjoint ones
    sv = sysview(file)
    rows = state_cells(sv)
    cols = rows if wrt == "states" else ["eps"]
    lam = {r: math.sin(0.7 * n + 0.3) for n, r in enumerate(rows)}
    v = {c: math.cos(0.4 * n + 1.1) for n, c in enumerate(cols)}
    vjp = doc_vjp(file, lam, wrt=wrt, states=g["u"], params=g["params"], t=g["t"])
    doc, io = jvp_document(file, wrt=wrt)
    d = _Doc(doc, io["model"], rows)
    out = d.call(g["u"], g["params"], g["t"], {_doc_cell(io["v"], c): x for c, x in v.items()})
    jv_lam = sum(out(_doc_cell(io["jvp"], r)) * lam[r] for r in rows)
    v_jtl = sum(v[c] * vjp[c] for c in cols)
    assert jv_lam == pytest.approx(v_jtl, rel=1e-12)


@pytest.mark.parametrize("wrt", [("states", "states"), ("states", "parameters")])
def test_contraction_matches_central_differences(wrt, valid_dir, golden_dir):
    gd = json.loads((golden_dir / f"{NAME}.{wrt[0]}-{wrt[1]}.hvals.json").read_text())
    pairs = [(p["mu"], p["w"]) for p in gd["pairs"]]
    file = _load(valid_dir)
    sv = sysview(file)
    rows = state_cells(sv)
    doc, io = jvp_document(file)
    d = _Doc(doc, io["model"], rows)

    def phi(states, params):
        tot = 0.0
        for mu, w in pairs:
            out = d.call(states, params, gd["t"], {_doc_cell(io["v"], c): x for c, x in w.items()})
            tot += sum(m * out(_doc_cell(io["jvp"], r)) for r, m in mu.items())
        return tot

    h = 1e-5
    for c, gv in gd["g"]:
        if wrt[1] == "states":
            fd = (
                phi({**gd["u"], c: gd["u"][c] + h}, {}) - phi({**gd["u"], c: gd["u"][c] - h}, {})
            ) / (2 * h)
        else:
            p0 = 0.8  # eps at its document default
            fd = (phi(gd["u"], {c: p0 + h}) - phi(gd["u"], {c: p0 - h})) / (2 * h)
        assert gv == pytest.approx(fd, rel=1e-7, abs=1e-7), c
    if wrt == ("states", "states"):
        a = doc_hvp(file, pairs, wrt=wrt, states=gd["u"], t=gd["t"])
        b = doc_hvp(file, pairs, wrt=wrt, form="symmetric", states=gd["u"], t=gd["t"])
        for k, v in a.items():
            assert b[k] == pytest.approx(v, rel=1e-12, abs=1e-14), k


# ── small documents ──────────────────────────────────────────────────────────


def _o(op, *a):
    return {"op": op, "args": list(a)}


def _ix(v, *a):
    return _o("index", v, *a)


def _recdoc(body):
    # One state x[lev], one recurrence V with body `body` (over k),
    # D(x)[k] = V[k].
    return {
        "esm": "1.1.0",
        "metadata": {"name": "R"},
        "index_sets": {"lev": {"kind": "interval", "size": 4}},
        "models": {
            "R": {
                "variables": {
                    "x": {"type": "unknown", "shape": ["lev"], "default": 0.5},
                    "V": {"type": "unknown", "shape": ["lev"]},
                },
                "equations": [
                    {
                        "lhs": "V",
                        "rhs": {
                            "op": "faq",
                            "args": [],
                            "output_idx": ["k"],
                            "ranges": {"k": {"from": "lev"}},
                            "expr": body,
                        },
                    },
                    {
                        "lhs": {"op": "D", "args": ["x"], "wrt": "t"},
                        "rhs": {
                            "op": "faq",
                            "args": [],
                            "output_idx": ["i"],
                            "ranges": {"i": {"from": "lev"}},
                            "expr": _ix("V", "i"),
                        },
                    },
                ],
            }
        },
    }


def test_a_running_max_tie_takes_the_earlier_argument():
    # V[k] = max(V[k-1], x[k]): at an exact tie V[k-1] == x[k] the
    # derivative follows the earlier argument, the carried maximum (spec
    # §3.2's left fold)
    body = _o(
        "ifelse", _o("<=", "k", 1), _ix("x", 1), _o("max", _ix("V", _o("-", "k", 1)), _ix("x", "k"))
    )
    file = ea.load_document(_recdoc(body))
    x = {"x[1]": 0.7, "x[2]": 0.4, "x[3]": 0.7, "x[4]": 0.9}
    J = doc_jacobian(file, states=x).J
    # V = [0.7, 0.7, 0.7 (tie), 0.9]
    assert J[("x[3]", "x[1]")] == 1.0
    assert J.get(("x[3]", "x[3]"), 0.0) == 0.0
    assert J[("x[4]", "x[4]")] == 1.0
    assert J.get(("x[4]", "x[1]"), 0.0) == 0.0


@pytest.mark.parametrize(
    "why,body",
    [
        ("same cell", _o("+", _ix("x", "k"), _ix("V", "k"))),
        ("bare self-reference", _o("+", _ix("x", "k"), "V")),
        ("not along a frame axis", _o("+", _ix("x", "k"), _ix("V", 1, "k"))),
        # a read of a later cell: its lag is -1, never "≥ 1", so its sweep
        # term would be gated to a silent zero
        (
            "a later cell",
            _o("ifelse", _o(">=", "k", 4), _ix("x", "k"), _ix("V", _o("+", "k", 1))),
        ),
    ],
)
def test_ill_founded_recurrences_are_refused(why, body):
    # EarthSciAST's Python loader refuses these documents itself; this
    # package must refuse them too, so the ill-founded body is put in place
    # after a well-founded document loads
    from dataclasses import replace

    from earthsci_ast.parse import _parse_expression

    good = _o("ifelse", _o("<=", "k", 1), _ix("x", 1), _ix("V", _o("-", "k", 1)))
    file = ea.load_document(_recdoc(good))
    model = file.models["R"]
    eq = model.equations[0]
    bad = _parse_expression(_recdoc(body)["models"]["R"]["equations"][0]["rhs"])
    model.equations[0] = replace(eq, rhs=bad)
    with pytest.raises(BandError, match="recurrence `V`"):
        jacobian_bands(file)


def test_recognition(valid_dir, golden_dir):
    sv = sysview(_load(valid_dir))
    recs = recurrences(sv, observed_defs(sv), ctx_of(sv))
    assert [(r.name, r.dim) for r in recs] == [
        ("bet", 1),
        ("uf", 1),
        ("yrev", 1),
        ("p", 1),
        ("phi", 1),
        ("r", 2),
    ]
    block = json.loads((golden_dir / f"{NAME}.column-lev.block.json").read_text())
    assert block["structure"] == "column_block"
    assert block["factorization"]["pattern"] == "dense"


def test_rank_one_plan(valid_dir):
    file = _load(valid_dir, "pbl_rank1")
    b = jacobian_block(file, column_axis="lev")
    f = b["factorization"]
    assert f["pattern"] == "diag_plus_rank1"
    terms = [t["entry"] for t in f["rank1"]]
    assert len(terms) == 2
    for t in terms:
        en = b["entries"][t - 1]
        assert en["row"] == en["col"] and len(en["contracted"]) == 1
    assert jacobian_block(file)["structure"] == "general"


def _lifted(u, lo, hi, rhs):
    # `faq{D(u[i])} ~ rhs` over i in [lo, hi]: one equation of an array state
    return {
        "lhs": {
            "op": "faq",
            "args": [],
            "output_idx": ["i"],
            "ranges": {"i": [lo, hi]},
            "expr": _o("D", _ix(u, "i")) | {"wrt": "t"},
        },
        "rhs": rhs,
    }


def _sweep(v, src, a):
    # `v[k] = src[k] + a·v[k − 1]` (src[1] at k = 1)
    return {
        "lhs": v,
        "rhs": {
            "op": "faq",
            "args": [],
            "output_idx": ["k"],
            "ranges": {"k": {"from": "lev"}},
            "expr": _o(
                "ifelse",
                _o("<=", "k", 1),
                _ix(src, 1),
                _o("+", _ix(src, "k"), _o("*", a, _ix(v, _o("-", "k", 1)))),
            ),
        },
    }


def _model(variables, equations):
    return {
        "esm": "1.1.0",
        "metadata": {"name": "R"},
        "index_sets": {"lev": {"kind": "interval", "size": 4}},
        "models": {"R": {"variables": variables, "equations": equations}},
    }


def test_transfer_names_do_not_depend_on_activity():
    # `A` (inactive on the state axis: it reads only `p`) sorts before `A_`,
    # and the source already holds `_tr_A`: each recurrence's transfer name
    # is the same in the coefficients and in the `auxiliary` field, so the
    # evaluation document's J is A_'s chain 0.5^(k − j), not A's 0.25^(k − j)
    lev = {"type": "unknown", "shape": ["lev"]}
    doc = _model(
        {
            "x": lev | {"default": 0.5},
            "p": {"type": "parameter", "shape": ["lev"], "default": 1.0},
            "_tr_A": {"type": "parameter", "default": 0.0},
            "A": lev,
            "A_": lev,
        },
        [
            _sweep("A", "p", 0.25),
            _sweep("A_", "x", 0.5),
            {
                "lhs": {"op": "D", "args": ["x"], "wrt": "t"},
                "rhs": {
                    "op": "faq",
                    "args": [],
                    "output_idx": ["i"],
                    "ranges": {"i": {"from": "lev"}},
                    "expr": _o("+", _ix("A_", "i"), _ix("A", "i")),
                },
            },
        ],
    )
    sv = sysview(ea.load_document(doc))
    block = jacobian_block(sv)
    read = set()
    for en in block["entries"]:
        read |= set(json.dumps(en["coef"]).replace('"', " ").split())
    aux = [a["name"] for a in block["auxiliary"]]
    assert aux and set(aux) <= read
    ctx = ctx_of(sv)
    entries = jacobian_bands(sv)
    edoc, names = _evaluation_document(sv, entries)
    cells = state_cells(sv)
    out = _Doc(edoc, "JacobianEval", cells).call({c: 0.5 for c in cells}, {}, 0.0, {})
    J: dict = {}
    for en, nm in zip(entries, names, strict=True):
        shape = ctx.shapes.get(en.v) or []
        for cell, cc, _, cidx in foreach_point(en.band):
            if any(not (1 <= c <= n) for c, n in zip(cidx, shape, strict=False)):
                continue
            key = (_cellname(en.u, cell), _cellname(en.v, cidx))
            J[key] = J.get(key, 0.0) + out(_cellname(nm, [*cell, *cc]))
    for k in range(1, 5):
        for j in range(1, 5):
            want = 0.5 ** (k - j) if j <= k else 0.0
            assert J.get((f"x[{k}]", f"x[{j}]"), 0.0) == pytest.approx(want, rel=1e-12), (k, j)


def test_entries_keep_each_equation_of_a_state_apart():
    # two equations of `x` (its first level, then the rest): the second's
    # direct band on `x` must not move ahead of the first's band on `y`
    lev = {"type": "unknown", "shape": ["lev"], "default": 0.5}
    doc = _model(
        {"x": lev, "y": lev, "V": {"type": "unknown", "shape": ["lev"]}},
        [
            _sweep("V", "y", 0.5),
            _lifted("x", 1, 1, _ix("y", "i")),
            _lifted("x", 2, 4, _o("+", _o("-", _ix("x", "i")), _ix("V", "i"))),
        ],
    )
    cols = [en.v for en in jacobian_bands(ea.load_document(doc)) if en.u == "x"]
    # the first equation's `y`, then the second's `x` and its composites on
    # `y` (merging the two would sort the second's `x` first)
    assert cols[0] == "y" and cols[1] == "x" and set(cols[2:]) == {"y"} and len(cols) > 2
