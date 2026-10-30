"""Second-order bands (spec §4.4) against the Julia reference's goldens:
`<fixture>.<a1>-<a2>.{bands,block}.json` byte-identical, and the hvp
contraction against `<fixture>.<a1>-<a2>.hvals.json` (exact key set, values
to 1e-10 relative)."""

import json

import earthsci_ast as ea
import pytest
from earthsci_ast.expression import free_variables
from earthsci_astdiff import (
    BandError,
    JacEntry2,
    hvp,
    jacobian_bands2,
    jacobian_block,
    parse_jacobian_block,
    parse_jacobian_blocks,
    prepare_hvp,
    serialize_band,
    stable_json,
    sysview,
)
from earthsci_astdiff.expr_helpers import ser_expr
from test_values import _cells_to_arrays

SECOND_ORDER = (
    ("bd_chem", ("states", "states")),
    ("arrhenius_chem", ("states", "states")),
    ("arrhenius_chem", ("states", "parameters")),
    ("forcing_bracket", ("states", "forcing")),
    ("forcing_bracket", ("time", "states")),
    ("minmod_adv", ("states", "states")),
)


def _load(valid_dir, name):
    file = ea.load_path(str(valid_dir / f"{name}.esm"))
    return file, next(iter(file.models))


@pytest.mark.parametrize("name,wrt", SECOND_ORDER)
def test_second_order_goldens(name, wrt, valid_dir, golden_dir):
    file, model = _load(valid_dir, name)
    tag = f"{wrt[0]}-{wrt[1]}"
    entries = jacobian_bands2(file, model, wrt=wrt)
    bands = stable_json([serialize_band(en) for en in entries])
    assert bands == (golden_dir / f"{name}.{tag}.bands.json").read_text()
    block = jacobian_block(file, model, wrt=wrt)
    assert block["wrt"] == list(wrt)
    assert "factorization" not in block
    assert ("columns" in block) == (wrt[1] in ("parameters", "forcing"))
    assert stable_json(block) == (golden_dir / f"{name}.{tag}.block.json").read_text()
    w, back = parse_jacobian_block(block)
    assert w == wrt
    assert stable_json([serialize_band(en) for en in back]) == bands


def _triples(entries):
    return {(en.u, en.v, en.w) for en in entries}


def test_what_the_entries_are(valid_dir):
    # bd_chem's bimolecular terms: k3·B·C gives both orientations of (B, C)
    # in A's row, and B² a single (B, B) entry; the linear k1 term none
    file, model = _load(valid_dir, "bd_chem")
    assert _triples(jacobian_bands2(file, model)) == {
        ("A", "B", "C"),
        ("A", "C", "B"),
        ("B", "B", "B"),
        ("C", "B", "B"),
    }
    # every orientation comes out on its own: (r, c, s) ⇒ (r, s, c)
    for name in ("bd_chem", "arrhenius_chem"):
        t = _triples(jacobian_bands2(*_load(valid_dir, name)))
        assert all((r, s, c) in t for r, c, s in t)
    # the Arrhenius rate is an observed of parameters only: opaque on the
    # state pair, inlined on the parameter pair so T's sites are seen
    file, model = _load(valid_dir, "arrhenius_chem")
    ss = jacobian_bands2(file, model)
    assert any("k1" in free_variables(en.band.coef) for en in ss)
    sp = jacobian_bands2(file, model, wrt=("states", "parameters"))
    assert any(en.w == "T" and [ser_expr(c) for c in en.col2_idx] == ["i"] for en in sp)
    assert all("k1" not in free_variables(en.band.coef) for en in sp)
    # minmod's limiter is piecewise linear: the second-order pattern is
    # empty (§3.2)
    assert jacobian_bands2(*_load(valid_dir, "minmod_adv")) == []
    # the bracket's records stay distinct second columns
    file, model = _load(valid_dir, "forcing_bracket")
    fb = jacobian_bands2(file, model, wrt=("states", "forcing"))
    assert any(en.w == "G" and en.band.contracted for en in fb)
    assert {stable_json(ser_expr(en.col2_idx[0])) for en in fb if en.u == "c" and en.w == "F"} == {
        "1",
        "2",
    }


def test_structure(valid_dir):
    def blk(name, wrt):
        return jacobian_block(*_load(valid_dir, name), wrt=wrt)

    assert blk("bd_chem", ("states", "states"))["structure"] == "block_diagonal"
    assert blk("arrhenius_chem", ("states", "states"))["structure"] == "block_diagonal"
    # a scalar parameter column is shared by every lane: not lane-local
    assert "structure" not in blk("arrhenius_chem", ("states", "parameters"))
    assert "structure" not in blk("forcing_bracket", ("time", "states"))


def _golden_point(golden_dir, name, wrt):
    gd = json.loads((golden_dir / f"{name}.{wrt[0]}-{wrt[1]}.hvals.json").read_text())
    assert gd["wrt"] == list(wrt)
    pairs = [(pr["mu"], pr["w"]) for pr in gd["pairs"]]
    return gd, pairs, _cells_to_arrays(gd["u"])


@pytest.mark.parametrize("name,wrt", SECOND_ORDER)
def test_hvp_value_goldens(name, wrt, valid_dir, golden_dir):
    gd, pairs, states = _golden_point(golden_dir, name, wrt)
    hv = prepare_hvp(*_load(valid_dir, name), wrt=wrt)
    g = hvp({}, hv, pairs, states=states, t=gd["t"])
    want = dict(gd["g"])
    assert set(g) == set(want)  # every reached second column, exactly
    for k, v in want.items():
        assert g[k] == pytest.approx(v, rel=1e-10, abs=1e-13), k
    # one pair at a time accumulates to the same
    g1: dict = {}
    for pr in pairs:
        hvp(g1, hv, pr, states=states, t=gd["t"])
    assert set(g1) == set(g)
    for k, v in g.items():
        assert g1[k] == pytest.approx(v, rel=1e-12, abs=1e-14), k
    if wrt == ("states", "states"):
        # ∇²(μᵀf) is symmetric: scattering by the first column gives the
        # same gradient on every column either form reaches
        gs = hvp({}, hv, pairs, states=states, t=gd["t"], form="symmetric")
        assert bool(gs) == bool(want)
        for k in set(gs) | set(want):
            assert gs.get(k, 0.0) == pytest.approx(want.get(k, 0.0), rel=1e-10, abs=1e-13), k
    if hv.col2names is not None:
        # the flat form over the canonical second-axis ordinal
        flat = hvp([0.0] * len(hv.col2names), hv, pairs, states=states, t=gd["t"])
        for n, v in zip(hv.col2names, flat):
            assert v == pytest.approx(want.get(n, 0.0), rel=1e-10, abs=1e-13), n


def test_document_forms(valid_dir, golden_dir):
    file, model = _load(valid_dir, "forcing_bracket")
    jb = jacobian_block(file, model, wrt=["states", ("states", "forcing"), ["time", "states"]])
    assert [b["wrt"] for b in jb] == ["states", ["states", "forcing"], ["time", "states"]]
    assert (
        stable_json(jb[1]) == (golden_dir / "forcing_bracket.states-forcing.block.json").read_text()
    )
    parsed = parse_jacobian_blocks(jb)
    assert [w for w, _ in parsed] == ["states", ("states", "forcing"), ("time", "states")]
    assert all(isinstance(en, JacEntry2) for en in parsed[1][1])
    with pytest.raises(ValueError):
        jacobian_block(file, model, wrt=[("states", "states"), ["states", "states"]])


def test_refusals(valid_dir):
    sv = sysview(*_load(valid_dir, "forcing_bracket"))
    for bad in (("forcing", "states"), ("states", "time"), ("states",)):
        with pytest.raises(ValueError):
            jacobian_bands2(sv, wrt=bad)
        with pytest.raises(ValueError):
            prepare_hvp(sv, wrt=bad)
    hv = prepare_hvp(*_load(valid_dir, "bd_chem"))
    ones = {n: 1.0 for n in ("A", "B", "C")}
    with pytest.raises(ValueError):
        hvp({}, hv, (ones, ones), form="nope")
    with pytest.raises(ValueError):
        hvp({}, hv, ({"A": 1.0}, ones))  # μ misses a row a point needs
    with pytest.raises(ValueError):
        hvp([0.0], hv, (ones, ones))  # a states second axis has no flat form
    with pytest.raises(ValueError):
        hvp({}, hv, (ones, ones), wrt=("states", "parameters"))
    ht = prepare_hvp(sv, wrt=("time", "states"))
    mu = {n: 1.0 for n in ("S", "c[1]", "c[2]", "c[3]", "c[4]")}
    with pytest.raises(ValueError):
        hvp({}, ht, (mu, 1.0), t=0.37, form="symmetric")
    with pytest.raises(ValueError):
        hvp({}, ht, (mu, mu), t=0.37)  # w is a number on a time pair
    hp = prepare_hvp(sv, wrt=("states", "forcing"))
    with pytest.raises(ValueError):
        hvp([0.0] * (len(hp.col2names) + 1), hp, (mu, mu), t=0.37)


def test_unclosed_second_column_on_forcing_is_refused():
    # a second column read through a parameter offset cannot be placed
    def ix(v, *i):
        return {"op": "index", "args": [v, *i]}

    body = {
        "op": "*",
        "args": [ix("c", "i"), ix("c", "i"), ix("F", 1, {"op": "+", "args": ["i", "s"]})],
    }
    doc = ea.load_string(
        json.dumps(
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
                        "variables": {
                            "c": {"type": "unknown", "shape": ["x"], "default": 0.5},
                            "F": {"type": "parameter", "shape": ["rec", "xf"], "default": 1.0},
                            "s": {"type": "parameter", "default": 1},
                        },
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
    )
    with pytest.raises(BandError, match="second column index"):
        jacobian_bands2(sysview(doc, "B"), wrt=("states", "forcing"), forcing=["F"])
