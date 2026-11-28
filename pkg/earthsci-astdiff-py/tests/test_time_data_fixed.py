"""Time bands with data held fixed (spec §4.2): only observed that move with
``t`` are followed, and an observed declared without a shape is read at a
cell. Byte-identical to the Julia reference's golden. And the time
companions (``tgrad_named_steps``): the chain through the moving observed
kept as named steps, checked against a central difference in ``t`` of the
source right-hand side."""

import json

import earthsci_ast as ea
import pytest
from earthsci_astdiff import (
    BandError,
    InlineError,
    jacobian_bands,
    jacobian_bands2,
    jacobian_block,
    serialize_band,
    stable_json,
)
from earthsci_astdiff.expr_helpers import ser_expr


def test_tgrad_data_fixed_time_bands(valid_dir, golden_dir):
    file = ea.load_path(str(valid_dir / "tgrad_data_fixed.esm"))
    entries = jacobian_bands(file, "Box", wrt="time")
    assert len(entries) == 1
    text = stable_json([serialize_band(e) for e in entries])
    assert text == (golden_dir / "tgrad_data_fixed.time.bands.json").read_text()


def _variant(valid_dir, edit):
    """tgrad_data_fixed with ``edit`` applied to the ``Box`` model."""
    d = json.loads((valid_dir / "tgrad_data_fixed.esm").read_text())
    edit(d["models"]["Box"])
    return ea.load_string(json.dumps(d))


_COS_T = {"op": "cos", "args": [{"op": "*", "args": ["omega", "t"]}]}
_JR_T = {"op": "*", "args": ["J0", _COS_T]}


def test_time_pair_follows_what_its_second_axis_seeds(valid_dir):
    # kc = c·c moves only with the states: the plain time pass leaves
    # `index(kc, i, j)` an opaque read, the (time, states) pair follows it.
    # Each pair keeps its own pass, so the order of emission is immaterial.
    def edit(m):
        m["variables"]["kc"] = {"type": "unknown"}
        m["equations"].append({"lhs": "kc", "rhs": {"op": "*", "args": ["c", "c"]}})
        body = m["equations"][3]["rhs"]["expr"]
        m["equations"][3]["rhs"]["expr"] = {
            "op": "+",
            "args": [
                body,
                {"op": "*", "args": [{"op": "index", "args": ["kc", "i", "j"]}, _COS_T]},
            ],
        }

    file = _variant(valid_dir, edit)
    assert jacobian_bands2(file, "Box", ("time", "states"))
    pair = jacobian_block(file, "Box", wrt=[("time", "states")])
    time = jacobian_block(file, "Box", wrt=["time"])
    both = jacobian_block(file, "Box", wrt=["time", ("time", "states")])
    one = lambda b: b[0] if isinstance(b, list) else b  # noqa: E731
    assert both[1] == one(pair)
    assert both[0] == one(time)


def test_elementwise_read_at_a_cell_refuses_what_it_cannot_read(valid_dir):
    # a partial index (still an array) inside the definition
    def partial(m):
        m["equations"][0]["rhs"] = {"op": "+", "args": [_JR_T, {"op": "index", "args": ["Eb", 1]}]}

    with pytest.raises((BandError, InlineError)):
        jacobian_bands(_variant(valid_dir, partial), "Box", wrt="time")

    # a global named like the read site's index `j`: the read is a column
    # (spec §4.2, companions), not inlined, and the companion row of `jr`
    # names its own indices clear of `j`, so nothing is captured
    def capture(m):
        m["variables"]["j"] = {"type": "parameter", "default": 2.0}
        m["equations"][0]["rhs"] = {"op": "*", "args": [_JR_T, "j"]}

    cap = _variant(valid_dir, capture)
    assert len(jacobian_bands(cap, "Box", wrt="time")) == 1
    rhs = jacobian_block(cap, "Box", wrt="time")["intermediates"][0]["rhs"]
    assert rhs["output_idx"] == ["_r1", "_r2"]
    assert '"j"' in stable_json(rhs)

    # the full index of an array inside the definition is a scalar
    def full(m):
        m["equations"][0]["rhs"] = {
            "op": "*",
            "args": [_JR_T, {"op": "index", "args": ["Eb", 1, 2]}],
        }

    assert len(jacobian_bands(_variant(valid_dir, full), "Box", wrt="time")) == 1


def _companion_reads(e) -> list:
    """The names ``_dt_<o>`` a serialized expression holds, sorted."""
    acc = set()

    def walk(x):
        if isinstance(x, str):
            if x.startswith("_dt_"):
                acc.add(x)
        elif isinstance(x, list):
            for y in x:
                walk(y)
        elif isinstance(x, dict):
            for y in x.values():
                walk(y)

    walk(e)
    return sorted(acc)


def test_companions_keep_the_chain_as_named_steps(valid_dir, golden_dir):
    # tgrad_named_steps (spec §4.2): w(t) → Tb → kr → c (at its cell) and s
    # (under a sum over k). The bands read `_dt_kr`; each companion's
    # definition reads only the next one down the chain; Eb (data and a
    # parameter weight) gets none.
    file = ea.load_path(str(valid_dir / "tgrad_named_steps.esm"))
    entries = jacobian_bands(file, "Column", wrt="time")
    assert [(en.u, en.v) for en in entries] == [("c", "t"), ("s", "t")]
    bands = [serialize_band(e) for e in entries]
    assert _companion_reads(bands[0]["coef"]) == ["_dt_kr"]
    assert bands[1]["coef"]["op"] == "faq"
    assert stable_json(bands) == (golden_dir / "tgrad_named_steps.time.bands.json").read_text()
    comps = jacobian_block(file, "Column", wrt="time")["intermediates"]
    assert [c["name"] for c in comps] == ["_dt_Tb", "_dt_kr", "_dt_w"]
    assert [c["of"] for c in comps] == ["Tb", "kr", "w"]
    assert _companion_reads(comps[1]["rhs"]) == ["_dt_Tb"]
    assert _companion_reads(comps[0]["rhs"]) == ["_dt_w"]
    assert _companion_reads(comps[2]["rhs"]) == []
    # a time pair takes `t` second (spec §4.4): the time bands of each state
    # entry's coefficient, through the same companions, never inlined
    e2 = jacobian_bands2(file, "Column", ("time", "states"))
    assert e2
    for en in e2:
        assert en.v == "t"
        assert _companion_reads(ser_expr(en.band.coef)) != []
    (pblk,) = jacobian_block(file, "Column", wrt=[("time", "states")])
    assert [c["name"] for c in pblk["intermediates"]] == ["_dt_Tb", "_dt_kr", "_dt_w"]


def test_companion_extents_are_read_or_refused(valid_dir):
    # operands over different index sets: an undeclared shape is their union
    # by index-set name, in order of first appearance (esm-spec §4.3.4):
    # `jr = Ea·cos(…) + Eb` spans [x, y], and its companion is read at the
    # row's cell
    def mixed(m):
        m["equations"][0]["rhs"] = {"op": "+", "args": [{"op": "*", "args": ["Ea", _COS_T]}, "Eb"]}

    es = jacobian_bands(_variant(valid_dir, mixed), "Box", wrt="time")
    assert '"_dt_jr","i","j"' in stable_json([serialize_band(e) for e in es])
    # a band whose column index does not read every dimension of the observed
    from earthsci_astdiff.bands import Band
    from earthsci_astdiff.companions import _through_companion

    with pytest.raises(BandError):
        _through_companion(Band([(1, 3)], ["i"], ["i"], 1, [], None), "jr", "_dt_jr", [3, 2], True)
    # and in a time pair: the first `n` column indices are the pair's own
    from earthsci_astdiff.companions import _pair_through_companion

    b = Band([(1, 3)], ["i"], ["i"], 1, [], None)
    with pytest.raises(BandError):
        _pair_through_companion(b, 0, "jr", "_dt_jr", [3, 2])
    assert _pair_through_companion(b, 0, "jr", "_dt_jr", [3]).cidx == []


def test_moving_observed_that_read_each_other_are_refused(valid_dir):
    # EarthSciAST's loader refuses an observed cycle; the companion pass
    # refuses one too (a companion reached again while its bands are being
    # built), never recursing without end
    from earthsci_astdiff import sysview
    from earthsci_astdiff.companions import companion_bands, companion_setup
    from earthsci_astdiff.jacobian import declared_scalars
    from earthsci_astdiff.system import ctx_of, observed_defs

    sv = sysview(ea.load_path(str(valid_dir / "tgrad_named_steps.esm")))
    obs = observed_defs(sv)
    cs = companion_setup(sv, ctx_of(sv), obs, declared_scalars(sv, obs), True)
    cs.pending.add("Tb")
    with pytest.raises(BandError, match="cyclic"):
        companion_bands(cs, "kr")


def test_companion_name_is_clear_of_the_source(valid_dir):
    d = json.loads((valid_dir / "tgrad_named_steps.esm").read_text())
    d["models"]["Column"]["variables"]["_dt_w"] = {"type": "parameter", "default": 1.0}
    file = ea.load_string(json.dumps(d))
    comps = jacobian_block(file, "Column", wrt="time")["intermediates"]
    assert [c["name"] for c in comps] == ["_dt_Tb", "_dt_kr", "_dt_w_"]
    # a component path is taken too: `P._dt_v` names a subsystem of a source
    # with a variable `P._dt_v.x`
    from earthsci_astdiff.companions import component_paths

    assert component_paths(set(), ["P._dt_v.x", "w"]) == {"P", "P._dt_v"}


def test_companions_match_a_central_difference_in_t(valid_dir, golden_dir):
    # The composed f_t at the value golden's point equals the derivative in
    # t of the source right-hand side, which EarthSciAST evaluates.
    from docvals import _Doc, doc_jacobian, state_cells
    from earthsci_astdiff import sysview

    file = ea.load_path(str(valid_dir / "tgrad_named_steps.esm"))
    raw = json.loads((valid_dir / "tgrad_named_steps.esm").read_text())
    src = _Doc(raw, "Column", state_cells(sysview(file)))
    g = json.loads((golden_dir / "tgrad_named_steps.time.jvals.json").read_text())
    h = 1e-5
    up = src.call(g["u"], g["params"], g["t"] + h, {})
    dn = src.call(g["u"], g["params"], g["t"] - h, {})
    assert len(g["entries"]) == 12
    # this binding's own bands and companions, through its own JvpEval
    # document (not the reference's numbers)
    mine = doc_jacobian(file, "Column", wrt="time", states=g["u"], params=g["params"], t=g["t"]).J
    for row, _, j in g["entries"]:
        fd = (up(row) - dn(row)) / (2 * h)
        assert j == pytest.approx(fd, rel=1e-6, abs=1e-12), row
        assert mine[(row, "t")] == pytest.approx(fd, rel=1e-6, abs=1e-12), row
