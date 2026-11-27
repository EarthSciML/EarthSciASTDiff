"""An elementwise observed read at a cell, its array operands aligned by
index-set name (esm-spec §4.3.4). Byte-identical to the Julia reference's
golden."""

import json

import earthsci_ast as ea
import pytest
from earthsci_ast.esm_types import ExprNode
from earthsci_astdiff import BandError, InlineError, jacobian_bands, serialize_band, stable_json
from earthsci_astdiff.bands import Ctx, align_whole
from earthsci_astdiff.system import ctx_of, sysview


def test_elementwise_aligned_bands(valid_dir, golden_dir):
    file = ea.load_path(str(valid_dir / "elementwise_aligned.esm"))
    bands = [serialize_band(e) for e in jacobian_bands(file, "Box")]
    assert stable_json(bands) == (golden_dir / "elementwise_aligned.bands.json").read_text()
    # row c reads q at its own y index and r with its axes swapped
    byc = {b["col"]: b for b in bands if b["row"] == "c"}
    assert byc["q"]["col_idx"] == ["j"]
    assert byc["r"]["col_idx"] == ["j", "i"]
    # row s, a whole-array right-hand side, aligns its operands the same way
    bys = [b for b in bands if b["row"] == "s"]
    assert {tuple(b["col_idx"]) for b in bys if b["col"] == "q"} == {("_r2",)}
    assert [b["col_idx"] for b in bys if b["col"] == "r"] == [["_r2", "_r1"]]


def _variant(valid_dir, edit):
    d = json.loads((valid_dir / "elementwise_aligned.esm").read_text())
    edit(d)
    return ea.load_string(json.dumps(d))


def test_elementwise_aligned_refuses_what_cannot_be_aligned(valid_dir):
    # an operand over an index set the observed is not shaped over
    def foreign(d):
        d["index_sets"]["z"] = {"kind": "interval", "size": 2}
        m = d["models"]["Box"]
        m["variables"]["p"] = {"type": "parameter", "shape": ["z"], "default": 0.1}
        m["equations"][0]["rhs"]["args"].append("p")

    # refused by this binding, or already by an EarthSciAST loader that
    # checks array-level index sets itself (esm-spec §4.3.4)
    with pytest.raises(
        (BandError, InlineError, ea.SchemaValidationError), match=r"index set [`']z[`']"
    ):
        jacobian_bands(_variant(valid_dir, foreign), "Box")

    # an observed declared without a shape that mixes ranks
    def undeclared(d):
        del d["models"]["Box"]["variables"]["w"]["shape"]

    with pytest.raises((BandError, InlineError), match="declared without a shape"):
        jacobian_bands(_variant(valid_dir, undeclared), "Box")


def test_elementwise_aligned_refuses_arity_mixed_ranks_and_whole_arrays(valid_dir):
    # a read whose arity is not the observed's rank
    for ix in (["w", "i"], ["w", "i", "j", "i"]):

        def arity(d, ix=ix):
            d["models"]["Box"]["equations"][1]["rhs"]["expr"]["args"][0]["args"] = ix

        with pytest.raises(
            InlineError, match=f"of rank 2 is read at a cell with {len(ix) - 1} indices"
        ):
            jacobian_bands(_variant(valid_dir, arity), "Box")

    # an observed declared without a shape mixing ranks below its top node
    def nested(d):
        d["index_sets"]["z"] = {"kind": "interval", "size": 2}
        m = d["models"]["Box"]
        m["variables"]["p"] = {"type": "parameter", "shape": ["z"], "default": 0.1}
        m["variables"]["u"] = {"type": "unknown"}
        m["equations"].append(
            {"lhs": "u", "rhs": {"op": "-", "args": [{"op": "*", "args": ["c", "p"]}]}}
        )
        m["equations"][0]["rhs"]["args"].append("u")

    with pytest.raises(
        InlineError, match="observed u, declared without a shape, combines arrays of rank 2 and 1"
    ):
        jacobian_bands(_variant(valid_dir, nested), "Box")

    # a whole-array operand over an index set the row is not shaped over
    def foreign(d):
        d["index_sets"]["z"] = {"kind": "interval", "size": 2}
        m = d["models"]["Box"]
        m["variables"]["p"] = {"type": "parameter", "shape": ["z"], "default": 0.1}
        m["equations"][-1]["rhs"]["args"][0]["args"].append("p")

    # (this binding's loader may already refuse it, by the same rule)
    with pytest.raises(Exception, match=r"index set .z.,? .*is not shaped over"):
        jacobian_bands(_variant(valid_dir, foreign), "Box")


def test_elementwise_aligned_synthetic_lift_set_aligns_by_position(valid_dir):
    # `g` over the pointwise lift's synthetic `_liftdim1_3` (an anonymous
    # dimension of x's extent), read by `h` over [x] at a cell
    def cell(n, x):
        return {
            "op": "faq",
            "output_idx": ["i"],
            "args": [],
            "ranges": {"i": {"from": x}},
            "expr": {"op": "-", "args": [{"op": "index", "args": [n, "i"]}]},
        }

    def lifted(d):
        d["index_sets"]["_liftdim1_3"] = {"kind": "interval", "size": 3}
        m = d["models"]["Box"]
        v = m["variables"]
        v["g"] = {"type": "unknown", "shape": ["_liftdim1_3"], "default": 1.0}
        v["h"] = {"type": "unknown", "shape": ["x"]}
        v["e"] = {"type": "unknown", "shape": ["x"], "default": 1.0}
        m["equations"] += [
            {"lhs": "h", "rhs": {"op": "*", "args": [2, "g"]}},
            {"lhs": {"op": "D", "args": ["g"], "wrt": "t"}, "rhs": cell("g", "_liftdim1_3")},
            {"lhs": {"op": "D", "args": ["e"], "wrt": "t"}, "rhs": cell("h", "x")},
        ]

    # A loader that checks array-level index sets by name refuses this
    # hand-built document (`flatten` writes the synthetic set itself); the
    # positional fallback is then never reached from a document. A check of
    # the loader's behavior, not of a version.
    try:
        f = _variant(valid_dir, lifted)
    except ea.SchemaValidationError as err:
        if "_liftdim1_3" not in str(err):
            raise
        pytest.skip("the installed EarthSciAST loader refuses the hand-built lift document")
    assert "g" not in ctx_of(sysview(f, "Box")).axes
    bands = [serialize_band(e) for e in jacobian_bands(f, "Box")]
    assert [b["col_idx"] for b in bands if b["row"] == "e" and b["col"] == "g"] == [["i"]]


def test_align_whole_refuses_a_foreign_index_set():
    # the band calculus's own refusal, whatever this binding's loader checks
    ctx = Ctx(
        {"x": 3, "y": 2, "z": 2}, {"s": [3, 2], "p": [2]}, set(), {"s": ["x", "y"], "p": ["z"]}
    )
    with pytest.raises(BandError, match="whose index set `z` `s` is not shaped over"):
        align_whole(ExprNode(op="+", args=["s", "p"]), "s", ctx)
