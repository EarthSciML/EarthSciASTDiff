"""An elementwise observed read at a cell, its array operands aligned by
index-set name (esm-spec §4.3.4). Byte-identical to the Julia reference's
golden."""

import json

import earthsci_ast as ea
import pytest
from earthsci_astdiff import BandError, InlineError, jacobian_bands, serialize_band, stable_json


def test_elementwise_aligned_bands(valid_dir, golden_dir):
    file = ea.load_path(str(valid_dir / "elementwise_aligned.esm"))
    bands = [serialize_band(e) for e in jacobian_bands(file, "Box")]
    assert stable_json(bands) == (golden_dir / "elementwise_aligned.bands.json").read_text()
    # row c reads q at its own y index and r with its axes swapped
    byc = {b["col"]: b for b in bands if b["row"] == "c"}
    assert byc["q"]["col_idx"] == ["j"]
    assert byc["r"]["col_idx"] == ["j", "i"]


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

    with pytest.raises((BandError, InlineError), match="index set `z`"):
        jacobian_bands(_variant(valid_dir, foreign), "Box")

    # an observed declared without a shape that mixes ranks
    def undeclared(d):
        del d["models"]["Box"]["variables"]["w"]["shape"]

    with pytest.raises((BandError, InlineError), match="declared without a shape"):
        jacobian_bands(_variant(valid_dir, undeclared), "Box")
