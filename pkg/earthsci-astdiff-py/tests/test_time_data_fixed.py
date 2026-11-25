"""Time bands with data held fixed (spec §4.2): only observed that move with
``t`` are followed, and an observed declared without a shape is read at a
cell. Byte-identical to the Julia reference's golden."""

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

    # a global the read site's index `j` would capture
    def capture(m):
        m["variables"]["j"] = {"type": "parameter", "default": 2.0}
        m["equations"][0]["rhs"] = {"op": "*", "args": [_JR_T, "j"]}

    with pytest.raises((BandError, InlineError)):
        jacobian_bands(_variant(valid_dir, capture), "Box", wrt="time")

    # the full index of an array inside the definition is a scalar
    def full(m):
        m["equations"][0]["rhs"] = {
            "op": "*",
            "args": [_JR_T, {"op": "index", "args": ["Eb", 1, 2]}],
        }

    assert len(jacobian_bands(_variant(valid_dir, full), "Box", wrt="time")) == 1
