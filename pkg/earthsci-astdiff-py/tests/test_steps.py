"""Named steps (spec §4.6): bands composed through the observed graph.
Structure-only passes compose; a chain with many paths stays linear, and the
time companions align elementwise operands by index set."""

import json

import earthsci_ast as ea
from earthsci_astdiff import jacobian_bands, jacobian_block, serialize_band, stable_json, sysview


def test_diamond_chain_structure(valid_dir, golden_dir):
    file = ea.load_path(str(valid_dir / "diamond_chain.esm"))
    entries = jacobian_bands(sysview(file, "Chain"), coefficients=False)
    text = stable_json([serialize_band(e) for e in entries])
    assert text == (golden_dir / "diamond_chain.structure.bands.json").read_text()


def test_time_companions_align_by_index_set(valid_dir):
    # w = c·q + k·t + r over [x, y]: its companion `_dt_w` reads k at the x
    # index, aligned by name (elementwise_aligned's operands)
    d = json.loads((valid_dir / "elementwise_aligned.esm").read_text())
    for e in d["models"]["Box"]["equations"]:
        if e["lhs"] == "w":
            e["rhs"]["args"][1] = {"op": "*", "args": ["k", "t"]}
    file = ea.load_string(json.dumps(d))
    comps = jacobian_block(file, "Box", wrt="time")["intermediates"]
    assert [c["name"] for c in comps] == ["_dt_w"]
    assert '"k"' in stable_json(comps[0]["rhs"])
