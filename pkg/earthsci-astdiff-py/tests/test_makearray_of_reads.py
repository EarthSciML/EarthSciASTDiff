"""A makearray whose region values are cell reads (scalars), read at a cell
through an observed: each value is read as itself, never wrapped in an
`index` with no subscripts. Byte-identical to the Julia reference's golden."""

import json

import earthsci_ast as ea
import pytest
from earthsci_astdiff import InlineError, jacobian_bands, serialize_band, stable_json


def test_makearray_of_reads_bands(valid_dir, golden_dir):
    file = ea.load_path(str(valid_dir / "makearray_of_reads.esm"))
    entries = jacobian_bands(file, "M")
    text = stable_json([serialize_band(e) for e in entries])
    assert text == (golden_dir / "makearray_of_reads.bands.json").read_text()
    assert [(e.u, e.v) for e in entries] == [("u", "u"), ("u", "v"), ("v", "v")]


def test_makearray_of_reads_values(valid_dir):
    # this binding's own JvpEval document against the Jacobian by hand:
    # m[k, 1] = u[k], m[k, 2:3] = v[k]; D(u) = -u v, D(v) = v - 0.5 v
    from docvals import doc_jacobian

    file = ea.load_path(str(valid_dir / "makearray_of_reads.esm"))
    U, V = [0.3, 0.7, 1.1], [0.2, 0.9, 1.3]
    cells = {f"u[{k}]": U[k - 1] for k in (1, 2, 3)} | {f"v[{k}]": V[k - 1] for k in (1, 2, 3)}
    J = doc_jacobian(file, "M", states=cells).J
    want = {}
    for k in (1, 2, 3):
        want[(f"u[{k}]", f"u[{k}]")] = -V[k - 1]
        want[(f"u[{k}]", f"v[{k}]")] = -U[k - 1]
        want[(f"v[{k}]", f"v[{k}]")] = 0.5
    assert {k for k, v in J.items() if v != 0.0} == set(want)
    for k, v in want.items():
        assert J[k] == pytest.approx(v, abs=1e-12), k


def _variant(valid_dir, vals, edit=None):
    d = json.loads((valid_dir / "makearray_of_reads.esm").read_text())
    m = d["models"]["M"]
    m["equations"][0]["rhs"]["expr"]["args"][0]["values"] = vals
    if edit:
        edit(m)
    return ea.load_string(json.dumps(d))


def _ix(*a):
    return {"op": "index", "args": list(a)}


def test_makearray_value_base_of_no_declared_shape(valid_dir):
    # a cell read of an observed declared without a shape (`w = 2u`): ranked
    # as an operand of an elementwise definition, so it is a scalar
    def edit(m):
        m["variables"]["w"] = {"type": "unknown"}
        m["equations"].append({"lhs": "w", "rhs": {"op": "*", "args": [2, "u"]}})

    file = _variant(valid_dir, [_ix("w", "k"), _ix("v", "k")], edit)
    entries = jacobian_bands(file, "M")
    assert [(e.u, e.v) for e in entries] == [("u", "u"), ("u", "v"), ("v", "v")]
    # more subscripts than the array has dimensions
    with pytest.raises(InlineError):
        jacobian_bands(_variant(valid_dir, [_ix("u", "k", "k"), _ix("v", "k")]), "M")
    # a partial read of an array whose ranges do not all start at 1
    shifted = {
        "op": "faq",
        "args": [],
        "output_idx": ["a", "b"],
        "ranges": {"a": {"from": "x"}, "b": [2, 3]},
        "expr": _ix("u", "a"),
    }
    with pytest.raises(InlineError):
        jacobian_bands(_variant(valid_dir, [_ix("u", "k"), _ix(shifted, "k")]), "M")
