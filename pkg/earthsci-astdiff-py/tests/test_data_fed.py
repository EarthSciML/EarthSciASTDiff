"""Data-fed parameters in the evaluation document (spec §8.1; the
reference's data_fed_test.jl, the Rust tests/data_fed.rs): a parameter whose
``update`` has a ``data`` rule keeps it, and the document carries the
``data_sources`` entries those rules name, so EarthSciAST builds the
parameter as an externally fed input, not a constant. Byte-identical to the
reference's ``data_fed_flat.{evaldoc,bands}.json``."""

import json

import earthsci_ast as ea
import pytest
from earthsci_ast.simulation_array import evaluate_rhs
from earthsci_astdiff import (
    BandError,
    evaluation_document,
    jacobian_bands,
    jvp_document,
    serialize_band,
    stable_json,
    sysview,
)
from earthsci_astdiff.emit import _evaluation_document

NAME = "data_fed_flat"
FED = ["Box.P", "Chem.E", "Chem.T"]


def _file(valid_dir):
    return ea.load_path(str(valid_dir / f"{NAME}.esm"))


def _without_rules(valid_dir):
    """The fixture without its data rules and ``data_sources``: the same model
    with every data-fed parameter a constant at its default."""
    d = json.loads((valid_dir / f"{NAME}.esm").read_text())
    del d["data_sources"]
    for m in d["models"].values():
        for v in m["variables"].values():
            v.pop("update", None)
    return ea.load_string(json.dumps(d))


@pytest.mark.parametrize("cse", [True, False])
def test_evaluation_document_carries_the_data_rules(cse, valid_dir, golden_dir):
    file = _file(valid_dir)
    flat = ea.flatten(file)
    sv = sysview(flat, data_sources=file.data_sources)
    doc, _ = _evaluation_document(sv, jacobian_bands(sv), cse=cse)
    assert stable_json(doc) == (golden_dir / f"{NAME}.evaldoc.json").read_text()
    doc, _ = evaluation_document(flat, cse=cse, data_sources=file.data_sources)
    assert stable_json(doc) == (golden_dir / f"{NAME}.evaldoc.json").read_text()
    chem = doc["models"]["Chem"]["variables"]
    assert chem["T"]["update"] == {
        "kind": "data",
        "source": "MET",
        "from": {"file_variable": "T2M"},
    }
    assert chem["E"]["update"]["from"] == {"file_variable": "NOX", "unit_conversion": 0.001}
    assert "update" not in chem["k"]
    assert sorted(doc["data_sources"]) == ["EMIS", "MET"]


def test_a_flattened_system_needs_its_data_sources(valid_dir):
    """A flattened system carries no ``data_sources``: without them the
    document is refused, never written with a parameter baked in; an
    ``EsmFile`` gives its own."""
    flat = ea.flatten(_file(valid_dir))
    with pytest.raises(BandError, match="data source `EMIS`"):
        evaluation_document(flat)
    with pytest.raises(BandError):
        jvp_document(flat)
    doc, _ = evaluation_document(ea.load_path(str(valid_dir / "forcing_bracket.esm")))
    assert sorted(doc["data_sources"]) == ["MET"]


def test_data_fed_parameters_are_constants_of_the_state_jacobian(valid_dir, golden_dir):
    want = (golden_dir / f"{NAME}.bands.json").read_text()
    for file in (_file(valid_dir), _without_rules(valid_dir)):
        entries = jacobian_bands(sysview(ea.flatten(file)))
        assert stable_json([serialize_band(e) for e in entries]) == want


def test_data_fed_parameters_are_inputs_of_the_build(valid_dir):
    """Loaded and flattened, the document's data-fed parameters are the
    loaded fields a provider fills (``loader_fields``), as the source's are;
    without the rules they are none. Its right-hand side builds, and at the
    parameters' defaults gives the band fields of the document with the
    parameters baked in."""
    file = _file(valid_dir)
    doc, _ = evaluation_document(ea.flatten(file), cse=False, data_sources=file.data_sources)
    built = ea.flatten(ea.load_string(stable_json(doc)))
    got = sorted((f.name, f.subkey, f.var) for f in built.loader_fields)
    assert got == [("Box.P", "MET", "PS"), ("Chem.E", "EMIS", "NOX"), ("Chem.T", "MET", "T2M")]
    assert got == sorted((f.name, f.subkey, f.var) for f in ea.flatten(file).loader_fields)
    old, _ = evaluation_document(ea.flatten(_without_rules(valid_dir)), cse=False)
    baked = ea.flatten(ea.load_string(stable_json(old)))
    assert not baked.loader_fields
    u = {"Box.S": 2.0, **{f"Chem.c[{i}]": 1.0 + i for i in (1, 2, 3)}}
    u.update({"JacobianEval.J_2": 0.0, **{f"JacobianEval.J_1[{i}]": 0.0 for i in (1, 2, 3)}})
    du = evaluate_rhs(built, dict(u), 0.0, compiler="interpreter")
    assert du == evaluate_rhs(baked, dict(u), 0.0, compiler="interpreter")
    assert any(du[k] != 0.0 for k in du if k.startswith("JacobianEval."))


@pytest.mark.parametrize("product", [False, True])
def test_kept_rule_keeps_the_observed_it_reads(product, valid_dir):
    """A kept rule's ``unit_conversion`` may be an expression reading an
    observed the coefficients do not: the derived documents keep it."""
    d = json.loads((valid_dir / f"{NAME}.esm").read_text())
    chem = d["models"]["Chem"]
    chem["variables"]["f"] = {"type": "unknown"}
    chem["equations"].append({"lhs": "f", "rhs": {"op": "*", "args": [0.002, "k"]}})
    chem["variables"]["E"]["update"]["from"]["unit_conversion"] = "f"
    file = ea.load_string(json.dumps(d))
    make = jvp_document if product else evaluation_document
    doc, _ = make(ea.flatten(file), data_sources=file.data_sources)
    chem = doc["models"]["Chem"]["variables"]
    assert chem["E"]["update"]["from"]["unit_conversion"] == "f"
    assert "f" in chem
    ea.load_string(stable_json(doc))
