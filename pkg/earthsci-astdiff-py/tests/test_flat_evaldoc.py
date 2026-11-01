"""The evaluation document of a flattened system (spec §8.1; the reference's
flat_evaldoc_test.jl, the Rust tests/flat_evaldoc.rs): its scoped names
declared in the component tree, byte-identical to the reference's goldens
(`coupled_chem_advection.evaldoc[-nocse].json`), and loaded, flattened and
evaluated by EarthSciAST, whose right-hand side gives the band values of
`coupled_chem_advection.jvals.json` once each band point places its field
cell (the consumer's work, done here to check the document by value)."""

import json
import math

import earthsci_ast as ea
import pytest
from earthsci_ast.simulation_array import evaluate_rhs
from earthsci_astdiff import BandError, evaluation_document, jacobian_bands, stable_json, sysview
from earthsci_astdiff.emit import _cellname, _evaluation_document, foreach_point
from earthsci_astdiff.system import ctx_of, view_states

NAME = "coupled_chem_advection"


def _flat(valid_dir):
    return ea.flatten(ea.load_path(str(valid_dir / f"{NAME}.esm")))


@pytest.mark.parametrize("cse,tag", [(True, "evaldoc"), (False, "evaldoc-nocse")])
def test_flat_evaldoc_golden_and_layout(cse, tag, valid_dir, golden_dir):
    flat = _flat(valid_dir)
    sv = sysview(flat)
    entries = jacobian_bands(sv)
    doc, fields = _evaluation_document(sv, entries, cse=cse)
    assert stable_json(doc) == (golden_dir / f"{NAME}.{tag}.json").read_text()
    assert fields == [f"J_{k + 1}" for k in range(len(entries))]
    models = doc["models"]
    assert sorted(models) == ["Advection", "JacobianEval", "SimpleOzone"]
    assert not any("." in n for m in models.values() for n in m["variables"])
    assert sorted(models["SimpleOzone"]["variables"]) == ["M", "NO", "NO2", "O3", "T", "jNO2"]
    assert models["SimpleOzone"]["variables"]["O3"]["shape"] == ["lon", "lat"]
    back = ea.flatten(ea.load_string(stable_json(doc)))
    for n in list(flat.state_variables) + list(flat.parameters):
        assert n in back.state_variables or n in back.parameters
    assert all(f"JacobianEval.{f}" in back.state_variables for f in fields)


def _state_cells(sv) -> list[str]:
    """The canonical state order of spec §8: states sorted, each state's cells
    column-major."""
    shapes = ctx_of(sv).shapes
    out = []
    for n in view_states(sv):
        sh = shapes.get(n) or []
        if not sh:
            out.append(n)
            continue
        cells = [[]]
        for ext in sh:  # first index fastest
            cells = [[*c, i] for i in range(1, ext + 1) for c in cells]
        out.extend(_cellname(n, c) for c in cells)
    return out


def _band_jacobian(flat, u: dict, t: float, cse: bool, compiler: str) -> dict:
    """J at ``u`` from one right-hand-side call of the evaluation document,
    each band point adding its field cell into its (row cell, column cell); a
    ghost read has no cell."""
    sv = sysview(flat)
    entries = jacobian_bands(sv)
    doc, fields = evaluation_document(flat, cse=cse)
    built = ea.flatten(ea.load_string(stable_json(doc)))
    du = evaluate_rhs(built, dict(u), float(t), compiler=compiler)
    shapes = ctx_of(sv).shapes
    J: dict = {}
    for en, nm in zip(entries, fields, strict=True):
        shape = shapes.get(en.v) or []
        for cell, cc, _, cidx in foreach_point(en.band):
            if any(not 1 <= c <= s for c, s in zip(cidx, shape, strict=True)):
                continue
            key = (_cellname(en.u, cell), _cellname(en.v, cidx))
            J[key] = J.get(key, 0.0) + du[_cellname(f"JacobianEval.{nm}", list(cell) + list(cc))]
    return J


@pytest.mark.parametrize("cse", [True, False])
@pytest.mark.parametrize("compiler", ["native", "interpreter"])
def test_flat_evaldoc_values(cse, compiler, valid_dir, golden_dir):
    flat = _flat(valid_dir)
    g = json.loads((golden_dir / f"{NAME}.jvals.json").read_text())
    assert sorted(g["u"]) == sorted(_state_cells(sysview(flat)))
    want = {(r, c): v for r, c, v in g["entries"]}
    got = _band_jacobian(flat, g["u"], g["t"], cse, compiler)
    assert set(got) == set(want)
    for k, v in want.items():
        assert math.isclose(got[k], v, rel_tol=1e-10, abs_tol=1e-13), k


NESTED = {
    "esm": "1.1.0",
    "metadata": {"name": "Nested"},
    "models": {
        "Chem": {
            "variables": {
                "c": {"type": "unknown", "default": 1.0},
                "r": {"type": "parameter", "default": 0.5},
            },
            "equations": [
                {
                    "lhs": {"op": "D", "args": ["c"], "wrt": "t"},
                    "rhs": {"op": "*", "args": [{"op": "-", "args": ["Inner.k"]}, "c", "c", "r"]},
                }
            ],
            "subsystems": {
                "Inner": {
                    "variables": {"k": {"type": "unknown", "default": 0.3}},
                    "equations": [
                        {
                            "lhs": {"op": "D", "args": ["k"], "wrt": "t"},
                            "rhs": {"op": "*", "args": [-0.1, "k"]},
                        }
                    ],
                }
            },
        },
        "Source": {
            "variables": {
                "s": {"type": "unknown", "default": 2.0},
                "q": {"type": "parameter", "default": 0.1},
            },
            "equations": [
                {
                    "lhs": {"op": "D", "args": ["s"], "wrt": "t"},
                    "rhs": {"op": "*", "args": [{"op": "-", "args": ["q"]}, "s"]},
                }
            ],
        },
    },
    "coupling": [
        {"type": "variable_map", "from": "Source.s", "to": "Chem.r", "transform": "param_to_var"}
    ],
}


def test_flat_evaldoc_nested_and_coupled():
    """Two coupled models, one with a nested subsystem: a name three deep
    (``Chem.Inner.k``) goes into ``subsystems``, and the evaluation document
    gives the same J as the reference's flat_evaldoc_test.jl (there checked
    against ForwardDiff): ``D(Chem.c) = -k c² s``, ``D(Inner.k) = -0.1 k``,
    ``D(Source.s) = -q s``."""
    flat = ea.flatten(ea.load_string(json.dumps(NESTED)))
    doc, _ = evaluation_document(flat, cse=False)
    assert "k" in doc["models"]["Chem"]["subsystems"]["Inner"]["variables"]
    u = {"Chem.Inner.k": 0.5, "Chem.c": 0.6, "Source.s": 0.7}
    assert sorted(u) == _state_cells(sysview(flat))
    J = _band_jacobian(flat, u, 0.0, False, "native")
    k, c, s = u["Chem.Inner.k"], u["Chem.c"], u["Source.s"]
    want = {
        ("Chem.c", "Chem.c"): -2 * k * c * s,
        ("Chem.c", "Chem.Inner.k"): -c * c * s,
        ("Chem.c", "Source.s"): -k * c * c,
        ("Chem.Inner.k", "Chem.Inner.k"): -0.1,
        ("Source.s", "Source.s"): -0.1,
    }
    assert set(J) == set(want)
    for key, v in want.items():
        assert math.isclose(J[key], v, rel_tol=1e-12), key


def test_flat_evaldoc_refuses_a_component_named_like_the_derived_model():
    clash = {
        "esm": "1.1.0",
        "metadata": {"name": "Clash"},
        "models": {
            "JacobianEval": {
                "variables": {"c": {"type": "unknown", "default": 1.0}},
                "equations": [
                    {
                        "lhs": {"op": "D", "args": ["c"], "wrt": "t"},
                        "rhs": {"op": "*", "args": [-1.0, "c", "c"]},
                    }
                ],
            }
        },
    }
    with pytest.raises(BandError):
        evaluation_document(ea.flatten(ea.load_string(json.dumps(clash))))


DIURNAL = {
    "esm": "1.1.0",
    "metadata": {"name": "Diurnal"},
    "domain": {
        "independent_variable": "t",
        "temporal": {"start": "2024-05-01T06:00:00Z", "end": "2024-05-02T06:00:00Z"},
    },
    "models": {
        "Chem": {
            "variables": {
                "c": {"type": "unknown", "default": 1.0},
                "k": {"type": "parameter", "default": 0.5},
            },
            "equations": [
                {
                    "lhs": {"op": "D", "args": ["c"], "wrt": "t"},
                    "rhs": {
                        "op": "*",
                        "args": [
                            {"op": "-", "args": ["k"]},
                            {"op": "fn", "name": "datetime.hour", "args": ["t"]},
                            "c",
                        ],
                    },
                }
            ],
        }
    },
}


def _j1(doc: dict, t: float) -> float:
    built = ea.flatten(ea.load_string(stable_json(doc)))
    du = evaluate_rhs(built, {}, t, compiler="interpreter")
    return du["JacobianEval.J_1"]


def test_evaluation_document_carries_the_domain(valid_dir):
    """The evaluation document carries the source's ``domain`` (spec §8.1),
    so it has the source's time origin; the right-hand side passes ``t`` to
    the calendar functions as UTC seconds whatever the origin, so a
    ``datetime.hour`` coefficient reads the same with or without it (the
    reference's flat_evaldoc_test.jl)."""
    file = ea.load_string(json.dumps(DIURNAL))
    for src in (file, ea.flatten(file)):
        doc, fields = evaluation_document(src)
        assert fields == ["J_1"]
        assert stable_json(doc["domain"]) == (
            '{"temporal":{"end":"2024-05-02T06:00:00Z","start":"2024-05-01T06:00:00Z"}}'
        )
        assert doc["esm"] == "1.1.0"
        if hasattr(ea, "time_origin"):  # earthsci-ast >= 0.7
            built = ea.load_string(stable_json(doc))
            assert ea.time_origin(built) == ea.time_origin(file) == 1714543200.0
        # 13:00 UTC on 2024-05-01, in UTC seconds: -k · 13
        assert _j1(doc, 1714568400.0) == -6.5
        bare = {k: v for k, v in doc.items() if k != "domain"}
        assert _j1(bare, 1714568400.0) == -6.5
    doc, _ = evaluation_document(ea.load_path(str(valid_dir / "bd_chem.esm")))
    assert "domain" not in doc
