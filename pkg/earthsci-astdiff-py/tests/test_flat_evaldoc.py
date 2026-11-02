"""The evaluation document of a flattened system (spec §8.1; the reference's
flat_evaldoc_test.jl, the Rust tests/flat_evaldoc.rs): its scoped names
declared in the component tree, byte-identical to the reference's goldens
(`coupled_chem_advection.evaldoc[-nocse].json`), and loaded, flattened and
evaluated by EarthSciAST, whose right-hand side gives the band values of
`coupled_chem_advection.jvals.json` once each band point places its field
cell (the consumer's work, done here to check the document by value)."""

import dataclasses
import json
import math

import earthsci_ast as ea
import pytest
from earthsci_ast.simulation_array import evaluate_rhs
from earthsci_astdiff import (
    BandError,
    evaluation_document,
    hvp_document,
    jacobian_bands,
    jvp_document,
    serialize_band,
    stable_json,
    sysview,
    vjp_document,
)
from earthsci_astdiff.emit import _cellname, _evaluation_document, foreach_point
from earthsci_astdiff.system import SysView, ctx_of, view_states

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


@pytest.mark.parametrize("recorded", [True, False])
@pytest.mark.parametrize("wrt,tag", [("states", ""), ("parameters", ".parameters")])
def test_flat_band_golden(wrt, tag, recorded, valid_dir, golden_dir):
    """The flattened coupled document differentiates, its pointwise-lifted
    chemistry read as the grid arrays it is (#36), byte-identical to the
    reference's ``coupled_chem_advection[.parameters].bands.json``: with the
    lifted shapes ``flatten`` records (EarthSciAST >= 0.8) and without them
    (before 0.8, the view's fallback; every grid extent here names one index
    set, so the fallback gives the same shapes)."""
    flat = _flat(valid_dir)
    if not recorded:
        for n in flat.lifted_shapes:
            flat.state_variables[n] = dataclasses.replace(flat.state_variables[n], shape=None)
    entries = jacobian_bands(sysview(flat), wrt=wrt)
    assert entries
    text = stable_json([serialize_band(e) for e in entries])
    assert text == (golden_dir / f"{NAME}{tag}.bands.json").read_text()


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


def _cell(io_map: dict, cell: str) -> str:
    """An input or output's name for a cell of a source variable."""
    v, sep, rest = cell.partition("[")
    return io_map[v] + sep + rest


def _doc_call(doc: dict, io: dict, u: dict, inputs: dict, t: float, compiler: str):
    """One right-hand-side call of a derived document of a flattened system:
    ``u`` sets the source's state cells, ``inputs`` the derived model's input
    cells (by their names in ``io["model"]``). A lookup of ``du`` by a cell
    name of the derived model."""
    built = ea.flatten(ea.load_string(stable_json(doc)))
    pfx = f"{io['model']}."
    x = dict(u)
    x.update({pfx + k: v for k, v in inputs.items()})
    du = evaluate_rhs(built, x, float(t), compiler=compiler)
    return lambda n: du[pfx + n]


def _layout_ok(doc: dict, model: str) -> bool:
    ms = doc["models"]
    return sorted(ms) == sorted(["Advection", "SimpleOzone", model]) and not any(
        "." in n for m in ms.values() for n in m["variables"]
    )


@pytest.mark.parametrize("cse", [True, False])
def test_flat_product_documents(cse, valid_dir, golden_dir):
    """The product documents (spec §9) of a flattened system are laid out as
    its component tree too, their inputs and outputs unscoped in the derived
    model (``_v_SimpleOzone_O3``); each loads and gives ``J v`` and ``Jᵀ λ``
    of the golden Jacobian (the reference's flat_evaldoc_test.jl)."""
    flat = _flat(valid_dir)
    g = json.loads((golden_dir / f"{NAME}.jvals.json").read_text())
    want = {(r, c): v for r, c, v in g["entries"]}
    cells = _state_cells(sysview(flat))
    v = {c: math.sin(1.7 * (i + 1)) for i, c in enumerate(cells)}
    lam = {c: math.cos(0.9 * (i + 1)) for i, c in enumerate(cells)}

    jv, io = jvp_document(flat, cse=cse)
    assert _layout_ok(jv, "JvpEval")
    assert io["v"]["SimpleOzone.O3"] == "_v_SimpleOzone_O3"
    assert io["jvp"]["SimpleOzone.O3"] == "_jvp_SimpleOzone_O3"
    out = _doc_call(jv, io, g["u"], {_cell(io["v"], c): v[c] for c in cells}, g["t"], "native")
    for r in cells:
        exp = sum(x * v[c] for (rr, c), x in want.items() if rr == r)
        assert math.isclose(out(_cell(io["jvp"], r)), exp, rel_tol=1e-10, abs_tol=1e-13), r

    vj, io = vjp_document(flat, cse=cse)
    assert _layout_ok(vj, "VjpEval")
    out = _doc_call(
        vj, io, g["u"], {_cell(io["lam"], r): lam[r] for r in cells}, g["t"], "interpreter"
    )
    for c in cells:
        exp = sum(lam[r] * x for (r, cc), x in want.items() if cc == c)
        assert math.isclose(out(_cell(io["vjp"], c)), exp, rel_tol=1e-10, abs_tol=1e-13), c


def test_flat_hvp_document(valid_dir, golden_dir):
    """The second-order product document of a flattened system:
    ``∇_u (λᵀ J(u) v)`` against a central difference of ``λᵀ J v`` through
    the ``JvpEval`` document."""
    flat = _flat(valid_dir)
    g = json.loads((golden_dir / f"{NAME}.jvals.json").read_text())
    u, t = g["u"], g["t"]
    cells = _state_cells(sysview(flat))
    v = {c: math.sin(1.7 * (i + 1)) for i, c in enumerate(cells)}
    lam = {c: math.cos(0.9 * (i + 1)) for i, c in enumerate(cells)}
    hv, io = hvp_document(flat, stages=1)
    assert _layout_ok(hv, "HvpEval")
    inputs = {_cell(io["mu"][0], c): lam[c] for c in cells}
    inputs.update({_cell(io["w"][0], c): v[c] for c in cells})
    out = _doc_call(hv, io, u, inputs, t, "interpreter")
    jv, jio = jvp_document(flat, cse=False)
    vin = {_cell(jio["v"], c): v[c] for c in cells}

    def ljv(uu):
        o = _doc_call(jv, jio, uu, vin, t, "native")
        return sum(lam[r] * o(_cell(jio["jvp"], r)) for r in cells)

    h = 1e-5
    for k in cells[::7]:
        up, um = dict(u), dict(u)
        up[k] += h
        um[k] -= h
        fd = (ljv(up) - ljv(um)) / (2 * h)
        assert math.isclose(out(_cell(io["hvp"], k)), fd, rel_tol=1e-6, abs_tol=1e-9), k


def test_flat_documents_refuse_bad_sources(valid_dir):
    """Refused (spec §8.1): a flattened system mixing scoped and unscoped
    names, a scoped name that is both a variable and a component, and a
    source variable named like a band field."""
    sv = sysview(_flat(valid_dir))
    entries = jacobian_bands(sv)
    par = next(n for n in sorted(sv.variables) if n.startswith("Advection."))
    mixed = SysView({**sv.variables, "x": sv.variables[par]}, sv.equations, sv.index_sets)
    with pytest.raises(BandError, match="`x`"):
        _evaluation_document(mixed, entries)
    both = SysView(
        {
            **sv.variables,
            "Advection.Inner": sv.variables[par],
            "Advection.Inner.q": sv.variables[par],
        },
        sv.equations,
        sv.index_sets,
    )
    with pytest.raises(BandError, match=r"`Advection\.Inner` is both a variable and a component"):
        _evaluation_document(both, entries)
    clash = {
        "esm": "1.1.0",
        "metadata": {"name": "Clash"},
        "models": {
            "JacobianEval": {
                "variables": {"J_1": {"type": "unknown", "default": 1.0}},
                "equations": [
                    {
                        "lhs": {"op": "D", "args": ["J_1"], "wrt": "t"},
                        "rhs": {"op": "*", "args": [-1.0, "J_1", "J_1"]},
                    }
                ],
            }
        },
    }
    with pytest.raises(BandError, match="band field 1"):
        evaluation_document(ea.load_string(json.dumps(clash)))


def test_product_documents_carry_the_domain(valid_dir):
    """Every product document carries the source's ``domain``, as the
    evaluation document does (spec §8.1, §9.1)."""
    file = ea.load_string(json.dumps(DIURNAL))
    want = '{"temporal":{"end":"2024-05-02T06:00:00Z","start":"2024-05-01T06:00:00Z"}}'
    for src in (file, ea.flatten(file)):
        for doc, _ in (jvp_document(src), vjp_document(src), hvp_document(src, stages=1)):
            assert stable_json(doc["domain"]) == want
    doc, _ = jvp_document(ea.load_path(str(valid_dir / "bd_chem.esm")))
    assert "domain" not in doc


def test_domain_follows_earthsci_ast_serialization():
    """The ``domain`` is built from EarthSciAST's serialization, which drops
    an empty ``reference_time``, so every binding writes the same object."""
    from earthsci_astdiff.emit import _domain

    src = json.loads(json.dumps(DIURNAL))
    src["domain"]["temporal"]["reference_time"] = ""
    doc = _domain({}, ea.load_string(json.dumps(src)).domain)
    assert "reference_time" not in doc["domain"]["temporal"]


# ── a lifted state on one of two same-size index sets ──────────────────────
#
# `lift_same_size_index_sets` (EarthSciAST's pointwise-lift conformance case of
# that name): `y` and `x` both have size 3, `y` declared first, and the operand
# `Wind.u`, declared on `[x, y]` and gathered at the lift's loops, names the
# grid. EarthSciAST's `flatten` records `Chem.c` on `[x, y]` from 0.8
# (esm-libraries-spec §4.7.5); the view, its context, the bands and the
# evaluation document take that shape as recorded, byte-identical to the
# reference's goldens. A `flatten` that records no lifted shape (before 0.8)
# leaves the view to its fallback, which cannot tell `x` from `y`, so the
# goldens do not apply there.
LIFT = "lift_same_size_index_sets"


def _lift_flat(valid_dir):
    flat = ea.flatten(ea.load_path(str(valid_dir / f"{LIFT}.esm")))
    if not getattr(flat.state_variables["Chem.c"], "shape", None):
        pytest.skip("this earthsci-ast's flatten records no lifted shape (before 0.8)")
    return flat


@pytest.mark.parametrize("cse", [True, False])
def test_lifted_shape_is_flattens(cse, valid_dir, golden_dir):
    flat = _lift_flat(valid_dir)
    assert list(flat.state_variables["Chem.c"].shape) == ["x", "y"]
    sv = sysview(flat)
    assert list(sv.variables["Chem.c"].shape) == ["x", "y"]
    assert ctx_of(sv).shapes["Chem.c"] == [3, 3]
    assert not any(str(k).startswith("_liftdim") for k in sv.index_sets)
    entries = jacobian_bands(sv)
    doc, fields = _evaluation_document(sv, entries, cse=cse)
    # nothing to hoist: the unhoisted document is the same (no nocse golden)
    assert not (golden_dir / f"{LIFT}.evaldoc-nocse.json").exists()
    assert stable_json(doc) == (golden_dir / f"{LIFT}.evaldoc.json").read_text()
    assert doc["models"]["Chem"]["variables"]["c"]["shape"] == ["x", "y"]
    assert all(doc["models"]["JacobianEval"]["variables"][f]["shape"] == ["x", "y"] for f in fields)


@pytest.mark.parametrize("compiler", ["native", "interpreter"])
def test_lifted_shape_values(compiler, valid_dir, golden_dir):
    flat = _lift_flat(valid_dir)
    g = json.loads((golden_dir / f"{LIFT}.jvals.json").read_text())
    assert sorted(g["u"]) == sorted(_state_cells(sysview(flat)))
    want = {(r, c): v for r, c, v in g["entries"]}
    got = _band_jacobian(flat, g["u"], g["t"], True, compiler)
    assert set(got) == set(want)
    for k, v in want.items():
        assert math.isclose(got[k], v, rel_tol=1e-10, abs_tol=1e-13), k


def test_lifted_shape_fallback(valid_dir):
    """A ``flatten`` that records no lifted shape (before 0.8) leaves the view
    to its fallback, as the Rust tests/flat_view.rs checks: with the recorded
    shape and synthetic index sets removed, ``x`` and ``y`` share the size 3,
    so each dimension takes a synthetic ``_liftdim<d>_3``."""
    sv = sysview(_fallback_flat(valid_dir))
    assert list(sv.variables["Chem.c"].shape) == ["_liftdim1_3", "_liftdim2_3"]
    assert ctx_of(sv).shapes["Chem.c"] == [3, 3]


def _fallback_flat(valid_dir):
    """The lift fixture as a ``flatten`` before 0.8 leaves it: no recorded
    shape, no synthetic index set."""
    flat = ea.flatten(ea.load_path(str(valid_dir / f"{LIFT}.esm")))
    flat.state_variables["Chem.c"] = dataclasses.replace(flat.state_variables["Chem.c"], shape=None)
    flat.index_sets = {
        k: v for k, v in flat.index_sets.items() if not str(k).startswith("_liftdim")
    }
    return flat


def _with_shape(flat, shape):
    flat.state_variables["Chem.c"] = dataclasses.replace(
        flat.state_variables["Chem.c"], shape=shape
    )
    return flat


def test_recorded_lift_shape_stands(valid_dir):
    """A recorded shape is used as it stands, never re-derived: ``[y, x]``
    (which no rule would pick) survives, and the index sets are untouched."""
    flat = _with_shape(_fallback_flat(valid_dir), ["y", "x"])
    sv = sysview(flat)
    assert list(sv.variables["Chem.c"].shape) == ["y", "x"]
    assert set(sv.index_sets) == set(flat.index_sets)


def test_recorded_synthetic_lift_axis_is_declared(valid_dir):
    """A recorded ``_liftdim<d>_<n>`` the flattened ``index_sets`` lacks (as
    Julia's ``flatten`` before 0.8 leaves it) is added, of extent ``n``."""
    flat = _with_shape(_fallback_flat(valid_dir), ["_liftdim1_3", "_liftdim2_3"])
    sv = sysview(flat)
    assert ctx_of(sv).shapes["Chem.c"] == [3, 3]


@pytest.mark.parametrize(
    "shape",
    [["x"], ["x", "y", "x"], ["x", "z"], ["_liftdim2_3", "y"], ["_liftdim1_4", "y"]],
)
def test_recorded_lift_shape_is_checked(shape, valid_dir):
    """A recorded shape of the wrong rank, naming an undeclared index set, or
    of another extent than ``lifted_shapes`` is refused."""
    flat = _with_shape(_fallback_flat(valid_dir), shape)
    with pytest.raises(BandError):
        sysview(flat)


def test_recorded_lift_shape_extent_mismatch(valid_dir):
    flat = _with_shape(_fallback_flat(valid_dir), ["x", "y"])
    flat.index_sets = {**flat.index_sets, "x": {"kind": "interval", "size": 4}}
    with pytest.raises(BandError):
        sysview(flat)


def test_declared_liftdim_of_another_size_is_refused(valid_dir):
    """esm-libraries-spec §4.7.5 rule 3: a document declaring the synthetic
    name with a different size is refused, not reused at that size."""
    flat = _fallback_flat(valid_dir)
    flat.index_sets = {**flat.index_sets, "_liftdim1_3": {"kind": "interval", "size": 5}}
    with pytest.raises(BandError, match="dimension promotion"):
        sysview(flat)


def test_synthetic_lift_axis_is_not_matched_by_size(valid_dir):
    """A synthetic set one lifted state adds is not "declared": a second
    lifted state of an extent no declared set has takes its own
    ``_liftdim<d>_<n>``, not the first state's by size."""
    flat = _fallback_flat(valid_dir)
    flat.lifted_shapes = {**flat.lifted_shapes, "Chem.k": [4, 4]}
    flat.lifted_shapes["Chem.c"] = [4]
    flat.index_sets = {}
    sv = sysview(flat)
    assert list(sv.variables["Chem.c"].shape) == ["_liftdim1_4"]
    assert list(sv.variables["Chem.k"].shape) == ["_liftdim1_4", "_liftdim2_4"]
