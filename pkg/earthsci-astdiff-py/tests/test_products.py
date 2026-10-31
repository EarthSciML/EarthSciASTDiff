"""The derived documents (spec §8, §9) against the Julia reference's goldens:
the evaluation document (`<fixture>[.<axis>].evaldoc[-nocse].json`) and the
product documents (`<fixture>[.<axis>].{jvp,vjp}.evaldoc.json`,
`<fixture>.<a1>-<a2>.{hvp,hvp-sym}.evaldoc.json`) byte-identical, and each
product document evaluated by EarthSciAST and compared with the product formed
from the value goldens themselves (`*.jvals.json`, `*.hvals.json`), so no
evaluator of this package takes part. Mirrors products_test.jl and
evaldoc_test.jl."""

import json
import math

import earthsci_ast as ea
import pytest
from conftest import REPO_ROOT
from earthsci_ast.simulation_array import evaluate_rhs
from earthsci_astdiff import (
    BandError,
    column_names,
    ctx_of,
    evaluation_document,
    hvp_document,
    jacobian_bands,
    jacobian_bands2,
    jacobian_columns,
    jvp_document,
    stable_json,
    sysview,
    vjp_document,
)
from earthsci_astdiff.emit import _evaluation_document
from earthsci_astdiff.products import _hvp_document, _jvp_document, _vjp_document
from earthsci_astdiff.system import lhs_state
from test_goldens import AXIS_GOLDENS, GOLDEN_NAMES
from test_second_order import SECOND_ORDER

# Every axis with a value golden (spec §9.6).
PRODUCT_AXES = (
    ("bd_chem", "states"),
    ("adv_interior", "states"),
    ("contracted_ops", "states"),
    ("flux_form_adv", "states"),
    ("minmod_adv", "states"),
    ("contracting_observed", "states"),
    ("tgrad_photolysis", "states"),
    ("forcing_bracket", "states"),
    ("filtered_scan", "states"),
    ("semiring_reduce", "states"),
    ("tgrad_photolysis", "time"),
    ("forcing_bracket", "forcing"),
    ("forcing_bracket", "parameters"),
)


def _axistag(wrt: str) -> str:
    return "" if wrt == "states" else f".{wrt}"


def _pairtag(wrt) -> str:
    return f"{wrt[0]}-{wrt[1]}"


def _sv(valid_dir, name):
    file = ea.load_path(str(valid_dir / f"{name}.esm"))
    return file, sysview(file, next(iter(file.models)))


# ── the evaluation document ──────────────────────────────────────────────────


@pytest.mark.parametrize("name,wrt", [(n, "states") for n in GOLDEN_NAMES] + list(AXIS_GOLDENS))
def test_evaldoc_goldens(name, wrt, valid_dir, golden_dir):
    _, sv = _sv(valid_dir, name)
    entries = jacobian_bands(sv, wrt=wrt)
    stem = f"{name}{_axistag(wrt)}"
    doc, fields = _evaluation_document(sv, entries)
    assert stable_json(doc) == (golden_dir / f"{stem}.evaldoc.json").read_text()
    assert fields == [f"J_{k + 1}" for k in range(len(entries))]
    # without hoisting; no `-nocse` golden means nothing to hoist
    doc0, _ = _evaluation_document(sv, entries, cse=False)
    p0 = golden_dir / f"{stem}.evaldoc-nocse.json"
    want = p0 if p0.exists() else golden_dir / f"{stem}.evaldoc.json"
    assert stable_json(doc0) == want.read_text()


def test_evaldoc_public_entry(valid_dir, golden_dir):
    file, _ = _sv(valid_dir, "tgrad_photolysis")
    doc, _ = evaluation_document(file, wrt="time")
    assert stable_json(doc) == (golden_dir / "tgrad_photolysis.time.evaldoc.json").read_text()


@pytest.mark.parametrize("name", [*GOLDEN_NAMES, "adv_react_2d"])
def test_evaldoc_no_source_rhs_survives(name, valid_dir):
    _, sv = _sv(valid_dir, name)
    states = {u for u in (lhs_state(eq) for eq in sv.equations) if u is not None}
    assert states
    doc, _ = _evaluation_document(sv, jacobian_bands(sv))
    eqs = doc["models"]["JacobianEval"]["equations"]
    for u in states:
        mine = [
            eq
            for eq in eqs
            if isinstance(eq["lhs"], dict) and eq["lhs"]["op"] == "D" and eq["lhs"]["args"] == [u]
        ]
        assert len(mine) == 1  # exactly one, and zero
        rhs = mine[0]["rhs"]
        assert rhs == 0.0 or (rhs["op"] == "makearray" and rhs["values"] == [0.0])
    # no lifted (faq-LHS) source derivative is left behind either
    assert not any(
        isinstance(eq["lhs"], dict)
        and eq["lhs"]["op"] == "faq"
        and isinstance(eq["lhs"].get("expr"), dict)
        and eq["lhs"]["expr"]["op"] == "D"
        for eq in eqs
    )


@pytest.mark.parametrize(
    "name,gone,kept",
    [
        ("flux_form_adv", ["F"], []),
        ("contracting_observed", ["s", "c"], []),
        ("contracted_ops", ["conn"], []),
        ("minmod_adv", [], ["dx"]),
    ],
)
def test_evaldoc_only_the_observed_the_coefficients_read(name, gone, kept, valid_dir):
    _, sv = _sv(valid_dir, name)
    doc, _ = _evaluation_document(sv, jacobian_bands(sv))
    m = doc["models"]["JacobianEval"]
    defs = {eq["lhs"] for eq in m["equations"] if isinstance(eq["lhs"], str)}
    for n in gone:
        assert n not in m["variables"] and n not in defs
    for n in kept:
        assert n in m["variables"] and n in defs


def test_evaldoc_element_derivative_lhs_refused():
    # `D(index(u, 1)) = …` is no form lhs_state knows: dropping it would lose
    # its rows and keep its RHS in the evaluation document.
    doc = {
        "esm": "1.1.0",
        "metadata": {"name": "Elem"},
        "index_sets": {"x": {"kind": "interval", "size": 3}},
        "models": {
            "Elem": {
                "variables": {"u": {"type": "unknown", "shape": ["x"], "default": 1.0}},
                "equations": [
                    {
                        "lhs": {"op": "D", "wrt": "t", "args": [{"op": "index", "args": ["u", 1]}]},
                        "rhs": {"op": "-", "args": [{"op": "index", "args": ["u", 1]}]},
                    }
                ],
            }
        },
    }
    sv = sysview(ea.load_string(json.dumps(doc)), "Elem")
    with pytest.raises(BandError):
        jacobian_bands(sv)
    with pytest.raises(BandError):
        _evaluation_document(sv, [])


# ── product documents: goldens ───────────────────────────────────────────────


@pytest.mark.parametrize("name,wrt", PRODUCT_AXES)
@pytest.mark.parametrize("kind", ["jvp", "vjp"])
def test_product_goldens(name, wrt, kind, valid_dir, golden_dir):
    _, sv = _sv(valid_dir, name)
    entries = jacobian_bands(sv, wrt=wrt)
    build = _jvp_document if kind == "jvp" else _vjp_document
    doc, _ = build(sv, entries, wrt, None)
    want = golden_dir / f"{name}{_axistag(wrt)}.{kind}.evaldoc.json"
    assert stable_json(doc) == want.read_text()


def _forms(wrt):
    return ("contract", "symmetric") if tuple(wrt) == ("states", "states") else ("contract",)


@pytest.mark.parametrize("name,wrt,form", [(n, w, f) for n, w in SECOND_ORDER for f in _forms(w)])
def test_hvp_goldens(name, wrt, form, valid_dir, golden_dir):
    _, sv = _sv(valid_dir, name)
    entries = jacobian_bands2(sv, wrt=wrt)
    doc, _ = _hvp_document(sv, entries, wrt, None, 2, form=form)
    kind = "hvp" if form == "contract" else "hvp-sym"
    want = golden_dir / f"{name}.{_pairtag(wrt)}.{kind}.evaldoc.json"
    assert stable_json(doc) == want.read_text()


# ── product documents: values ────────────────────────────────────────────────
#
# An equality-filtered `faq` is refused by EarthSciAST's native compiler until
# it implements §9.3's equality-filter scatter; the interpreter evaluates it.


def _doc_cell(io_map: dict, cell: str) -> str:
    """The product document's name for one cell of a source variable:
    ``u[3]`` → ``<io_map["u"]>[3]``."""
    v, sep, rest = cell.partition("[")
    return io_map[v] + sep + rest


def _eval_product(doc, model, u: dict, inputs: dict, t: float, params=None):
    if params is not None:
        variables = doc["models"][model]["variables"]
        for k, v in params.items():
            assert variables[k]["default"] == v  # the goldens pin the defaults
    state = dict(u)
    state.update(inputs)
    du = evaluate_rhs(ea.load_string(stable_json(doc)), state, float(t), compiler="interpreter")
    pfx = f"{model}."
    return lambda n: du[pfx + n]


def _colcells(sv, wrt, rows):
    if wrt == "states":
        return rows
    if wrt == "time":
        return ["t"]
    return column_names(jacobian_columns(sv, wrt=wrt))


def _seed(n: int, a: float) -> list[float]:
    return [math.sin(a * k + 0.4) for k in range(1, n + 1)]


def _close(got: float, want: float) -> bool:
    return math.isclose(got, want, rel_tol=1e-10, abs_tol=1e-13)


# Every value golden of a product axis: each axis has one, and the forcing
# and parameters axes a second point (`.w0`).
_VALUE_CASES = [
    (n, w, sfx)
    for n, w in PRODUCT_AXES
    for sfx in ("", ".w0")
    if (REPO_ROOT / "tests" / "goldens" / f"{n}{_axistag(w)}{sfx}.jvals.json").exists()
]


def test_every_product_axis_has_a_value_golden():
    assert {(n, w) for n, w, sfx in _VALUE_CASES if not sfx} == set(PRODUCT_AXES)


@pytest.mark.parametrize("name,wrt,sfx", _VALUE_CASES)
def test_jvp_vjp_values(name, wrt, sfx, valid_dir, golden_dir):
    g = json.loads((golden_dir / f"{name}{_axistag(wrt)}{sfx}.jvals.json").read_text())
    file, sv = _sv(valid_dir, name)
    rows = sorted(g["u"])
    cols = _colcells(sv, wrt, rows)
    ri = {n: i for i, n in enumerate(rows)}
    ci = {n: i for i, n in enumerate(cols)}
    J = [[0.0] * len(cols) for _ in rows]
    for r, c, v in g["entries"]:
        J[ri[r]][ci[c]] += v
    v = _seed(len(cols), 1.3)
    lam = _seed(len(rows), 0.7)

    doc, io = jvp_document(file, wrt=wrt)
    jv = _eval_product(
        doc,
        io["model"],
        g["u"],
        {_doc_cell(io["v"], c): v[k] for k, c in enumerate(cols)},
        g["t"],
        params=g["params"],
    )
    for i, r in enumerate(rows):
        want = sum(J[i][k] * v[k] for k in range(len(cols)))
        assert _close(jv(_doc_cell(io["jvp"], r)), want), r

    doc, io = vjp_document(file, wrt=wrt)
    vj = _eval_product(
        doc,
        io["model"],
        g["u"],
        {_doc_cell(io["lam"], r): lam[i] for i, r in enumerate(rows)},
        g["t"],
        params=g["params"],
    )
    for k, c in enumerate(cols):
        want = sum(J[i][k] * lam[i] for i in range(len(rows)))
        assert _close(vj(_doc_cell(io["vjp"], c)), want), c


@pytest.mark.parametrize("name,wrt,form", [(n, w, f) for n, w in SECOND_ORDER for f in _forms(w)])
def test_hvp_values(name, wrt, form, valid_dir, golden_dir):
    gd = json.loads((golden_dir / f"{name}.{_pairtag(wrt)}.hvals.json").read_text())
    file, sv = _sv(valid_dir, name)
    doc, io = hvp_document(file, stages=len(gd["pairs"]), wrt=wrt, form=form)
    inputs: dict = {}
    for i, pr in enumerate(gd["pairs"]):
        for r, x in pr["mu"].items():
            inputs[_doc_cell(io["mu"][i], r)] = x
        if isinstance(pr["w"], (int, float)):
            inputs[io["w"][i]["t"]] = pr["w"]
        else:
            for c, x in pr["w"].items():
                inputs[_doc_cell(io["w"][i], c)] = x
    h = _eval_product(doc, io["model"], gd["u"], inputs, gd["t"])
    want = dict(gd["g"])
    a2 = wrt[1]
    if a2 == "states":
        shapes = ctx_of(sv).shapes
        cells = column_names([(n, shapes.get(n, [])) for n in io["hvp"]])
    else:
        cells = column_names(jacobian_columns(sv, wrt=a2))
    assert set(want) <= set(cells)
    for c in cells:
        assert _close(h(_doc_cell(io["hvp"], c)), want.get(c, 0.0)), c


# ── product documents: form ──────────────────────────────────────────────────


def test_form(valid_dir):
    file = ea.load_path(str(valid_dir / "adv_interior.esm"))
    doc, io = vjp_document(file)
    m = doc["models"]["VjpEval"]
    assert io == {"model": "VjpEval", "lam": {"u": "_lam_u"}, "vjp": {"u": "_vjp_u"}}
    # the inputs are states with a zero derivative; the source's D(u) is zeroed
    for n in ("_lam_u", "_vjp_u"):
        assert m["variables"][n] == {"type": "unknown", "default": 0.0, "shape": ["x"]}
    (rhs,) = [
        eq["rhs"]
        for eq in m["equations"]
        if isinstance(eq["lhs"], dict) and eq["lhs"]["args"] == ["_vjp_u"]
    ]
    # two entries, a left fold of their scatters, each filtered by its column
    assert rhs["op"] == "+" and len(rhs["args"]) == 2
    f = rhs["args"][0]
    assert f["op"] == "faq" and f["output_idx"] == ["_j1"]
    assert f["ranges"] == {"_j1": [1, 8], "i": [2, 7]}
    assert f["filter"]["op"] == "==" and f["filter"]["args"][1] == "_j1"
    # names already taken get `_` appended
    d2 = json.loads((valid_dir / "adv_interior.esm").read_text())
    d2["models"]["Adv"]["variables"]["_v_u"] = {"type": "parameter", "default": 0.0}
    _, io2 = jvp_document(ea.load_string(json.dumps(d2)))
    assert io2["v"]["u"] == "_v_u_" and io2["jvp"]["u"] == "_jvp_u"


def test_ifelse_guard_makes_every_reduced_name_occur(valid_dir):
    # contracting_observed's third entry reads its contracted name only in
    # its column index: Jᵀλ's body must still mention it
    file = ea.load_path(str(valid_dir / "contracting_observed.esm"))
    doc, io = vjp_document(file)
    m = doc["models"][io["model"]]
    (rhs,) = [
        eq["rhs"]
        for eq in m["equations"]
        if isinstance(eq["lhs"], dict) and eq["lhs"]["args"] == [io["vjp"]["u"]]
    ]
    assert '"ifelse"' in stable_json(rhs)


def test_refusals(valid_dir):
    file = ea.load_path(str(valid_dir / "bd_chem.esm"))
    with pytest.raises(ValueError):
        hvp_document(file, stages=2, wrt=("states", "parameters"), form="symmetric")
    with pytest.raises(ValueError):
        hvp_document(file, stages=0)
    with pytest.raises(ValueError):
        hvp_document(file, stages=2, form="dense")
