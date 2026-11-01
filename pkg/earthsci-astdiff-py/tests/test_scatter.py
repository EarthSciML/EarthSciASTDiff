"""Scatter plans (scatter.py, spec §8): band fields → a consumer's storage.
Mirrors scatter_test.jl and the Rust tests/scatter.rs: every plan golden
(`<fixture>.<layout>.plan.json`) byte-identical, the refusals, and a bound
plan placing every value of J. Executing a bound plan is the consumer's work:
here the test does it, over band fields EarthSciAST evaluates from the
evaluation document, and checks the result against the Jacobian formed from
the value goldens (`*.jvals.json`) — no evaluator of this package takes part."""

import dataclasses
import json
import math

import earthsci_ast as ea
import pytest
from earthsci_ast.esm_types import ExprNode
from earthsci_ast.simulation_array import evaluate_rhs
from earthsci_astdiff import (
    BoundScatter,
    ColumnBanded,
    Csr,
    CsrPlan,
    JacEntry,
    LaneBlocks,
    LaneSparseBlocks,
    bind_plan,
    evaluation_document,
    jacobian_bands,
    plan_factorization,
    plan_lanes,
    scatter_plan,
    scatter_targets,
    serialize_scatter_plan,
    stable_json,
    sysview,
)

# Which layouts each golden fixture gets a plan golden for (the rest refuse;
# see the refusal tests). Kept in step with scripts/regenerate-goldens.jl,
# scatter_test.jl and the Rust tests/scatter.rs.
PLAN_LAYOUTS = [
    ("bd_chem", "lane_blocks", LaneBlocks()),
    ("bd_chem", "lane_sparse_blocks", LaneSparseBlocks()),
    ("bd_chem", "column_banded", ColumnBanded("z")),
    ("bd_chem", "csr", Csr()),
    ("adv_interior", "column_banded", ColumnBanded("x")),
    ("adv_interior", "csr", Csr()),
    ("flux_form_adv", "csr", Csr()),
    ("minmod_adv", "csr", Csr()),
    ("contracted_ops", "csr", Csr()),
    ("contracting_observed", "csr", Csr()),
    ("filtered_scan", "csr", Csr()),
    ("semiring_reduce", "csr", Csr()),
    ("recurrence_sweep", "csr", Csr()),
    ("pbl_rank1", "csr", Csr()),
]
_IDS = [f"{n}.{t}" for n, t, _ in PLAN_LAYOUTS]


def _load(valid_dir, name):
    return ea.load_path(str(valid_dir / f"{name}.esm"))


def _refuses(f) -> bool:
    try:
        f()
    except ValueError:
        return True
    return False


@pytest.mark.parametrize("name,tag,layout", PLAN_LAYOUTS, ids=_IDS)
def test_plan_goldens(name, tag, layout, valid_dir, golden_dir):
    plan = scatter_plan(_load(valid_dir, name), layout)
    want = (golden_dir / f"{name}.{tag}.plan.json").read_text()
    assert stable_json(serialize_scatter_plan(plan)) == want


def test_every_plan_golden_is_listed(golden_dir):
    on_disk = {p.name for p in golden_dir.glob("*.plan.json")}
    assert on_disk == {f"{n}.{t}.plan.json" for n, t, _ in PLAN_LAYOUTS}


# ── a bound plan places every value of J ─────────────────────────────────────


def _run_plan(b: BoundScatter, field) -> list[float]:
    """A consumer's execution of a bound plan: zero `out`, then `+=` each
    band-field cell into its position, in plan order."""
    out = [0.0] * b.len
    for f, pos in b.pairs:
        out[pos] += field(f)
    return out


def _band_fields(file, g):
    """The band fields `J_k`, evaluated by EarthSciAST from the evaluation
    document at the value golden's point."""
    doc, _ = evaluation_document(file)
    model = "JacobianEval"
    variables = doc["models"][model]["variables"]
    for k, v in g["params"].items():
        assert variables[k]["default"] == v  # the goldens pin the defaults
    du = evaluate_rhs(
        ea.load_string(stable_json(doc)), dict(g["u"]), float(g["t"]), compiler="interpreter"
    )
    return lambda n: du[f"{model}.{n}"]


def _close(got: float, want: float) -> bool:
    return math.isclose(got, want, rel_tol=1e-10, abs_tol=1e-13)


@pytest.fixture(scope="module")
def _fields_cache():
    return {}


@pytest.mark.parametrize("name,tag,layout", PLAN_LAYOUTS, ids=_IDS)
def test_bound_plan_places_every_value(name, tag, layout, valid_dir, golden_dir, _fields_cache):
    g = json.loads((golden_dir / f"{name}.jvals.json").read_text())
    J: dict[tuple[str, str], float] = {}
    for r, c, v in g["entries"]:
        J[(r, c)] = J.get((r, c), 0.0) + v
    file = _load(valid_dir, name)
    if name not in _fields_cache:
        _fields_cache[name] = _band_fields(file, g)
    field = _fields_cache[name]
    plan = scatter_plan(file, layout)
    nl = plan_lanes(plan)
    vals_nl = None
    for L in (nl, nl + 3):
        b = bind_plan(file, plan, lanes=L)
        assert b.lanes == L
        out = _run_plan(b, field)
        tg = scatter_targets(plan, lanes=L)
        assert len(tg) == len(out) == b.len
        covered = set()
        for pos, t in enumerate(tg):
            if t is None:
                assert out[pos] == 0.0, f"padding / out-of-range slot {pos}"
            else:
                assert t not in covered, f"{t} has two slots"
                covered.add(t)
                assert _close(out[pos], J.get(t, 0.0)), (L, t, out[pos], J.get(t))
        # every entry of J has a home in the layout
        assert set(J) <= covered
        if isinstance(layout, Csr):
            assert covered == set(J)  # CSR stores exactly the pattern
        vals = {t: out[i] for i, t in enumerate(tg) if t is not None}
        if vals_nl is None:
            vals_nl = vals
        else:
            # padding lanes never change a real lane's value
            assert vals == vals_nl


# ── refusals ─────────────────────────────────────────────────────────────────


def test_refusals_layout_does_not_fit(valid_dir):
    adv = _load(valid_dir, "adv_interior")
    bd = _load(valid_dir, "bd_chem")
    assert _refuses(lambda: scatter_plan(adv, LaneBlocks()))
    assert _refuses(lambda: scatter_plan(adv, LaneSparseBlocks()))
    # periodic wrap
    assert _refuses(lambda: scatter_plan(_load(valid_dir, "minmod_adv"), ColumnBanded("x")))
    # scalar state
    assert _refuses(lambda: scatter_plan(_load(valid_dir, "adv_react_2d"), ColumnBanded("lon")))
    assert _refuses(lambda: scatter_plan(bd, ColumnBanded("nope")))


def test_refusals_band_lists(valid_dir):
    bd = _load(valid_dir, "bd_chem")
    sv = sysview(bd, "Chem")
    # an empty band list on a model with array states is never a valid plan
    with pytest.raises(ValueError, match="empty band list"):
        scatter_plan(sv, Csr(), entries=[])
    # columns that are not states
    with pytest.raises(ValueError, match="is not a state"):
        scatter_plan(sv, Csr(), entries=jacobian_bands(sv, wrt="parameters"))
    # a contracted band whose column is its row cell is block-diagonal, but a
    # lane template reads one field cell per lane: it would drop k = 2..hi
    bde = jacobian_bands(sv, wrt="states")
    con = [
        JacEntry(bde[0].u, bde[0].v, dataclasses.replace(bde[0].band, contracted=[("kk", 1, 3)])),
        *bde[1:],
    ]
    for layout in (LaneBlocks(), LaneSparseBlocks()):
        with pytest.raises(ValueError, match="contracted dimensions"):
            scatter_plan(sv, layout, entries=con)
    # a guarded band's points differ from lane to lane: the lane and column
    # layouts refuse it, the CSR plan takes the guarded point set
    r = bde[0].band.ridx[0]
    assert isinstance(r, str)
    guard = ExprNode(op=">", args=[r, 1])
    gd = [JacEntry(bde[0].u, bde[0].v, dataclasses.replace(bde[0].band, guard=guard)), *bde[1:]]
    for layout in (LaneBlocks(), LaneSparseBlocks(), ColumnBanded("z")):
        with pytest.raises(ValueError, match="`where` guard"):
            scatter_plan(sv, layout, entries=gd)
    pg = scatter_plan(sv, Csr(), entries=gd)
    p0 = scatter_plan(sv, Csr(), entries=bde)
    assert len(pg.items) + 1 == len(p0.items)  # the one excluded point (row 1)


def test_refusals_binding(valid_dir):
    bd = _load(valid_dir, "bd_chem")
    adv = _load(valid_dir, "adv_interior")
    # a plan from another document, a short capacity
    with pytest.raises(ValueError, match="different band list"):
        bind_plan(bd, scatter_plan(adv, Csr()))
    with pytest.raises(ValueError, match="lane capacity 2 < 4"):
        bind_plan(bd, scatter_plan(bd, LaneBlocks()), lanes=2)
    # same variables and field shapes, mirrored stencil (u[i-1] ↔ u[i+1]):
    # the band list check alone passes; decoding the contributions does not
    asv = sysview(adv, "Adv")
    ae = jacobian_bands(asv, wrt="states")
    mirrored = [
        JacEntry(ae[0].u, ae[0].v, dataclasses.replace(ae[0].band, cidx=ae[1].band.cidx)),
        JacEntry(ae[1].u, ae[1].v, dataclasses.replace(ae[1].band, cidx=ae[0].band.cidx)),
    ]
    for layout in (Csr(), ColumnBanded("x")):
        bind_plan(adv, scatter_plan(asv, layout, entries=ae))  # the real one binds
        with pytest.raises(ValueError, match="bind_plan: band"):
            bind_plan(adv, scatter_plan(asv, layout, entries=mirrored))
    # a plan whose state cells repeat: refused with the intended message
    p0 = scatter_plan(bd, Csr())
    dup = CsrPlan(p0.states + p0.states, p0.bands, p0.rowptr, p0.colind, p0.items)
    with pytest.raises(ValueError, match=r"not this source's \(order or duplicate state cells\)"):
        bind_plan(bd, dup)
    # a plan that drops a band point
    short = dataclasses.replace(p0, items=p0.items[1:])
    with pytest.raises(ValueError, match="the plan drops band"):
        bind_plan(bd, short)


# ── lane templates ───────────────────────────────────────────────────────────


def test_lane_templates_are_per_cell(valid_dir):
    bd = _load(valid_dir, "bd_chem")
    plan = scatter_plan(bd, LaneBlocks())
    assert len(plan.items) == 6  # one per band, whatever the grid
    assert plan.block_vars == ["C", "A", "B"]  # the factorization plan's Markowitz order
    sp = scatter_plan(bd, LaneSparseBlocks())
    fp = plan_factorization(jacobian_bands(bd, "Chem"), sysview(bd, "Chem"))
    assert len(sp.lu_pattern) == sum(sum(row) for row in fp.lu)


def test_bound_pairs_name_band_field_cells(valid_dir):
    bd = _load(valid_dir, "bd_chem")
    b = bind_plan(bd, scatter_plan(bd, Csr()))
    assert all(f.startswith("J_") for f, _ in b.pairs)
    assert {pos for _, pos in b.pairs} == set(range(b.len))
