"""Scatter plans (scatter.jl, spec §8): the precomputed, checked map from
band-field values to a consumer's own Jacobian storage.

The derived evaluation document (:func:`~earthsci_astdiff.emit.evaluation_document`)
holds one band field ``J_k`` per entry, which an EarthSciAST evaluator
computes. A plan maps those fields to a storage layout (lane-SoA per-cell
blocks, a KPP-style sparse block, per-column bands, CSR); it is built once
from the band list, serialized byte-identically by every binding
(``tests/goldens/<fixture>.<layout>.plan.json``) and bound to a source
(:func:`bind_plan`) only after its band list and state names have been
checked. Executing a bound plan is the consumer's: this package evaluates
nothing.

Indices in a plan are 0-based throughout:

- band field ``k``, cell ``f``: field ``J_{k+1}``, column-major flat cell
  ``f`` of its shape (the row variable's shape, then each contracted extent).
- canonical state order: :func:`~earthsci_astdiff.system.view_states`
  (sorted names), species-major, each state's cells column-major — the rows
  and columns of a :class:`Csr` plan.

Lane layouts are per-cell TEMPLATES (O(nnz), not O(nnz·cells)): lane ``ℓ``
is cell ``ℓ`` of the lane domain, and the lane capacity ``L`` is chosen at
bind time, so one plan serves every leaf whatever its capacity."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import prod

from .emit import _cellname, _eval_cidx, _product, foreach_point
from .factorization import _block_diagonal_plan
from .jacobian import JacEntry, jacobian_bands
from .structure import _offset_class, detect_structure
from .system import ctx_of, sysview, view_states

# ── layouts ─────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class LaneBlocks:
    """Dense ``n × n`` block per cell, lane-SoA: ``J[r, c]`` of lane ``ℓ`` at
    ``(r + n·c)·L + ℓ`` (0-based). Block order is the factorization plan's
    Markowitz ``block_vars``, then any state no band touches (sorted). Needs a
    ``"block_diagonal"`` Jacobian with every state on one cell shape."""


@dataclass(frozen=True)
class LaneSparseBlocks:
    """KPP-style sparse block per cell: ``J[r, c]`` of lane ``ℓ`` at
    ``q·L + ℓ``, ``q`` the position of ``(r, c)`` in the plan's
    ``lu_pattern`` (fill-in included; fill slots stay zero). Same
    preconditions as :class:`LaneBlocks`."""


@dataclass(frozen=True)
class ColumnBanded:
    """Block-banded per column along index set ``axis``; lanes are the
    columns (every other dimension, column-major). ``J[r, c]`` at row level
    ``k``, column level ``k + d``, lane ``ℓ``:
    ``((((c·nvar + r)·(kl+ku+1) + (d+kl))·N + k)·L + ℓ``, ``r`` and ``c`` in
    sorted state order. Needs every coupling within a column, at a literal
    offset along ``axis``."""

    axis: str


@dataclass(frozen=True)
class Csr:
    """General sparse rows over the canonical state order (sorted names,
    species-major, cells column-major): ``rowptr``, sorted ``colind``,
    duplicates merged. Ghost reads (a column outside its variable) have no
    entry."""


StorageLayout = LaneBlocks | LaneSparseBlocks | ColumnBanded | Csr

# ── plans ───────────────────────────────────────────────────────────────────
#
# Every plan carries the canonical `states` (name, shape) and the `bands`
# (row var, column var, field shape) it was built from, which is what
# binding checks.


@dataclass
class LaneBlocksPlan:
    states: list[tuple[str, list[int]]]
    bands: list[tuple[str, str, list[int]]]
    cells: list[int]  # the lane domain (common state shape)
    block_vars: list[str]
    items: list[tuple[int, int]]  # (band k, block offset r + n·c)


@dataclass
class LaneSparseBlocksPlan:
    states: list[tuple[str, list[int]]]
    bands: list[tuple[str, str, list[int]]]
    cells: list[int]
    block_vars: list[str]
    lu_pattern: list[tuple[int, int]]  # (r, c), sorted; position = q
    items: list[tuple[int, int]]  # (band k, q)


@dataclass
class ColumnBandedPlan:
    states: list[tuple[str, list[int]]]
    bands: list[tuple[str, str, list[int]]]
    cells: list[int]
    axis: str
    axis_dim: int  # 0-based dimension of `axis`
    vars: list[str]
    kl: int
    ku: int
    items: list[tuple[int, int, int]]  # (band k, row level, offset without lane)


@dataclass
class CsrPlan:
    states: list[tuple[str, list[int]]]
    bands: list[tuple[str, str, list[int]]]
    rowptr: list[int]
    colind: list[int]
    items: list[tuple[int, int, int]]  # (band k, field cell, nz position)


ScatterPlan = LaneBlocksPlan | LaneSparseBlocksPlan | ColumnBandedPlan | CsrPlan


def _field_shape(en: JacEntry, ctx) -> list[int]:
    return list(ctx.shapes.get(en.u, [])) + [hi for _, _, hi in en.band.contracted]


def _flat(cell, shape) -> int:
    """Column-major 0-based flat index of a 1-based cell."""
    f, stride = 0, 1
    for c, n in zip(cell, shape, strict=False):
        f += (c - 1) * stride
        stride *= n
    return f


def _unflat(f: int, shape) -> list[int]:
    """1-based cell of a 0-based column-major flat index."""
    cell = []
    for n in shape:
        cell.append(f % n + 1)
        f //= n
    return cell


def _plan_error(layout, msg: str):
    raise ValueError(f"{type(layout).__name__}: {msg}")


def scatter_plan(
    obj,
    layout: StorageLayout,
    model_name: str | None = None,
    entries: list[JacEntry] | None = None,
) -> ScatterPlan:
    """Build the scatter plan of a model's state Jacobian for ``layout``.

    ``obj`` is an ``EsmFile``, a flattened system or a :class:`SysView`;
    ``entries`` (the reference's ``scatter_plan(entries, sv, layout)``) is a
    state band list to plan instead of the model's own. Raises
    ``ValueError`` naming the reason when the layout does not fit the
    Jacobian (a stencil under :class:`LaneBlocks`, a periodic wrap under
    :class:`ColumnBanded`, …), and always for an empty band list on a model
    with array states — that is the silent all-zero Jacobian, never a valid
    plan."""
    sv = sysview(obj, model_name)
    if entries is None:
        entries = jacobian_bands(sv, wrt="states")
    ctx = ctx_of(sv)
    names = view_states(sv)
    if not names:
        _plan_error(layout, "the model has no states")
    states = [(u, list(ctx.shapes.get(u, []))) for u in names]
    if not entries and any(s for _, s in states):
        _plan_error(
            layout,
            "empty band list for a model with array states (an all-zero Jacobian "
            "is never a valid plan — check the band calculus against this document)",
        )
    known = set(names)
    for en in entries:
        if en.v not in known:
            _plan_error(
                layout,
                f"band column `{en.v}` is not a state; scatter plans cover wrt = :states",
            )
    bands = [(en.u, en.v, _field_shape(en, ctx)) for en in entries]
    if isinstance(layout, LaneBlocks):
        return _lane_blocks(layout, entries, sv, states, bands)
    if isinstance(layout, LaneSparseBlocks):
        return _lane_sparse_blocks(layout, entries, sv, states, bands)
    if isinstance(layout, ColumnBanded):
        return _column_banded(layout, entries, sv, states, bands)
    if isinstance(layout, Csr):
        return _csr(entries, states, bands)
    raise TypeError(f"unknown storage layout {layout!r}")


def _lane_cells(layout, states) -> list[int]:
    shapes: list[list[int]] = []
    for _, s in states:
        if s not in shapes:
            shapes.append(s)
    if len(shapes) != 1:
        _plan_error(
            layout,
            "every state must share one cell shape; got "
            + ", ".join(sorted(str(s) for s in shapes)),
        )
    return shapes[0]


def _no_contracted_or_guard(layout, k: int, en: JacEntry) -> None:
    # A lane template reads one field cell per lane; a contracted band's
    # field has a cell per contracted point too, which it would drop, and a
    # guarded band's points differ from lane to lane.
    if en.band.contracted:
        _plan_error(layout, f"band {k + 1} ({en.u} ← {en.v}) has contracted dimensions")
    if en.band.guard is not None:
        _plan_error(layout, f"band {k + 1} ({en.u} ← {en.v}) has a `where` guard")


def _block_order(layout, entries, sv, states):
    for k, en in enumerate(entries):
        _no_contracted_or_guard(layout, k, en)
    st = detect_structure(entries, sv)
    if st != "block_diagonal":
        _plan_error(
            layout,
            f"needs a :block_diagonal Jacobian (per-cell coupling only); this one is :{st}",
        )
    fp = _block_diagonal_plan(entries)
    extra = sorted(u for u, _ in states if u not in fp.block_vars)
    return fp, list(fp.block_vars) + extra


def _lane_blocks(layout, entries, sv, states, bands) -> LaneBlocksPlan:
    cells = _lane_cells(layout, states)
    _, bv = _block_order(layout, entries, sv, states)
    pos = {u: i for i, u in enumerate(bv)}
    n = len(bv)
    items = sorted(
        ((k, pos[en.u] + n * pos[en.v]) for k, en in enumerate(entries)),
        key=lambda it: (it[1], it[0]),
    )
    return LaneBlocksPlan(states, bands, cells, bv, items)


def _lane_sparse_blocks(layout, entries, sv, states, bands) -> LaneSparseBlocksPlan:
    cells = _lane_cells(layout, states)
    fp, bv = _block_order(layout, entries, sv, states)
    n0 = len(fp.block_vars)
    lu = sorted(
        [(i, j) for i in range(n0) for j in range(n0) if fp.lu[i][j]]
        + [(i, i) for i in range(n0, len(bv))]
    )
    q = {rc: i for i, rc in enumerate(lu)}
    pos = {u: i for i, u in enumerate(bv)}
    items = sorted(
        ((k, q[(pos[en.u], pos[en.v])]) for k, en in enumerate(entries)),
        key=lambda it: (it[1], it[0]),
    )
    return LaneSparseBlocksPlan(states, bands, cells, bv, lu, items)


def _column_banded(layout: ColumnBanded, entries, sv, states, bands) -> ColumnBandedPlan:
    scalars = [u for u, s in states if not s]
    if scalars:
        _plan_error(
            layout,
            f"every state must be an array over `{layout.axis}`; {', '.join(scalars)} is scalar",
        )
    dimnames: list[list[str]] = []
    for u, _ in states:
        d = [str(x) for x in sv.variables[u].shape]
        if d not in dimnames:
            dimnames.append(d)
    if len(dimnames) != 1:
        _plan_error(
            layout,
            "every state must share one shape; got " + ", ".join(sorted(str(d) for d in dimnames)),
        )
    if layout.axis not in dimnames[0]:
        _plan_error(layout, f"`{layout.axis}` is not a dimension of the states {dimnames[0]}")
    ax = dimnames[0].index(layout.axis)
    cells = list(states[0][1])
    N = cells[ax]
    vars = [u for u, _ in states]
    vidx = {u: i for i, u in enumerate(vars)}
    nvar = len(vars)
    pts: list[tuple[int, int, int, int, int]] = []  # (band k, r, c, row level, offset d)
    for k, en in enumerate(entries):
        b = en.band
        _no_contracted_or_guard(layout, k, en)
        for m, (c, r) in enumerate(zip(b.cidx, b.ridx, strict=True)):
            cls = _offset_class(c, r)
            ok = cls in ("diag", "affine") if m == ax else cls == "diag"
            if not ok:
                where = (
                    f"at a non-literal offset along `{layout.axis}`"
                    if m == ax
                    else f"across columns (dimension {m + 1} is not the axis)"
                )
                _plan_error(layout, f"band {k + 1} ({en.u} ← {en.v}) couples cells {where}")
        lo, hi = b.rows[ax]
        for lev in range(lo, hi + 1):
            bind = {b.ridx[ax]: float(lev)} if isinstance(b.ridx[ax], str) else {}
            col = _eval_cidx(b.cidx[ax], bind)
            if not 1 <= col <= N:
                continue  # ghost read: no column
            pts.append((k, vidx[en.u], vidx[en.v], lev - 1, col - lev))
    kl = max([-p[4] for p in pts] + [0])
    ku = max([p[4] for p in pts] + [0])
    W = kl + ku + 1
    items = sorted(
        ((k, lev, ((c * nvar + r) * W + (d + kl)) * N + lev) for k, r, c, lev, d in pts),
        key=lambda it: (it[2], it[0]),
    )
    return ColumnBandedPlan(states, bands, cells, layout.axis, ax, vars, kl, ku, items)


def _csr(entries, states, bands) -> CsrPlan:
    base: dict[str, int] = {}
    n = 0
    for u, s in states:
        base[u] = n
        n += prod(s)
    shape = dict(states)
    pts: list[tuple[int, int, int, int]] = []  # (row, col, band k, field cell)
    for k, en in enumerate(entries):
        vshape = shape[en.v]
        for cell, cc, _, cidx in foreach_point(en.band):
            if not all(1 <= c <= s for c, s in zip(cidx, vshape, strict=False)):
                continue  # ghost read
            pts.append(
                (
                    base[en.u] + _flat(cell, shape[en.u]),
                    base[en.v] + _flat(cidx, vshape),
                    k,
                    _flat(list(cell) + list(cc), bands[k][2]),
                )
            )
    nz = sorted({(p[0], p[1]) for p in pts})
    pos = {rc: i for i, rc in enumerate(nz)}
    rowptr = [0] * (n + 1)
    for r, _ in nz:
        rowptr[r + 1] += 1
    for i in range(1, n + 1):
        rowptr[i] += rowptr[i - 1]
    items = sorted(
        ((k, f, pos[(r, c)]) for r, c, k, f in pts),
        key=lambda it: (it[2], it[0], it[1]),
    )
    return CsrPlan(states, bands, rowptr, [c for _, c in nz], items)


# ── serialization (the golden format, spec §8) ──────────────────────────────

_LAYOUT_TAG = {
    LaneBlocksPlan: "lane_blocks",
    LaneSparseBlocksPlan: "lane_sparse_blocks",
    ColumnBandedPlan: "column_banded",
    CsrPlan: "csr",
}


def serialize_scatter_plan(p: ScatterPlan) -> dict:
    """The plan as a JSON-ready dict — ``stable_json`` of it is the
    ``tests/goldens/<fixture>.<layout>.plan.json`` golden every binding
    reproduces."""
    d = {
        "layout": _LAYOUT_TAG[type(p)],
        "wrt": "states",
        "states": [{"name": u, "shape": list(s)} for u, s in p.states],
        "bands": [{"u": u, "v": v, "shape": list(s)} for u, v, s in p.bands],
        "items": [list(it) for it in p.items],
    }
    if isinstance(p, (LaneBlocksPlan, LaneSparseBlocksPlan)):
        d["cells"] = list(p.cells)
        d["block_vars"] = list(p.block_vars)
        if isinstance(p, LaneSparseBlocksPlan):
            d["lu_pattern"] = [list(rc) for rc in p.lu_pattern]
    elif isinstance(p, ColumnBandedPlan):
        d["cells"] = list(p.cells)
        d["axis"] = p.axis
        d["axis_dim"] = p.axis_dim
        d["vars"] = list(p.vars)
        d["kl"] = p.kl
        d["ku"] = p.ku
    else:
        d["rowptr"] = list(p.rowptr)
        d["colind"] = list(p.colind)
    return d


# ── binding to a source ─────────────────────────────────────────────────────


def plan_lanes(p: ScatterPlan) -> int:
    """The lane count of a plan: cells of a lane plan, columns (cells off the
    axis) of a column plan, 1 for CSR."""
    if isinstance(p, (LaneBlocksPlan, LaneSparseBlocksPlan)):
        return prod(p.cells)
    if isinstance(p, ColumnBandedPlan):
        return prod(p.cells) // p.cells[p.axis_dim]
    return 1


def _outlen(p: ScatterPlan, L: int) -> int:
    if isinstance(p, LaneBlocksPlan):
        return len(p.block_vars) ** 2 * L
    if isinstance(p, LaneSparseBlocksPlan):
        return len(p.lu_pattern) * L
    if isinstance(p, ColumnBandedPlan):
        return len(p.vars) ** 2 * (p.kl + p.ku + 1) * p.cells[p.axis_dim] * L
    return len(p.colind)


def _column_cell(p: ColumnBandedPlan, lev: int, lane: int) -> int:
    # Field cell (0-based flat) of column `lane` at row level `lev`.
    f, stride, rest = 0, 1, lane
    for m, n in enumerate(p.cells):
        if m == p.axis_dim:
            c = lev
        else:
            c = rest % n
            rest //= n
        f += c * stride
        stride *= n
    return f


def _contributions(p: ScatterPlan, L: int) -> list[tuple[int, int, int]]:
    # (band k, field cell, output position) per contribution, in accumulation
    # order: plan order, lanes innermost — so every output element sums its
    # contributions in the same order whatever the lane capacity.
    nl = plan_lanes(p)
    if isinstance(p, (LaneBlocksPlan, LaneSparseBlocksPlan)):
        return [(k, ln, off * L + ln) for k, off in p.items for ln in range(nl)]
    if isinstance(p, ColumnBandedPlan):
        return [
            (k, _column_cell(p, lev, ln), off * L + ln)
            for k, lev, off in p.items
            for ln in range(nl)
        ]
    return list(p.items)


@dataclass
class BoundScatter:
    """A scatter plan checked against one source's band list at a lane
    capacity (:func:`bind_plan`). ``pairs`` lists ``(field, pos)``: the
    evaluation document's band-field cell ``field`` (a name of its variable
    map, ``"J_k[…]"``) goes to the 0-based output position ``pos``. Executing
    it is the consumer's: zero the ``len`` outputs, then ``out[pos] +=
    du[field]`` for each pair in order (several fields may land on one
    position)."""

    plan: ScatterPlan
    lanes: int
    len: int
    pairs: list[tuple[str, int]] = field(default_factory=list)


def _state_cells(states) -> list[str]:
    # Every state cell name, in canonical order (the rows of a `Csr` plan).
    names: list[str] = []
    for u, s in states:
        if not s:
            names.append(u)
        else:
            names.extend(_cellname(u, c) for c in _product([(1, n) for n in s]))
    return names


def _bind_error(msg: str):
    raise ValueError("bind_plan: " + msg)


def bind_plan(
    obj, plan: ScatterPlan, lanes: int | None = None, model_name: str | None = None
) -> BoundScatter:
    """Check ``plan`` against the state Jacobian of ``obj`` (an ``EsmFile``, a
    flattened system or a :class:`SysView`) and precompute its gather.

    Binding is strict: the plan's band list (row var, column var, field shape
    per band) and canonical states must equal the source's, and every
    contribution is decoded against the source's own bands — each must land
    on the (row, column) cell its band point differentiates, and every band
    point with a column must be covered exactly once. A plan from another
    build, another document or another stencil is refused (``ValueError``),
    never silently misapplied. ``lanes`` is the lane capacity ``L`` of a lane
    or column layout (default: exactly the lane count; padding lanes stay
    zero)."""
    sv = sysview(obj, model_name)
    ctx = ctx_of(sv)
    entries = jacobian_bands(sv, wrt="states")
    field_shapes = [_field_shape(en, ctx) for en in entries]
    if not (
        len(plan.bands) == len(entries)
        and all(
            en.u == u and en.v == v and field_shapes[k] == list(s)
            for k, (en, (u, v, s)) in enumerate(zip(entries, plan.bands, strict=True))
        )
    ):
        _bind_error(
            "the plan was built from a different band list than this "
            f"source ({len(plan.bands)} vs {len(entries)} bands, or a "
            "band's variables or field shape differ)"
        )
    names = _state_cells(plan.states)
    mine = _state_cells([(u, list(ctx.shapes.get(u, []))) for u in view_states(sv)])
    if names != mine:
        extra = sorted(set(names) ^ set(mine))
        _bind_error(
            "the plan's states are not this source's"
            + (f" (e.g. `{extra[0]}`)" if extra else " (order or duplicate state cells)")
        )
    nl = plan_lanes(plan)
    L = nl if lanes is None else int(lanes)
    if L < nl:
        _bind_error(f"lane capacity {L} < {nl} lanes")
    # The source's own points: (band k, field cell) → (row, col) cell names,
    # or None for a ghost read (a value with no storage position).
    vshape = {u: list(s) for u, s in plan.states}
    want: dict[tuple[int, int], tuple[str, str] | None] = {}
    for k, en in enumerate(entries):
        sh = field_shapes[k]
        vs = vshape[en.v]
        for cell, cc, _, cidx in foreach_point(en.band):
            inside = all(1 <= c <= n for c, n in zip(cidx, vs, strict=False))
            want[(k, _flat(list(cell) + list(cc), sh))] = (
                (_cellname(en.u, cell), _cellname(en.v, cidx)) if inside else None
            )
    targets = scatter_targets(plan, lanes=L)
    seen: set[tuple[int, int]] = set()
    pairs: list[tuple[str, int]] = []
    for k, f, pos in _contributions(plan, L):
        if (k, f) in want:
            t = want[(k, f)]
            if t is None:
                _bind_error(
                    f"band {k + 1} field cell {f} is a ghost read; the plan gives it a "
                    "storage position"
                )
            if t != targets[pos]:
                at = targets[pos] if targets[pos] is not None else "padding"
                _bind_error(
                    f"band {k + 1} field cell {f} is {t[0]} ← {t[1]} here; the plan puts it at {at}"
                )
            if (k, f) in seen:
                _bind_error(f"band {k + 1} field cell {f} is scattered twice")
            seen.add((k, f))
        # else: outside the band's region, so zero
        pairs.append((_cellname(f"J_{k + 1}", _unflat(f, plan.bands[k][2])), pos))
    for kf in sorted(want):
        t = want[kf]
        if t is not None and kf not in seen:
            _bind_error(f"the plan drops band {kf[0] + 1} field cell {kf[1]} ({t[0]} ← {t[1]})")
    return BoundScatter(plan, L, _outlen(plan, L), pairs)


def scatter_targets(p: ScatterPlan, lanes: int | None = None) -> list[tuple[str, str] | None]:
    """The (row, column) state-cell names each output position of ``p`` holds
    at lane capacity ``lanes`` (``None`` for padding lanes and out-of-range
    band slots). For checking a consumer's decode, and for tests."""
    L = plan_lanes(p) if lanes is None else int(lanes)
    out: list[tuple[str, str] | None] = [None] * _outlen(p, L)
    nl = plan_lanes(p)
    if isinstance(p, LaneBlocksPlan):
        n = len(p.block_vars)
        for c in range(n):
            for r in range(n):
                for ln in range(nl):
                    cell = _unflat(ln, p.cells)
                    out[(r + n * c) * L + ln] = (
                        _cellname(p.block_vars[r], cell),
                        _cellname(p.block_vars[c], cell),
                    )
    elif isinstance(p, LaneSparseBlocksPlan):
        for q, (r, c) in enumerate(p.lu_pattern):
            for ln in range(nl):
                cell = _unflat(ln, p.cells)
                out[q * L + ln] = (
                    _cellname(p.block_vars[r], cell),
                    _cellname(p.block_vars[c], cell),
                )
    elif isinstance(p, ColumnBandedPlan):
        nvar = len(p.vars)
        N = p.cells[p.axis_dim]
        W = p.kl + p.ku + 1
        for c in range(nvar):
            for r in range(nvar):
                for j in range(W):
                    for lev in range(N):
                        col = lev + j - p.kl
                        if not 0 <= col < N:
                            continue
                        for ln in range(nl):
                            rc = _unflat(_column_cell(p, lev, ln), p.cells)
                            cc = list(rc)
                            cc[p.axis_dim] = col + 1
                            out[(((c * nvar + r) * W + j) * N + lev) * L + ln] = (
                                _cellname(p.vars[r], rc),
                                _cellname(p.vars[c], cc),
                            )
    else:
        names = _state_cells(p.states)
        for r in range(len(p.rowptr) - 1):
            for q in range(p.rowptr[r], p.rowptr[r + 1]):
                out[q] = (names[r], names[p.colind[q]])
    return out
