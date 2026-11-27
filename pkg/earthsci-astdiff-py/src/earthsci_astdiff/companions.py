"""Time companions (spec §4.2; the reference's companions.jl).

The time bands keep the chain through the observed that move with ``t`` as
named steps. Each such observed ``o`` gets ONE derived observed, its
companion ``_dt_<o>`` = ∂o/∂t (data held fixed), and every band reads it
through the chain rule instead of differentiating ``o``'s definition again
inside every row::

    a read of o at column index φ, with ∂f/∂o = c there   ↦   c · _dt_o[φ]

A companion's own definition is built the same way from ``o``'s definition:
its direct derivative in ``t`` plus, for each moving observed ``o′`` it
reads, ``∂def_o/∂o′ · _dt_o′``. Nothing is inlined and then deduplicated:
the moving observed are extra COLUMNS of the band calculus (as a recurrence
is, §4.5), so each definition is differentiated once.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import replace

from earthsci_ast.esm_types import ExprNode
from earthsci_ast.expression import free_variables

from .bands import Band, BandError, Ctx, _range_of, bands_in, merge_bands, normalize_band
from .clip import _index_bounds, _index_box
from .expr_helpers import mul, op, region_bound, ser_expr, skey
from .inline import all_names, inline_observed, subst_free
from .reductions import conj_list, conjuncts, fresh
from .simplify_branches import simp
from .system import _moving_observed, _subscript_observed


class Companions:
    """The time pass's companion layout (companions.jl ``_Companions``):
    ``names`` maps every observed that moves with ``t`` to its companion's
    name (assigned over all of them, so a name never depends on the pass);
    ``cols`` is the set that are columns of this pass (those a column index
    does not need inlined); ``inline`` the pass's inlining map (the observed
    a column index reads); ``ctx`` the band context with every column's
    extents; ``bands[o]`` the companion's time bands once computed (empty:
    ``∂o/∂t ≡ 0``, and a read of ``o`` contributes nothing)."""

    def __init__(self, names, cols, inline, obs, rank_obs, ctx, scalars, co):
        self.names = names
        self.cols = cols
        self.inline = inline
        self.obs = obs  # the observed definitions, recurrences left out
        self.rank_obs = rank_obs  # every observed definition (ranks, extents)
        self.ctx = ctx
        self.scalars = scalars
        self.co = co
        self.bands: dict[str, list[Band]] = {}
        self.pending: set[str] = set()  # companions whose bands are being built


def companion_names(sv, moving) -> dict[str, str]:
    """The companion name of each observed in ``moving``, in name order:
    ``_dt_<o>`` (``P._dt_<v>`` for a scoped ``P.v``, so it lives in ``o``'s
    own component), with ``_`` appended until clear of every variable,
    component path and index-set name of the source and of the companion
    names before it."""
    taken = {str(n) for n in sv.variables}
    taken |= {str(n) for n in sv.index_sets}
    # a companion named like a component path would be both a variable and a
    # subsystem of a derived model
    component_paths(taken, sv.variables)
    out = {}
    for o in sorted(moving):
        i = o.rfind(".")
        base = f"_dt_{o}" if i < 0 else f"{o[:i]}._dt_{o[i + 1 :]}"
        out[o] = fresh(taken, base, "_")
    return out


def component_paths(taken: set, names) -> set:
    """Every component path of the scoped names ``names`` (``A`` and ``A.B``
    of ``A.B.v``), added to ``taken``."""
    for n in names:
        n = str(n)
        for i, ch in enumerate(n):
            if ch == ".":
                taken.add(n[:i])
    return taken


def companion_setup(
    sv, ctx: Ctx, obs: dict, scalars, co: bool, rank_obs: dict | None = None
) -> Companions:
    """The companion layout of the time pass over the view. ``obs`` is every
    observed definition with the recurrences left out (the pass never
    inlines a recurrence), ``rank_obs`` every observed definition,
    recurrences included (what an elementwise read at a cell and an
    undeclared shape are ranked from); ``scalars`` the declared scalars that
    are not observed."""
    if rank_obs is None:
        rank_obs = obs
    moving = _moving_observed(obs, {"t"})
    sub = _subscript_observed(set(), obs, sv.equations, sorted(moving))
    cols = moving - sub
    shapes = dict(ctx.shapes)
    memo: dict = {}
    for o in sorted(cols):
        ext = _obs_extents(o, sv, rank_obs, ctx, memo, set())
        if ext:
            shapes[o] = ext
    ctx2 = replace(ctx, shapes=shapes)
    return Companions(
        companion_names(sv, moving),
        cols,
        {n: obs[n] for n in sub},
        obs,
        rank_obs,
        ctx2,
        set(scalars),
        co,
    )


def _refuse(o: str) -> BandError:
    return BandError(
        f"the extents of observed `{o}` (declared without a shape) cannot be read "
        "from its definition; declare its shape"
    )


def _obs_extents(o: str, sv, obs: dict, ctx: Ctx, memo: dict, seen: set) -> list[int]:
    """The extents of observed ``o``: declared, or inferred from its
    definition — a ``faq``'s output ranges, a ``makearray``'s regions, or the
    one extent list every array operand of an elementwise definition shares.
    A definition whose extents cannot be read at build time (operands of
    different extents, whose union needs index-set names; a partial
    ``index`` read) is refused."""
    if o in ctx.shapes:
        return list(ctx.shapes[o])
    var = sv.variables.get(o)
    # a named intermediate (a time pair's `a2` steps, §4.4) is no variable of
    # the view: declared without a shape, its extents are its definition's
    if var is None and o not in obs:
        return []
    if var is not None and getattr(var, "shape", None) is not None:
        return []  # declared `[]`: a scalar
    if o not in obs:
        return []
    if o in memo:
        return memo[o]
    if o in seen:
        raise BandError(f"cyclic observed definition through `{o}`")
    seen.add(o)
    ext = _expr_extents(obs[o], o, sv, obs, ctx, memo, seen)
    memo[o] = ext
    return ext


def _expr_extents(e, o: str, sv, obs: dict, ctx: Ctx, memo: dict, seen: set) -> list[int]:
    if isinstance(e, str):
        return _obs_extents(e, sv, obs, ctx, memo, seen)
    if not isinstance(e, ExprNode):
        return []
    if e.op == "faq":
        out = []
        for x in e.output_idx or []:
            if not isinstance(x, str):
                raise _refuse(o)
            r = (e.ranges or {}).get(x)
            if r is None:
                raise _refuse(o)
            lo, hi = _range_of(r, ctx)
            if lo != 1:
                raise _refuse(o)
            out.append(hi)
        return out
    if e.op == "makearray":
        if not e.regions:
            raise _refuse(o)
        out = [0] * len(e.regions[0])
        for region in e.regions:
            for d, r in enumerate(region):
                out[d] = max(out[d], region_bound(r[1]))
        return out
    if e.op == "index":
        # a read at a cell is a scalar; a read with fewer indices than its
        # array's rank is an array slice, whose extents are not read here
        args = e.args or []
        if args and isinstance(args[0], str):
            ext = _obs_extents(args[0], sv, obs, ctx, memo, seen)
            if len(args) - 1 < len(ext):
                raise _refuse(o)
        return []
    out: list[int] = []
    for a in e.args or []:
        ext = _expr_extents(a, o, sv, obs, ctx, memo, seen)
        if not ext:
            continue
        if out and out != ext:
            raise _refuse(o)
        out = ext
    return out


def _faq_over(names: list, shape: list, body) -> ExprNode:
    return ExprNode(
        op="faq",
        args=[],
        output_idx=list(names),
        expr=body,
        ranges={n: [1, shape[d]] for d, n in enumerate(names)},
    )


def companion_bands(cs: Companions, o: str) -> list[Band]:
    """The time bands of companion ``o`` (row ``o`` over its shape), computed
    once."""
    if o in cs.bands:
        return cs.bands[o]
    # moving observed that read each other (an algebraic loop that is not a
    # recognized recurrence) have no companion: refused, never followed
    if o in cs.pending:
        raise BandError(f"cyclic observed definition through `{o}`")
    cs.pending.add(o)
    shape = cs.ctx.shapes.get(o, [])
    d = cs.obs[o]
    if not shape:
        rhs = o
    else:
        names = _row_names_of(d, len(shape))
        rhs = _faq_over(names, shape, op("index", o, *names))
    inl = dict(cs.inline)
    inl[o] = d
    bs = time_row_bands(cs, rhs, o, inl, shape)
    cs.pending.discard(o)
    cs.bands[o] = bs
    return bs


def _row_names_of(d, k: int) -> list[str]:
    """The row index names of a companion row: the definition's own output
    indices when it is a ``faq`` naming one per dimension, else ``_r<d>``
    clear of every name of the definition."""
    if isinstance(d, ExprNode) and d.op == "faq":
        oi = list(d.output_idx or [])
        if len(oi) == k and all(isinstance(x, str) for x in oi) and len(set(oi)) == len(oi):
            return oi
    taken = all_names(d)
    return [fresh(taken, f"_r{i}", "_") for i in range(1, k + 1)]


def time_row_bands(cs: Companions, rhs, u: str, inl: dict, shape_u: list) -> list[Band]:
    """The time bands of one row (a state equation's right-hand side
    ``rhs``, or a companion row), all with column ``t``: the direct bands in
    ``t`` and each band of a moving-observed column composed with that
    observed's companion, summed where they share a box, in column-name
    order."""
    ctx = cs.ctx
    from .jacobian import inline_row

    rhs = inline_row(rhs, u, inl, ctx, rank_obs=cs.rank_obs, scalars=cs.scalars)
    out: list[Band] = []
    for v in sorted(x for x in free_variables(rhs) if x == "t" or x in cs.cols):
        raw = bands_in(rhs, v, ctx, shape_u, cs.co)
        if not cs.co:
            raw = [replace(b, coef=None) for b in raw]
        bs = [normalize_band(b) for b in merge_bands(raw)]
        if v == "t":
            out.extend(bs)
            continue
        if not companion_bands(cs, v):
            continue
        shape_v = ctx.shapes.get(v, [])
        for b in bs:
            if b.join is not None:
                raise BandError("a time companion read through a joined faq node is not supported")
            out.append(_through_companion(b, v, cs.names[v], shape_v, cs.co))
    return [normalize_band(b) for b in merge_bands(_sum_same_box(out, cs.co))]


def _check_read_rank(b: Band, o: str, shape: list) -> None:
    """A band's column index must read every dimension of the moving
    observed ``o``, whose extents are ``shape``: a read of another rank has
    no companion cell (an undeclared observed whose extents were read wrong,
    or a malformed read), refused rather than read out of bounds."""
    if len(b.cidx) != len(shape):
        raise BandError(
            f"observed `{o}` of rank {len(shape)} is read with {len(b.cidx)} indices; declare its shape"
        )


def _through_companion(b: Band, o: str, name: str, shape: list, co: bool) -> Band:
    """Band ``b`` (column: a moving observed of shape ``shape``, read at
    ``b.cidx``) composed with that observed's companion ``name``: column
    ``t``, coefficient ``coef · name[cidx…]`` (the bare name for a scalar).
    A column index not provably inside the observed's extent over the band's
    box guards the term by ``1 <= φ_d <= n_d``. Contracted dimensions become
    a scalar ``faq`` sum over their ranges, the band's ``where`` its filter,
    so a time band keeps the row's shape."""
    _check_read_rank(b, o, shape)
    if not co:
        return Band(list(b.rows), list(b.ridx), [], None, [], b.guard if not b.contracted else None)
    read = name if not b.cidx else op("index", name, *b.cidx)
    term = simp(mul(b.coef, read))
    box = _index_box(b)
    conds = []
    for d, c in enumerate(b.cidx):
        bd = _index_bounds(c, box)
        if bd is not None and bd[0] >= 1 and bd[1] <= shape[d]:
            continue
        conds.append(op(">=", c, 1))
        conds.append(op("<=", c, shape[d]))
    g = conj_list(conds)
    if g is not None:
        term = op("ifelse", g, term, 0)
    if not b.contracted:
        return Band(list(b.rows), list(b.ridx), [], term, [], b.guard)
    total = ExprNode(
        op="faq",
        args=[],
        output_idx=[],
        reduce="+",
        expr=term,
        ranges={n: [lo, hi] for n, lo, hi in b.contracted},
        filter=b.guard,
    )
    return Band(list(b.rows), list(b.ridx), [], total, [], None)


def _sum_same_box(bs: list[Band], co: bool) -> list[Band]:
    """Bands of one row with the same box (rows, row names and ``where``)
    summed into one: an n-ary ``+`` of their coefficients in order, at the
    first one's position."""
    order: list = []
    groups: dict = {}
    for i, b in enumerate(bs):
        k = (
            tuple(tuple(r) for r in b.rows),
            tuple(b.ridx),
            None if b.guard is None else skey(b.guard),
        )
        if k not in groups:
            order.append(k)
            groups[k] = []
        groups[k].append(i)
    out = []
    for k in order:
        idx = groups[k]
        b = bs[idx[0]]
        if len(idx) > 1:
            b = replace(b, coef=op("+", *(bs[i].coef for i in idx)) if co else None)
        out.append(b)
    return out


def _companion_definition(bs: list[Band], shape: list):
    """A companion's definition from its time bands: the scalar sum of their
    terms, or one ``faq`` over the observed's whole shape whose body sums
    them, each band's term gated by its region's membership (and its
    ``where``) unless it covers the whole shape. Row names are the bands'
    own when they all share them, else ``_r<d>`` clear of every name the
    bands hold."""

    def gate(coef, conds):
        g = conj_list(conds)
        return coef if g is None else op("ifelse", g, coef, 0)

    def total(ts):
        return ts[0] if len(ts) == 1 else op("+", *ts)

    if not shape:
        return total([gate(b.coef, conjuncts(b.guard) if b.guard is not None else []) for b in bs])
    k = len(shape)
    first = list(bs[0].ridx)
    if (
        all(isinstance(r, str) for r in first)
        and len(set(first)) == len(first)
        and all(list(b.ridx) == first for b in bs)
    ):
        names = first
    else:
        taken: set = set()
        for b in bs:
            taken |= {r for r in b.ridx if isinstance(r, str)}
            all_names(b.coef, taken)
            if b.guard is not None:
                all_names(b.guard, taken)
        names = [fresh(taken, f"_r{i}", "_") for i in range(1, k + 1)]
    # each band: its box over the shape and its term (row names unified,
    # gated by its `where` only)
    boxes = []
    terms = []
    for b in bs:
        m = {}
        box = []
        for d in range(k):
            r = b.ridx[d]
            if isinstance(r, str):
                if r != names[d]:
                    m[r] = names[d]
                box.append(tuple(b.rows[d]))
            else:
                box.append((r, r))
        coef = subst_free(b.coef, m) if m else b.coef
        conds = [] if b.guard is None else conjuncts(subst_free(b.guard, m) if m else b.guard)
        boxes.append(box)
        terms.append(gate(coef, conds))

    def faq_box(box, body):
        return ExprNode(
            op="faq",
            args=[],
            output_idx=list(names),
            expr=body,
            ranges={n: [box[d][0], box[d][1]] for d, n in enumerate(names)},
        )

    full = [(1, n) for n in shape]
    if all(bx == full for bx in boxes):
        return faq_box(full, total(terms))
    # A region narrower than the shape: a `makearray`, never a whole-shape
    # gate (a reader built per cell, a lane program, needs the region). The
    # boxes are cut at every box's bounds into disjoint cells, each the sum of
    # the bands that cover it, over zero where none does
    # (companions.jl `_companion_definition`).
    cuts = []
    for d in range(k):
        c = {1, shape[d] + 1}
        for bx in boxes:
            c |= {bx[d][0], bx[d][1] + 1}
        cuts.append(sorted(c))
    regions = []
    values = []
    ncells = 0
    for pos in itertools.product(*(range(len(cuts[d]) - 1) for d in range(k))):
        cell = [(cuts[d][pos[d]], cuts[d][pos[d] + 1] - 1) for d in range(k)]
        cover = [
            i
            for i, bx in enumerate(boxes)
            if all(bx[d][0] <= cell[d][0] and cell[d][1] <= bx[d][1] for d in range(k))
        ]
        if not cover:
            continue
        ncells += math.prod(hi - lo + 1 for lo, hi in cell)
        regions.append([[lo, hi] for lo, hi in cell])
        values.append(faq_box(cell, total([terms[i] for i in cover])))
    if ncells < math.prod(shape):
        regions.insert(0, [[1, n] for n in shape])
        values.insert(0, 0)
    return region_array(regions, values, shape)


def region_array(regions: list, values: list, shape: list):
    """An array over ``shape`` given by ``makearray`` regions: one ``faq``
    over the whole extent whose body reads the ``makearray`` at its own
    cell, so the definition stays an array aggregate (a lane program builds
    an observed only from one) while each region keeps its own box. Its names
    are ``_m<d>``, ``_`` appended until clear of every name of the regions'
    values (companions.jl ``_region_array``)."""
    taken: set = set()
    for v in values:
        all_names(v, taken)
    names = [fresh(taken, f"_m{d}", "_") for d in range(1, len(shape) + 1)]
    ma = ExprNode(op="makearray", args=[], regions=regions, values=values)
    return _faq_over(names, shape, op("index", ma, *names))


def time_companions(sv, bands: list[Band], extra: dict | None = None) -> list:
    """The companions the band list ``bands`` reads, directly or through
    other companions' definitions, as ``(name, observed, definition)`` in the
    observed's name order — what the block's ``intermediates`` field and every
    derived document of the list carry (spec §4.2, §8.1). Empty unless some
    coefficient (or ``where``) reads a companion name, which no source name
    can be."""
    from .recurrence import recurrences
    from .system import ctx_of, observed_defs

    out: list = []
    if not bands:
        return out
    # Every companion name's last segment starts with `_dt_`: a list none of
    # whose coefficients reads such a name (every block of another axis)
    # reads no companion, and the analysis below is skipped.
    cand = []
    for b in bands:
        for x in (b.coef, b.guard):
            if x is None:
                continue
            cand.extend(n for n in free_variables(x) if n.rsplit(".", 1)[-1].startswith("_dt_"))
    if not cand:
        return out
    obs = observed_defs(sv)
    ctx = ctx_of(sv)
    rn = {r.name for r in recurrences(sv, obs, ctx)}
    obs2 = {k: v for k, v in obs.items() if k not in rn}
    # a time pair's `a2` intermediates are observed of the time pass (§4.4)
    if extra:
        obs2.update(extra)
        obs = {**obs, **extra}
    moving = _moving_observed(obs2, {"t"})
    if not moving:
        return out
    names = companion_names(sv, moving)
    byname = {n: o for o, n in names.items()}
    read = [n for n in cand if n in byname]
    if not read:
        return out
    from .jacobian import declared_scalars

    cs = companion_setup(sv, ctx, obs2, declared_scalars(sv, obs), True, rank_obs=obs)
    defs: dict = {}
    while read:
        n = read.pop()
        if n in defs:
            continue
        o = byname[n]
        d = _companion_definition(companion_bands(cs, o), cs.ctx.shapes.get(o, []))
        defs[n] = d
        read.extend(m for m in free_variables(d) if m in byname and m not in defs)
    for o in sorted(moving):
        n = names[o]
        if n in defs:
            out.append((n, o, defs[n]))
    return out


def add_companions(m: dict, sv, comps: list) -> None:
    """The companions and named intermediates as observed of a derived model
    (spec §8.1): each declared like its observed (``{"type": "unknown"}``,
    with the observed's declared ``shape`` when it has one, unless it adds
    contracted dimensions, §4.6) and defined by a bare-LHS equation, after
    the source's equations."""
    for c in comps:
        n, o = c[0], c[1]
        d: dict = {"type": "unknown"}
        # a named step of another axis (§4.6) is declared without a shape:
        # its definition gives its extents
        noshape = len(c) >= 4 and c[3] != 0
        v = sv.variables.get(o)
        shp = getattr(v, "shape", None) if v is not None else None
        if not noshape and shp:
            d["shape"] = list(shp)
        m["variables"][n] = d
    for c in comps:
        m["equations"].append({"lhs": c[0], "rhs": ser_expr(c[2])})


def companions_field(comps: list) -> list:
    """The block's ``intermediates`` field (spec §4, §4.2, §4.6): ``[{name,
    of, rhs}]``."""
    return [{"name": c[0], "of": c[1], "rhs": ser_expr(c[2])} for c in comps]


# ── second-order pairs ``["time", a2]`` (spec §4.4) ──────────────────────────


def time_pair_bands(sv, a2: str, forcing, memo, t2) -> list | None:
    """A time pair taken with ``t`` SECOND (companions.jl
    ``_time_pair_bands``). With every branch frozen (spec §3.2) each branch
    is smooth, so the mixed partials commute: ∂²f/∂t∂x = ∂/∂t (∂f/∂x). Each
    first-order entry of the ``a2`` axis has its coefficient's time bands
    taken by the time pass's own rule — the observed that move with ``t``
    are columns composed with their companions, never inlined. The ``a2``
    pass keeps its named steps (§4.6), and its named intermediates are
    observed of the time pass: one that moves with ``t`` is a column composed
    with its own companion. ``None`` when a recurrence is active on either axis: that
    pair keeps the inlining path, as the time axis does."""
    from .jacobian import declared_scalars
    from .recurrence import active, recurrences
    from .second_order import JacEntry2
    from .system import ctx_of, observed_defs

    ctx = ctx_of(sv)
    obs = observed_defs(sv)
    recs = recurrences(sv, obs, ctx)
    if recs and active(recs, obs, set(t2) | {"t"}):
        return None
    # the a2 pass keeps its named steps (§4.6): its coefficients read named
    # intermediates, which are observed of the time pass here, each moving
    # with `t` a column composed with its own companion — never inlined
    from .jacobian import _extended_pass
    from .steps import intermediates

    e2 = _extended_pass(sv, ctx, obs, a2, forcing, [], companions=True).rows
    inter = {n: d for n, _, d, k in intermediates(sv, [en.band for en in e2], forcing) if k == 1}
    names = {r.name for r in recs}
    cs = companion_setup(
        sv,
        ctx,
        {**{k: v for k, v in obs.items() if k not in names}, **inter},
        declared_scalars(sv, obs),
        True,
        rank_obs={**obs, **inter},
    )
    out = []
    for en in e2:
        for b in _time_coef_bands(cs, en.band):
            out.append(JacEntry2(en.u, "t", en.v, b, 0))
    return out


def _time_coef_bands(cs: Companions, b: Band) -> list[Band]:
    """The time bands of a first-order band's coefficient, broadcast over its
    rows (its contracted names bound too): :func:`time_row_bands` on a
    coefficient. Every band keeps the first-order column indices, which
    become the pair's ``col2_idx``; a companion read whose index is not
    provably inside the observed's extent is gated as on the time axis. A
    site under a nested sum adds contracted dimensions (spec §4.4), which
    stay contracted."""
    from .second_order import _coef_bands

    ctx = cs.ctx
    coef = inline_observed(b.coef, cs.inline, [None], ctx.shapes, cs.rank_obs, cs.scalars, ctx.axes)
    b0 = replace(b, coef=coef)
    n = len(b.cidx)
    out: list[Band] = []
    for v in sorted(x for x in free_variables(coef) if x == "t" or x in cs.cols):
        bs = [normalize_band(x) for x in merge_bands(_coef_bands(b0, v, ctx))]
        if v == "t":
            out.extend(bs)
            continue
        if not companion_bands(cs, v):
            continue
        shape_v = ctx.shapes.get(v, [])
        for x in bs:
            out.append(_pair_through_companion(x, n, v, cs.names[v], shape_v))
    return [normalize_band(x) for x in merge_bands(_sum_same_box2(out))]


def _pair_through_companion(x: Band, n: int, o: str, name: str, shape: list) -> Band:
    """Band ``x`` of a coefficient (column indices: the first-order ones,
    then the moving observed's ``φ``) composed with that observed's
    companion ``name``: coefficient ``coef · name[φ…]``, gated by
    ``1 <= φ_d <= n_d`` where ``φ_d`` is not provably in extent over the
    band's box, column indices back to the first ``n``."""
    phi = list(x.cidx[n:])
    if len(phi) != len(shape):
        raise BandError(
            f"observed `{o}` of rank {len(shape)} is read with {len(phi)} indices; declare its shape"
        )
    read = name if not phi else op("index", name, *phi)
    term = simp(mul(x.coef, read))
    box = _index_box(x)
    conds = []
    for d, c in enumerate(phi):
        bd = _index_bounds(c, box)
        if bd is not None and bd[0] >= 1 and bd[1] <= shape[d]:
            continue
        conds.append(op(">=", c, 1))
        conds.append(op("<=", c, shape[d]))
    g = conj_list(conds)
    if g is not None:
        term = op("ifelse", g, term, 0)
    return replace(x, cidx=list(x.cidx[:n]), coef=term)


def _sum_same_box2(bs: list[Band]) -> list[Band]:
    """:func:`_sum_same_box` for pair bands: the box also holds the column
    indices and the contracted dimensions."""
    order: list = []
    groups: dict = {}
    for i, b in enumerate(bs):
        k = (
            tuple(tuple(r) for r in b.rows),
            tuple(b.ridx),
            tuple(skey(c) for c in b.cidx),
            tuple(tuple(c) for c in b.contracted),
            None if b.guard is None else skey(b.guard),
        )
        if k not in groups:
            order.append(k)
            groups[k] = []
        groups[k].append(i)
    out = []
    for k in order:
        idx = groups[k]
        b = bs[idx[0]]
        if len(idx) > 1:
            b = replace(b, coef=op("+", *(bs[i].coef for i in idx)))
        out.append(b)
    return out
