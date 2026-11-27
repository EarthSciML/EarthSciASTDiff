"""Named steps on every axis (spec §4.6; the reference's steps.jl).

The bands of a row are composed through the observed graph rather than read
off the row with every observed inlined. Each observed that moves with the
axis is a column of the band calculus; its own definition is differentiated
once, and its bands onto the axis's targets are composed once and memoized,
so the cost grows with the number of observed, not with the number of paths
through them. Nothing is inlined and then deduplicated, and there is no
threshold.
"""

from __future__ import annotations

import re
from dataclasses import replace

from earthsci_ast.esm_types import ExprNode
from earthsci_ast.expression import free_variables

from .bands import (
    Band,
    BandError,
    Ctx,
    bands_in,
    join_key,
    join_names,
    merge_bands,
    normalize_band,
    rename_join,
)
from .companions import _faq_over, _obs_extents, _row_names_of, region_array
from .expr_helpers import is_zero, mul, op, skey, stable_json
from .inline import InlineError, all_names, subst_free
from .reductions import conj, conj_list, conjuncts, fresh, split_guard
from .simplify_branches import canon_lits, simp
from .simplify_port import simplify_jl
from .system import _moving_observed, _subscript_observed

_CMP = {
    ">=": lambda a, b: a >= b,
    "<=": lambda a, b: a <= b,
}


class Steps:
    """One pass's named-step layout (steps.jl ``_Steps``): ``targets`` the
    axis's columns; ``cols`` the observed that move with them and are columns
    too (those a column index does not need inlined); ``inline`` the pass's
    inlining map (the observed a column index reads); ``ctx`` the band
    context with every column's extents; ``composed[o]`` the bands of
    observed ``o`` onto the targets once composed."""

    def __init__(self, targets, cols, inline, obs, ctx, scalars, co, letter="u", taken=None):
        self.letter = letter
        self.taken = taken if taken is not None else set()
        self.inters: dict = {}
        self.targets = targets
        self.cols = cols
        self.inline = inline
        self.obs = obs
        self.ctx = ctx
        self.scalars = scalars
        self.co = co
        self.composed: dict = {}
        self.busy: set = set()


def steps_setup(sv, ctx: Ctx, obs: dict, targets, scalars, co: bool, letter: str = "u") -> Steps:
    """The layout of a pass over ``targets``. ``obs`` is every observed
    definition with the recurrences left out; ``scalars`` the declared
    scalars that are not observed."""
    tset = set(targets)
    moving = _moving_observed(obs, tset)
    sub = _subscript_observed(set(), obs, sv.equations, sorted(moving))
    cols = moving - sub
    shapes = dict(ctx.shapes)
    memo: dict = {}
    for o in sorted(cols):
        ext = _obs_extents(o, sv, obs, ctx, memo, set())
        if ext:
            shapes[o] = ext
    taken = {str(n) for n in sv.variables} | {str(n) for n in sv.index_sets}
    return Steps(
        tset,
        cols,
        {n: obs[n] for n in sub},
        obs,
        replace(ctx, shapes=shapes),
        set(scalars),
        co,
        letter,
        taken,
    )


def steps_row(st: Steps, u: str, rhs, inl: dict, shape_u: list) -> list:
    """The bands of one row (``rhs``, rows ``u`` of shape ``shape_u``) onto
    the targets: its direct bands onto each target, and each direct band onto
    an observed column composed with that observed's composed bands. Columns
    in code-point order; within a column the direct bands, then the
    composites in the order of the observed they pass through; merged and
    normalized."""
    from .jacobian import JacEntry, inline_row

    ctx = st.ctx
    # inlined, and the operands of a whole-array term aligned with `u` by
    # index-set name (``inline_row``), as on every axis
    rhs = inline_row(rhs, u, inl, ctx, rank_obs=st.obs, scalars=st.scalars)
    bycol: dict = {}
    for v in sorted(x for x in free_variables(rhs) if x in st.targets or x in st.cols):
        raw = bands_in(rhs, v, ctx, shape_u, st.co)
        if not st.co:
            raw = [replace(b, coef=None) for b in raw]
        bs = [normalize_band(b) for b in merge_bands(raw)]
        if v in st.targets:
            bycol.setdefault(v, []).extend(bs)
            continue
        # a kept observed is read with one index per dimension of its shape:
        # another arity selects no cell, refused as inlining refuses it
        # (steps.jl ``_check_cell_read``)
        sh = ctx.shapes.get(v)
        if sh is not None:
            for b in bs:
                if len(b.cidx) != len(sh):
                    raise InlineError(
                        f"observed `{v}` of rank {len(sh)} is read at a cell with "
                        f"{len(b.cidx)} indices"
                    )
        inner = steps_composed(st, v)
        for w in sorted({e.v for e in inner}):
            ib = [e.band for e in inner if e.v == w]
            bycol.setdefault(w, []).extend(
                _steps_sum_free(x, st.co) for x in compose(bs, ib, st.co)
            )
    out = []
    for w in sorted(bycol):
        for b in merge_bands(_steps_combine(bycol[w], st.co)):
            out.append(JacEntry(u, w, normalize_band(b)))
    return out


def _steps_sum_free(b: Band, co: bool) -> Band:
    """A composite's contracted names that its column indices do not read are
    summed into its coefficient: a scalar ``faq`` (reduce ``+``) over their
    ranges whose filter is the ``where`` conjuncts that read them (in order);
    the other conjuncts stay the band's ``where`` (steps.jl
    ``_steps_sum_free``)."""
    if not b.contracted:
        return b
    read: set = set(join_names(b))  # a joined name is never summed away
    for c in b.cidx:
        read |= set(free_variables(c))
    free = {n for n, _, _ in b.contracted if n not in read}
    if not free:
        return b
    keep, filt = [], []
    if b.guard is not None:
        for c in conjuncts(b.guard):
            (filt if set(free_variables(c)) & free else keep).append(c)
    con = [x for x in b.contracted if x[0] not in free]
    coef = None
    if co:
        coef = ExprNode(
            op="faq",
            args=[],
            output_idx=[],
            reduce="+",
            expr=b.coef,
            ranges={n: [lo, hi] for n, lo, hi in b.contracted if n in free},
            filter=conj_list(filt),
        )
    return normalize_band(
        Band(list(b.rows), list(b.ridx), list(b.cidx), coef, con, conj_list(keep), b.join)
    )


def _steps_combine(bs: list, co: bool) -> list:
    """Bands of one column with the same structure (rows, row names, column
    indices, contracted dimensions and ``where``), reached by different paths
    through the observed graph, combined into the first of them: their
    coefficients summed (an n-ary ``+``, in order), or, structure-only, kept
    once (steps.jl ``_steps_combine``)."""
    order: list = []
    groups: dict = {}
    for i, b in enumerate(bs):
        k = (
            tuple(tuple(r) for r in b.rows),
            tuple(b.ridx),
            tuple(skey(c) for c in b.cidx),
            tuple(tuple(c) for c in b.contracted),
            None if b.guard is None else skey(b.guard),
            join_key(b.join),
        )
        if k not in groups:
            order.append(k)
            groups[k] = []
        groups[k].append(i)
    out = []
    for k in order:
        idx = groups[k]
        b = bs[idx[0]]
        if len(idx) > 1 and co:
            b = replace(b, coef=op("+", *(bs[i].coef for i in idx)))
        out.append(b)
    return out


def steps_composed(st: Steps, o: str) -> list:
    """The composed bands of observed ``o`` onto the targets, computed once:
    the bands of ``o``'s own row (its definition over its shape, as a
    companion row is written, §4.2) by :func:`steps_row`."""
    hit = st.composed.get(o)
    if hit is not None:
        return hit
    if o in st.busy:
        raise BandError(f"cyclic observed definition through `{o}`")
    st.busy.add(o)
    shape = st.ctx.shapes.get(o, [])
    d = st.obs[o]
    if not shape:
        rhs = o
    else:
        names = _row_names_of(d, len(shape))
        rhs = _faq_over(names, shape, op("index", o, *names))
    inl = dict(st.inline)
    if (
        isinstance(d, ExprNode)
        and d.op == "faq"
        and d.join is not None
        and isinstance(rhs, ExprNode)
        and list(rhs.output_idx or []) == list(d.output_idx or [])
    ):
        # a joined definition over its own row names is its row (a joined
        # node is never read through beta-reduction, §3.5)
        rhs = d
    else:
        inl[o] = d
    try:
        out = steps_row(st, o, rhs, inl, shape)
    finally:
        st.busy.discard(o)
    if st.co:
        out = _steps_name(st, o, shape, out)
    st.composed[o] = out
    return out


# ── named intermediates (full mode) ──────────────────────────────────────────


def _steps_name(st: Steps, o: str, shape: list, bands: list) -> list:
    """The composed bands of observed ``o`` in their canonical order (the
    code-point order of each band's structure-only entry written as stable
    JSON), each coefficient that is not a literal written once as a named
    intermediate ``_d<letter>_<o>_<k>`` (k its 1-based position) and replaced
    by a read of it at the band's row and contracted names. One intermediate
    per BAND, never one per (column, column index): two bands of one column
    index may overlap with different ``where``, and a summed definition read
    by each would count the overlap twice (steps.jl ``_steps_name!``)."""
    from .emit import serialize_band
    from .jacobian import JacEntry

    def key(en):
        d = serialize_band(en)
        d.pop("coef", None)
        return stable_json(d)

    out = []
    for k, en in enumerate(sorted(bands, key=key), start=1):
        b = en.band
        if isinstance(b.coef, (int, float)) and not isinstance(b.coef, bool):
            out.append(en)
            continue
        name = _inter_name(st, o, k)
        dims = _inter_dims(b, shape)
        st.inters[name] = (o, _inter_definition(b, dims), 1)
        out.append(JacEntry(en.u, en.v, replace(b, coef=_inter_read(name, dims))))
    return out


def _inter_name(st: Steps, o: str, k: int) -> str:
    i = o.rfind(".")
    base = f"_d{st.letter}_{o}_{k}" if i < 0 else f"{o[:i]}._d{st.letter}_{o[i + 1 :]}_{k}"
    return fresh(st.taken, base, "_")


def _inter_dims(b: Band, shape: list) -> list:
    """The dimensions of a band's intermediate: the band's row names (in row
    order) and then contracted names that its coefficient or ``where`` reads,
    each with its box and its whole extent. Only those: an intermediate
    depends on what its coefficient depends on (steps.jl ``_inter_dims``)."""
    fv = set(free_variables(b.coef))
    if b.guard is not None:
        fv |= set(free_variables(b.guard))
    dims = []
    for d, r in enumerate(b.ridx):
        if isinstance(r, str) and r in fv:
            dims.append((r, tuple(b.rows[d]), shape[d]))
    for n, lo, hi in b.contracted:
        if n in fv:
            dims.append((n, (lo, hi), hi))
    return dims


def _inter_read(name: str, dims: list):
    return op("index", name, *[n for n, _, _ in dims]) if dims else name


def _inter_definition(b: Band, dims: list):
    """The definition of a band's intermediate over its dimensions: one
    ``faq`` when the box is the whole extent, else the ``makearray`` of zero
    over the whole extent and the box over it, read at each cell (steps.jl
    ``_inter_definition``)."""
    body = b.coef if b.guard is None else op("ifelse", b.guard, b.coef, 0)
    if not dims:
        return body
    names = [n for n, _, _ in dims]
    box = [bx for _, bx, _ in dims]
    full = [n for _, _, n in dims]

    def faq(bx):
        return ExprNode(
            op="faq",
            args=[],
            output_idx=list(names),
            expr=body,
            ranges={n: [lo, hi] for n, (lo, hi) in zip(names, bx, strict=True)},
        )

    if all(lo == 1 and hi == n for (lo, hi), n in zip(box, full, strict=True)):
        return faq(box)
    return region_array([[[1, n] for n in full], [[lo, hi] for lo, hi in box]], [0, faq(box)], full)


# ── the intermediates a band list reads ──────────────────────────────────────

_INTER_RE = re.compile(r"^_d([upf])_(.*)_([0-9]+)_*$")
_AXIS = {"u": "states", "p": "parameters", "f": "forcing"}


def dt_observed(n: str):
    """The observed a companion name ``_dt_<o>`` (``P._dt_<v>`` for ``P.v``)
    is of, or ``None`` for another name (steps.jl ``_dt_observed``)."""
    i = n.rfind(".")
    scope, base = ("", n) if i < 0 else (n[: i + 1], n[i + 1 :])
    return scope + base[4:] if base.startswith("_dt_") else None


def inter_parse(n: str):
    """The axis and observed of intermediate name ``n``, or ``None``."""
    i = n.rfind(".")
    scope, base = ("", n) if i < 0 else (n[: i + 1], n[i + 1 :])
    m = _INTER_RE.match(base)
    if m is None:
        return None
    return _AXIS[m.group(1)], scope + m.group(2)


def intermediates(sv, bands: list, forcing=None) -> list:
    """Every named intermediate the coefficients and ``where`` of ``bands``
    read, directly or through another intermediate's definition, as
    ``(name, observed, definition, added contracted dimensions)``, in the
    code-point order of the observed, then of the name: the time companions
    (§4.2) and the named steps of the other axes (§4.6) (steps.jl
    ``_intermediates``)."""
    from .companions import time_companions
    from .jacobian import _targets, declared_scalars
    from .recurrence import recurrences
    from .system import ctx_of, observed_defs

    work = []
    for b in bands:
        for x in (b.coef, b.guard):
            if x is None:
                continue
            for n in free_variables(x):
                if inter_parse(n) is not None:
                    work.append(n)
                # a time pair's companion of a named intermediate (§4.4)
                o = dt_observed(n)
                if o is not None and inter_parse(o) is not None:
                    work.append(o)
    if not work:
        return [(n, o, d, 0) for n, o, d in time_companions(sv, bands)]
    ctx = ctx_of(sv)
    obs = observed_defs(sv)
    recs = recurrences(sv, obs, ctx)
    rn = {r.name for r in recs}
    scalars = declared_scalars(sv, obs)

    # A pair's second-axis step of a first-axis intermediate (§4.4) is named
    # after that intermediate (`_dp__du_o_1_1`): the intermediates whose
    # observed is a source observed are resolved first, then those over them,
    # each level's passes taking the levels below as observed (steps.jl).
    def level(n):
        o = inter_parse(n)[1]
        return 0 if inter_parse(o) is None else 1 + level(o)

    def chain(acc, n):
        acc.append(n)
        o = inter_parse(n)[1]
        if inter_parse(o) is not None:
            chain(acc, o)

    passes: dict = {}
    defs: dict = {}
    pending: list = []
    for n in work:
        chain(pending, n)
    base = {k: v for k, v in obs.items() if k not in rn}
    while pending:
        lv = min(level(n) for n in pending)
        passes = {k: v for k, v in passes.items() if k[1] <= lv}
        here = [n for n in pending if level(n) == lv]
        pending = [n for n in pending if level(n) != lv]
        while here:
            n = here.pop()
            if n in defs:
                continue
            wrt, o = inter_parse(n)
            if (wrt, lv) not in passes:
                letter = {"states": "u", "parameters": "p", "forcing": "f"}[wrt]
                o2 = dict(base)
                if lv > 0:
                    o2.update({m: d for m, (_, d, _) in defs.items()})
                passes[(wrt, lv)] = steps_setup(
                    sv, ctx, o2, _targets(sv, wrt, forcing, obs), scalars, True, letter
                )
            st = passes[(wrt, lv)]
            if n not in st.inters:
                steps_composed(st, o)
            hit = st.inters.get(n)
            if hit is None:
                raise BandError(f"no named intermediate `{n}` (observed `{o}`)")
            for m in free_variables(hit[1]):
                if inter_parse(m) is None or m in defs:
                    continue
                if level(m) == lv:
                    here.append(m)
                else:
                    chain(pending, m)
            defs[n] = hit
    extra = {n: d for n, (_, d, _) in defs.items()}
    out = [(n, o, d, 0) for n, o, d in time_companions(sv, bands, extra)]
    out += [(n, o, d, c) for n, (o, d, c) in defs.items()]
    return sorted(out, key=lambda x: (x[1], x[0]))


def _int_lit(e):
    if isinstance(e, bool):
        return None
    if isinstance(e, int):
        return e
    if isinstance(e, float) and e.is_integer():
        return int(e)
    return None


def _is_guard(e) -> bool:
    """Whether ``e`` is an ``ifelse`` with a literal-zero branch: a guard."""
    return (
        isinstance(e, ExprNode)
        and e.op == "ifelse"
        and len(e.args) == 3
        and (is_zero(e.args[1]) or is_zero(e.args[2]))
    )


def guarded_mul(outer, inner):
    """``outer · inner``, with a guard of ``outer`` (an ``ifelse`` with a
    literal-zero branch, ``outer`` itself or the first such factor of a
    product) kept outside the product: ``ifelse(c, a, 0) · y`` is
    ``ifelse(c, a · y, 0)``. The guard of a definition
    (``ifelse(s > 0, …/s, …)``) then also guards its composed derivative
    where the inner derivative ``y`` is NaN (spec §3.6)."""
    if _is_guard(outer):
        return op(
            "ifelse",
            outer.args[0],
            guarded_mul(outer.args[1], inner),
            guarded_mul(outer.args[2], inner),
        )
    if isinstance(outer, ExprNode) and outer.op == "*":
        k = next((i for i, a in enumerate(outer.args) if _is_guard(a)), None)
        if k is not None:
            g = outer.args[k]

            def branch(x):
                return guarded_mul(mul(*outer.args[:k], x, *outer.args[k + 1 :]), inner)

            return op("ifelse", g.args[0], branch(g.args[1]), branch(g.args[2]))
    return mul(outer, inner)


def compose(outer: list, inner: list, co: bool) -> list:
    """The chain rule through an intermediate array ``W`` (operator_refs.jl
    ``_compose``): ``outer`` are bands of an expression with respect to ``W``
    (their column indices are cells of ``W``), ``inner`` the bands of ``W``
    (rows: cells of ``W``). Each pair gives the bands of the expression
    where the outer column lands in the inner row box: inner row names read
    the outer column indices, the coefficients multiply, contractions
    concatenate (the inner's renamed apart), and the row-box membership joins
    the guard, decided over the outer box as far as it can be."""
    out = []
    for o in outer:
        for b in inner:
            if len(o.cidx) != len(b.ridx):
                raise BandError(
                    "composing through an intermediate array: a read of rank "
                    f"{len(o.cidx)} of an array of rank {len(b.ridx)}"
                )
            taken = {r for r in o.ridx if isinstance(r, str)}
            taken |= {n for n, _, _ in o.contracted}
            for c in o.cidx:
                taken |= set(free_variables(c))
            if o.coef is not None:
                all_names(o.coef, taken)
            sigma: dict = {}
            bcon = []
            for n, lo, hi in b.contracted:
                if n in taken:
                    f = fresh(taken, n + "_c", "_c")
                else:
                    taken.add(n)
                    f = n
                if f != n:
                    sigma[n] = f
                bcon.append((f, lo, hi))
            conds = []
            dead = False
            jnames = join_names(b)
            tie = []
            for d, r in enumerate(b.ridx):
                lo, hi = b.rows[d]
                if isinstance(r, str) and r in jnames and not isinstance(o.cidx[d], str):
                    # a joined row name read at a non-bare index stays a name
                    # of the composite (the join reads it): contracted over
                    # its row range, tied to the index by an equality (§3.5)
                    if r in taken:
                        f = fresh(taken, r + "_c", "_c")
                    else:
                        taken.add(r)
                        f = r
                    sigma[r] = f
                    bcon.append((f, lo, hi))
                    tie.append(op("==", f, o.cidx[d]))
                elif isinstance(r, str):
                    sigma[r] = o.cidx[d]
                x = _int_lit(o.cidx[d])
                for cmp, bound in ((">=", lo), ("<=", hi)):
                    if x is not None:
                        if not _CMP[cmp](x, bound):
                            dead = True
                            break
                        continue
                    conds.append(op(cmp, o.cidx[d], bound))
                if dead:
                    break
            if dead:
                continue

            def sub(x, sigma=sigma):
                return x if x is None or not sigma else subst_free(x, sigma)

            # a join travels with the band it gates (§3.5): the inner's names
            # renamed as the composite reads them; two joins are refused
            jn = o.join
            if b.join is not None:
                if o.join is not None:
                    raise BandError("a band through two joined faq nodes is not supported")
                jm = {}
                for n in join_names(b):
                    x = sigma.get(n, n)
                    if not isinstance(x, str):
                        raise BandError(
                            "a joined band composed at a non-bare index is not supported"
                        )
                    jm[n] = x
                jn = rename_join(b.join, jm)
            g = conj(o.guard, sub(b.guard))
            if conds:
                g = conj(g, conj_list(conds))
            if tie:
                g = conj(g, conj_list(tie))
            cidx = [canon_lits(simplify_jl(sub(c))) for c in b.cidx]
            coef = simp(guarded_mul(o.coef, sub(b.coef))) if co else None
            if co and is_zero(coef):
                continue
            contracted = list(o.contracted) + bcon
            names = [r for r in o.ridx if isinstance(r, str)]
            nr = len(names)
            names += [n for n, _, _ in contracted]
            box = [o.rows[k] for k, r in enumerate(o.ridx) if isinstance(r, str)]
            box += [(lo, hi) for _, lo, hi in contracted]
            pieces = [(box, None)] if g is None else split_guard(g, names, box)
            for cell, g2 in pieces:
                rows = list(o.rows)
                j = 0
                for k, r in enumerate(o.ridx):
                    if isinstance(r, str):
                        rows[k] = cell[j]
                        j += 1
                con = [(n, *cell[nr + i]) for i, (n, _, _) in enumerate(contracted)]
                out.append(normalize_band(Band(rows, list(o.ridx), cidx, coef, con, g2, jn)))
    return out
