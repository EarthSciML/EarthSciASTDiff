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

from dataclasses import replace

from earthsci_ast.expression import free_variables

from .bands import Band, BandError, Ctx, bands_in, merge_bands, normalize_band
from .companions import _faq_over, _obs_extents, _row_names_of
from .expr_helpers import is_zero, mul, op, skey
from .inline import all_names, inline_observed, subst_free
from .reductions import conj, conj_list, fresh, split_guard
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

    def __init__(self, targets, cols, inline, obs, ctx, scalars, co):
        self.targets = targets
        self.cols = cols
        self.inline = inline
        self.obs = obs
        self.ctx = ctx
        self.scalars = scalars
        self.co = co
        self.composed: dict = {}
        self.busy: set = set()


def steps_setup(sv, ctx: Ctx, obs: dict, targets, scalars, co: bool) -> Steps:
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
    return Steps(
        tset, cols, {n: obs[n] for n in sub}, obs, replace(ctx, shapes=shapes), set(scalars), co
    )


def steps_row(st: Steps, u: str, rhs, inl: dict, shape_u: list) -> list:
    """The bands of one row (``rhs``, rows ``u`` of shape ``shape_u``) onto
    the targets: its direct bands onto each target, and each direct band onto
    an observed column composed with that observed's composed bands. Columns
    in code-point order; within a column the direct bands, then the
    composites in the order of the observed they pass through; merged and
    normalized."""
    from .jacobian import JacEntry

    ctx = st.ctx
    rhs = inline_observed(rhs, inl, [None], ctx.shapes, st.obs, st.scalars, ctx.axes)
    bycol: dict = {}
    for v in sorted(x for x in free_variables(rhs) if x in st.targets or x in st.cols):
        raw = bands_in(rhs, v, ctx, shape_u, st.co)
        if not st.co:
            raw = [replace(b, coef=None) for b in raw]
        bs = [normalize_band(b) for b in merge_bands(raw)]
        if v in st.targets:
            bycol.setdefault(v, []).extend(bs)
            continue
        inner = steps_composed(st, v)
        for w in sorted({e.v for e in inner}):
            ib = [e.band for e in inner if e.v == w]
            bycol.setdefault(w, []).extend(compose(bs, ib, st.co))
    out = []
    for w in sorted(bycol):
        for b in merge_bands(_steps_combine(bycol[w], st.co)):
            out.append(JacEntry(u, w, normalize_band(b)))
    return out


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
    inl[o] = d
    try:
        out = steps_row(st, o, rhs, inl, shape)
    finally:
        st.busy.discard(o)
    st.composed[o] = out
    return out


def _int_lit(e):
    if isinstance(e, bool):
        return None
    if isinstance(e, int):
        return e
    if isinstance(e, float) and e.is_integer():
        return int(e)
    return None


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
            for d, r in enumerate(b.ridx):
                lo, hi = b.rows[d]
                if isinstance(r, str):
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

            g = conj(o.guard, sub(b.guard))
            if conds:
                g = conj(g, conj_list(conds))
            cidx = [canon_lits(simplify_jl(sub(c))) for c in b.cidx]
            coef = simp(mul(o.coef, sub(b.coef))) if co else None
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
                out.append(normalize_band(Band(rows, list(o.ridx), cidx, coef, con, g2)))
    return out
