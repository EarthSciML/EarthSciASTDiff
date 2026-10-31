"""The array-level band calculus (bands.jl): differentiate an array-valued
RHS w.r.t. one variable, producing structured bands."""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from earthsci_ast.esm_types import Expr, ExprNode
from earthsci_ast.expr_walk import map_children
from earthsci_ast.expression import free_variables
from earthsci_ast.substitute import substitute

from .clip import clip_regions
from .expr_helpers import Site, add, is_zero, mul, op, region_bound, skey
from .inline import all_names
from .inline import subst_free as _subst_free
from .reductions import conj, conj_list, conjuncts, index_only, one_hot, reduce_op, split_guard
from .reductions import window_weight as _window_weight
from .scalar_rules import dscalar
from .simplify_branches import canon_lits, simp
from .simplify_port import simplify_jl as simplify


@dataclass
class Band:
    """One structured block of ∂(rhs of the equation for u)/∂v — see the
    Julia reference docstring and esm-jacobian-spec.md §4. ``contracted``
    holds free contracted column dimensions ``(name, lo, hi)``; ``guard`` is
    an exact point predicate (the entry's ``where`` field) over the row and
    contracted index names — what a filtered faq node's non-rectangular
    window leaves after its box is split at affine breakpoints — or ``None``
    for every rectangular band. Rebuild with :func:`dataclasses.replace` so
    the guard travels."""

    rows: list[tuple[int, int]]
    ridx: list  # str names or int singletons
    cidx: list  # column index expressions
    coef: Expr
    contracted: list[tuple[str, int, int]] = field(default_factory=list)
    guard: Expr | None = None


@dataclass
class Ctx:
    index_sets: dict[str, int]
    shapes: dict[str, list[int]]
    params: set[str]


class BandError(Exception):
    """Raised when an expression has a form the band calculus cannot differentiate
    (for example a joined ``faq`` node or a boolean semiring)."""


def _set_size(name: str, ctx: Ctx) -> int:
    if name not in ctx.index_sets:
        raise BandError(f"range over undeclared index set `{name}`")
    return ctx.index_sets[name]


def _range_of(r, ctx: Ctx) -> tuple[int, int]:
    if isinstance(r, dict):
        return (1, _set_size(str(r["from"]), ctx))
    if isinstance(r, (list, tuple)):
        return (region_bound(r[0]), region_bound(r[-1]))
    # typed IndexSetRef-like object with a `from`/`from_` attribute
    frm = getattr(r, "from_", None) or getattr(r, "from_set", None)
    if frm is not None:
        return (1, _set_size(str(frm), ctx))
    raise BandError(f"unsupported range spec {r!r}")


def is_array(ctx: Ctx, name) -> bool:
    return isinstance(name, str) and name in ctx.shapes


def sites_of(e: Expr, v: str, acc: dict[str, Site] | None = None) -> dict[str, Site]:
    if acc is None:
        acc = {}
    if isinstance(e, str) and e == v:
        s = Site.of(e)
        acc[s.key] = s
    elif isinstance(e, ExprNode) and e.op == "index" and e.args[0] == v:
        s = Site.of(e)
        acc[s.key] = s
    elif isinstance(e, ExprNode):
        map_children(e, lambda x: (sites_of(x, v, acc), x)[1])
    return acc


def sorted_sites(e: Expr, v: str) -> list[Site]:
    d = sites_of(e, v)
    return [d[k] for k in sorted(d)]


_AMBIGUOUS = object()
_MIXED = object()


class _Nested:
    """``_nested_sums``' result (bands.jl ``_Nested``): per site key, the
    indices the enclosing sums bind (``ranges``) and the filters of those
    sums (``filters``: the same list at every occurrence under a sum, or
    ``_MIXED``), compared by keys taken once per filter (``fkeys``)."""

    __slots__ = ("filters", "fkeys", "ranges")

    def __init__(self):
        self.ranges: dict = {}
        self.filters: dict = {}
        self.fkeys: dict = {}


def _nested_sums(e: Expr, v: str, outer) -> _Nested:
    """For each site of ``v`` under a nested scalar faq sum, the indices the
    sums ENCLOSING it bind (bands.jl ``_nested_sums``): site key → {name →
    range spec, or ``_AMBIGUOUS`` for a name also bound by ``outer``, bound
    twice on the way down, or over different ranges at different
    occurrences}, plus the filters of those sums. Position-aware: a sum
    elsewhere in the body does not affect a site outside it."""
    acc = _Nested()
    _nested_walk(acc, e, v, set(outer), [], [], [])
    return acc


def _nested_walk(
    acc: _Nested, e: Expr, v: str, outer: set, stack: list, fstack: list, fkeys: list
) -> None:
    if (isinstance(e, str) and e == v) or (
        isinstance(e, ExprNode) and e.op == "index" and e.args[0] == v
    ):
        if not stack:
            return
        key = Site.of(e).key
        f = acc.filters.get(key)
        if f is None:
            acc.filters[key] = list(fstack)
            acc.fkeys[key] = list(fkeys)
        elif f is not _MIXED and acc.fkeys[key] != fkeys:
            acc.filters[key] = _MIXED
        d = acc.ranges.setdefault(key, {})
        seen: set = set()
        for n, r in stack:
            amb = n in outer or n in seen or (n in d and d[n] != r)
            d[n] = _AMBIGUOUS if amb else r
            seen.add(n)
        return
    if not isinstance(e, ExprNode):
        return
    pushed = 0
    filtered = False
    if e.op == "faq" and not (e.output_idx or []) and e.ranges is not None:
        for n, r in e.ranges.items():
            stack.append((str(n), r))
            pushed += 1
        if e.filter is not None:
            fstack.append(e.filter)
            fkeys.append(skey(e.filter))
            filtered = True
    map_children(e, lambda x: (_nested_walk(acc, x, v, outer, stack, fstack, fkeys), x)[1])
    del stack[len(stack) - pushed :]
    if filtered:
        fstack.pop()
        fkeys.pop()


def _lift_nested(
    nested: dict, s: Site, body: Expr, cidx: list, coef: Expr, ctx: Ctx, outer, taken: list
):
    """A site under a nested sum whose column reads the sum's index: that
    index becomes a CONTRACTED column dimension of the band, renamed apart in
    ``cidx`` and in the coefficient's free occurrences (bands.jl
    ``_lift_nested``); the sums' index-only filter conjuncts over a lifted
    name become the band's guard (``_nested_guard``). ``taken`` is a per-body
    lazy cache (``[None]``) of the body's names. Returns ``(cidx, coef,
    extra, guard)``."""
    d = nested.ranges.get(s.key)
    if d is None:
        return cidx, coef, [], None
    read: set[str] = set()
    for c in cidx:
        read |= set(free_variables(c))
    names = sorted(n for n in d if n in read)
    if not names:
        return cidx, coef, [], None
    if taken[0] is None:
        taken[0] = all_names(body, set(outer))
    ren: dict = {}
    extra = []
    for n in names:
        if d[n] is _AMBIGUOUS:
            raise BandError(
                f"a column reads `{n}`, which the sums around it bind "
                "ambiguously (shadowing an enclosing index, or over two "
                "different ranges)"
            )
        f = n + "_j"
        while f in taken[0] or f in ren.values():
            f += "_j"
        ren[n] = f
        extra.append((f, *_range_of(d[n], ctx)))
    guard = _nested_guard(nested.filters.get(s.key, _MIXED), d, names, outer, ren)
    return ([substitute(c, ren) for c in cidx], _subst_free(coef, ren), extra, guard)


def _nested_guard(filters, d: dict, lifted: list, outer, ren: dict):
    """The index-only conjuncts of the filters around a site that read a
    lifted name — and no name a sum around it still binds — renamed like the
    column: the band's guard (bands.jl ``_nested_guard``). A site under
    different filters at different occurrences gets none."""
    if filters is _MIXED:
        return None
    ok = set(outer) | set(lifted)
    gs = []
    for f in filters:
        for c in conjuncts(f):
            fv = set(free_variables(c))
            if not any(n in fv for n in lifted):
                continue
            if not fv <= ok:
                continue
            if any(n in d and n not in lifted for n in fv):
                continue
            gs.append(substitute(c, ren))
    return conj_list(gs)


def is_array_expr(e: Expr, ctx: Ctx) -> bool:
    if isinstance(e, str):
        return is_array(ctx, e)
    if not isinstance(e, ExprNode):
        return False
    if e.op in ("faq", "makearray", "broadcast", "reshape", "transpose", "concat"):
        return True
    if e.op == "index":
        return False
    return any(is_array_expr(x, ctx) for x in e.args)


def occurs_var(e: Expr, v: str) -> bool:
    return v in free_variables(e)


def bands(rhs: Expr, v: str, ctx: Ctx, shape_u: list[int] | None = None) -> list[Band]:
    """Differentiate an equation RHS (already observed-inlined) w.r.t. ``v``."""
    shape_u = shape_u or []
    out: list[Band] = []
    if not shape_u:
        nested = _nested_sums(rhs, v, [])
        taken: list = [None]
        for s in sorted_sites(rhs, v):
            cidx, d, extra, g = _lift_nested(
                nested, s, rhs, _site_cidx(s), dscalar(rhs, s), ctx, [], taken
            )
            coef = simp(d)
            if is_zero(coef):
                continue
            _push_band(out, [], [], cidx, coef, extra, g, v, ctx)
        return out
    _bands_array(out, rhs, v, ctx, shape_u, [1])
    return out


def _site_cidx(s: Site) -> list:
    return [] if isinstance(s.expr, str) else list(s.expr.args[1:])


def _push_band(out, rows, ridx, cidx, coef, contracted, guard, v: str, ctx: Ctx) -> None:
    """Push one band, its value-reading column dimensions made one-hot first."""
    ridx = list(ridx)
    cidx, coef, contracted = one_hot(cidx, coef, contracted, ridx, v, ctx)
    out.append(Band(list(rows), ridx, cidx, coef, list(contracted), guard))


def _scalar_bands(out, e: Expr, v: str, ctx: Ctx, rows, ridx, scale) -> None:
    outer = [r for r in ridx if isinstance(r, str)]
    for rows2, body in clip_regions(e, ridx, rows):
        nested = _nested_sums(body, v, outer)
        taken: list = [None]
        for s in sorted_sites(body, v):
            cidx, d, extra, g = _lift_nested(
                nested, s, body, _site_cidx(s), dscalar(body, s), ctx, outer, taken
            )
            coef = simp(mul(d, *scale))
            if is_zero(coef):
                continue
            _push_band(out, rows2, ridx, cidx, coef, extra, g, v, ctx)


def _bands_array(out, e: Expr, v: str, ctx: Ctx, shape_u, scale) -> None:
    if isinstance(e, str):
        if e != v:
            return
        names = [f"_r{k + 1}" for k in range(len(shape_u))]
        out.append(Band([(1, n) for n in shape_u], list(names), [n for n in names], mul(*scale)))
        return
    if not isinstance(e, ExprNode):
        return
    o = e.op
    if o == "+":
        for x in e.args:
            _bands_array(out, x, v, ctx, shape_u, scale)
    elif o == "-" and len(e.args) == 1:
        _bands_array(out, e.args[0], v, ctx, shape_u, [*scale, -1])
    elif o == "-":
        _bands_array(out, e.args[0], v, ctx, shape_u, scale)
        _bands_array(out, e.args[1], v, ctx, shape_u, [*scale, -1])
    elif o == "neg":
        _bands_array(out, e.args[0], v, ctx, shape_u, [*scale, -1])
    elif o == "*":
        arr = [x for x in e.args if is_array_expr(x, ctx)]
        sc = [x for x in e.args if not is_array_expr(x, ctx)]
        if len(arr) != 1:
            raise BandError(
                f"elementwise product of {len(arr)} array-valued factors at "
                "array level; wrap in a faq node"
            )
        if any(occurs_var(x, v) for x in sc):
            raise BandError(f"scalar factor depends on `{v}` in an array-level product")
        _bands_array(out, arr[0], v, ctx, shape_u, scale + sc)
    elif o == "/":
        if is_array_expr(e.args[1], ctx):
            raise BandError("array-valued divisor at array level")
        _bands_array(out, e.args[0], v, ctx, shape_u, [*scale, op("/", 1, e.args[1])])
    elif o == "makearray":
        _bands_makearray(out, e, v, ctx, scale)
    elif o == "faq":
        _bands_faq(out, e, v, ctx, shape_u, scale)
    elif o == "index":
        names = [f"_r{k + 1}" for k in range(len(shape_u))]
        _scalar_bands(out, e, v, ctx, [(1, n) for n in shape_u], names, scale)
    else:
        if is_array_expr(e, ctx):
            names = [f"_r{k + 1}" for k in range(len(shape_u))]
            idx = [n for n in names]
            body = _push_index(e, idx, ctx)
            agg = ExprNode(
                op="faq",
                args=[],
                output_idx=list(names),
                expr=body,
                ranges={n: [1, shape_u[k]] for k, n in enumerate(names)},
            )
            _bands_array(out, agg, v, ctx, shape_u, scale)
        else:
            names = [f"_r{k + 1}" for k in range(len(shape_u))]
            _scalar_bands(out, e, v, ctx, [(1, n) for n in shape_u], names, scale)


def _bands_faq(out, e: ExprNode, v: str, ctx: Ctx, shape_u, scale) -> None:
    """A faq node at array level (bands.jl ``_bands_faq!``). Each site of its
    body differentiates as a term of the node's reduction over its window
    (reductions.py): the derivative is weighted per term
    (``window_weight``), and the filter's index-only conjuncts split the
    band box (``split_guard``) while its value-dependent conjuncts gate the
    coefficient. Per box piece, a site whose column reads no contracted name
    yields one band whose coefficient sums the weighted derivative over the
    piece's contraction (filtered by the residual guard's contracted
    conjuncts; its row-only conjuncts guard the band); a gathered column
    yields a band with contracted dimensions and the whole residual guard."""
    oidx = [str(x) for x in e.output_idx if isinstance(x, str)]
    if len(oidx) != len(e.output_idx):
        raise BandError("singleton `1` output_idx entries not supported yet")
    red = reduce_op(e, BandError)
    ranges = e.ranges or {}
    cnames = sorted(str(n) for n in ranges if str(n) not in oidx)
    crange0 = [(n, *_range_of(ranges[n], ctx)) for n in cnames]
    rows0 = []
    for k, n in enumerate(oidx):
        rows0.append(_range_of(ranges[n], ctx) if n in ranges else (1, shape_u[k]))
    outer = oidx + cnames
    gidx: list = []
    gval: list = []
    if e.filter is not None:
        for c in conjuncts(e.filter):
            (gidx if index_only(c, outer) else gval).append(c)
    if not gidx:
        pieces = [(rows0, crange0, None)]
    else:
        no = len(oidx)
        box = list(rows0) + [(lo, hi) for _, lo, hi in crange0]
        pieces = [
            (cell[:no], [(n, *cell[no + k]) for k, n in enumerate(cnames)], g)
            for cell, g in split_guard(conj_list(gidx), outer, box)
        ]
    vfilt = conj_list(gval)
    body = e.expr
    for rows_p, crange, guard in pieces:
        for rows2, body2 in clip_regions(body, oidx, rows_p):
            nested = _nested_sums(body2, v, outer)
            taken: list = [None]
            for s in sorted_sites(body2, v):
                if isinstance(s.expr, str) and is_array(ctx, v):
                    raise BandError(f"whole-array reference to `{v}` inside a faq body")
                cidx, d, extra, g = _lift_nested(
                    nested, s, body2, _site_cidx(s), dscalar(body2, s), ctx, outer, taken
                )
                d = _window_weight(red, e, cnames, d, BandError)
                if vfilt is not None and not is_zero(d):
                    d = op("ifelse", vfilt, d, 0)
                if not crange:
                    coef = simp(mul(d, *scale))
                    if is_zero(coef):
                        continue
                    _push_band(out, rows2, oidx, cidx, coef, extra, conj(guard, g), v, ctx)
                elif not any(any(n in free_variables(c) for c in cidx) for n, _, _ in crange):
                    # The column is independent of every contracted index: a
                    # plain band whose coefficient is the (scalar) `reduce:
                    # "+"` of the site derivative over the contraction. A
                    # guard conjunct reading a contracted name — the piece's
                    # own, or a nested sum's filter tied to it (`Σ_k Σ_{j≤k}`)
                    # — is no condition on the band's point: it filters that
                    # sum.
                    gc: list = []
                    gr: list = []
                    gg = conj(guard, g)
                    if gg is not None:
                        for c in conjuncts(gg):
                            fv = free_variables(c)
                            (gc if any(n in cnames for n in fv) else gr).append(c)
                    red_sum = ExprNode(
                        op="faq",
                        args=[],
                        output_idx=[],
                        reduce="+",
                        expr=d,
                        ranges={n: [lo, hi] for n, lo, hi in crange},
                        filter=conj_list(gc),
                    )
                    coef = simp(mul(red_sum, *scale))
                    if is_zero(coef):
                        continue
                    _push_band(out, rows2, oidx, cidx, coef, extra, conj_list(gr), v, ctx)
                else:
                    # A gathered column (`u[j]`, `u[i+k]`, `u[conn[i,k]]`): a
                    # band with free contracted column dimensions; entries
                    # landing on one column accumulate at assembly.
                    coef = simp(mul(d, *scale))
                    if is_zero(coef):
                        continue
                    _push_band(
                        out, rows2, oidx, cidx, coef, list(crange) + extra, conj(guard, g), v, ctx
                    )


def _bands_makearray(out, e: ExprNode, v: str, ctx: Ctx, scale) -> None:
    for region, val in zip(e.regions, e.values):
        rows = [(region_bound(r[0]), region_bound(r[1])) for r in region]
        if any(hi < lo for lo, hi in rows):
            continue  # folded-empty region
        nonsing = [k for k, (lo, hi) in enumerate(rows) if hi > lo]
        if is_array_expr(val, ctx):
            sub: list[Band] = []
            val_fullrank = (
                isinstance(val, ExprNode) and val.op == "faq" and len(val.output_idx) == len(rows)
            )
            if val_fullrank:
                sub_shape = [hi - lo + 1 for lo, hi in rows]
            else:
                sub_shape = [rows[k][1] - rows[k][0] + 1 for k in nonsing]
            _bands_array(sub, val, v, ctx, sub_shape, scale)
            if isinstance(val, ExprNode) and val.op == "faq":
                vr = val.ranges or {}
                origins = [
                    _range_of(vr[str(n)], ctx)[0] if str(n) in vr else 1 for n in val.output_idx
                ]
            else:
                origins = [1] * len(nonsing)
            for b in sub:
                ridx: list = []
                subst: dict[str, Expr] = {}
                rows2: list[tuple[int, int]] = []
                fullrank = len(b.ridx) == len(rows)
                kk = -1
                for _k, (lo, hi) in enumerate(rows):
                    if fullrank or hi > lo:
                        kk += 1
                        r = b.ridx[kk]
                        rlo, rhi = b.rows[kk]
                        shift = lo - origins[kk]
                        if isinstance(r, str):
                            ridx.append(r)
                            if shift != 0:
                                subst[r] = add(r, -shift)
                            rows2.append((rlo + shift, rhi + shift))
                        else:
                            ridx.append(r + shift)
                            rows2.append((r + shift, r + shift))
                    else:
                        ridx.append(lo)
                        rows2.append((lo, lo))
                coef = b.coef if not subst else substitute(b.coef, subst)
                cidx = b.cidx if not subst else [substitute(c, subst) for c in b.cidx]
                guard = b.guard if not subst or b.guard is None else substitute(b.guard, subst)
                out.append(Band(rows2, ridx, cidx, coef, b.contracted, guard))
        else:
            _scalar_bands(out, val, v, ctx, rows, [lo for lo, _ in rows], scale)


def _push_index(e: Expr, idx: list, ctx: Ctx) -> Expr:
    if isinstance(e, str):
        return op("index", e, *idx) if is_array(ctx, e) else e
    if isinstance(e, ExprNode):
        if e.op in ("faq", "makearray", "index"):
            raise BandError(f"cannot push an index into `{e.op}` nested under an elementwise op")
        return map_children(e, lambda x: _push_index(x, idx, ctx))
    return e


def normalize_band(b: Band) -> Band:
    """Pin singleton named row dims (and one-value contractions) to constants,
    substituting them through ``cidx``, ``coef`` and ``guard``."""
    subst: dict[str, Expr] = {}
    ridx: list = []
    for k, r in enumerate(b.ridx):
        lo, hi = b.rows[k]
        if isinstance(r, str) and lo == hi:
            subst[r] = lo
            ridx.append(lo)
        else:
            ridx.append(r)
    contracted = []
    for n, lo, hi in b.contracted:
        if lo == hi:
            subst[n] = lo
        else:
            contracted.append((n, lo, hi))
    if not subst:
        return b
    return Band(
        b.rows,
        ridx,
        [canon_lits(simplify(substitute(c, subst))) for c in b.cidx],
        simp(substitute(b.coef, subst)),
        contracted,
        None if b.guard is None else canon_lits(simplify(substitute(b.guard, subst))),
    )


def merge_bands(bs: list[Band]) -> list[Band]:
    """Coalesce bands identical in (ridx, cidx, coef, contracted, guard) whose
    row boxes are adjacent along exactly one dimension. Order-preserving."""
    if len(bs) < 2:
        return bs

    def key(b: Band):
        return (
            tuple(b.ridx),
            tuple(skey(c) for c in b.cidx),
            skey(b.coef),
            tuple(b.contracted),
            None if b.guard is None else skey(b.guard),
        )

    ks = [key(b) for b in bs]
    out = list(bs)
    alive = [True] * len(out)
    changed = True
    while changed:
        changed = False
        for a in range(len(out)):
            if not alive[a]:
                continue
            for b in range(a + 1, len(out)):
                if not alive[b] or ks[a] != ks[b]:
                    continue
                ra, rb = out[a].rows, out[b].rows
                if len(ra) != len(rb):
                    continue
                d = -1
                ok = True
                for k in range(len(ra)):
                    if ra[k] == rb[k]:
                        continue
                    if d < 0 and (ra[k][1] + 1 == rb[k][0] or rb[k][1] + 1 == ra[k][0]):
                        d = k
                    else:
                        ok = False
                        break
                if not ok or d < 0:
                    continue
                lo = min(ra[d][0], rb[d][0])
                hi = max(ra[d][1], rb[d][1])
                rows = list(ra)
                rows[d] = (lo, hi)
                out[a] = replace(out[a], rows=rows)
                alive[b] = False
                changed = True
    return [b for k, b in enumerate(out) if alive[k]]
