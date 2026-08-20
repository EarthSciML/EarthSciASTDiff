"""The array-level band calculus (bands.jl): differentiate an array-valued
RHS w.r.t. one variable, producing structured bands."""
from __future__ import annotations

from dataclasses import dataclass, field

from earthsci_ast.esm_types import Expr, ExprNode
from earthsci_ast.expr_walk import map_children
from earthsci_ast.expression import free_variables
from earthsci_ast.substitute import substitute

from .clip import clip_regions
from .expr_helpers import Site, add, ckey, is_zero, mul, neg, op, region_bound, skey
from .scalar_rules import dscalar
from .simplify_branches import canon_lits, simp
from .simplify_port import simplify_jl as simplify


@dataclass
class Band:
    """One structured block of ∂(rhs of the equation for u)/∂v — see the
    Julia reference docstring and esm-jacobian-spec.md §4. ``contracted``
    holds free contracted column dimensions ``(name, lo, hi)``."""

    rows: list[tuple[int, int]]
    ridx: list                        # str names or int singletons
    cidx: list                        # column index expressions
    coef: Expr
    contracted: list[tuple[str, int, int]] = field(default_factory=list)


@dataclass
class Ctx:
    index_sets: dict[str, int]
    shapes: dict[str, list[int]]
    params: set[str]


class BandError(Exception):
    pass


def _range_of(r, ctx: Ctx) -> tuple[int, int]:
    if isinstance(r, dict):
        return (1, ctx.index_sets[str(r["from"])])
    if isinstance(r, (list, tuple)):
        return (region_bound(r[0]), region_bound(r[-1]))
    # typed IndexSetRef-like object with a `from`/`from_` attribute
    frm = getattr(r, "from_", None) or getattr(r, "from_set", None)
    if frm is not None:
        return (1, ctx.index_sets[str(frm)])
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


def is_array_expr(e: Expr, ctx: Ctx) -> bool:
    if isinstance(e, str):
        return is_array(ctx, e)
    if not isinstance(e, ExprNode):
        return False
    if e.op in ("aggregate", "makearray", "arrayop", "broadcast",
                "reshape", "transpose", "concat"):
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
        for s in sorted_sites(rhs, v):
            coef = simp(dscalar(rhs, s))
            if is_zero(coef):
                continue
            cidx = [] if isinstance(s.expr, str) else list(s.expr.args[1:])
            out.append(Band([], [], cidx, coef))
        return out
    _bands_array(out, rhs, v, ctx, shape_u, [1])
    return out


def _site_cidx(s: Site) -> list:
    return [] if isinstance(s.expr, str) else list(s.expr.args[1:])


def _scalar_bands(out, e: Expr, v: str, rows, ridx, scale) -> None:
    for rows2, body in clip_regions(e, ridx, rows):
        for s in sorted_sites(body, v):
            coef = simp(mul(dscalar(body, s), *scale))
            if is_zero(coef):
                continue
            out.append(Band(rows2, list(ridx), _site_cidx(s), coef))


def _bands_array(out, e: Expr, v: str, ctx: Ctx, shape_u, scale) -> None:
    if isinstance(e, str):
        if e != v:
            return
        names = [f"_r{k + 1}" for k in range(len(shape_u))]
        out.append(Band([(1, n) for n in shape_u], list(names),
                        [n for n in names], mul(*scale)))
        return
    if not isinstance(e, ExprNode):
        return
    o = e.op
    if o == "+":
        for x in e.args:
            _bands_array(out, x, v, ctx, shape_u, scale)
    elif o == "-":
        _bands_array(out, e.args[0], v, ctx, shape_u, scale)
        if len(e.args) == 2:
            _bands_array(out, e.args[1], v, ctx, shape_u, scale + [-1])
    elif o == "neg":
        _bands_array(out, e.args[0], v, ctx, shape_u, scale + [-1])
    elif o == "*":
        arr = [x for x in e.args if is_array_expr(x, ctx)]
        sc = [x for x in e.args if not is_array_expr(x, ctx)]
        if len(arr) != 1:
            raise BandError(
                f"elementwise product of {len(arr)} array-valued factors at "
                "array level; wrap in an aggregate")
        if any(occurs_var(x, v) for x in sc):
            raise BandError(
                f"scalar factor depends on `{v}` in an array-level product")
        _bands_array(out, arr[0], v, ctx, shape_u, scale + sc)
    elif o == "/":
        if is_array_expr(e.args[1], ctx):
            raise BandError("array-valued divisor at array level")
        _bands_array(out, e.args[0], v, ctx, shape_u,
                     scale + [op("/", 1, e.args[1])])
    elif o == "makearray":
        _bands_makearray(out, e, v, ctx, scale)
    elif o == "aggregate":
        _bands_aggregate(out, e, v, ctx, shape_u, scale)
    elif o == "index":
        names = [f"_r{k + 1}" for k in range(len(shape_u))]
        _scalar_bands(out, e, v, [(1, n) for n in shape_u], names, scale)
    else:
        if is_array_expr(e, ctx):
            names = [f"_r{k + 1}" for k in range(len(shape_u))]
            idx = [n for n in names]
            body = _push_index(e, idx, ctx)
            agg = ExprNode(op="aggregate", args=[],
                           output_idx=list(names), expr=body,
                           ranges={n: [1, shape_u[k]]
                                   for k, n in enumerate(names)})
            _bands_array(out, agg, v, ctx, shape_u, scale)
        else:
            names = [f"_r{k + 1}" for k in range(len(shape_u))]
            _scalar_bands(out, e, v, [(1, n) for n in shape_u], names, scale)


def _bands_aggregate(out, e: ExprNode, v: str, ctx: Ctx, shape_u, scale) -> None:
    oidx = [str(x) for x in e.output_idx if isinstance(x, str)]
    if len(oidx) != len(e.output_idx):
        raise BandError("singleton `1` output_idx entries not supported yet")
    if e.filter is not None:
        raise BandError("filtered aggregates not supported (the filter gate "
                        "would be dropped from the derivative)")
    ranges = e.ranges or {}
    cnames = sorted(str(n) for n in ranges if str(n) not in oidx)
    if e.reduce is not None and e.reduce != "+":
        raise BandError(f"reduce=`{e.reduce}` (non-smooth semiring reductions "
                        "have no bands)")
    crange = [(n, *_range_of(ranges[n], ctx)) for n in cnames]
    rows = []
    for k, n in enumerate(oidx):
        rows.append(_range_of(ranges[n], ctx) if n in ranges
                    else (1, shape_u[k]))
    body = e.expr
    for rows2, body2 in clip_regions(body, oidx, rows):
        for s in sorted_sites(body2, v):
            d = dscalar(body2, s)
            if isinstance(s.expr, str) and is_array(ctx, v):
                raise BandError(
                    f"whole-array reference to `{v}` inside an aggregate body")
            cidx = _site_cidx(s)
            if not crange:
                coef = simp(mul(d, *scale))
                if not is_zero(coef):
                    out.append(Band(rows2, list(oidx), cidx, coef))
            elif not any(any(n in free_variables(c) for c in cidx)
                         for n, _, _ in crange):
                red = ExprNode(op="aggregate", args=[], output_idx=[],
                               reduce="+", expr=d,
                               ranges={n: [lo, hi] for n, lo, hi in crange})
                coef = simp(mul(red, *scale))
                if not is_zero(coef):
                    out.append(Band(rows2, list(oidx), cidx, coef))
            else:
                coef = simp(mul(d, *scale))
                if not is_zero(coef):
                    out.append(Band(rows2, list(oidx), cidx, coef,
                                    list(crange)))


def _bands_makearray(out, e: ExprNode, v: str, ctx: Ctx, scale) -> None:
    for region, val in zip(e.regions, e.values):
        rows = [(region_bound(r[0]), region_bound(r[1])) for r in region]
        if any(hi < lo for lo, hi in rows):
            continue                                # folded-empty region
        nonsing = [k for k, (lo, hi) in enumerate(rows) if hi > lo]
        if is_array_expr(val, ctx):
            sub: list[Band] = []
            val_fullrank = (isinstance(val, ExprNode) and val.op == "aggregate"
                            and len(val.output_idx) == len(rows))
            if val_fullrank:
                sub_shape = [hi - lo + 1 for lo, hi in rows]
            else:
                sub_shape = [rows[k][1] - rows[k][0] + 1 for k in nonsing]
            _bands_array(sub, val, v, ctx, sub_shape, scale)
            if isinstance(val, ExprNode) and val.op == "aggregate":
                vr = val.ranges or {}
                origins = [_range_of(vr[str(n)], ctx)[0] if str(n) in vr else 1
                           for n in val.output_idx]
            else:
                origins = [1] * len(nonsing)
            for b in sub:
                ridx: list = []
                subst: dict[str, Expr] = {}
                rows2: list[tuple[int, int]] = []
                fullrank = len(b.ridx) == len(rows)
                kk = -1
                for k, (lo, hi) in enumerate(rows):
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
                cidx = b.cidx if not subst else [substitute(c, subst)
                                                 for c in b.cidx]
                out.append(Band(rows2, ridx, cidx, coef, b.contracted))
        else:
            _scalar_bands(out, val, v, rows, [lo for lo, _ in rows], scale)


def _push_index(e: Expr, idx: list, ctx: Ctx) -> Expr:
    if isinstance(e, str):
        return op("index", e, *idx) if is_array(ctx, e) else e
    if isinstance(e, ExprNode):
        if e.op in ("aggregate", "makearray", "index"):
            raise BandError(f"cannot push an index into `{e.op}` nested "
                            "under an elementwise op")
        return map_children(e, lambda x: _push_index(x, idx, ctx))
    return e


def normalize_band(b: Band) -> Band:
    """Pin singleton named row dims (and one-value contractions) to constants."""
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
    return Band(b.rows, ridx,
                [canon_lits(simplify(substitute(c, subst))) for c in b.cidx],
                simp(substitute(b.coef, subst)), contracted)


def merge_bands(bs: list[Band]) -> list[Band]:
    """Coalesce bands identical in (ridx, cidx, coef, contracted) whose row
    boxes are adjacent along exactly one dimension. Order-preserving."""
    if len(bs) < 2:
        return bs

    def key(b: Band):
        return (tuple(b.ridx), tuple(skey(c) for c in b.cidx), skey(b.coef),
                tuple(b.contracted))

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
                    if d < 0 and (ra[k][1] + 1 == rb[k][0]
                                  or rb[k][1] + 1 == ra[k][0]):
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
                out[a] = Band(rows, out[a].ridx, out[a].cidx, out[a].coef,
                              out[a].contracted)
                alive[b] = False
                changed = True
    return [b for k, b in enumerate(out) if alive[k]]
