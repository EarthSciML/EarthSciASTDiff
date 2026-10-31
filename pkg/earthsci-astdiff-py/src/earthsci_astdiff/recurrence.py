"""Causal self-reference recurrences (spec §4.5; esm-spec §4.3.1.1): an
observed ``V`` whose defining ``faq`` reads ``V`` itself at a strictly
earlier cell along one of its output axes. Such an observed has no inlining
fixpoint, so it is differentiated as an INTERMEDIATE: the band calculus runs
on its definition like on an equation (rows ``V``), and on every reader with
``V`` as an extra column. The chain rule through the sweep is then either
materialized — composed with the transfer recurrence ``T = (I − G)⁻¹`` of
the line (the block's ``auxiliary``, §4.5) — or left matrix-free as tangent
and adjoint sweeps in the product documents (§9.7). Everything here is an
AST: the sweeps are ESM recurrences that EarthSciAST executes. Mirrors the
Rust binding's recurrence.rs."""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from earthsci_ast.esm_types import Expr, ExprNode
from earthsci_ast.expr_walk import map_children

from .bands import Band, BandError, _range_of, bands, merge_bands, normalize_band
from .clip import _index_bounds, _index_box
from .expr_helpers import add, mul, op, skey
from .inline import all_names, binders, inline_observed, subst_free
from .reductions import conj, conjuncts
from .simplify_branches import canon_lits, simp


@dataclass
class Recurrence:
    """One recurrence: the observed ``name``, defined by a ``faq`` over the
    cell frame ``frame`` (its ``output_idx``) with ranges ``rows``, folding
    along frame dimension ``dim`` (0-based). ``shape`` is the declared shape
    of ``name``, as written (index-set names)."""

    name: str
    frame: list[str]
    rows: list[tuple[int, int]]
    dim: int
    shape: list[str]

    def flip(self) -> int:
        """``lo + hi`` of the recurrence axis: the reversal ``k ↦ lo + hi − k``."""
        return self.rows[self.dim][0] + self.rows[self.dim][1]

    def lo(self) -> int:
        return self.rows[self.dim][0]


def free_vars(e: Expr) -> set[str]:
    """The free variables of ``e`` with every node's own binders subtracted
    at that node (a faq's ranges and output indices, an integral's
    variable), plus any ``wrt`` — the Rust binding's ``helpers::free_vars``.
    (``earthsci_ast``'s ``free_variables`` subtracts no binder.)"""
    if isinstance(e, str):
        return {e}
    if not isinstance(e, ExprNode):
        return set()
    out: set[str] = set()
    map_children(e, lambda x: (out.update(free_vars(x)), x)[1])
    iv = getattr(e, "int_var", None)
    if iv is not None:
        out.discard(str(iv))
    for k in e.ranges or {}:
        out.discard(str(k))
    for k in e.output_idx or []:
        if isinstance(k, str):
            out.discard(k)
    w = getattr(e, "wrt", None)
    if isinstance(w, str):
        out.add(w)
    return out


def _index(v: str, idx: list) -> Expr:
    return op("index", v, *idx) if idx else v


def _and_all(cs: list):
    if not cs:
        return None
    return cs[0] if len(cs) == 1 else op("and", *cs)


def _faq(output_idx: list, ranges: list, expr: Expr) -> ExprNode:
    """A ``faq`` node with the given output indices and ranges and body."""
    return op(
        "faq",
        output_idx=list(output_idx),
        ranges={n: [lo, hi] for n, lo, hi in ranges},
        expr=expr,
    )


def plus_fold(terms: list, zero) -> Expr:
    """A left fold of binary ``+`` (§9.1), ``zero`` for no term."""
    if not terms:
        return zero
    acc = terms[0]
    for t in terms[1:]:
        acc = op("+", acc, t)
    return acc


# ── recognition ──────────────────────────────────────────────────────────────


def _self_reads(e: Expr, v: str, reads: list, bare: list) -> None:
    # The self-reads of `v` in `e` (each read's index arguments) and whether
    # `v` also occurs bare. A faq that binds `v` as an index hides it: there
    # the name is that index, never a read of the observed.
    if isinstance(e, str):
        if e == v:
            bare[0] = True
        return
    if not isinstance(e, ExprNode):
        return
    if e.op == "faq" and v in binders(e):
        return
    if e.op == "index" and e.args and isinstance(e.args[0], str) and e.args[0] == v:
        reads.append(list(e.args[1:]))
        for x in e.args[1:]:
            _self_reads(x, v, reads, bare)
        return
    map_children(e, lambda x: (_self_reads(x, v, reads, bare), x)[1])


def _is_candidate(d: Expr, v: str) -> bool:
    reads: list = []
    bare = [False]
    _self_reads(d, v, reads, bare)
    return bool(reads) or bare[0]


def _defined_name(eq) -> str | None:
    from .system import _indexed_definition, lhs_definition

    n = lhs_definition(eq)
    if n is not None:
        return str(n)
    d = _indexed_definition(eq)
    return None if d is None else str(d[0])


def recurrences(sv, obs: dict, ctx) -> list[Recurrence]:
    """Every recurrence of the view, in equation order: each observed whose
    definition reads itself, with its frame and axis read off the
    definition. A self-read the producer cannot place on one axis is refused
    (EarthSciAST rejects such a document anyway; the producer never
    guesses)."""
    out: list[Recurrence] = []
    for eq in sv.equations:
        n = _defined_name(eq)
        if n is None or n not in obs:
            continue
        if any(r.name == n for r in out) or not _is_candidate(obs[n], n):
            continue
        out.append(_recognize(n, obs[n], sv, ctx))
    return out


def _recognize(v: str, d: Expr, sv, ctx) -> Recurrence:
    def fail(why: str) -> BandError:
        return BandError(f"recurrence `{v}`: {why} (esm-spec §4.3.1.1)")

    var = sv.variables.get(v)
    shape = [str(x) for x in (getattr(var, "shape", None) or [])]
    ext = list(ctx.shapes.get(v, []))
    if not (
        isinstance(d, ExprNode) and d.op == "faq" and d.output_idx and len(d.output_idx) == len(ext)
    ):
        raise fail("its definition is not a faq over its frame")
    if not all(isinstance(x, str) for x in d.output_idx):
        raise fail("a singleton output index is not a frame symbol")
    frame = [str(x) for x in d.output_idx]
    ranges = d.ranges or {}
    rows = [_range_of(ranges[f], ctx) if f in ranges else (1, ext[m]) for m, f in enumerate(frame)]
    reads: list = []
    bare = [False]
    _self_reads(d, v, reads, bare)
    if bare[0]:
        raise fail("a bare self-reference has no cell")
    dims: list[int] = []
    for args in reads:
        if len(args) != len(frame):
            raise fail("a self-read does not index every frame dimension")
        for m, a in enumerate(args):
            if not (isinstance(a, str) and a == frame[m]) and m not in dims:
                dims.append(m)
    if len(dims) == 1:
        return Recurrence(v, frame, [tuple(r) for r in rows], dims[0], shape)
    if not dims:
        raise fail("a self-read reads its own cell")
    raise fail("self-reads move along more than one axis")


def reader_closure(obs: dict, seeds: set) -> set[str]:
    """The names reachable from ``seeds`` through the observed definitions,
    read backwards: every observed whose definition (transitively) reads a
    seed."""
    deps = {n: free_vars(e) for n, e in obs.items()}
    keep: set[str] = set()
    while True:
        before = len(keep)
        for n, fv in deps.items():
            if n not in keep and any(x in seeds or x in keep for x in fv):
                keep.add(n)
        if len(keep) == before:
            return keep


def active(recs: list[Recurrence], obs: dict, seeds: set) -> list[Recurrence]:
    """The recurrences that move with ``seeds`` (an axis's targets, or a
    pair's two target sets): the active intermediates of a pass."""
    moving = reader_closure(obs, seeds)
    return [r for r in recs if r.name in moving]


# ── the lag of a self band ───────────────────────────────────────────────────


def _lag(rec: Recurrence, b: Band) -> Expr:
    """``lag = k − (column index along the axis)`` of a self band over its
    row index ``k`` (a name or a pinned literal). The column index must be
    affine in ``k`` with coefficient 1, and every other dimension the
    identity (esm-spec §4.3.1.1), or the band is refused; so is a literal
    lag below 1, a self-read that is not strictly earlier on the axis (its
    sweep term would be gated to zero, never evaluated)."""
    lag = _lag_expr(rec, b)
    s = simp(lag)
    if isinstance(s, int) and not isinstance(s, bool) and s < 1:
        raise BandError(
            f"recurrence `{rec.name}`: a self-read at lag {s} along `{rec.frame[rec.dim]}` "
            "is not strictly earlier on its axis (esm-spec §4.3.1.1)"
        )
    return lag


def _lag_expr(rec: Recurrence, b: Band) -> Expr:
    # The lag expression of `_lag`, before its sign check.
    def fail() -> BandError:
        return BandError(
            f"recurrence `{rec.name}`: a self-read is not `{rec.frame[rec.dim]}` minus a "
            "lag along its axis (esm-spec §4.3.1.1)"
        )

    for m, (r, c) in enumerate(zip(b.ridx, b.cidx)):
        if m == rec.dim:
            continue
        same = (isinstance(r, str) and isinstance(c, str) and c == r) or (
            isinstance(r, int) and isinstance(c, int) and not isinstance(c, bool) and c == r
        )
        if not same:
            raise BandError(
                f"recurrence `{rec.name}`: a self-read moves along dimension {m + 1} "
                "besides its axis (esm-spec §4.3.1.1)"
            )
    c = b.cidx[rec.dim]
    r = b.ridx[rec.dim]
    if not isinstance(r, str):
        if free_vars(c) & set(rec.frame):
            raise fail()
        return simp(op("-", r, c))
    k = r

    def has(e) -> bool:
        return k in free_vars(e)

    if not isinstance(c, ExprNode):
        raise fail()
    if c.op == "-" and len(c.args) == 2 and c.args[0] == k and not has(c.args[1]):
        return c.args[1]
    if c.op == "+":
        ks = [a for a in c.args if isinstance(a, str) and a == k]
        rest = [a for a in c.args if not (isinstance(a, str) and a == k)]
        if len(ks) != 1 or any(has(a) for a in rest):
            raise fail()
        return simp(op("-", add(*rest)))
    raise fail()


def _lit_at_least(e: Expr, lo: int) -> bool:
    return isinstance(e, int) and not isinstance(e, bool) and e >= lo


# ── sweep terms ──────────────────────────────────────────────────────────────


def _place(rec: Recurrence, b: Band, at: list, plain: list):
    # A band of V's definition placed on the sweep frame: the substitution of
    # its row names by frame expressions and the conditions that keep the
    # frame point inside its row box.
    sigma: dict = {}
    conds: list = []
    for m, r in enumerate(b.ridx):
        if isinstance(r, str):
            sigma[r] = at[m]
            lo, hi = b.rows[m]
            flo, fhi = rec.rows[m]
            if not plain[m] or lo > flo:
                conds.append(op(">=", at[m], lo))
            if not plain[m] or hi < fhi:
                conds.append(op("<=", at[m], hi))
        else:
            conds.append(op("==", at[m], r))
    return sigma, conds


def _term(b: Band, sigma: dict, conds: list, read: Expr, zero) -> Expr:
    # One band's term in a sweep body: `coef · read` at the frame point,
    # under `conds`, summed over the band's contracted names. The conditions
    # gate the read through `ifelse`, which a recurrence body evaluates
    # branch-first, so a guarded self-read is never reached outside its axis.
    conds = list(conds)
    if b.guard is not None:
        conds.append(subst_free(b.guard, sigma))
    body = op("*", subst_free(b.coef, sigma), read)
    c = _and_all(conds)
    if c is not None:
        body = op("ifelse", c, body, zero)
    if not b.contracted:
        return body
    return _faq([], list(b.contracted), body)


def _self_read(rec: Recurrence, b: Band, sigma: dict, target: str, extra: list):
    # The forward read of a self band: `self[frame with axis − lag, extra…]`,
    # with the conditions that the lag is positive and the read cell is on
    # the axis.
    lag = subst_free(_lag(rec, b), sigma)
    d = rec.dim
    pos = op("-", rec.frame[d], lag)
    conds = []
    if not _lit_at_least(lag, 1):
        conds.append(op(">=", lag, 1))
    row_lo = b.rows[d][0] if isinstance(b.ridx[d], str) else b.ridx[d]
    provable = isinstance(lag, int) and not isinstance(lag, bool) and row_lo - lag >= rec.lo()
    if not provable:
        conds.append(op(">=", pos, rec.lo()))
    idx = list(rec.frame)
    idx[d] = pos
    idx.extend(extra)
    return _index(target, idx), conds


def forward_sweep(rec: Recurrence, own: list, own_cols: list, target: str, col, zero) -> Expr:
    """The forward (tangent) sweep of recurrence ``rec``: one ``faq`` over
    its frame whose body adds, per band of ``V``'s definition (``own``, rows
    ``V``), ``coef · col(read)``; a self band reads the sweep variable
    ``target`` itself one lag back. ``col(b, c, ci)`` gives ``(read,
    conds)`` for another column, or ``None`` to skip it."""
    at = list(rec.frame)
    plain = [True] * len(at)
    terms = []
    for b, c in zip(own, own_cols):
        sigma, conds = _place(rec, b, at, plain)
        if c == rec.name:
            r, cs = _self_read(rec, b, sigma, target, [])
        else:
            got = col(b, c, [subst_free(x, sigma) for x in b.cidx])
            if got is None:
                continue
            r, cs = got
        terms.append(_term(b, sigma, conds + cs, r, zero))
    ranges = [(f, lo, hi) for f, (lo, hi) in zip(rec.frame, rec.rows)]
    return _faq(rec.frame, ranges, plus_fold(terms, zero))


def adjoint_sweep(rec: Recurrence, own_self: list, target: str, src: Expr, zero) -> Expr:
    """The adjoint sweep of recurrence ``rec``, written ascending on the
    REVERSED axis (the spec admits no backward lag): ``target[o, k̃]`` is the
    accumulated cotangent of ``V[o, lo + hi − k̃]``, ``src`` plus, per self
    point ``p`` whose column is that cell, ``coef(p) · V̂[row(p)]`` — row
    ``m + lag(p)`` on the axis, i.e. ``k̃ − lag`` reversed."""
    d = rec.dim
    actual = op("-", rec.flip(), rec.frame[d])
    terms = [src]
    for b in own_self:
        lag0 = _lag(rec, b)
        at = list(rec.frame)
        plain = [True] * len(at)
        o_sigma = {r: at[m] for m, r in enumerate(b.ridx) if m != d and isinstance(r, str)}
        lag = subst_free(lag0, o_sigma)
        at[d] = op("+", actual, lag)
        plain[d] = False
        sigma, conds = _place(rec, b, at, plain)
        pos = op("-", rec.frame[d], lag)
        if not _lit_at_least(lag, 1):
            conds.append(op(">=", lag, 1))
        conds.append(op(">=", pos, rec.lo()))
        idx = list(rec.frame)
        idx[d] = pos
        terms.append(_term(b, sigma, conds, _index(target, idx), zero))
    ranges = [(f, lo, hi) for f, (lo, hi) in zip(rec.frame, rec.rows)]
    return _faq(rec.frame, ranges, plus_fold(terms, zero))


def reversed_read(rec: Recurrence, target: str, rho: list) -> Expr:
    """The actual cell of ``rec`` a reversed sweep ``target`` holds at row
    point ``rho``: ``target[rho with the axis reversed]``."""
    idx = list(rho)
    idx[rec.dim] = op("-", rec.flip(), idx[rec.dim])
    return _index(target, idx)


# ── the transfer recurrence and materialization ─────────────────────────────


def _source_names(sv) -> set[str]:
    t = {str(n) for n in sv.variables}
    t |= {str(n) for n in sv.index_sets}
    for eq in sv.equations:
        all_names(eq.lhs, t)
        all_names(eq.rhs, t)
    return t


def transfer_names(sv, recs: list[Recurrence]) -> dict[str, str]:
    """The transfer variable of each recurrence: ``_tr_<V>``, ``_`` appended
    until clear of every name of the source and of the names before it."""
    taken = _source_names(sv)
    out = {}
    for r in recs:
        nm = f"_tr_{r.name}"
        while nm in taken:
            nm += "_"
        taken.add(nm)
        out[r.name] = nm
    return out


def _self_bands(rec: Recurrence, obs: dict, recs: list[Recurrence], ctx) -> list[Band]:
    # The self bands of `rec`, from its definition with every other observed
    # inlined (no recurrence): the line operator `G`, independent of the axis
    # a pass differentiates for.
    m = {k: v for k, v in obs.items() if k not in {r.name for r in recs}}
    d = inline_observed(obs[rec.name], m, [None])
    shape = list(ctx.shapes.get(rec.name, []))
    return [normalize_band(b) for b in merge_bands(bands(d, rec.name, ctx, shape_u=shape))]


def _transfer_j(rec: Recurrence, bs: list[Band]) -> str:
    taken = set(rec.frame)
    for b in bs:
        all_names(b.coef, taken)
        taken |= {n for n, _, _ in b.contracted}
    j = "_tj"
    while j in taken:
        j += "_j"
    return j


def transfer_definition(rec: Recurrence, obs: dict, recs: list[Recurrence], ctx, tname: str):
    """The transfer recurrence ``T`` of ``rec`` (spec §4.5): over the frame
    plus one output index ``j`` on the axis, ``T[o, k, j] = [k == j] +
    Σ_{self bands} coef · T[o, k − lag, j]``, so ``T[o, ·, ·] = (I − G)⁻¹``
    on each line. Lower triangular on each line."""
    bs = _self_bands(rec, obs, recs, ctx)
    j = _transfer_j(rec, bs)
    d = rec.dim
    at = list(rec.frame)
    plain = [True] * len(at)
    terms = [op("ifelse", op("==", rec.frame[d], j), 1, 0)]
    for b in bs:
        sigma, conds = _place(rec, b, at, plain)
        r, cs = _self_read(rec, b, sigma, tname, [j])
        terms.append(_term(b, sigma, conds + cs, r, 0))
    ranges = [(f, lo, hi) for f, (lo, hi) in zip(rec.frame, rec.rows)]
    ranges.append((j, rec.rows[d][0], rec.rows[d][1]))
    return canon_lits(_faq([*rec.frame, j], ranges, plus_fold(terms, 0)))


def _transfer_band(rec: Recurrence, tname: str) -> Band:
    d = rec.dim
    kap = f"{rec.frame[d]}_t"
    while kap in rec.frame:
        kap += "_t"
    cidx = list(rec.frame)
    cidx[d] = kap
    tidx = list(cidx)
    tidx[d] = rec.frame[d]
    tidx.append(kap)
    return Band(
        list(rec.rows),
        list(rec.frame),
        cidx,
        _index(tname, tidx),
        [(kap, rec.rows[d][0], rec.rows[d][1])],
        op("<=", kap, rec.frame[d]),
    )


def _binder_names(e: Expr, acc: set) -> None:
    if isinstance(e, ExprNode):
        if e.op == "faq":
            acc.update(binders(e))
        map_children(e, lambda x: (_binder_names(x, acc), x)[1])


def _band_names(b: Band) -> set[str]:
    t = {r for r in b.ridx if isinstance(r, str)}
    t |= {n for n, _, _ in b.contracted}
    all_names(b.coef, t)
    for c in b.cidx:
        all_names(c, t)
    if b.guard is not None:
        all_names(b.guard, t)
    return t


def compose(outer: Band, inner: Band, shape_v: list) -> Band | None:
    """The chain rule on two bands: ``outer`` (rows ``R``, column ``V`` at
    ``φ``) followed by ``inner`` (rows ``V``, column ``x``). Each inner row
    name becomes the outer column index it is read at; the inner contracted
    names are renamed apart (``_c`` appended) and become contracted names of
    the composite. The inner row box and ``V``'s extent restrict the
    composite exactly. ``None`` when the composite is empty."""
    taken = _band_names(outer) | _band_names(inner)
    sigma: dict = {}
    extra = []
    for n, lo, hi in inner.contracted:
        if n in _band_names(outer):
            f = f"{n}_c"
            while f in taken:
                f += "_c"
            taken.add(f)
            sigma[n] = f
            nm = f
        else:
            nm = n
        extra.append((nm, lo, hi))
    rows = list(outer.rows)
    ctr = [list(c) for c in outer.contracted]
    conds: list = []

    def restrict(e, lo, hi):
        if isinstance(e, str):
            for k, r in enumerate(outer.ridx):
                if isinstance(r, str) and r == e:
                    rows[k] = (max(rows[k][0], lo), min(rows[k][1], hi))
                    return
            for c in ctr:
                if c[0] == e:
                    c[1] = max(c[1], lo)
                    c[2] = min(c[2], hi)
                    return
        if lo == hi:
            conds.append(op("==", e, lo))
        else:
            conds.append(op("<=", lo, e))
            conds.append(op("<=", e, hi))

    for m, e in enumerate(outer.cidx):
        n = shape_v[m] if m < len(shape_v) else 1
        r = inner.ridx[m]
        lo, hi = inner.rows[m] if isinstance(r, str) else (r, r)
        lo, hi = max(lo, 1), min(hi, n)
        if not _in_range(e, outer, lo, hi):
            restrict(e, lo, hi)
        if isinstance(r, str):
            sigma[r] = e
    if any(hi < lo for lo, hi in rows) or any(c[2] < c[1] for c in ctr):
        return None
    bound: set = set()
    _binder_names(inner.coef, bound)
    for c in inner.cidx:
        _binder_names(c, bound)
    if inner.guard is not None:
        _binder_names(inner.guard, bound)
    read = set().union(*(free_vars(v) for v in sigma.values())) if sigma else set()
    hit = sorted(bound & read)
    if hit:
        raise BandError(
            f"composing through a recurrence would capture `{hit[0]}` under a nested sum"
        )
    guard = conj(
        conj(outer.guard, _and_all(conds)),
        None if inner.guard is None else subst_free(inner.guard, sigma),
    )
    b = Band(
        rows,
        list(outer.ridx),
        [subst_free(c, sigma) for c in inner.cidx],
        simp(mul(outer.coef, subst_free(inner.coef, sigma))),
        [tuple(c) for c in ctr] + extra,
        guard,
    )
    return _tidy(b)


def _span(e: Expr, b: Band):
    # The span of index expression `e` over `b`'s box (its row and
    # contracted names): the interval bounds of the `+ - *` core
    # (clip._index_bounds, the bounds §9.1's extent decision uses); None
    # outside it or when `e` reads another name. The bounds always enclose
    # `e`'s values, so a comparison they decide holds (or fails) at every
    # point.
    r = _index_bounds(e, _index_box(b))
    return None if r is None else (r[0], r[1])


def _in_range(e: Expr, b: Band, lo: int, hi: int) -> bool:
    s = _span(e, b)
    return s is not None and lo <= s[0] and s[1] <= hi


def _tighten(b: Band, nm: str, lo: int, hi: int) -> bool:
    for k, r in enumerate(b.ridx):
        if isinstance(r, str) and r == nm:
            b.rows[k] = (max(b.rows[k][0], lo), min(b.rows[k][1], hi))
            return True
    for k, (n, clo, chi) in enumerate(b.contracted):
        if n == nm:
            b.contracted[k] = (n, max(clo, lo), min(chi, hi))
            return True
    return False


_INF = 1 << 62


def _tidy(b: Band) -> Band | None:
    # Decide what the box decides of a composite's `where` guard: a conjunct
    # that holds at every point is dropped, one that holds at none empties
    # the band, and a bound on a bare name tightens that name's range
    # instead. Repeated to a fixpoint; the remaining conjuncts are kept once
    # each, in order.
    b = replace(b, rows=list(b.rows), contracted=list(b.contracted))
    while True:
        g = b.guard
        if g is None:
            break
        b.guard = None
        keep: list = []
        seen: set = set()
        changed = False
        for c in conjuncts(g):
            c = canon_lits(c)
            if isinstance(c, int) and not isinstance(c, bool):
                if c == 0:
                    return None
                changed = True
                continue
            if (
                isinstance(c, ExprNode)
                and len(c.args) == 2
                and c.op in ("<=", "<", ">=", ">", "==")
            ):
                if c.op in ("<=", "=="):
                    a, z, off = c.args[0], c.args[1], 0
                elif c.op == "<":
                    a, z, off = c.args[0], c.args[1], 1
                elif c.op == ">=":
                    a, z, off = c.args[1], c.args[0], 0
                else:
                    a, z, off = c.args[1], c.args[0], 1
                sa, sz = _span(a, b), _span(z, b)
                if sa is not None and sz is not None:
                    if c.op == "==":
                        if sa[0] == sa[1] and sz[0] == sz[1]:
                            if sa[0] != sz[0]:
                                return None
                            changed = True
                            continue
                        pin = None
                        if isinstance(a, str) and isinstance(z, int):
                            pin = (a, z)
                        elif isinstance(a, int) and isinstance(z, str):
                            pin = (z, a)
                        if pin is not None and _tighten(b, pin[0], pin[1], pin[1]):
                            changed = True
                            continue
                    else:
                        if sa[1] <= sz[0] - off:
                            changed = True
                            continue
                        if sa[0] > sz[1] - off:
                            return None
                        bound = None
                        if isinstance(a, str) and isinstance(z, int):
                            bound = (a, -_INF, z - off)
                        elif isinstance(a, int) and isinstance(z, str):
                            bound = (z, a + off, _INF)
                        if bound is not None and _tighten(b, *bound):
                            changed = True
                            continue
            k = skey(c)
            if k in seen:
                changed = True
            else:
                seen.add(k)
                keep.append(c)
        if any(hi < lo for lo, hi in b.rows) or any(hi < lo for _, lo, hi in b.contracted):
            return None
        b.guard = _and_all(keep)
        if not changed:
            break
    return normalize_band(b)


@dataclass
class ExtendedBands:
    """The extended first-order system of one pass: the active recurrences,
    the bands of the state equations (columns: the axis's targets and the
    active recurrences) and the bands of the recurrences' own definitions
    (rows: a recurrence). ``row_eq[k]`` is the equation ``rows[k]`` came
    from (two equations may share a state), and ``transfer`` the transfer
    variable name of EVERY recurrence of the view (:func:`transfer_names`,
    so a name never depends on which recurrences are active). Without an
    active recurrence ``rows`` is the ordinary band list and the rest is
    empty."""

    recurrences: list = field(default_factory=list)
    rows: list = field(default_factory=list)
    rec_rows: list = field(default_factory=list)
    row_eq: list = field(default_factory=list)
    transfer: dict = field(default_factory=dict)

    def is_recurrence(self, v: str) -> bool:
        return any(r.name == v for r in self.recurrences)

    def recurrence(self, v: str) -> Recurrence | None:
        return next((r for r in self.recurrences if r.name == v), None)


def materialize(sv, ctx, ext: ExtendedBands) -> list:
    """The materialized band list of an extended system (spec §4.5): every
    state band whose column is a target, plus, for each state band reading a
    recurrence ``V``, its composition with ``dV/dx``. Per row equation (two
    equations of one state are never merged) the entries are stably sorted
    by column name, a column's direct bands before its composites."""
    from .jacobian import JacEntry

    names = ext.transfer
    memo: dict = {}
    out: list = []
    group: list = []  # (column, 0 direct | 1 composite, entry)

    def flush():
        group.sort(key=lambda g: (g[0], g[1]))
        out.extend(g[2] for g in group)
        group.clear()

    for k, en in enumerate(ext.rows):
        if k > 0 and ext.row_eq[k - 1] != ext.row_eq[k]:
            flush()
        rec = ext.recurrence(en.v)
        if rec is None:
            group.append((en.v, 0, en))
            continue
        shape = list(ctx.shapes.get(rec.name, []))
        for dd in _total(rec, ext, ctx, names, memo):
            b = compose(en.band, dd.band, shape)
            if b is not None:
                group.append((dd.v, 1, JacEntry(en.u, dd.v, b)))
    flush()
    return out


def _total(rec: Recurrence, ext: ExtendedBands, ctx, names: dict, memo: dict) -> list:
    # `dV/dx` of `rec` as bands with rows V and target columns: the transfer
    # band composed with each band of V's definition on a target, and with
    # each band on another recurrence W composed with `dW/dx` first.
    from .jacobian import JacEntry

    if rec.name in memo:
        return memo[rec.name]
    tb = _transfer_band(rec, names[rec.name])
    shape = list(ctx.shapes.get(rec.name, []))
    out = []
    for en in ext.rec_rows:
        if en.u != rec.name or en.v == rec.name:
            continue
        w = ext.recurrence(en.v)
        if w is None:
            direct = [en]
        else:
            sw = list(ctx.shapes.get(w.name, []))
            direct = []
            for dd in _total(w, ext, ctx, names, memo):
                b = compose(en.band, dd.band, sw)
                if b is not None:
                    direct.append(JacEntry(en.u, dd.v, b))
        for dd in direct:
            b = compose(tb, dd.band, shape)
            if b is not None:
                out.append(JacEntry(rec.name, dd.v, b))
    memo[rec.name] = out
    return out


def auxiliary(sv, obs: dict, ctx, entries: list) -> list:
    """The ``auxiliary`` definitions a band list reads (spec §4.5): for each
    recurrence whose transfer variable some coefficient reads, ``(name,
    shape, definition)`` — the shape is the recurrence's followed by its
    axis once more. In recurrence order."""
    recs = recurrences(sv, obs, ctx)
    if not recs:
        return []
    names = transfer_names(sv, recs)
    read: set = set()
    for en in entries:
        read |= free_vars(en.band.coef)
    out = []
    for rec in recs:
        t = names[rec.name]
        if t not in read:
            continue
        out.append(
            (t, [*rec.shape, rec.shape[rec.dim]], transfer_definition(rec, obs, recs, ctx, t))
        )
    return out
