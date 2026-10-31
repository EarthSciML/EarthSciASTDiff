"""Second-order bands (second_order.jl, esm-jacobian-spec.md §4.4): the band
calculus applied once more, to the coefficients of the first-order bands. A
Rosenbrock step contains J(uₙ) explicitly, so its discrete adjoint needs
∇(μᵀ J(u) w) once per step; that is a contraction of these entries (see
:func:`~earthsci_astdiff.numeric.hvp`), never a Hessian tensor."""

from __future__ import annotations

from dataclasses import dataclass

from earthsci_ast.expression import free_variables
from earthsci_ast.parse import _parse_expression as parse_expression

from .bands import (
    Band,
    BandError,
    Ctx,
    _lift_nested,
    _nested_sums,
    _site_cidx,
    is_array,
    merge_bands,
    normalize_band,
    sorted_sites,
)
from .clip import clip_regions
from .expr_helpers import is_zero, ser_expr
from .jacobian import JacEntry, _check_closed, _FirstOrder, _json, _targets
from .scalar_rules import dscalar
from .simplify_branches import simp
from .structure import _offset_class
from .system import SysView, ctx_of, observed_defs, sysview


@dataclass
class JacEntry2:
    """One second-order band: ∂²(equation for ``u``)/∂``v`` ∂``w``. ``band``
    is the first-order entry's band with its coefficient replaced by
    ∂coef/∂(site of ``w``): the same ``rows``, ``ridx`` and ``contracted``
    (plus any dimensions the second site adds under a nested sum), and
    ``cidx`` holding ``v``'s column indices FOLLOWED by ``w``'s. The first
    ``ncol`` of them are ``v``'s (:attr:`col_idx` / :attr:`col2_idx` split
    them). Keeping both in one :class:`Band` lets clipping, merging and
    normalization run unchanged."""

    u: str
    v: str
    w: str
    band: Band
    ncol: int

    @property
    def col_idx(self) -> list:
        return self.band.cidx[: self.ncol]

    @property
    def col2_idx(self) -> list:
        return self.band.cidx[self.ncol :]


def axis_pair(wrt) -> tuple[str, str]:
    """A second-order ``wrt`` is a pair: the first derivative's axis, then the
    second's (spec §4.4)."""
    if not (isinstance(wrt, (tuple, list)) and len(wrt) == 2):
        raise ValueError(f"a second-order wrt is a pair (a1, a2), got {wrt!r}")
    a1, a2 = str(wrt[0]), str(wrt[1])
    if a1 not in ("states", "time"):
        raise ValueError(f"the first axis of a pair must be states or time, got {a1}")
    if a2 not in ("states", "parameters", "forcing"):
        raise ValueError(
            f"the second axis of a pair must be states, parameters or forcing, got {a2}"
        )
    return a1, a2


def jacobian_bands2(
    obj, model_name: str | None = None, wrt=("states", "states"), forcing=None
) -> list[JacEntry2]:
    """Second-order bands for the axis pair ``wrt = (a1, a2)``, ``a1`` in
    ``("states", "time")`` and ``a2`` in ``("states", "parameters",
    "forcing")``: every first-order entry of the ``a1`` axis
    (:func:`jacobian_bands`) with its coefficient differentiated by the same
    calculus with respect to every site of every ``a2`` target, broadcast
    over the entry's rows with region clipping and re-merging. No new rules:
    the non-smooth policy (spec §3.2) applies again, and the pattern is the
    global structural union over branches.

    ``(r, c, s)`` and ``(r, s, c)`` come out independently, from distinct
    first-order entries; consumers must not assume either is deduplicated.
    On the ``"parameters"`` and ``"forcing"`` second axes every second column
    index must be closed at build time (spec §4.3), else :class:`BandError`.
    """
    a1, a2 = axis_pair(wrt)
    sv = obj if isinstance(obj, SysView) else sysview(obj, model_name)
    return _bands2(sv, a1, a2, forcing)[1]


def _bands2(sv: SysView, a1: str, a2: str, forcing, memo: _FirstOrder | None = None):
    # (first-order entries, second-order entries). The first-order pass
    # inlines the observed that move with EITHER axis: one that reads only
    # parameters (an Arrhenius rate) is opaque on the state axis, and would
    # hide its parameter sites from the second derivative. `memo` shares
    # that pass with the other blocks of one emission.
    ctx = ctx_of(sv)
    t2 = _targets(sv, a2, forcing, observed_defs(sv))
    e1 = (memo or _FirstOrder()).bands(sv, a1, a2, forcing, seed=t2)
    tset = set(t2)
    out: list[JacEntry2] = []
    for en in e1:
        for w in sorted(free_variables(en.band.coef) & tset):
            for b in merge_bands(_coef_bands(en.band, w, ctx)):
                out.append(JacEntry2(en.u, en.v, w, normalize_band(b), len(en.band.cidx)))
    if a2 in ("parameters", "forcing"):
        for en in out:
            _check_closed_col2(en)
    return e1, out


def _coef_bands(b: Band, w: str, ctx: Ctx) -> list[Band]:
    # One band per site of `w` in a first-order coefficient, broadcast over
    # the entry's rows: `_scalar_bands` with the entry's contracted names
    # bound too.
    out: list[Band] = []
    outer = [r for r in b.ridx if isinstance(r, str)] + [n for n, _, _ in b.contracted]
    for rows2, body in clip_regions(b.coef, b.ridx, b.rows):
        nested = _nested_sums(body, w, outer)
        taken: list = [None]
        for s in sorted_sites(body, w):
            if isinstance(s.expr, str) and is_array(ctx, w):
                raise BandError(f"whole-array reference to `{w}` in a band coefficient")
            c2, d, extra = _lift_nested(
                nested, s, body, _site_cidx(s), dscalar(body, s), ctx, outer, taken
            )
            coef = simp(d)
            if is_zero(coef):
                continue
            out.append(
                Band(rows2, list(b.ridx), list(b.cidx) + c2, coef, list(b.contracted) + extra)
            )
    return out


def _check_closed_col2(en: JacEntry2) -> None:
    # `_check_closed_columns` for the second column of a pair.
    _check_closed(
        en.band,
        en.col2_idx,
        lambda c: f"second column index {_json(c)} of `{en.w}` in ∂²({en.u})/∂{en.v}∂{en.w}",
    )


def structure2(
    e1: list[JacEntry],
    entries: list[JacEntry2],
    sv: SysView,
    a1: str,
    memo: _FirstOrder | None = None,
):
    """``"block_diagonal"`` when the first-order state block is and every
    second column sits on the row's own cell, so the contraction is
    lane-local; ``None`` (field omitted) otherwise, and always on a time
    pair."""
    if a1 != "states" or (memo or _FirstOrder()).structure(e1, sv) != "block_diagonal":
        return None
    ctx = ctx_of(sv)
    for en in entries:
        if (en.u in ctx.shapes) != (en.w in ctx.shapes):
            return None
        c2 = en.col2_idx
        if not c2:
            continue
        if len(c2) != len(en.band.ridx):
            return None
        if not all(_offset_class(c, r) == "diag" for c, r in zip(c2, en.band.ridx)):
            return None
    return "block_diagonal"


# ── emission ─────────────────────────────────────────────────────────────────


def serialize_band2(en: JacEntry2) -> dict:
    """One second-order entry of a pair block, as a JSON-ready dict."""
    b = en.band
    d = {
        "row": en.u,
        "col": en.v,
        "col2": en.w,
        "rows": [[lo, hi] for lo, hi in b.rows],
        "row_idx": list(b.ridx),
        "col_idx": [ser_expr(c) for c in en.col_idx],
        "col2_idx": [ser_expr(c) for c in en.col2_idx],
        "coef": ser_expr(b.coef),
    }
    if b.contracted:
        d["contracted"] = [[n, lo, hi] for n, lo, hi in b.contracted]
    return d


def parse_band2(d: dict) -> JacEntry2:
    """Inverse of :func:`serialize_band` on a second-order entry."""
    from .emit import parse_band

    en = parse_band(d)
    b = en.band
    c2 = [parse_expression(c) for c in d["col2_idx"]]
    return JacEntry2(
        en.u,
        en.v,
        str(d["col2"]),
        Band(b.rows, b.ridx, list(b.cidx) + c2, b.coef, b.contracted),
        len(b.cidx),
    )


def jacobian_block2(
    sv: SysView, wrt, forcing, cse: bool, cse_min_nodes: int, memo: _FirstOrder | None = None
) -> dict:
    # The `jacobians` block of one axis pair (spec §4.4): no factorization,
    # `structure` only when the contraction is lane-local, and the second
    # axis's `columns` when it is a parameter or forcing axis.
    from .emit import _block, _columns_field

    a1, a2 = wrt
    memo = memo or _FirstOrder()
    e1, entries = _bands2(sv, a1, a2, forcing, memo)
    return _block(
        [a1, a2],
        entries,
        serialize_band2,
        _columns_field(sv, a2, forcing) if a2 in ("parameters", "forcing") else None,
        structure2(e1, entries, sv, a1, memo),
        cse,
        cse_min_nodes,
    )
