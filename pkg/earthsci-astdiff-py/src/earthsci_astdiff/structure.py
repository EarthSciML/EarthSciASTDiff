"""Matrix-structure classification from the bands' column-index expressions
(structure.jl) — never probed numerically."""

from __future__ import annotations

from earthsci_ast.esm_types import Expr, ExprNode

from .jacobian import JacEntry
from .system import SysView, ctx_of


def _is_lit(x: Expr) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _offset_class(c: Expr, r) -> str:
    if isinstance(r, str):
        if isinstance(c, str) and c == r:
            return "diag"
        if isinstance(c, ExprNode) and c.op in ("+", "-") and len(c.args) == 2:
            a, b = c.args
            if isinstance(a, str) and a == r and _is_lit(b):
                return "affine"
            if c.op == "+" and isinstance(b, str) and b == r and _is_lit(a):
                return "affine"
        if _is_lit(c):
            return "fixed"
        return "other"
    if _is_lit(c):
        return "diag" if int(round(float(c))) == r else "fixed"
    return "other"


def detect_structure(entries: list[JacEntry], sv: SysView, column_axis: str | None = None) -> str:
    """Classify the band list: ``"block_diagonal"``, ``"banded"``, ``"general"``
    (or ``"empty"`` / ``"other"``), read off the column-index expressions
    without evaluating anything. Use it to choose a matrix type.

    Given ``column_axis`` (an index-set name, from the caller's locality
    pass), ``"column_block"`` replaces ``"banded"`` / ``"general"`` when
    every coupling stays within one LINE along that axis
    (:func:`column_lines`), so a solver can factorize line by line."""
    st = _detect_structure(entries, sv)
    if column_axis is None or st in ("empty", "block_diagonal"):
        return st
    return st if column_lines(entries, sv, str(column_axis)) is None else "column_block"


def column_lines(entries: list[JacEntry], sv: SysView, axis: str) -> dict | None:
    """The per-line structure of a state Jacobian whose couplings all stay
    within one line along index set ``axis``, or ``None`` when one does not
    (structure.jl ``column_lines``): every band couples two array states of
    one common shape that holds ``axis`` once, and at every band point the
    column equals the row in every other dimension. A point reading past the
    axis is a ghost read and does not count. Returns a dict with ``axis``,
    ``dim`` (1-based), ``line_length``, ``block_vars``, ``kl`` / ``ku`` (the
    largest distances below / above the diagonal any point reaches along the
    axis) and ``pattern`` — ``"dense"``, ``"lower"``, ``"upper"`` or
    ``"banded"`` — read off the exact band points, never probed."""
    from .emit import foreach_point

    if not entries:
        return None
    shape = None
    for en in entries:
        for n in (en.u, en.v):
            var = sv.variables.get(n)
            s = None if var is None else getattr(var, "shape", None)
            if not s:
                return None
            sh = [str(x) for x in s]
            if shape is None:
                shape = sh
            if sh != shape:
                return None
    dims = [k for k, x in enumerate(shape) if x == axis]
    if len(dims) != 1:
        return None
    dim = dims[0]
    N = ctx_of(sv).index_sets[axis]
    kl = ku = 0
    for en in entries:
        for cell, _, _, cidx in foreach_point(en.band):
            for d in range(len(cidx)):
                if d != dim and cidx[d] != cell[d]:
                    return None
            if not (1 <= cidx[dim] <= N):
                continue  # ghost read
            off = cidx[dim] - cell[dim]
            kl = max(kl, -off)
            ku = max(ku, off)
    if kl == N - 1 and ku == N - 1:
        pattern = "dense"
    elif ku == 0 and kl == N - 1:
        pattern = "lower"
    elif kl == 0 and ku == N - 1:
        pattern = "upper"
    else:
        pattern = "banded"
    return {
        "axis": axis,
        "dim": dim + 1,
        "line_length": N,
        "block_vars": sorted({en.u for en in entries} | {en.v for en in entries}),
        "kl": kl,
        "ku": ku,
        "pattern": pattern,
    }


def _detect_structure(entries: list[JacEntry], sv: SysView) -> str:
    if not entries:
        return "empty"
    ctx = ctx_of(sv)
    seen_affine = False
    for en in entries:
        rowarr = en.u in ctx.shapes
        colarr = en.v in ctx.shapes
        if rowarr != colarr:
            return "general"
        b = en.band
        if not b.cidx:
            continue
        per = [_offset_class(c, r) for c, r in zip(b.cidx, b.ridx)]
        if any(p in ("other", "fixed") for p in per):
            return "general"
        if any(p == "affine" for p in per):
            seen_affine = True
    return "banded" if seen_affine else "block_diagonal"
