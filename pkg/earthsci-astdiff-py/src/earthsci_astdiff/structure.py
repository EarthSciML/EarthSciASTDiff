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


def detect_structure(entries: list[JacEntry], sv: SysView) -> str:
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
