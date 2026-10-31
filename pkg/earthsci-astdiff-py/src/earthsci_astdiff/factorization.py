"""Symbolic factorization info (factorization.jl): a general interface with
the per-cell block (chemistry-shaped) LU as the first implementation —
Markowitz elimination ordering + LU pattern including fill-in, decided at
generation time (the KPP precedent)."""

from __future__ import annotations

from dataclasses import dataclass

from .jacobian import JacEntry
from .structure import column_lines, detect_structure
from .system import SysView


@dataclass
class BlockDiagonalPlan:
    """Per-cell block LU of a block-diagonal Jacobian: the block variables in
    Markowitz order, the block pattern, the 1-based ordering and the pattern
    of the LU factors including fill-in."""

    block_vars: list[str]
    pattern: list[list[bool]]
    ordering: list[int]  # 1-based, matching the reference
    lu: list[list[bool]]


@dataclass
class ColumnBlockPlan:
    """Line-by-line factorization info for a ``"column_block"`` Jacobian:
    every coupling stays within one line along ``axis``, so the matrix is one
    ``(len(block_vars)·line_length)``-square block per line. ``pattern``
    names the coupling along the line (``"dense"``, ``"lower"``,
    ``"upper"``, ``"banded"``) and ``kl`` / ``ku`` bound it below and above
    the diagonal."""

    axis: str
    line_length: int
    block_vars: list[str]
    pattern: str
    kl: int
    ku: int


def lu_fill(pattern: list[list[bool]]) -> list[list[bool]]:
    """Symbolic (no-pivoting) LU: the pattern plus every fill-in entry."""
    P = [row[:] for row in pattern]
    n = len(P)
    for k in range(n):
        for i in range(k + 1, n):
            if P[i][k]:
                for j in range(k + 1, n):
                    P[i][j] = P[i][j] or P[k][j]
    return P


def markowitz_ordering(pattern: list[list[bool]]) -> list[int]:
    """Greedy Markowitz ordering (1-based), first minimal cost wins."""
    P = [row[:] for row in pattern]
    n = len(P)
    remaining = list(range(n))
    order: list[int] = []
    while remaining:
        best, bestcost = remaining[0], None
        for i in remaining:
            r = sum(1 for j in remaining if P[i][j]) - 1
            c = sum(1 for j in remaining if P[j][i]) - 1
            cost = r * c
            if bestcost is None or cost < bestcost:
                best, bestcost = i, cost
        order.append(best)
        rest = [i for i in remaining if i != best]
        for i in rest:
            if P[i][best]:
                for j in rest:
                    P[i][j] = P[i][j] or P[best][j]
        remaining = rest
    return [i + 1 for i in order]


def plan_factorization(
    entries: list[JacEntry],
    sv: SysView,
    structure: str | None = None,
    column_axis: str | None = None,
):
    """Symbolic factorization info for the detected structure.

    Returns a :class:`BlockDiagonalPlan` when the structure is
    ``"block_diagonal"``, a :class:`ColumnBlockPlan` for ``"column_block"``
    (which needs the ``column_axis`` it was detected with), and ``None``
    otherwise; banded and general plans are not implemented yet.
    """
    if structure is None:
        structure = detect_structure(entries, sv, column_axis=column_axis)
    if structure == "block_diagonal":
        return _block_diagonal_plan(entries)
    if structure == "column_block":
        if column_axis is None:
            raise ValueError("a column_block plan needs its `column_axis`")
        cl = column_lines(entries, sv, str(column_axis))
        if cl is None:
            raise ValueError(f"the Jacobian is not column_block along `{column_axis}`")
        return _column_block_plan(cl)
    return None


def _column_block_plan(cl: dict) -> ColumnBlockPlan:
    return ColumnBlockPlan(
        cl["axis"], cl["line_length"], cl["block_vars"], cl["pattern"], cl["kl"], cl["ku"]
    )


def _block_diagonal_plan(entries: list[JacEntry]) -> BlockDiagonalPlan:
    vars = sorted({en.u for en in entries} | {en.v for en in entries})
    slot = {v: i for i, v in enumerate(vars)}
    ns = len(vars)
    P = [[i == j for j in range(ns)] for i in range(ns)]  # W = I − γJ diag
    for en in entries:
        P[slot[en.u]][slot[en.v]] = True
    ordering = markowitz_ordering(P)
    idx = [o - 1 for o in ordering]
    Pp = [[P[i][j] for j in idx] for i in idx]
    return BlockDiagonalPlan([vars[i] for i in idx], Pp, ordering, lu_fill(Pp))


def serialize_plan(p) -> dict:
    if isinstance(p, ColumnBlockPlan):
        return {
            "type": "column_block",
            "axis": p.axis,
            "line_length": p.line_length,
            "block_vars": list(p.block_vars),
            "pattern": p.pattern,
            "bandwidth": [p.kl, p.ku],
        }

    def nzpairs(M):
        return [
            [i + 1, j + 1]
            for i, j in sorted((i, j) for i in range(len(M)) for j in range(len(M[i])) if M[i][j])
        ]

    return {
        "type": "block_diagonal_lu",
        "block_vars": list(p.block_vars),
        "ordering": list(p.ordering),
        "pattern": nzpairs(p.pattern),
        "lu_pattern": nzpairs(p.lu),
    }
