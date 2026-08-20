"""Symbolic factorization info (factorization.jl): a general interface with
the per-cell block (chemistry-shaped) LU as the first implementation —
Markowitz elimination ordering + LU pattern including fill-in, decided at
generation time (the KPP precedent)."""
from __future__ import annotations

from dataclasses import dataclass

from .jacobian import JacEntry
from .structure import detect_structure
from .system import SysView


@dataclass
class BlockDiagonalPlan:
    block_vars: list[str]
    pattern: list[list[bool]]
    ordering: list[int]                    # 1-based, matching the reference
    lu: list[list[bool]]


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


def plan_factorization(entries: list[JacEntry], sv: SysView,
                       structure: str | None = None):
    if structure is None:
        structure = detect_structure(entries, sv)
    if structure == "block_diagonal":
        return _block_diagonal_plan(entries)
    return None


def _block_diagonal_plan(entries: list[JacEntry]) -> BlockDiagonalPlan:
    vars = sorted({en.u for en in entries} | {en.v for en in entries})
    slot = {v: i for i, v in enumerate(vars)}
    ns = len(vars)
    P = [[i == j for j in range(ns)] for i in range(ns)]   # W = I − γJ diag
    for en in entries:
        P[slot[en.u]][slot[en.v]] = True
    ordering = markowitz_ordering(P)
    idx = [o - 1 for o in ordering]
    Pp = [[P[i][j] for j in idx] for i in idx]
    return BlockDiagonalPlan([vars[i] for i in idx], Pp, ordering, lu_fill(Pp))


def serialize_plan(p: BlockDiagonalPlan) -> dict:
    def nzpairs(M):
        return [[i + 1, j + 1]
                for i, j in sorted((i, j)
                                   for i in range(len(M))
                                   for j in range(len(M[i])) if M[i][j])]

    return {
        "type": "block_diagonal_lu",
        "block_vars": list(p.block_vars),
        "ordering": list(p.ordering),
        "pattern": nzpairs(p.pattern),
        "lu_pattern": nzpairs(p.lu),
    }
