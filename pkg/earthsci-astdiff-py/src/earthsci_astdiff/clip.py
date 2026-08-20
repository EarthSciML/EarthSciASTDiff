"""Static region clipping (clip.jl): decidable membership guards are decided
exactly by splitting band row ranges at their affine breakpoints. Everything
here is conservative — an undecidable condition contributes no breakpoints
and the result degrades to branch-union behavior."""
from __future__ import annotations

from itertools import product

from earthsci_ast.esm_types import Expr, ExprNode
from earthsci_ast.expr_walk import map_children

from .expr_helpers import op

_CMP_OPS = ("<", "<=", ">", ">=")


def _affine(e: Expr, name: str):
    """``e`` as (α, β) over α·name + β, integer arithmetic only; else None."""
    if isinstance(e, str):
        return (1, 0) if e == name else None
    if isinstance(e, bool):
        return None
    if isinstance(e, int):
        return (0, e)
    if isinstance(e, float):
        return (0, int(e)) if e.is_integer() else None
    if isinstance(e, ExprNode):
        a = e.args
        if e.op == "+":
            al, be = 0, 0
            for x in a:
                t = _affine(x, name)
                if t is None:
                    return None
                al += t[0]
                be += t[1]
            return (al, be)
        if e.op == "-" and len(a) == 1:
            t = _affine(a[0], name)
            return None if t is None else (-t[0], -t[1])
        if e.op == "-" and len(a) == 2:
            t1 = _affine(a[0], name)
            t2 = _affine(a[1], name) if t1 is not None else None
            if t1 is None or t2 is None:
                return None
            return (t1[0] - t2[0], t1[1] - t2[1])
        if e.op == "*":
            al, be = 0, 1     # product of affines: at most one sloped factor
            for x in a:
                t = _affine(x, name)
                if t is None:
                    return None
                if t[0] == 0:
                    al *= t[1]
                    be *= t[1]
                elif al == 0 and be != 0:
                    al = be * t[0]
                    be = be * t[1]
                else:
                    return None
            return (al, be)
    return None


def _sole_name(e: Expr, names) -> str | None:
    found: str | None = None
    ok = True

    def walk(x: Expr) -> None:
        nonlocal found, ok
        if isinstance(x, str) and x in names:
            if found is None:
                found = x
            elif found != x:
                ok = False
        if isinstance(x, ExprNode):
            map_children(x, lambda c: (walk(c), c)[1])

    walk(e)
    return found if ok else None


def _cmp_truth(o: str, al: int, be: int, i: int) -> bool:
    v = al * i + be
    if o == "<":
        return v < 0
    if o == "<=":
        return v <= 0
    if o == ">":
        return v > 0
    return v >= 0


def _floor_div(a: int, b: int) -> int:
    return a // b            # Python // is floor division, like Julia fld


def _breakpoints(bp: dict[str, set[int]], e: Expr, names) -> None:
    if not isinstance(e, ExprNode):
        return
    if e.op in _CMP_OPS and len(e.args) == 2:
        n = _sole_name(e, names)
        if n is not None:
            diff = op("-", e.args[0], e.args[1])
            t = _affine(diff, n)
            if t is not None and t[0] != 0:
                q = _floor_div(-t[1], t[0])
                for c in (q, q + 1):
                    if (_cmp_truth(e.op, t[0], t[1], c - 1)
                            != _cmp_truth(e.op, t[0], t[1], c)):
                        bp.setdefault(n, set()).add(c)
    map_children(e, lambda c: (_breakpoints(bp, c, names), c)[1])


def _uniform(cond: Expr, env: dict[str, tuple[int, int]]):
    """Truth over the box, three-valued (True / False / None)."""
    if not isinstance(cond, ExprNode):
        return None
    if cond.op in _CMP_OPS and len(cond.args) == 2:
        n = _sole_name(cond, env.keys())
        if n is None or n not in env:
            return None
        t = _affine(op("-", cond.args[0], cond.args[1]), n)
        if t is None:
            return None
        lo, hi = env[n]
        vlo, vhi = t[0] * lo + t[1], t[0] * hi + t[1]
        a, b = min(vlo, vhi), max(vlo, vhi)
        f = {"<": lambda x: x < 0, "<=": lambda x: x <= 0,
             ">": lambda x: x > 0, ">=": lambda x: x >= 0}[cond.op]
        return f(a) if f(a) == f(b) else None
    if cond.op == "and":
        anynone = False
        for x in cond.args:
            u = _uniform(x, env)
            if u is False:
                return False
            if u is None:
                anynone = True
        return None if anynone else True
    if cond.op == "or":
        anynone = False
        for x in cond.args:
            u = _uniform(x, env)
            if u is True:
                return True
            if u is None:
                anynone = True
        return None if anynone else False
    if cond.op == "not" and len(cond.args) == 1:
        u = _uniform(cond.args[0], env)
        return None if u is None else (not u)
    return None


def _decide(e: Expr, env: dict[str, tuple[int, int]]) -> Expr:
    if not isinstance(e, ExprNode):
        return e
    if e.op == "ifelse" and len(e.args) == 3:
        u = _uniform(e.args[0], env)
        if u is True:
            return _decide(e.args[1], env)
        if u is False:
            return _decide(e.args[2], env)
    elif e.op in _CMP_OPS:
        u = _uniform(e, env)
        if u is not None:
            return 1 if u else 0
    return map_children(e, lambda x: _decide(x, env))


def clip_regions(body: Expr, ridx, rows):
    """Partition ``rows`` at the breakpoints of every statically-decidable
    comparison in ``body``; collapse decided guards per cell. Returns a list
    of ``(rows', body')`` in ascending order per dimension."""
    names = [r for r in ridx if isinstance(r, str)]
    if not names:
        return [(list(rows), body)]
    bp: dict[str, set[int]] = {}
    _breakpoints(bp, body, names)
    if not bp:
        return [(list(rows), body)]

    dims: list[list[tuple[int, int]]] = []
    dimname: list[str | None] = []
    for k, r in enumerate(ridx):
        lo, hi = rows[k]
        if isinstance(r, str) and r in bp:
            cuts = sorted(c for c in bp[r] if lo < c <= hi)
            ivs = []
            a = lo
            for c in cuts:
                ivs.append((a, c - 1))
                a = c
            ivs.append((a, hi))
            dims.append(ivs)
            dimname.append(r)
        else:
            dims.append([(lo, hi)])
            dimname.append(r if isinstance(r, str) else None)

    out = []
    # Julia's Iterators.product cycles the FIRST dimension fastest; Python's
    # cycles the last. Golden band order pins the Julia order.
    for cell_r in product(*reversed(dims)):
        rows2 = list(reversed(cell_r))
        env = {dimname[k]: rows2[k] for k in range(len(rows2))
               if dimname[k] is not None}
        out.append((rows2, _decide(body, env)))
    return out
