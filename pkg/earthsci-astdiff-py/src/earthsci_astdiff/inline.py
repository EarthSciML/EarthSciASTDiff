"""Observed inlining and `index`-of-array-expression β-reduction (inline.jl)."""
from __future__ import annotations

from earthsci_ast.esm_types import Expr, ExprNode
from earthsci_ast.expr_walk import map_children
from earthsci_ast.substitute import substitute

from .expr_helpers import op, region_bound, skey

MAX_INLINE_PASSES = 64


class InlineError(Exception):
    pass


def inline_observed(e: Expr, obs: dict[str, Expr]) -> Expr:
    """Inline observed reads, β-reduce `index`-of-`aggregate`, lower
    `index`-of-`makearray` to region membership `ifelse`. Fixpoint-iterated."""
    if not obs and not _needs_inline(e):
        return e
    for _ in range(MAX_INLINE_PASSES):
        e2 = _inline_once(e, obs)
        if skey(e2) == skey(e):
            return e2
        e = e2
    raise InlineError(
        f"no fixpoint after {MAX_INLINE_PASSES} passes (cyclic observed?)")


def _needs_inline(e: Expr) -> bool:
    if not isinstance(e, ExprNode):
        return False
    if e.op == "index" and isinstance(e.args[0], ExprNode):
        return True
    found = False

    def visit(c: Expr) -> Expr:
        nonlocal found
        if not found:
            found = _needs_inline(c)
        return c

    map_children(e, visit)
    return found


def _index_makearray(ma: ExprNode, idx: list[Expr], obs: dict[str, Expr]) -> Expr:
    """index(makearray(regions, values), e…) → nested region ifelse.
    Fold from the FIRST region outward so later regions override (§4.3.2)."""
    result: Expr = 0
    for region, val in zip(ma.regions, ma.values):
        conds: list[Expr] = []
        sub_idx: list[Expr] = []
        for k, r in enumerate(region):
            conds.append(op(">=", idx[k], r[0]))
            conds.append(op("<=", idx[k], r[1]))
            if region_bound(r[1]) > region_bound(r[0]):
                sub_idx.append(idx[k])
        if isinstance(val, ExprNode) and val.op == "aggregate":
            oidx = [str(x) for x in val.output_idx]
            if len(oidx) != len(sub_idx):
                raise InlineError(
                    f"makearray value rank {len(oidx)} ≠ region free rank "
                    f"{len(sub_idx)}")
            v = substitute(val.expr, {n: sub_idx[k] for k, n in enumerate(oidx)})
        elif isinstance(val, str) or (isinstance(val, ExprNode) and
                                      val.op in ("makearray", "index")):
            v = _inline_once(op("index", val, *sub_idx), obs)
        else:
            v = val                      # scalar broadcast over the region
        cond = conds[0] if len(conds) == 1 else op("and", *conds)
        result = op("ifelse", cond, v, result)
    return result


def _inline_once(e: Expr, obs: dict[str, Expr]) -> Expr:
    if isinstance(e, str):
        return obs.get(e, e)
    if not isinstance(e, ExprNode):
        return e
    if e.op == "index" and isinstance(e.args[0], ExprNode):
        t = e.args[0]
        if t.op == "makearray":
            return _index_makearray(
                t, [_inline_once(x, obs) for x in e.args[1:]], obs)
        if t.op == "aggregate":
            oidx = [str(x) for x in t.output_idx]
            subst = {n: _inline_once(e.args[k + 1], obs)
                     for k, n in enumerate(oidx)}
            return _inline_once(substitute(t.expr, subst), obs)
    if (e.op == "index" and isinstance(e.args[0], str) and e.args[0] in obs):
        name = e.args[0]
        de = obs[name]
        if isinstance(de, ExprNode) and de.op == "aggregate":
            oidx = [str(x) for x in de.output_idx]
            if len(oidx) != len(e.args) - 1:
                raise InlineError(
                    f"index arity {len(e.args) - 1} ≠ rank {len(oidx)} of "
                    f"observed `{name}`")
            subst = {n: _inline_once(e.args[k + 1], obs)
                     for k, n in enumerate(oidx)}
            return substitute(de.expr, subst)
        if isinstance(de, ExprNode) and de.op == "makearray":
            return _index_makearray(
                de, [_inline_once(x, obs) for x in e.args[1:]], obs)
        if isinstance(de, ExprNode) and de.op == "const":
            return op("index", de, *[_inline_once(x, obs) for x in e.args[1:]])
        raise InlineError(
            f"index into observed `{name}` whose definition is not an "
            "aggregate, makearray or const")
    return map_children(e, lambda x: _inline_once(x, obs))
