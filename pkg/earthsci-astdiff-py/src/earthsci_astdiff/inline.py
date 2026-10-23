"""Observed inlining and `index`-of-array-expression β-reduction (inline.jl)."""
from __future__ import annotations

from dataclasses import replace

from earthsci_ast.esm_types import Expr, ExprNode
from earthsci_ast.expr_walk import map_children
from earthsci_ast.substitute import substitute

from .expr_helpers import op, region_bound, skey

MAX_INLINE_PASSES = 64


class InlineError(Exception):
    pass


def inline_observed(e: Expr, obs: dict[str, Expr]) -> Expr:
    """Inline observed reads, β-reduce `index`-of-`faq`, lower
    `index`-of-`makearray` to region membership `ifelse`. Fixpoint-iterated."""
    if not obs and not _needs_inline(e):
        return e
    fresh = _Fresh(e)
    for _ in range(MAX_INLINE_PASSES):
        e2 = _inline_once(e, obs, fresh)
        if skey(e2) == skey(e):
            return e2
        e = e2
    raise InlineError(
        f"no fixpoint after {MAX_INLINE_PASSES} passes (cyclic observed?)")


class _Fresh:
    """Fresh names for renamed-apart contraction indices: ``_c<n>_<name>``,
    skipping every name the tree being inlined already uses (collected on
    the first request only)."""

    def __init__(self, root: Expr):
        self.root = root
        self.taken: set[str] | None = None
        self.n = 0

    def __call__(self, base: str) -> str:
        if self.taken is None:
            self.taken = all_names(self.root)
        while True:
            self.n += 1
            name = f"_c{self.n}_{base}"
            if name not in self.taken:
                self.taken.add(name)
                return name


def all_names(e: Expr, acc: set[str] | None = None) -> set[str]:
    """Every variable name and every faq-bound index name in ``e``."""
    if acc is None:
        acc = set()
    if isinstance(e, str):
        acc.add(e)
    elif isinstance(e, ExprNode):
        if e.op == "faq":
            acc.update(str(n) for n in (e.ranges or {}))
            acc.update(str(x) for x in (e.output_idx or []) if isinstance(x, str))
        map_children(e, lambda x: (all_names(x, acc), x)[1])
    return acc


def _beta_faq(de: ExprNode, sub: list[Expr], fresh: _Fresh) -> Expr:
    """β-reduce a read ``F[sub…]`` of a faq-defined array F (inline.jl
    ``_beta_faq``): output indices take the subscripts; any other index F's
    ranges bind is CONTRACTED, so the read is the scalar faq summing over it
    (``reduce``/``filter``/``semiring`` carried), with the contracted names
    renamed to fresh ones in the same substitution — ``substitute`` is not
    capture-avoiding, and a faq around the reading site may bind them."""
    oidx = [str(x) for x in de.output_idx]
    if len(sub) != len(oidx):
        raise InlineError(
            f"index arity {len(sub)} ≠ rank {len(oidx)} of a faq read")
    ranges = de.ranges or {}
    cnames = sorted(str(n) for n in ranges if str(n) not in oidx)
    subst = {n: sub[k] for k, n in enumerate(oidx)}
    if not cnames and de.filter is None:
        return substitute(de.expr, subst)
    if de.join is not None:
        raise InlineError("cannot inline a read of a joined faq definition "
                          "(join keys name its indices)")
    newranges = {}
    for n in cnames:
        f = fresh(n)
        subst[n] = f
        newranges[f] = ranges[n]
    return replace(de, output_idx=[], ranges=newranges,
                   expr=substitute(de.expr, subst),
                   filter=None if de.filter is None
                   else substitute(de.filter, subst))


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


def _index_makearray(ma: ExprNode, idx: list[Expr], obs: dict[str, Expr],
                     fresh: _Fresh) -> Expr:
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
        if isinstance(val, ExprNode) and val.op == "faq":
            oidx = [str(x) for x in val.output_idx]
            if len(oidx) != len(sub_idx):
                raise InlineError(
                    f"makearray value rank {len(oidx)} ≠ region free rank "
                    f"{len(sub_idx)}")
            v = _beta_faq(val, sub_idx, fresh)
        elif isinstance(val, str) or (isinstance(val, ExprNode) and
                                      val.op in ("makearray", "index")):
            v = _inline_once(op("index", val, *sub_idx), obs, fresh)
        else:
            v = val                      # scalar broadcast over the region
        cond = conds[0] if len(conds) == 1 else op("and", *conds)
        result = op("ifelse", cond, v, result)
    return result


def _inline_once(e: Expr, obs: dict[str, Expr], fresh: _Fresh) -> Expr:
    if isinstance(e, str):
        de = obs.get(e, e)
        # A scalar sum renamed apart too: spliced under a faq binding the
        # same index, its sites would be the outer ones, structurally.
        if (isinstance(de, ExprNode) and de.op == "faq"
                and not (de.output_idx or [])):
            return _beta_faq(de, [], fresh)
        return de
    if not isinstance(e, ExprNode):
        return e
    if e.op == "index" and isinstance(e.args[0], ExprNode):
        t = e.args[0]
        if t.op == "makearray":
            return _index_makearray(
                t, [_inline_once(x, obs, fresh) for x in e.args[1:]], obs, fresh)
        if t.op == "faq":
            sub = [_inline_once(x, obs, fresh) for x in e.args[1:]]
            return _inline_once(_beta_faq(t, sub, fresh), obs, fresh)
    if (e.op == "index" and isinstance(e.args[0], str) and e.args[0] in obs):
        name = e.args[0]
        de = obs[name]
        if isinstance(de, ExprNode) and de.op == "faq":
            oidx = [str(x) for x in de.output_idx]
            if len(oidx) != len(e.args) - 1:
                raise InlineError(
                    f"index arity {len(e.args) - 1} ≠ rank {len(oidx)} of "
                    f"observed `{name}`")
            return _beta_faq(
                de, [_inline_once(x, obs, fresh) for x in e.args[1:]], fresh)
        if isinstance(de, ExprNode) and de.op == "makearray":
            return _index_makearray(
                de, [_inline_once(x, obs, fresh) for x in e.args[1:]], obs, fresh)
        if isinstance(de, ExprNode) and de.op == "const":
            return op("index", de,
                      *[_inline_once(x, obs, fresh) for x in e.args[1:]])
        raise InlineError(
            f"index into observed `{name}` whose definition is not an "
            "faq, makearray or const")
    return map_children(e, lambda x: _inline_once(x, obs, fresh))
