"""Observed inlining and `index`-of-array-expression β-reduction (inline.jl)."""

from __future__ import annotations

import re
from dataclasses import replace

from earthsci_ast.esm_types import Expr, ExprNode
from earthsci_ast.expr_walk import map_children
from earthsci_ast.expression import free_variables

from .expr_helpers import op, region_bound, skey

MAX_INLINE_PASSES = 64


class InlineError(Exception):
    pass


def inline_observed(e: Expr, obs: dict[str, Expr], obs_names: list | None = None) -> Expr:
    """Inline observed reads, β-reduce `index`-of-`faq`, lower
    `index`-of-`makearray` to region membership `ifelse`. Fixpoint-iterated.
    Every substitution is capture-avoiding, and a faq node binding a name an
    enclosing faq already binds is renamed apart. ``obs_names`` caches the
    names ``obs``'s definitions use (a one-element list, ``[None]`` until
    filled); pass one to every call against the same ``obs``."""
    if not obs and not _needs_inline(e):
        return e
    fresh = _Fresh(e, obs, [None] if obs_names is None else obs_names)
    for _ in range(MAX_INLINE_PASSES):
        e2 = _inline_once(e, obs, fresh, frozenset())
        if skey(e2) == skey(e):
            return e2
        e = e2
    raise InlineError(f"no fixpoint after {MAX_INLINE_PASSES} passes (cyclic observed?)")


_GEN = re.compile(r"^_c\d+_")


class _Fresh:
    """Fresh names for renamed-apart binders (inline.jl ``_Fresh``):
    ``_c<n>_<name>``, the smallest ``n`` clear of every name the tree and the
    observed definitions use and of the caller's ``avoid`` set. Deterministic
    in (name, avoid), so copies of one sum renamed in the same scope stay
    structurally equal. The taken sets are collected on the first rename."""

    def __init__(self, root: Expr, obs: dict[str, Expr], obs_names: list):
        self.root = root
        self.obs = obs
        self.obs_names = obs_names
        self.taken: set[str] | None = None

    def __call__(self, name: str, avoid) -> str:
        if self.taken is None:
            if self.obs_names[0] is None:
                t: set[str] = set()
                for d in self.obs.values():
                    all_names(d, t)
                self.obs_names[0] = t
            self.taken = all_names(self.root)
        base = _GEN.sub("", name)
        n = 0
        while True:
            n += 1
            c = f"_c{n}_{base}"
            if c not in self.taken and c not in self.obs_names[0] and c not in avoid:
                return c


def binders(e: ExprNode) -> list[str]:
    """The index names a faq node binds: its ranges (sorted, so renaming is
    deterministic across bindings) and its named output indices."""
    b = sorted(str(n) for n in (e.ranges or {}))
    for x in e.output_idx or []:
        if isinstance(x, str) and x not in b:
            b.append(x)
    return b


def all_names(e: Expr, acc: set[str] | None = None) -> set[str]:
    """Every variable name and every faq-bound index name in ``e``."""
    if acc is None:
        acc = set()
    if isinstance(e, str):
        acc.add(e)
    elif isinstance(e, ExprNode):
        if e.op == "faq":
            acc.update(binders(e))
        map_children(e, lambda x: (all_names(x, acc), x)[1])
    return acc


def _rename_binders(e: ExprNode, names, fresh: _Fresh, avoid) -> ExprNode:
    """Rename the binders ``names`` of faq node ``e`` to fresh ones, clear of
    ``avoid`` and of every name under ``e``."""
    if e.join is not None:
        raise InlineError("cannot rename the indices of a joined faq node (join keys name them)")
    av = all_names(e, set(avoid))
    ren: dict[str, Expr] = {}
    for n in names:
        f = fresh(n, av)
        av.add(f)
        ren[n] = f
    fv = set(ren.values())
    return replace(
        e,
        ranges=None
        if e.ranges is None
        else {ren.get(str(n), str(n)): r for n, r in e.ranges.items()},
        output_idx=None
        if e.output_idx is None
        else [ren.get(x, x) if isinstance(x, str) else x for x in e.output_idx],
        expr=_csubst(e.expr, ren, fv, fresh),
        filter=None if e.filter is None else _csubst(e.filter, ren, fv, fresh),
    )


def _csubst(e: Expr, m: dict, fv: set, fresh: _Fresh) -> Expr:
    """CAPTURE-AVOIDING substitution (``substitute`` is not): a faq node
    shadows the names it binds, and one binding a name free in a replacement
    (``fv``) is renamed apart before the substitution enters it."""
    if not m:
        return e
    if isinstance(e, str):
        return m.get(e, e)
    if not isinstance(e, ExprNode):
        return e
    if e.op == "faq":
        b = binders(e)
        if any(n in m for n in b):
            m = {k: v for k, v in m.items() if k not in b}
            if not m:
                return e
        clash = [n for n in b if n in fv]
        if clash:
            e = _rename_binders(e, clash, fresh, fv)
    return map_children(e, lambda x: _csubst(x, m, fv, fresh))


def _beta_faq(de: ExprNode, sub: list[Expr], fresh: _Fresh, scope) -> Expr:
    """β-reduce a read ``F[sub…]`` of a faq-defined array F (inline.jl
    ``_beta_faq``): output indices take the subscripts; any other index F's
    ranges bind is CONTRACTED, so the read is the scalar faq summing over it
    (``reduce``/``filter``/``semiring`` carried). Capture-avoiding: a
    contracted name the subscripts read, or one shadowing an index bound
    around the reading site (``scope``), is renamed apart."""
    oidx = [str(x) for x in (de.output_idx or [])]
    if len(sub) != len(oidx):
        raise InlineError(f"index arity {len(sub)} ≠ rank {len(oidx)} of a faq read")
    ranges = de.ranges or {}
    cnames = sorted(str(n) for n in ranges if str(n) not in oidx)
    subst: dict[str, Expr] = {n: sub[k] for k, n in enumerate(oidx)}
    fv: set[str] = set()
    for x in sub:
        fv |= free_variables(x)
    if not cnames and de.filter is None:
        return _csubst(de.expr, subst, fv, fresh)
    if de.join is not None:
        raise InlineError(
            "cannot inline a read of a joined faq definition (join keys name its indices)"
        )
    newranges = {}
    av = fv | set(scope) | set(cnames)
    for n in cnames:
        if n in fv or n in scope:
            f = fresh(n, av)
            av.add(f)
            subst[n] = f
            newranges[f] = ranges[n]
        else:
            newranges[n] = ranges[n]
    fv2 = fv | {x for x in subst.values() if isinstance(x, str)}
    return replace(
        de,
        output_idx=[],
        ranges=newranges,
        expr=_csubst(de.expr, subst, fv2, fresh),
        filter=None if de.filter is None else _csubst(de.filter, subst, fv2, fresh),
    )


def _needs_inline(e: Expr, scope=frozenset()) -> bool:
    """Whether a pass could change ``e`` with no observed to inline: an
    `index`-of-array-expression, or a faq node shadowing an enclosing index."""
    if not isinstance(e, ExprNode):
        return False
    if e.op == "index" and isinstance(e.args[0], ExprNode):
        return True
    if e.op == "faq":
        b = binders(e)
        if any(n in scope for n in b):
            return True
        scope = scope | set(b)
    found = False

    def visit(c: Expr) -> Expr:
        nonlocal found
        if not found:
            found = _needs_inline(c, scope)
        return c

    map_children(e, visit)
    return found


def _index_makearray(
    ma: ExprNode, idx: list[Expr], obs: dict[str, Expr], fresh: _Fresh, scope
) -> Expr:
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
                    f"makearray value rank {len(oidx)} ≠ region free rank {len(sub_idx)}"
                )
            v = _beta_faq(val, sub_idx, fresh, scope)
        elif isinstance(val, str) or (
            isinstance(val, ExprNode) and val.op in ("makearray", "index")
        ):
            v = _inline_once(op("index", val, *sub_idx), obs, fresh, scope)
        else:
            v = val  # scalar broadcast over the region
        cond = conds[0] if len(conds) == 1 else op("and", *conds)
        result = op("ifelse", cond, v, result)
    return result


def _inline_once(e: Expr, obs: dict[str, Expr], fresh: _Fresh, scope: frozenset) -> Expr:
    """One pass; ``scope`` is the index names the enclosing faq nodes bind."""
    if isinstance(e, str):
        # A name an enclosing faq binds is that index, not an observed read.
        if e in scope:
            return e
        return obs.get(e, e)
    if not isinstance(e, ExprNode):
        return e

    def args() -> list[Expr]:
        return [_inline_once(x, obs, fresh, scope) for x in e.args[1:]]

    if e.op == "index" and isinstance(e.args[0], ExprNode):
        t = e.args[0]
        if t.op == "makearray":
            return _index_makearray(t, args(), obs, fresh, scope)
        if t.op == "faq":
            return _inline_once(_beta_faq(t, args(), fresh, scope), obs, fresh, scope)
    if e.op == "faq":
        # A faq node binding a name already bound around it is renamed apart:
        # the band calculus reads a free index in a derivative by name.
        b = binders(e)
        clash = [n for n in b if n in scope]
        if clash:
            e = _rename_binders(e, clash, fresh, scope)
            b = binders(e)
        inner = scope | frozenset(b)
        return map_children(e, lambda x: _inline_once(x, obs, fresh, inner))
    if (
        e.op == "index"
        and isinstance(e.args[0], str)
        and e.args[0] in obs
        and e.args[0] not in scope
    ):
        name = e.args[0]
        de = obs[name]
        if isinstance(de, ExprNode) and de.op == "faq":
            oidx = de.output_idx or []
            if len(oidx) != len(e.args) - 1:
                raise InlineError(
                    f"index arity {len(e.args) - 1} ≠ rank {len(oidx)} of observed `{name}`"
                )
            return _beta_faq(de, args(), fresh, scope)
        if isinstance(de, ExprNode) and de.op == "makearray":
            return _index_makearray(de, args(), obs, fresh, scope)
        if isinstance(de, ExprNode) and de.op == "const":
            return op("index", de, *args())
        raise InlineError(
            f"index into observed `{name}` whose definition is not an faq, makearray or const"
        )
    return map_children(e, lambda x: _inline_once(x, obs, fresh, scope))


def subst_free(e: Expr, m: dict) -> Expr:
    """Substitution that does not descend into a faq for the names it binds
    (bands.jl ``_subst_free``)."""
    if isinstance(e, str):
        return m.get(e, e)
    if not isinstance(e, ExprNode):
        return e
    if e.op == "faq":
        bound = binders(e)
        if any(n in m for n in bound):
            m = {k: v for k, v in m.items() if k not in bound}
            if not m:
                return e
    return map_children(e, lambda x: subst_free(x, m))
