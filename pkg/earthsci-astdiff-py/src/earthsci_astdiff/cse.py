"""Hash-consed template CSE over emitted band coefficients (cse.jl).

Deterministic and byte-compatible with the Julia reference: the same
extraction rule and naming (esm-jacobian-spec §4.1) — the cross-runtime
block goldens pin it.
"""

from __future__ import annotations

import math
import struct

from earthsci_ast.esm_types import Expr, ExprNode
from earthsci_ast.expr_walk import iter_children, map_children

from .expr_helpers import key_json, ser_expr


def _leaf_key(e: Expr) -> str:
    """A leaf by value, as the reference keys it: an integer is not a float
    (``1`` is not ``1.0``), ``0.0`` is not ``-0.0``, and every NaN is one
    leaf. (``ser_expr`` writes an integral float as an integer.) Never starts
    with ``{``, so it cannot meet a shell key."""
    if isinstance(e, float):
        return "f:nan" if math.isnan(e) else "f:" + struct.pack(">d", e).hex()
    if isinstance(e, int):
        return f"i:{e}"
    if isinstance(e, str):
        return "v:" + e
    return key_json(ser_expr(e))


_SCALAR_SLOTS = ("lower", "upper", "expr", "filter")


def cse_children(e: ExprNode) -> list[Expr]:
    """The child slots of ``e`` in the order of esm-jacobian-spec §4.1:
    ``args``, ``lower``, ``upper``, ``expr``, ``filter``, ``values``, ``axes``
    by ascending key, ``key``, ``bindings`` by ascending key. Fixed by the
    spec, not by ``map_children`` (which visits a map in insertion order), so
    every binding numbers subtrees alike. Raises if ``iter_children`` reaches a
    child this list misses (a field a newer EarthSciAST added)."""
    cs: list[Expr] = list(e.args or ())
    for f in _SCALAR_SLOTS:
        c = getattr(e, f, None)
        if c is not None:
            cs.append(c)
    cs.extend(getattr(e, "values", None) or ())
    ax = getattr(e, "table_axes", None)
    if ax:
        cs.extend(ax[k] for k in sorted(ax))
    kx = getattr(e, "key", None)
    if kx is not None:
        cs.append(kx)
    b = getattr(e, "bindings", None)
    if b:
        cs.extend(b[k] for k in sorted(b))
    if sum(1 for _ in iter_children(e)) != len(cs):
        raise ValueError(
            f"cse_templates: `{e.op}` node has an expression child outside the "
            "child slots of esm-jacobian-spec §4.1"
        )
    return cs


class _CSEIndex:
    """Every structurally distinct subtree with an id (1-based), in
    first-encounter post-order over the child slots of :func:`cse_children`,
    so a child's id is smaller than its parent's. A node is keyed by its shell
    (each child a placeholder carrying its id), O(node fields) per node, and
    memoized by identity: a subtree the inputs share in memory is walked
    once. The walk is iterative, so a deep chain costs no stack."""

    def __init__(self):
        self.ids: dict[str, int] = {}
        self.reps: list[Expr] = []
        self.kids: list[list[int]] = []
        self.nnodes: list[int] = []
        self.memo: dict[int, int] = {}  # id(node) -> id; the inputs outlive the index

    def _fresh(self, key: str, e: Expr, kids: list[int]) -> int:
        got = self.ids.get(key)
        if got is None:
            self.reps.append(e)
            self.kids.append(kids)
            self.nnodes.append(1 + sum(self.nnodes[c - 1] for c in kids))
            got = self.ids[key] = len(self.reps)
        return got

    def child_id(self, c: Expr) -> int:
        """The id of an already-interned child."""
        if isinstance(c, ExprNode):
            return self.memo[id(c)]
        return self.ids[_leaf_key(c)]

    def intern(self, e: Expr) -> int:
        if not isinstance(e, ExprNode):
            return self._fresh(_leaf_key(e), e, [])
        got = self.memo.get(id(e))
        if got is not None:
            return got
        # frame: [node, child slots, their ids so far]
        stack: list[tuple[ExprNode, list[Expr], list[int]]] = [(e, cse_children(e), [])]
        while True:
            n, cs, kids = stack[-1]
            if len(kids) < len(cs):
                c = cs[len(kids)]
                if isinstance(c, ExprNode):
                    got = self.memo.get(id(c))
                    if got is None:
                        stack.append((c, cse_children(c), []))
                    else:
                        kids.append(got)
                else:
                    kids.append(self._fresh(_leaf_key(c), c, []))
                continue
            stack.pop()
            # the shell: every child a placeholder carrying its id (matched by
            # identity / value, so map_children's own visiting order is moot)
            key = key_json(ser_expr(map_children(n, lambda c: f"__cse⟦{self.child_id(c)}⟧")))
            got = self._fresh(key, n, kids)
            self.memo[id(n)] = got
            if not stack:
                return got
            stack[-1][2].append(got)


def cse_templates(exprs: list[Expr], min_nodes: int = 12, prefix: str = "jt"):
    """Extract repeated subtrees by the rule of esm-jacobian-spec §4.1 (the
    reference's cse.jl). Every distinct subtree is visited once, largest
    first (ties in first-encounter order); its count is its number of
    occurrences left once every larger extracted subtree is a reference whose
    body is kept once; one of at least ``min_nodes`` nodes that still counts
    at least 2 is extracted and named ``<prefix><k>`` in visiting order.
    Returns (templates: {name: body}, rewritten exprs)."""
    t = _CSEIndex()
    roots = [t.intern(e) for e in exprs]
    n = len(t.reps)
    # largest first; sorted() is stable, so ties keep id order (equal sizes
    # cannot contain one another: the tie order only decides names)
    order = sorted(range(n), key=lambda i: -t.nnodes[i])
    uses = [0] * n
    for r in roots:
        uses[r - 1] += 1
    name: list[str | None] = [None] * n
    names: list[int] = []
    for i in order:
        # nothing from here on is large enough to extract, and a count only
        # decides extraction, so the rest need no counts
        if t.nnodes[i] < min_nodes:
            break
        r = t.reps[i]
        cand = (
            uses[i] >= 2
            and t.nnodes[i] >= min_nodes
            and isinstance(r, ExprNode)
            and r.op != "apply_expression_template"
        )
        if cand:
            names.append(i)
            name[i] = f"{prefix}{len(names)}"
        w = 1 if cand else uses[i]
        for k in t.kids[i]:
            uses[k - 1] += w
    if not names:
        return {}, list(exprs)
    # rewrite bottom-up in id order (children first)
    body: list[Expr] = []
    out: list[Expr] = []
    for i in range(n):
        r = t.reps[i]
        b = _rebuild(t, r, out) if isinstance(r, ExprNode) and t.kids[i] else r
        body.append(b)
        nm = name[i]
        out.append(b if nm is None else ExprNode(op="apply_expression_template", args=[], name=nm))
    templates = {name[i]: body[i] for i in names}
    return templates, [out[r - 1] for r in roots]


def _rebuild(t: _CSEIndex, r: ExprNode, out: list[Expr]) -> Expr:
    """``r`` with each operator child replaced by what its id rewrote to; ``r``
    itself (no copy) when no child changed. Leaves are never extracted, so
    they stay."""

    def repl(c: Expr) -> Expr:
        return out[t.memo[id(c)] - 1] if isinstance(c, ExprNode) else c

    if all(repl(c) is c for c in iter_children(r)):
        return r
    return map_children(r, repl)


def expand_templates(e: Expr, registry: dict[str, Expr]) -> Expr:
    if not isinstance(e, ExprNode):
        return e
    if e.op == "apply_expression_template":
        if e.name is None or e.name not in registry:
            raise ValueError(
                f"apply_expression_template reference `{e.name}` not in the "
                "block's expression_templates"
            )
        return expand_templates(registry[e.name], registry)
    return map_children(e, lambda x: expand_templates(x, registry))
