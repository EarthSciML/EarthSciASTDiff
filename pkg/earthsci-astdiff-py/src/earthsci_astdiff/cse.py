"""Hash-consed template CSE over emitted band coefficients (cse.jl).

Deterministic and byte-compatible with the Julia reference: the same
extraction rule and naming (esm-jacobian-spec §4.1) — the cross-runtime
block goldens pin it.
"""

from __future__ import annotations

import math
import struct

from earthsci_ast.esm_types import Expr, ExprNode
from earthsci_ast.expr_walk import map_children

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


class _CSEIndex:
    """Every structurally distinct subtree with an id (1-based), in
    first-encounter post-order (children in ``map_children`` order), so a
    child's id is smaller than its parent's. A node is keyed by its shell
    (each child a placeholder carrying its id), O(node fields) per node, and
    memoized by identity: a subtree the inputs share in memory is walked
    once."""

    def __init__(self):
        self.ids: dict[str, int] = {}
        self.reps: list[Expr] = []
        self.kids: list[list[int]] = []
        self.nnodes: list[int] = []
        self.memo: dict[int, int] = {}  # id(node) -> id; the inputs outlive the index

    def intern(self, e: Expr) -> int:
        if isinstance(e, ExprNode):
            got = self.memo.get(id(e))
            if got is not None:
                return got
        childids: list[int] = []
        if isinstance(e, ExprNode):
            map_children(e, lambda c: (childids.append(self.intern(c)), c)[1])
            i = [0]

            def repl(_c: Expr) -> Expr:
                i[0] += 1
                return f"__cse⟦{childids[i[0] - 1]}⟧"

            key = key_json(ser_expr(map_children(e, repl)))
        else:
            key = _leaf_key(e)
        got = self.ids.get(key)
        if got is None:
            self.reps.append(e)
            self.kids.append(childids)
            self.nnodes.append(1 + sum(self.nnodes[c - 1] for c in childids))
            got = self.ids[key] = len(self.reps)
        if isinstance(e, ExprNode):
            self.memo[id(e)] = got
        return got


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
        if t.nnodes[i] == 1:
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
        if isinstance(r, ExprNode) and t.kids[i]:
            ks = iter(t.kids[i])
            b = map_children(r, lambda _c, ks=ks: out[next(ks) - 1])
        else:
            b = r
        body.append(b)
        nm = name[i]
        out.append(b if nm is None else ExprNode(op="apply_expression_template", args=[], name=nm))
    templates = {name[i]: body[i] for i in names}
    return templates, [out[r - 1] for r in roots]


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
