"""Hash-consed template CSE over emitted band coefficients (cse.jl).

Deterministic and byte-compatible with the Julia reference: same interning
(first-encounter) order, same largest-first tie-break, same `jt` naming —
the cross-runtime block goldens pin it.
"""
from __future__ import annotations

from earthsci_ast.esm_types import Expr, ExprNode
from earthsci_ast.expr_walk import map_children

from .expr_helpers import ser_expr, stable_json


class _Intern:
    def __init__(self):
        self.ids: dict[str, int] = {}
        self.reps: list[Expr] = []
        self.nnodes: list[int] = []
        self.counts: list[int] = []

    def intern(self, e: Expr) -> int:
        childids: list[int] = []
        if isinstance(e, ExprNode):
            map_children(e, lambda c: (childids.append(self.intern(c)), c)[1])
        key, nn = self._shell_key(e, childids)
        got = self.ids.get(key)
        if got is None:
            self.reps.append(e)
            self.nnodes.append(nn)
            self.counts.append(0)
            got = self.ids[key] = len(self.reps)
        self.counts[got - 1] += 1
        return got

    def _shell_key(self, e: Expr, childids: list[int]) -> tuple[str, int]:
        if isinstance(e, ExprNode):
            i = [0]

            def repl(_c: Expr) -> Expr:
                i[0] += 1
                return f"__cse⟦{childids[i[0] - 1]}⟧"

            shell = map_children(e, repl)
            nn = 1 + sum(self.nnodes[cid - 1] for cid in childids)
            return stable_json(ser_expr(shell)), nn
        return stable_json(ser_expr(e)), 1


def _replace_id(e: Expr, it: _Intern, target: int, repl: Expr):
    """Replace every subtree whose id is ``target``; ids recomputed bottom-up
    against the read-only table. Returns (rewritten, original id of e)."""
    childids: list[int] = []
    newchildren: list[Expr] = []
    if isinstance(e, ExprNode):
        def visit(c: Expr) -> Expr:
            nc, cid = _replace_id(c, it, target, repl)
            childids.append(cid)
            newchildren.append(nc)
            return c

        map_children(e, visit)
    key, _ = it._shell_key(e, childids)
    eid = it.ids[key]
    if eid == target:
        return repl, eid
    if isinstance(e, ExprNode) and newchildren:
        i = [0]

        def take(_c: Expr) -> Expr:
            i[0] += 1
            return newchildren[i[0] - 1]

        e = map_children(e, take)
    return e, eid


def cse_templates(exprs: list[Expr], min_nodes: int = 12,
                  prefix: str = "jt"):
    """Greedy largest-first extraction of repeated subtrees. Returns
    (templates: {name: body}, rewritten exprs)."""
    work = list(exprs)
    nin = len(work)
    names: list[str] = []
    while True:
        it = _Intern()
        for e in work:
            it.intern(e)
        best = 0
        for i in range(1, len(it.reps) + 1):
            if it.counts[i - 1] < 2 or it.nnodes[i - 1] < min_nodes:
                continue
            r = it.reps[i - 1]
            if not isinstance(r, ExprNode):
                continue
            if r.op == "apply_expression_template":
                continue
            if best == 0 or it.nnodes[i - 1] > it.nnodes[best - 1]:
                best = i
        if best == 0:
            break
        name = f"{prefix}{len(names) + 1}"
        names.append(name)
        repl = ExprNode(op="apply_expression_template", args=[], name=name)
        body = it.reps[best - 1]
        work = [_replace_id(e, it, best, repl)[0] for e in work]
        work.append(body)          # later rounds may factor the body itself
    templates = {n: work[nin + k] for k, n in enumerate(names)}
    return templates, work[:nin]


def expand_templates(e: Expr, registry: dict[str, Expr]) -> Expr:
    if not isinstance(e, ExprNode):
        return e
    if e.op == "apply_expression_template":
        if e.name is None or e.name not in registry:
            raise ValueError(
                f"apply_expression_template reference `{e.name}` not in the "
                "block's expression_templates")
        return expand_templates(registry[e.name], registry)
    return map_children(e, lambda x: expand_templates(x, registry))
