"""Expression constructors, structural keys, and the differentiation site.

Mirrors the Julia reference's ``expr_helpers.jl``. Python atoms are the
wire scalars themselves (``int`` / ``float`` literals, ``str`` variable
references); only operator nodes are ``ExprNode``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from earthsci_ast.canonicalize import canonical_json
from earthsci_ast.esm_types import Expr, ExprNode
from earthsci_ast.expr_walk import map_children
from earthsci_ast.serialize import _serialize_expression


def op(o: str, *args: Expr, **kw: Any) -> ExprNode:
    return ExprNode(op=o, args=list(args), **kw)


def is_zero(e: Expr) -> bool:
    return isinstance(e, (int, float)) and not isinstance(e, bool) and e == 0


def is_one(e: Expr) -> bool:
    return isinstance(e, (int, float)) and not isinstance(e, bool) and e == 1


def add(*args: Expr) -> Expr:
    """n-ary ``+`` dropping literal zeros, collapsing to a single argument."""
    a = [x for x in args if not is_zero(x)]
    if not a:
        return 0
    if len(a) == 1:
        return a[0]
    return op("+", *a)


def mul(*args: Expr) -> Expr:
    """n-ary ``*`` short-circuiting on a literal zero, dropping literal ones."""
    if any(is_zero(x) for x in args):
        return 0
    a = [x for x in args if not is_one(x)]
    if not a:
        return 1
    if len(a) == 1:
        return a[0]
    return op("*", *a)


def neg(x: Expr) -> Expr:
    return 0 if is_zero(x) else op("-", x)


def stable_json(x: Any) -> str:
    """Deterministic JSON: sorted object keys at every level (emit.jl)."""
    return json.dumps(x, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False)


def ser_expr(e: Expr) -> Any:
    return _serialize_expression(e)


def skey(e: Expr) -> str:
    """Full-structure key for ANY expression (array ops included)."""
    return stable_json(ser_expr(e))


def ckey(e: Expr) -> str:
    """Term-identity key for SCALAR expressions (RFC §5.4 canonical form)."""
    return canonical_json(e)


@dataclass(frozen=True)
class Site:
    """A bare variable reference or one symbolic ``index(v, i…)`` cell."""

    expr: Expr
    key: str

    @staticmethod
    def of(e: Expr) -> "Site":
        return Site(e, ckey(e))


def is_site(e: Expr, s: Site) -> bool:
    if isinstance(e, str) or (isinstance(e, ExprNode) and e.op == "index"):
        return ckey(e) == s.key
    return False


def occurs(e: Expr, s: Site) -> bool:
    """Structural occurrence — THE sparsity primitive (never a zero test)."""
    if is_site(e, s):
        return True
    if not isinstance(e, ExprNode):
        return False
    found = False

    def visit(c: Expr) -> Expr:
        nonlocal found
        if not found:
            found = occurs(c, s)
        return c

    map_children(e, visit)
    return found


def region_bound(b: Any) -> int:
    """A literal region/range bound (post-load documents carry closed forms)."""
    if isinstance(b, bool):
        raise TypeError("boolean region bound")
    if isinstance(b, (int, float)):
        return int(round(b))
    from earthsci_ast.numpy_interpreter import evaluate

    return int(round(evaluate(b, {})))
