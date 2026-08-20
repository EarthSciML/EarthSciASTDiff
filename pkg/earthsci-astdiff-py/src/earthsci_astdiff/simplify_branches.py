"""Branch-aware simplification + wire-canonical literals (simplify_branches.jl).

``simp`` is THE coefficient simplifier of the band calculus: algebraic pass
(`earthsci_ast.simplify`), branch pass, a final algebraic pass, then literal
normalization to the parse-canonical form (CONFORMANCE_SPEC §5.5.3.1 rule 1).
"""
from __future__ import annotations

from earthsci_ast.esm_types import Expr, ExprNode
from earthsci_ast.expr_walk import map_children
from .simplify_port import simplify_jl as simplify

from .expr_helpers import is_zero, op, skey

_INT64_MIN, _INT64_MAX = -(2 ** 63), 2 ** 63 - 1


def _skc(e: Expr, cache: dict[int, str]) -> str:
    k = id(e)
    got = cache.get(k)
    if got is None:
        got = cache[k] = skey(e)
    return got


def simplify_branches(e: Expr) -> Expr:
    """Literal-condition fold, equal-branch collapse, same-condition pruning."""
    return _sbr(e, {})


def _sbr(e: Expr, cache: dict[int, str]) -> Expr:
    if not isinstance(e, ExprNode):
        return e
    e2 = map_children(e, lambda x: _sbr(x, cache))
    if not (e2.op == "ifelse" and len(e2.args) == 3):
        return e2
    c, a, b = e2.args
    if isinstance(c, (int, float)) and not isinstance(c, bool):
        return b if is_zero(c) else a
    ck = _skc(c, cache)
    a2 = _assume(a, ck, True, cache)
    b2 = _assume(b, ck, False, cache)
    if _skc(a2, cache) == _skc(b2, cache):
        return a2
    if a2 is a and b2 is b:
        return e2
    return op("ifelse", c, a2, b2)


def _assume(e: Expr, ck: str, val: bool, cache: dict[int, str]) -> Expr:
    if not isinstance(e, ExprNode):
        return e
    if e.op == "ifelse" and len(e.args) == 3 and _skc(e.args[0], cache) == ck:
        return _assume(e.args[1 if val else 2], ck, val, cache)
    return map_children(e, lambda x: _assume(x, ck, val, cache))


def canon_lit(e: Expr) -> Expr:
    if (isinstance(e, float) and e == e and abs(e) != float("inf")
            and float(e).is_integer() and _INT64_MIN <= e <= _INT64_MAX
            and float(int(e)) == e):
        return int(e)
    return e


def canon_lits(e: Expr) -> Expr:
    if isinstance(e, ExprNode):
        return map_children(e, canon_lits)
    return canon_lit(e)


def simp(e: Expr) -> Expr:
    return canon_lits(simplify(simplify_branches(simplify(e))))
