"""Faithful port of the Julia reference's `EarthSciAST.simplify`.

The two bindings' own ``simplify`` helpers differ (the Python one combines
constants and moves them to the end of ``+``/``*``; the Julia one only
strips exact 0/1 literals, order-preserving, and folds all-literal nodes
through the evaluator). Band goldens pin the JULIA behavior, so the
band calculus uses this port instead of ``earthsci_ast.expression.simplify``.
"""
from __future__ import annotations

import math

from earthsci_ast.esm_types import Expr, ExprNode
from earthsci_ast.expr_walk import map_children
from earthsci_ast.numpy_interpreter import evaluate

_TWO63 = 2.0 ** 63


def _is_lit(x: Expr) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _is_lit_val(x: Expr, v: float) -> bool:
    return _is_lit(x) and float(x) == v


def simplify_jl(e: Expr) -> Expr:
    if not isinstance(e, ExprNode):
        return e
    node = map_children(e, simplify_jl)
    args = node.args
    o = node.op

    # Constant folding, delegated to the official evaluator; a node the
    # evaluator cannot run without bindings declines to fold (algebraic
    # rules below). Mirrors expression.jl's `_foldable_failure` set.
    if all(_is_lit(a) for a in args):
        try:
            rv = float(evaluate(node, {}))
            all_int = all(isinstance(a, int) for a in args)
            if (all_int and math.isfinite(rv) and rv == math.trunc(rv)
                    and abs(rv) <= _TWO63):
                if abs(rv) == _TWO63:      # Julia Int64() InexactError
                    pass                   # → decline to fold
                else:
                    return int(rv)
            else:
                return rv
        except Exception:                  # any evaluator refusal (unbound
            pass                           # names, domain errors) declines

    if o == "+":
        nz = [a for a in args if not _is_lit_val(a, 0.0)]
        if not nz:
            return 0.0
        if len(nz) == 1:
            return nz[0]
        from dataclasses import replace
        return replace(node, args=nz)
    if o == "*":
        if any(_is_lit_val(a, 0.0) for a in args):
            return 0.0
        no = [a for a in args if not _is_lit_val(a, 1.0)]
        if not no:
            return 1.0
        if len(no) == 1:
            return no[0]
        from dataclasses import replace
        return replace(node, args=no)
    if o == "^" and len(args) == 2:
        base, expo = args
        if _is_lit_val(expo, 0.0):
            return 1.0
        if _is_lit_val(expo, 1.0):
            return base
        if _is_lit_val(base, 0.0) and _is_lit(expo) and float(expo) > 0.0:
            return 0.0
        if _is_lit_val(base, 1.0):
            return 1.0
        return node
    if o == "-" and len(args) == 2:
        if _is_lit_val(args[1], 0.0):
            return args[0]
        return node
    if o == "/" and len(args) == 2:
        if _is_lit_val(args[1], 1.0):
            return args[0]
        if isinstance(args[0], float) and args[0] == 0.0:
            return 0.0
        return node
    return node
