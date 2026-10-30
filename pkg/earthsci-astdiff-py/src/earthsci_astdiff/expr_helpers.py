"""Expression constructors, structural keys, and the differentiation site.

Mirrors the Julia reference's ``expr_helpers.jl``. Python atoms are the
wire scalars themselves (``int`` / ``float`` literals, ``str`` variable
references); only operator nodes are ``ExprNode``.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from decimal import Decimal
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


def canonical_float(f: float) -> str:
    """A float in the Julia reference's layout (EarthSciAST CONFORMANCE_SPEC
    §5.5.3.1 rule 2): shortest round-trip digits, positional iff the decimal
    exponent of the first significant digit is in ``-4..5``, otherwise
    ``d.f…e<exp>`` with at least one fraction digit (``1.0e-5``,
    ``2.4405875e6``). ``repr`` has the same digits but switches layout at
    other magnitudes (``7.27e-05``, ``2440587.5``)."""
    if f != f or f in (float("inf"), float("-inf")):
        raise ValueError(f"non-finite float {f!r} has no JSON form")
    if f == 0.0:
        return "-0.0" if math.copysign(1.0, f) < 0 else "0.0"
    sign, digits, exp = Decimal(repr(f)).as_tuple()
    ds = "".join(map(str, digits)).rstrip("0")
    exp += len(digits) - len(ds)
    e10 = len(ds) - 1 + exp
    if -4 <= e10 <= 5:
        if e10 < 0:
            body = "0." + "0" * (-e10 - 1) + ds
        else:
            body = ds[: e10 + 1].ljust(e10 + 1, "0") + "." + (ds[e10 + 1 :] or "0")
    else:
        body = ds[0] + "." + (ds[1:] or "0") + "e" + str(e10)
    return ("-" if sign else "") + body


def _key_str(k: Any) -> str:
    """An object key as JSON's string form, the way ``json.dumps`` coerces
    one (``1`` → ``"1"``, ``True`` → ``"true"``), so non-string keys still
    give valid JSON and sort as the strings they are written as."""
    if isinstance(k, str):
        return k
    if k is None or isinstance(k, (bool, int)):
        return json.dumps(k)
    if isinstance(k, float):
        return canonical_float(k)
    raise TypeError(f"JSON object key {k!r} is not a string or a number")


def _write_stable(x: Any, out: list) -> None:
    if isinstance(x, dict):
        items = sorted(((_key_str(k), v) for k, v in x.items()), key=lambda kv: kv[0])
        out.append("{")
        for i, (k, v) in enumerate(items):
            if i:
                out.append(",")
            out.append(json.dumps(k, ensure_ascii=False))
            out.append(":")
            _write_stable(v, out)
        out.append("}")
    elif isinstance(x, (list, tuple)):
        out.append("[")
        for i, v in enumerate(x):
            if i:
                out.append(",")
            _write_stable(v, out)
        out.append("]")
    elif isinstance(x, float):
        out.append(canonical_float(x))
    else:
        out.append(json.dumps(x, ensure_ascii=False))


def stable_json(x: Any) -> str:
    """Deterministic JSON: sorted object keys at every level, floats in the
    reference's layout (``canonical_float``) — emit.jl's ``stable_json``."""
    out: list = []
    _write_stable(x, out)
    return "".join(out)


def ser_expr(e: Expr) -> Any:
    return _serialize_expression(e)


def key_json(x: Any) -> str:
    """Deterministic JSON for equality keys only (``skey``, CSE interning):
    the C ``json.dumps`` with sorted keys. Never written out, so it need not
    match the reference's float layout, and a non-finite constant (``inf``
    from folding ``1.0/0.0``) still gets a key."""
    return json.dumps(x, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def skey(e: Expr) -> str:
    """Full-structure key for ANY expression (array ops included)."""
    return key_json(ser_expr(e))


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
