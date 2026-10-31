"""Derivatives through a faq node's WINDOW and REDUCTION (reductions.jl,
esm-jacobian-spec.md §3.5): filtered faq nodes, the ``max`` / ``min`` / ``*``
reductions beside ``+``, and reads whose index is a value (a level search
used as a subscript).

A filter is a gate whose Dirac term is dropped, exactly like an ``ifelse``
condition (§3.2), so the derivative of a filtered reduction is a ``+`` over
the same window of each admitted term's derivative, weighted by what the
reduction does with that term:

  +        1
  max/min  the indicator of the FIRST extremal term in ascending order
           (the left fold's tie rule), built from two max/min faq nodes
  *        the product of every OTHER admitted term (never a division)

An index-only filter conjunct is decided exactly — by splitting the band's
box at its affine breakpoints, and by the band's ``where`` guard for what is
left (a triangular window); a value-dependent conjunct stays in the
coefficient as ``ifelse(φ, d, 0)`` and its pattern is the whole window.
"""

from __future__ import annotations

from earthsci_ast.esm_types import Expr, ExprNode
from earthsci_ast.expression import free_variables
from earthsci_ast.substitute import substitute

from .clip import _decide, clip_regions
from .expr_helpers import is_zero, mul, op
from .inline import all_names, subst_free


def _reduction(e: ExprNode) -> str:
    return f"reduce=`{e.reduce}`" if e.semiring is None else f"semiring=`{e.semiring}`"


def reduce_op(e: ExprNode, E) -> str:
    """The ⊕ of faq node ``e`` (``"+"``, ``"*"``, ``"max"`` or ``"min"``); a
    ``semiring`` supersedes ``reduce``, and its ⊗ is already written out in
    the body. A join or a value-invention node has no rule. ``E`` is the
    caller's error type."""
    if e.join is not None:
        raise E("joined faq nodes are not differentiated")
    if e.distinct is True or e.key is not None:
        raise E("value-invention faq nodes (`distinct`, `key`) are not differentiated")
    if e.semiring is None:
        r = "+" if e.reduce is None else e.reduce
        if r in ("+", "*", "max", "min"):
            return r
    elif e.semiring == "sum_product":
        return "+"
    elif e.semiring in ("max_product", "max_sum"):
        return "max"
    elif e.semiring == "min_sum":
        return "min"
    raise E(f"{_reduction(e)} has no derivative rule")


def conjuncts(g: Expr) -> list:
    if isinstance(g, ExprNode) and g.op == "and":
        return [c for x in g.args for c in conjuncts(x)]
    return [g]


def conj_list(xs: list):
    """The conjunction of a list: ``None`` when empty, the sole element, or
    an ``and`` node."""
    if not xs:
        return None
    if len(xs) == 1:
        return xs[0]
    return op("and", *xs)


def conj(a, b):
    """The conjunction of two optional predicates, flattened."""
    if a is None:
        return b
    if b is None:
        return a
    return conj_list(conjuncts(a) + conjuncts(b))


def index_only(g: Expr, names) -> bool:
    """Reads no name but the given index names (literals and ``const``
    gathers carry none), so it can be decided at assembly, point by point."""
    return set(free_variables(g)) <= set(names)


def fresh(taken: set, base: str, suffix: str) -> str:
    """``base``, then ``suffix`` appended until clear of ``taken``, which it
    joins."""
    f = base
    while f in taken:
        f += suffix
    taken.add(f)
    return f


def window_weight(red: str, e: ExprNode, cn: list, d: Expr, E) -> Expr:
    """The derivative ``d`` of one term of the reduction ``red`` of faq node
    ``e`` over its contracted names ``cn``, weighted by what the reduction
    does with that term (reductions.jl ``_window_weight``). The max/min
    indicator and the exclusion product read the node's whole window
    through copies of the body with the contracted names renamed apart
    (``<name>_p``, ``_p`` repeated until clear of every name in ``e``). A
    one-point window weighs 1; a max/min over more than one contracted name
    has no single ascending order and is refused."""
    if red == "+" or not cn or is_zero(d):
        return d
    rng = e.ranges or {}
    if red != "*" and len(cn) != 1:
        raise E(
            f"a `{red}` reduction over {len(cn)} contracted indices "
            f"({', '.join(cn)}) has no first-extremum order"
        )
    taken = all_names(e, set())
    ren = {n: fresh(taken, n + "_p", "_p") for n in cn}
    b2 = subst_free(e.expr, ren)
    phi2 = None if e.filter is None else subst_free(e.filter, ren)
    r2 = {ren[n]: rng[n] for n in cn}
    if red == "*":
        ne = [op("!=", ren[n], n) for n in cn]
        others = ExprNode(
            op="faq",
            args=[],
            output_idx=[],
            reduce="*",
            ranges=r2,
            filter=conj(phi2, ne[0] if len(ne) == 1 else op("or", *ne)),
            expr=b2,
        )
        return mul(d, others)
    n = cn[0]
    m = ExprNode(op="faq", args=[], output_idx=[], reduce=red, ranges=r2, filter=phi2, expr=b2)
    m_before = ExprNode(
        op="faq",
        args=[],
        output_idx=[],
        reduce=red,
        ranges=dict(r2),
        filter=conj(phi2, op("<", ren[n], n)),
        expr=b2,
    )
    ge, gt = (">=", ">") if red == "max" else ("<=", "<")
    first_ext = op("and", op(ge, e.expr, m), op(gt, e.expr, m_before))
    return op("ifelse", first_ext, d, 0)


def bool_fold(g: Expr):
    """Truth of a decided predicate: ``True`` / ``False``, or the residual
    expression."""
    if isinstance(g, bool):
        return g
    if isinstance(g, (int, float)):
        return g != 0
    if not isinstance(g, ExprNode):
        return g
    if g.op in ("and", "or"):
        absorb = g.op == "or"  # the value that decides it
        rest = []
        for x in g.args:
            t = bool_fold(x)
            if t is absorb:
                return absorb
            if t is (not absorb):
                continue
            rest.append(t)
        if not rest:
            return not absorb
        return rest[0] if len(rest) == 1 else op(g.op, *rest)
    if g.op == "not" and len(g.args) == 1:
        t = bool_fold(g.args[0])
        return (not t) if isinstance(t, bool) else op("not", t)
    if g.op == "true":
        return True
    if g.op == "false":
        return False
    return g


def split_guard(g: Expr, names: list, box: list) -> list:
    """Decide the index-only predicate ``g`` over the box ``names[k] ∈
    box[k]`` as far as the box can: pin one-value names, split the box at
    the predicate's affine breakpoints (the region clipper's), and decide
    each cell — dropped where ``g`` is false throughout, unguarded
    (``None``) where it is true throughout, and guarded by the undecided
    residual otherwise. Cells are in the clipper's ascending order."""
    pins = {n: lo for n, (lo, hi) in zip(names, box) if lo == hi}
    if pins:
        g = substitute(g, pins)
    out = []
    for cell, g2 in clip_regions(g, names, box):
        t = bool_fold(_decide(g2, {n: cell[k] for k, n in enumerate(names)}))
        if t is False:
            continue
        out.append((cell, None if t is True else t))
    return out


def one_hot(cidx: list, coef: Expr, contracted: list, ridx, v: str, ctx):
    """A column index that reads a value — ``x[κ(u) + d]`` with ``κ`` a level
    search — is integer-valued, hence piecewise constant: its own derivative
    is 0, and the read is the one-hot sum ``Σ_k [k == κ(u)+d] x[k]``. Each
    such dimension becomes a contracted name (``_oh<dim>``, ``_j`` appended
    until clear) over the static extent of that axis of ``v``, and the
    selection moves into the coefficient (reductions.jl ``_one_hot``)."""
    bound = {r for r in ridx if isinstance(r, str)}
    bound.update(n for n, _, _ in contracted)
    hot = [k for k, c in enumerate(cidx) if not index_only(c, bound)]
    if not hot:
        return cidx, coef, contracted
    shape = ctx.shapes.get(v, [])
    taken = all_names(coef, set(bound))
    for c in cidx:
        all_names(c, taken)
    cidx = list(cidx)
    contracted = list(contracted)
    for k in hot:
        f = fresh(taken, f"_oh{k + 1}", "_j")
        coef = op("ifelse", op("==", f, cidx[k]), coef, 0)
        cidx[k] = f
        contracted.append((f, 1, shape[k]))
    return cidx, coef, contracted
