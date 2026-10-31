"""Evaluation-model CSE (eval_cse.jl): repeated coefficient subtrees become
observed-array variables of the derived evaluation document.

Template CSE at emission (cse.py) shrinks the SERIALIZED block; this pass
shrinks the derived EVALUATION model (:func:`~earthsci_astdiff.emit.
evaluation_document`). The same hash-consed extraction runs over the band
coefficients, but instead of ``expression_templates`` each repeated subtree
becomes an OBSERVED ARRAY variable of the derived document, defined by a
``makearray`` over the union of its use regions, and every occurrence
becomes an ``index`` read of it — so the evaluator computes each shared
subtree once per cell per right-hand-side call.

Free row-index names in a hoisted fragment become the observed array's axes
(sorted for determinism); a fragment with no free index names hoists to a
scalar observed. The pass is conservative: any inconsistency — a template
whose free index names are not bound by every band that (transitively) uses
it — abandons hoisting entirely and the caller falls back to the plain
(inline-coefficient) document. Byte-gated through the ``*.evaldoc.json``
goldens, so naming, axis order and region order follow the reference.
"""

from __future__ import annotations

from dataclasses import dataclass

from earthsci_ast.esm_types import Expr, ExprNode
from earthsci_ast.expr_walk import map_children
from earthsci_ast.expression import free_variables

from .cse import cse_templates
from .expr_helpers import op, ser_expr
from .jacobian import JacEntry
from .system import SysView, _iset_size


@dataclass
class Hoisted:
    """The hoisted observed of the derived document.

    - ``obs`` — variable name → serialized declaration, in hoisting order
    - ``obs_eqs`` — ``(name, serialized defining RHS)``, same order (a
      bare-LHS equation each: esm 1.0.0 keeps definitions in the equations)
    - ``axes`` — index-set additions (axis name → size)
    - ``coefs`` — the entries' coefficients, rewritten to read the hoisted
      observed
    """

    obs: dict[str, dict]
    obs_eqs: list[tuple[str, object]]
    axes: dict[str, int]
    coefs: list[Expr]


def _template_refs(e: Expr, acc: set[str] | None = None) -> set[str]:
    """Every ``apply_expression_template`` reference name in ``e``."""
    if acc is None:
        acc = set()
    if not isinstance(e, ExprNode):
        return acc
    if e.op == "apply_expression_template" and e.name is not None:
        acc.add(e.name)
    map_children(e, lambda c: (_template_refs(c, acc), c)[1])
    return acc


def _deref(e: Expr, readmap: dict[str, Expr]) -> Expr:
    """Replace template references by their reads (name → read expression)."""
    if not isinstance(e, ExprNode):
        return e
    if e.op == "apply_expression_template" and e.name is not None and e.name in readmap:
        return readmap[e.name]
    return map_children(e, lambda x: _deref(x, readmap))


def jcax_name(taken, sz: int) -> str:
    """The ``jcax<size>`` axis index-set name for extent ``sz``,
    ``x``-suffixed past any name ``taken(nm)`` (an existing set of another
    size). The one naming rule for hoisted-observed axes and contracted
    band-field axes (emit.py), so an extent gets the same name in both."""
    nm = f"jcax{sz}"
    while taken(nm):
        nm += "x"
    return nm


def hoist_observed(entries: list[JacEntry], sv: SysView, min_nodes: int = 12) -> Hoisted | None:
    """Extract repeated coefficient subtrees (≥ ``min_nodes`` AST nodes, ≥ 2
    occurrences; :func:`~earthsci_astdiff.cse.cse_templates`) and package
    each as an observed variable of the derived evaluation document.

    Returns ``None`` when there is nothing to hoist, or when a template's
    index names cannot be mapped consistently (the caller then emits the
    plain document); otherwise a :class:`Hoisted`."""
    coefs = [en.band.coef for en in entries]
    # Hoisted names must be fresh in the derived model's variable namespace.
    pfx = "Jcse"
    while any(str(k).startswith(pfx) for k in sv.variables):
        pfx += "x"
    templates, rewritten = cse_templates(coefs, min_nodes=min_nodes, prefix=pfx)
    if not templates:
        return None
    tnames = list(templates)

    # Per entry: index name → (lo, hi) box — row dims plus any free
    # contracted column dims (their names are just as free in the
    # coefficient, and the derived J_k field binds them the same way).
    boxof: list[dict[str, tuple[int, int]]] = []
    for en in entries:
        b = en.band
        d = {r: tuple(b.rows[k]) for k, r in enumerate(b.ridx) if isinstance(r, str)}
        for n, lo, hi in b.contracted:
            d[n] = (lo, hi)
        boxof.append(d)
    namesk = [set(d) for d in boxof]
    allnames = set().union(*namesk)

    # Which entries evaluate each template, transitively through the
    # reference DAG (a template read inside another template's body runs
    # wherever the outer one does).
    refs_body = {n: _template_refs(templates[n]) for n in tnames}
    uses: dict[str, set[int]] = {n: set() for n in tnames}
    for k, e in enumerate(rewritten):
        for n in _template_refs(e):
            uses[n].add(k)
    changed = True
    while changed:
        changed = False
        for n in tnames:
            for n2 in refs_body[n]:
                before = len(uses[n2])
                uses[n2] |= uses[n]
                if len(uses[n2]) != before:
                    changed = True
    live = [n for n in tnames if uses[n]]

    # Free row-index names per template (transitive; sorted = the axis order).
    S: dict[str, list[str]] = {}

    def s_of(n: str) -> list[str]:
        if n in S:
            return S[n]
        s = free_variables(templates[n]) & allnames
        for n2 in refs_body[n]:
            s |= set(s_of(n2))
        S[n] = sorted(s)
        return S[n]

    for n in live:
        s_of(n)
    # Consistency: every user band must bind every axis name. (Holds by
    # construction unless an index name collides with a variable name; then
    # hoisting is silently abandoned.)
    for n in live:
        for k in uses[n]:
            if not set(S[n]) <= namesk[k]:
                return None

    # Use boxes (deduplicated, ascending entry order) and per-axis extents.
    boxes: dict[str, list[list[tuple[int, int]]]] = {}
    for n in live:
        bs: list[list[tuple[int, int]]] = []
        for k in sorted(uses[n]):
            box = [boxof[k][x] for x in S[n]]
            if box not in bs:
                bs.append(box)
        boxes[n] = bs

    # Axis index sets: one per distinct extent, reusing an existing set of
    # exactly that size when the name is free.
    axes: dict[str, int] = {}
    existing: dict[str, int] = {}
    for nm, s in sv.index_sets.items():
        sz = _iset_size(s)
        if sz is not None:
            existing[str(nm)] = int(sz)

    def axis_for(sz: int) -> str:
        nm = jcax_name(lambda n: n in existing and existing[n] != sz, sz)
        if nm not in existing:
            axes[nm] = sz
            existing[nm] = sz
        return nm

    readmap: dict[str, Expr] = {n: (op("index", n, *S[n]) if S[n] else n) for n in live}

    obs: dict[str, dict] = {}
    obs_eqs: list[tuple[str, object]] = []
    for n in live:
        body = ser_expr(_deref(templates[n], readmap))
        if not S[n]:
            obs[n] = {"type": "unknown"}
            obs_eqs.append((n, body))
            continue
        exts = [max(b[j][1] for b in boxes[n]) for j in range(len(S[n]))]
        regions: list = [[[1, e] for e in exts]]
        values: list = [0.0]
        for b in boxes[n]:
            regions.append([[lo, hi] for lo, hi in b])
            values.append(
                {
                    "op": "faq",
                    "args": [],
                    "output_idx": list(S[n]),
                    "ranges": {x: [b[j][0], b[j][1]] for j, x in enumerate(S[n])},
                    "expr": body,
                }
            )
        obs[n] = {"type": "unknown", "shape": [axis_for(e) for e in exts]}
        obs_eqs.append((n, {"op": "makearray", "args": [], "regions": regions, "values": values}))
    return Hoisted(obs, obs_eqs, axes, [_deref(e, readmap) for e in rewritten])
