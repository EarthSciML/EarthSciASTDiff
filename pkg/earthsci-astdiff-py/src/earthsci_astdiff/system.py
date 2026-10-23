"""Uniform system view over an ``EsmFile`` model or a ``FlattenedSystem``,
plus the expansion step guaranteeing we differentiate the same tree the
evaluators run (system.jl)."""
from __future__ import annotations

from dataclasses import dataclass

from earthsci_ast import expand_document
from earthsci_ast.esm_types import EsmFile, Expr, ExprNode
from earthsci_ast.expr_walk import map_children
from earthsci_ast.expression import free_variables

from .bands import Ctx


@dataclass
class SysView:
    variables: dict[str, object]
    equations: list
    index_sets: dict[str, object]


def _only_model(file: EsmFile) -> str:
    if len(file.models) != 1:
        raise ValueError(
            f"document has {len(file.models)} models; pass model_name")
    return next(iter(file.models))


def sysview(obj, model_name: str | None = None) -> SysView:
    if isinstance(obj, EsmFile):
        expanded = expand_document(obj)
        name = model_name if model_name is not None else _only_model(expanded)
        model = expanded.models[name]
        return SysView(dict(model.variables), list(model.equations),
                       dict(expanded.index_sets))
    # FlattenedSystem
    vars: dict[str, object] = {}
    for d in (obj.state_variables, obj.parameters, obj.observed_variables):
        vars.update(d)
    return SysView(vars, list(obj.equations), dict(obj.index_sets))


def _iset_size(s) -> int | None:
    if isinstance(s, dict):
        return s.get("size")
    return getattr(s, "size", None)


def _var_type(var) -> str:
    t = getattr(var, "type", None)
    return t.value if hasattr(t, "value") else str(t)


def ctx_of(sv: SysView) -> Ctx:
    isets: dict[str, int] = {}
    for n, s in sv.index_sets.items():
        sz = _iset_size(s)
        if sz is not None:
            isets[str(n)] = int(sz)
    shapes: dict[str, list[int]] = {}
    params: set[str] = set()
    for n, var in sv.variables.items():
        shape = getattr(var, "shape", None)
        if shape:
            shapes[str(n)] = [int(x) if isinstance(x, int) else isets[str(x)]
                              for x in shape]
        if _var_type(var) == "parameter":
            params.add(str(n))
    return Ctx(isets, shapes, params)


def lhs_state(eq) -> str | None:
    """The state an equation integrates: plain ``D(u)`` or the pointwise-
    lifted ``faq{ expr: D(index(u, i…)) }`` form.

    Only the esm 1.1.0 ``faq`` tag is matched: EarthSciAST's loader rewrites
    the deprecated ``aggregate`` alias at the wire boundary. Matching the old
    tag silently classifies every array state equation as non-differential
    (zero bands, no error) — the reference's faq_tag_test.jl."""
    l = eq.lhs
    if not isinstance(l, ExprNode):
        return None
    if l.op == "faq" and isinstance(l.expr, ExprNode) and l.expr.op == "D":
        a = l.expr.args[0]
        if (isinstance(a, ExprNode) and a.op == "index"
                and isinstance(a.args[0], str)):
            return a.args[0]
        return None
    if l.op == "D" and isinstance(l.args[0], str):
        return l.args[0]
    return None


def lhs_definition(eq) -> str | None:
    """The variable an equation DEFINES by a bare-variable LHS (``y ~ f(…)``)."""
    return eq.lhs if isinstance(eq.lhs, str) else None


def _indexed_definition(eq) -> tuple[str, Expr | None] | None:
    """An arrayed-observed definition ``faq{expr: index(y, i…)} ~ f(i…)``
    (``y[i…] ~ f(i…)`` as documents spell it) rewritten to the whole-array
    form ``y ~ faq{output_idx: [i…], expr: f(i…)}`` the inliner β-reduces.
    Mirrors the indexed half of ``earthsci_ast.observed_definitions``.
    ``None`` when the LHS is not an indexed read of a bare name; ``(y, None)``
    when it is one whose subscripts are not exactly the output indices (a
    partial or pinned definition such as ``y[1] ~ …``, not rewritable)."""
    l = eq.lhs
    if not isinstance(l, ExprNode):
        return None
    faq = None
    if l.op == "faq":
        if not (isinstance(l.expr, ExprNode) and l.expr.op == "index"):
            return None
        faq, l = l, l.expr
    if l.op != "index" or not l.args or not isinstance(l.args[0], str):
        return None
    name, sub = l.args[0], list(l.args[1:])
    if (faq is None or [str(x) for x in faq.output_idx] != sub
            or len(set(sub)) != len(sub)):
        return name, None
    return name, ExprNode(op="faq", args=[], output_idx=sub, expr=eq.rhs,
                          ranges=dict(faq.ranges) if faq.ranges else None)


_UNKNOWN_TYPES = ("unknown",
                  # a FlattenedVariable carries its DERIVED role instead
                  "state", "observed", "species")


def _is_unknown(var) -> bool:
    return _var_type(var) in _UNKNOWN_TYPES


def observed_defs(sv: SysView) -> dict[str, Expr]:
    """Observed definitions, for inlining. esm 1.0.0 removed the ``observed``
    type and ``variable.expression``: an observed is an unknown some equation
    names on the left — bare, or indexed (an arrayed observed, rewritten by
    :func:`_indexed_definition`) — and that no ``D(·)`` equation also targets
    (mirrors ``earthsci_ast.classification``). First definition wins; an
    arrayed observed defined by more than one equation is rejected."""
    states = {lhs_state(eq) for eq in sv.equations} - {None}
    out: dict[str, Expr] = {}
    indexed: set[str] = set()
    for eq in sv.equations:
        bare = lhs_definition(eq)
        nd = (bare, eq.rhs) if bare is not None else _indexed_definition(eq)
        if nd is None:
            continue
        n, d = nd
        var = sv.variables.get(n)
        if n in states or var is None or not _is_unknown(var):
            continue
        if d is None:
            raise ValueError(
                f"arrayed observed `{n}`: defining LHS subscripts must be the "
                "distinct output indices (partial definitions are not supported)")
        if n in out:
            if n in indexed or bare is None:
                raise ValueError(
                    f"arrayed observed `{n}` is defined by more than one "
                    "equation (piecewise definitions are not supported)")
            continue
        out[n] = d
        if bare is None:
            indexed.add(n)
    return out


def view_states(sv: SysView, obs: dict[str, Expr] | None = None) -> list[str]:
    """The unknowns occupying a solver slot: every unknown MINUS the observed
    ones (the 1.0.0 replacement for the 0.x ``state`` type). Sorted. Pass
    ``obs`` (:func:`observed_defs`) when the caller already has it."""
    if obs is None:
        obs = observed_defs(sv)
    return sorted(str(n) for n, var in sv.variables.items()
                  if _is_unknown(var) and str(n) not in obs)


def _opaque_deps(e: Expr) -> bool:
    # A surviving template reference hides its body's free names: treat it as
    # target-DEPENDENT so it is kept (and loudly rejected by the inliner).
    if not isinstance(e, ExprNode):
        return False
    if e.op == "apply_expression_template":
        return True
    found = False

    def visit(c):
        nonlocal found
        found = found or _opaque_deps(c)
        return c
    map_children(e, visit)
    return found


def _index_arg_names(acc: set[str], e: Expr) -> None:
    # Names under an index SUBSCRIPT — ``args[1:]`` of an ``index`` node.
    if not isinstance(e, ExprNode):
        return
    if e.op == "index":
        _index_arg_names(acc, e.args[0])
        for x in e.args[1:]:
            acc |= free_variables(x)
        return

    def visit(c):
        _index_arg_names(acc, c)
        return c
    map_children(e, visit)


def inlinable_observed(obs: dict[str, Expr], equations,
                       targets: set[str]) -> dict[str, Expr]:
    """The subset of the observed the band calculus must inline (the
    reference's ``inlinable_observed``, system.jl). Kept: (1) the reflexive-
    transitive READER closure of the observed naming a ``targets`` member
    (all others are constant w.r.t. every target); (2) any observed reached
    through an index subscript, whose value a column index needs at build
    time. Joint fixpoint. The seed MUST be this ``wrt``'s targets."""
    readers: dict[str, list[str]] = {}
    work: list[str] = []
    for n, e in obs.items():
        fv = free_variables(e)
        if not fv.isdisjoint(targets) or _opaque_deps(e):
            work.append(n)
        for m in fv:
            if m in obs:
                readers.setdefault(m, []).append(n)
    keep = set(work)
    while work:                                   # (1) reader closure
        for n in readers.get(work.pop(), ()):
            if n not in keep:
                keep.add(n)
                work.append(n)
    # (2) subscript fixpoint. The equations are scanned once; each pass then
    # scans only the bodies it just added, since nothing else can change.
    acc: set[str] = set()
    for eq in equations:
        _index_arg_names(acc, eq.rhs)
    added = set(keep)
    while True:
        for n in added:
            _index_arg_names(acc, obs[n])
        added = {n for n in acc if n in obs and n not in keep}
        if not added:
            break
        keep |= added
    return {n: obs[n] for n in keep}
