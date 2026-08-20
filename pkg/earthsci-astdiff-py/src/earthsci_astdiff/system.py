"""Uniform system view over an ``EsmFile`` model or a ``FlattenedSystem``,
plus the expansion step guaranteeing we differentiate the same tree the
evaluators run (system.jl)."""
from __future__ import annotations

from dataclasses import dataclass

from earthsci_ast import expand_document
from earthsci_ast.esm_types import EsmFile, Expr, ExprNode

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
    lifted ``aggregate{ expr: D(index(u, i…)) }`` form."""
    l = eq.lhs
    if not isinstance(l, ExprNode):
        return None
    if l.op == "aggregate" and isinstance(l.expr, ExprNode) and l.expr.op == "D":
        a = l.expr.args[0]
        if (isinstance(a, ExprNode) and a.op == "index"
                and isinstance(a.args[0], str)):
            return a.args[0]
        return None
    if l.op == "D" and isinstance(l.args[0], str):
        return l.args[0]
    return None


def observed_defs(sv: SysView) -> dict[str, Expr]:
    out: dict[str, Expr] = {}
    for n, var in sv.variables.items():
        if (_var_type(var) == "observed"
                and getattr(var, "expression", None) is not None):
            out[str(n)] = var.expression
    return out
