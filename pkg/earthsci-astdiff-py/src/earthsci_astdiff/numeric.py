"""Numeric assembly of the band entries — a correctness surface, not a
performance one.

The Julia reference compiles a derived evaluation model through its
tree-walk; here each band coefficient is evaluated per (row, contracted)
cell by resolving every read (state/parameter/`t`/const-table gathers and
scalar ``reduce: "+"`` aggregates) to a literal and handing the closed tree
to the official ``numpy_interpreter`` evaluator — so operator semantics
(ifelse, min/max, ``fn`` lookups) stay the runtime's own. Complexity is
O(cells × coefficient size) per call; use the Julia binding for production
assembly.
"""
from __future__ import annotations

from itertools import product

from earthsci_ast.esm_types import Expr, ExprNode
from earthsci_ast.expr_walk import map_children
from earthsci_ast.numpy_interpreter import evaluate
from earthsci_ast.substitute import substitute

from .emit import _cellname, _eval_cidx, _product
from .jacobian import jacobian_bands
from .system import SysView, _var_type, ctx_of, sysview


class EvalError(Exception):
    pass


class _Env:
    """Point bindings: state arrays (nested lists, 1-based wire indices),
    scalar/array parameters, and the time value."""

    def __init__(self, states: dict, params: dict, t: float):
        self.states = states
        self.params = params
        self.t = t

    def scalar(self, name: str):
        if name == "t":
            return self.t
        if name in self.params and not isinstance(self.params[name], list):
            return self.params[name]
        if name in self.states and not isinstance(self.states[name], list):
            return self.states[name]
        return None

    def gather(self, name: str, cells: list[int]):
        arr = self.states.get(name, self.params.get(name))
        if arr is None:
            return None
        for c in cells:
            arr = arr[c - 1]
        return arr


def _resolve(e: Expr, env: _Env) -> Expr:
    """Fold every read in ``e`` to a literal (recursively), leaving a closed
    tree the numpy evaluator can run with no bindings."""
    if isinstance(e, str):
        v = env.scalar(e)
        if v is None:
            raise EvalError(f"unbound name `{e}` in coefficient")
        return v
    if not isinstance(e, ExprNode):
        return e
    if e.op == "index":
        t = e.args[0]
        cells = [int(round(float(_as_lit(_resolve(x, env)))))
                 for x in e.args[1:]]
        if isinstance(t, str):
            v = env.gather(t, cells)
            if v is None:
                raise EvalError(f"index into unbound array `{t}`")
            return v
        if isinstance(t, ExprNode) and t.op == "const":
            v = t.value
            for c in cells:
                v = v[c - 1]
            return v
        raise EvalError(f"index into unsupported target `{t}`")
    if e.op == "aggregate":
        if e.output_idx:
            raise EvalError("array-producing aggregate in scalar coefficient")
        if e.reduce not in (None, "+"):
            raise EvalError(f"reduce=`{e.reduce}` in scalar coefficient")
        names = sorted(str(n) for n in (e.ranges or {}))
        boxes = [range(int(e.ranges[n][0]), int(e.ranges[n][1]) + 1)
                 for n in names]
        total = 0.0
        for cell in product(*boxes):
            body = substitute(e.expr, {n: cell[k]
                                       for k, n in enumerate(names)})
            total += float(_as_lit(_resolve(body, env)))
        return total
    return map_children(e, lambda x: _resolve(x, env))


def _as_lit(e: Expr) -> float:
    if isinstance(e, (int, float)) and not isinstance(e, bool):
        return float(e)
    return float(evaluate(e, {}))


def _default_values(sv: SysView, ctx):
    states: dict = {}
    params: dict = {}
    for n, var in sv.variables.items():
        ty = _var_type(var)
        if ty not in ("state", "parameter"):
            continue
        default = getattr(var, "default", None)
        shape = ctx.shapes.get(str(n))
        if shape:
            val = _nested_full(shape, 0.0 if default is None else float(default))
        else:
            val = 0.0 if default is None else float(default)
        (states if ty == "state" else params)[str(n)] = val
    return states, params


def _nested_full(shape: list[int], v: float):
    if len(shape) == 1:
        return [v] * shape[0]
    return [_nested_full(shape[1:], v) for _ in range(shape[0])]


def assemble_jacobian(obj, model_name: str | None = None, wrt: str = "states",
                      states: dict | None = None, params: dict | None = None,
                      t: float = 0.0) -> dict[tuple[str, str], float]:
    """Evaluate the band entries at a point: ``{(rowname, colname): value}``
    with contributions to one (row, col) pair SUMMED (§4 additivity, incl.
    contracted-point accumulation). ``states``/``params`` map variable name
    → nested list (1-based cells) or scalar; defaults come from the document."""
    sv = obj if isinstance(obj, SysView) else sysview(obj, model_name)
    ctx = ctx_of(sv)
    dstates, dparams = _default_values(sv, ctx)
    if states:
        dstates.update(states)
    if params:
        dparams.update(params)
    env = _Env(dstates, dparams, t)
    entries = jacobian_bands(sv, wrt=wrt)
    out: dict[tuple[str, str], float] = {}
    for en in entries:
        b = en.band
        for cell in _product(b.rows):
            for cc in _product([(lo, hi) for _, lo, hi in b.contracted]):
                bind = {r: cell[k] for k, r in enumerate(b.ridx)
                        if isinstance(r, str)}
                for k, (nm, _, _) in enumerate(b.contracted):
                    bind[nm] = cc[k]
                fbind = {k2: float(v2) for k2, v2 in bind.items()}
                cidx = [_eval_cidx(c, fbind) for c in b.cidx]
                vshape = ctx.shapes.get(en.v)
                if vshape is not None and any(
                        not (1 <= cidx[d] <= vshape[d])
                        for d in range(len(cidx))):
                    continue                # ghost/boundary read: no column
                coef = substitute(b.coef, dict(bind)) if bind else b.coef
                val = float(_as_lit(_resolve(coef, env)))
                key = (_cellname(en.u, cell), _cellname(en.v, cidx))
                out[key] = out.get(key, 0.0) + val
    return out
