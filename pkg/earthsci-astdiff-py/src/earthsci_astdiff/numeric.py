"""Numeric assembly of the band entries — a correctness surface, not a
performance one.

The Julia reference compiles a derived evaluation model through its
tree-walk; here each band coefficient is evaluated per (row, contracted)
cell by resolving every read (state/parameter/`t`/const-table gathers and
scalar ``reduce: "+"`` faq nodes) to a literal and handing the closed tree
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

from .bands import _range_of
from .emit import _cellname, _eval_cidx, _product, check_column, column_names
from .inline import inline_observed
from .jacobian import _AXES, _jacobian_bands, jacobian_columns
from .scalar_rules import plain_sum, reduction
from .system import SysView, _var_type, ctx_of, observed_defs, sysview, view_states


class EvalError(Exception):
    pass


class _Env:
    """Point bindings: state arrays (nested lists, 1-based wire indices),
    scalar/array parameters, and the time value."""

    def __init__(self, states: dict, params: dict, t: float, ctx=None):
        self.states = states
        self.params = params
        self.t = t
        self.ctx = ctx  # resolves index-set range references ({"from": set})

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
        cells = [int(round(float(_as_lit(_resolve(x, env))))) for x in e.args[1:]]
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
    if e.op == "faq":
        if e.output_idx:
            raise EvalError("array-producing faq in scalar coefficient")
        if not plain_sum(e):
            raise EvalError(f"{reduction(e)} in scalar coefficient")
        names = sorted(str(n) for n in (e.ranges or {}))
        boxes = []
        for n in names:
            lo, hi = _range_of(e.ranges[n], env.ctx)
            boxes.append(range(lo, hi + 1))
        total = 0.0
        for cell in product(*boxes):
            body = substitute(e.expr, {n: cell[k] for k, n in enumerate(names)})
            total += float(_as_lit(_resolve(body, env)))
        return total
    return map_children(e, lambda x: _resolve(x, env))


def _as_lit(e: Expr) -> float:
    if isinstance(e, (int, float)) and not isinstance(e, bool):
        return float(e)
    return float(evaluate(e, {}))


def _default_values(sv: SysView, ctx, obs: dict):
    states: dict = {}
    params: dict = {}
    slots = set(view_states(sv, obs))
    for n, var in sv.variables.items():
        ty = "state" if str(n) in slots else _var_type(var)
        if ty not in ("state", "parameter"):
            continue
        default = getattr(var, "default", None)
        shape = ctx.shapes.get(str(n))
        if isinstance(default, (list, tuple)):
            # an inline array default: row-major nested lists, 1-based cells
            val = _nested_float(default)
        elif shape:
            val = _nested_full(shape, 0.0 if default is None else float(default))
        else:
            val = 0.0 if default is None else float(default)
        (states if ty == "state" else params)[str(n)] = val
    return states, params


def _nested_float(v):
    if isinstance(v, (list, tuple)):
        return [_nested_float(x) for x in v]
    return float(v)


def _nested_full(shape: list[int], v: float):
    if len(shape) == 1:
        return [v] * shape[0]
    return [_nested_full(shape[1:], v) for _ in range(shape[0])]


class JacobianEvaluator:
    """A prepared band evaluation for one model and axis, built once by
    :func:`prepare_jacobian`: the band calculus, the observed definitions,
    the column space and every band point's (row, column) placement. Each
    later evaluation (:meth:`values`, :func:`vjp`, :func:`vjp_forcing`) only
    resolves the coefficients at the point — the Julia reference's
    ``JacobianEvaluator`` in this binding's correctness-surface form.

    Fields of note: ``wrt``, ``entries``, ``columns`` (the
    :func:`jacobian_columns` space on the ``"parameters"`` / ``"forcing"``
    axes, else ``None``) and ``colnames`` (that space's cell names, ``["t"]``
    on ``"time"``, ``None`` on ``"states"``)."""

    def __init__(self, sv: SysView, wrt: str, forcing=None):
        if wrt not in _AXES:
            raise ValueError(f"wrt must be states, parameters, forcing or time, got {wrt}")
        ctx = ctx_of(sv)
        obs = observed_defs(sv)
        self.wrt = wrt
        self._ctx = ctx
        self._defaults = _default_values(sv, ctx, obs)
        self.entries = _jacobian_bands(sv, ctx, obs, wrt, forcing)
        if wrt in ("parameters", "forcing"):
            self.columns = jacobian_columns(sv, wrt=wrt, forcing=forcing)
            self.colnames = column_names(self.columns)
        else:
            self.columns = None
            self.colnames = ["t"] if wrt == "time" else None
        colset = set(self.colnames) if self.columns is not None else None
        # (row name, column name, coefficient with its index binding applied)
        # per band point. Pruning leaves target-independent observed as
        # opaque reads in the coefficients; resolve them from their own
        # definitions (what the reference's derived evaluation model does).
        # This splices in only the observed a coefficient actually reads.
        self._points: list[tuple[str, str, Expr]] = []
        for en in self.entries:
            b = en.band
            coef0 = inline_observed(b.coef, obs)
            vshape = ctx.shapes.get(en.v)
            for cell in _product(b.rows):
                for cc in _product([(lo, hi) for _, lo, hi in b.contracted]):
                    bind = {r: cell[k] for k, r in enumerate(b.ridx) if isinstance(r, str)}
                    for k, (nm, _, _) in enumerate(b.contracted):
                        bind[nm] = cc[k]
                    fbind = {k2: float(v2) for k2, v2 in bind.items()}
                    cidx = [_eval_cidx(c, fbind) for c in b.cidx]
                    row, col = _cellname(en.u, cell), _cellname(en.v, cidx)
                    if colset is not None:
                        if not check_column(col in colset, col, row, wrt):
                            continue  # parameter ghost read: no column
                    elif vshape is not None and any(
                        not (1 <= cidx[d] <= vshape[d]) for d in range(len(cidx))
                    ):
                        continue  # ghost/boundary read: no column
                    coef = substitute(coef0, dict(bind)) if bind else coef0
                    self._points.append((row, col, coef))

    def _env(self, states, params, t) -> _Env:
        dstates = dict(self._defaults[0])
        dparams = dict(self._defaults[1])
        if states:
            dstates.update(states)
        if params:
            dparams.update(params)
        return _Env(dstates, dparams, t, self._ctx)

    def _each_value(self, states, params, t):
        env = self._env(states, params, t)
        for row, col, coef in self._points:
            yield row, col, float(_as_lit(_resolve(coef, env)))

    def values(
        self, states: dict | None = None, params: dict | None = None, t: float = 0.0
    ) -> dict[tuple[str, str], float]:
        """The band entries at a point, as :func:`assemble_jacobian` returns
        them."""
        out: dict[tuple[str, str], float] = {}
        for key0, key1, val in self._each_value(states, params, t):
            key = (key0, key1)
            out[key] = out.get(key, 0.0) + val
        return out


def prepare_jacobian(
    obj, model_name: str | None = None, wrt: str = "states", forcing=None
) -> JacobianEvaluator:
    """Build a :class:`JacobianEvaluator` once — differentiation, column
    space and point placement — for repeated evaluation at different points
    (an adjoint loop calling :func:`vjp_forcing` once per stage)."""
    sv = obj if isinstance(obj, SysView) else sysview(obj, model_name)
    return JacobianEvaluator(sv, wrt, forcing)


def _prepared(obj, model_name, wrt, forcing, default_wrt: str) -> JacobianEvaluator:
    # An evaluator carries its own axis and forcing set; anything else is
    # prepared here, for one evaluation.
    if isinstance(obj, JacobianEvaluator):
        if wrt is not None and wrt != obj.wrt:
            raise ValueError(f"the evaluator is on wrt = {obj.wrt}, not {wrt}")
        return obj
    return prepare_jacobian(obj, model_name, default_wrt if wrt is None else wrt, forcing)


def assemble_jacobian(
    obj,
    model_name: str | None = None,
    wrt: str | None = None,
    states: dict | None = None,
    params: dict | None = None,
    t: float = 0.0,
    forcing=None,
) -> dict[tuple[str, str], float]:
    """Evaluate the band entries at a point: ``{(rowname, colname): value}``
    with contributions to one (row, col) pair SUMMED (§4 additivity, incl.
    contracted-point accumulation). ``states``/``params`` map variable name
    → nested list (1-based cells, row-major) or scalar; defaults come from
    the document (array parameters and forcing buffers included, from their
    inline array ``default``).

    On the ``"parameters"`` and ``"forcing"`` axes the column names are the
    cells of the canonical column space (:func:`jacobian_columns`). A
    forcing column outside its buffer's extent raises :class:`BandError` (no
    ghost skip); a parameter or state column read past its extent is a ghost
    read and is skipped. ``wrt`` defaults to ``"states"``; ``obj`` may also
    be a :class:`JacobianEvaluator`, whose own axis applies."""
    return _prepared(obj, model_name, wrt, forcing, "states").values(states, params, t)


def vjp(
    g,
    obj,
    lam: dict,
    model_name: str | None = None,
    wrt: str | None = None,
    states: dict | None = None,
    params: dict | None = None,
    t: float = 0.0,
    forcing=None,
):
    """The transposed product ``g += Jᵀ λ`` at a point, as a scatter-add over
    the band points: ``g[col] += coef · λ[row]``, with no matrix formed.

    ``obj`` is a :class:`JacobianEvaluator` (built once with
    :func:`prepare_jacobian`; its axis and forcing set apply) or anything
    :func:`assemble_jacobian` takes, prepared for this one call with ``wrt``
    (default ``"forcing"``). ``lam`` maps row (state cell) names to values; a
    row a band point needs that ``lam`` does not bind raises ``ValueError``.
    ``g`` is a flat mutable sequence (list or numpy vector) over the
    canonical column ordinal of the ``"parameters"`` / ``"forcing"`` axis
    (:func:`jacobian_columns`, cells column-major) or of ``"time"`` (``[t]``).
    ``g`` ACCUMULATES — zero it first for a single product. Returns ``g``."""
    jac = _prepared(obj, model_name, wrt, forcing, "forcing")
    if jac.wrt not in ("parameters", "forcing", "time"):
        raise ValueError(f"vjp needs wrt = parameters, forcing or time, got {jac.wrt}")
    if len(g) != len(jac.colnames):
        raise ValueError(f"g has length {len(g)}, the column space {len(jac.colnames)}")
    slot = {n: i for i, n in enumerate(jac.colnames)}
    for r, c, v in jac._each_value(states, params, t):
        lr = lam.get(r)
        if lr is None:
            raise ValueError(f"λ binds no row `{r}`")
        g[slot[c]] += v * lr
    return g


def vjp_forcing(
    g,
    obj,
    lam: dict,
    model_name: str | None = None,
    states: dict | None = None,
    params: dict | None = None,
    t: float = 0.0,
    forcing=None,
):
    """:func:`vjp` on the forcing axis: the forcing cotangent of one stage
    evaluation, ``g_B += (∂f/∂B)ᵀ λ`` (spec §4.3). ``obj`` is a
    ``wrt="forcing"`` :class:`JacobianEvaluator` — build it once with
    :func:`prepare_jacobian` and reuse it across stages — or anything
    :func:`assemble_jacobian` takes. ``g`` is either a flat sequence over the
    canonical column ordinal or a dict of buffer-shaped numpy accumulators
    keyed by buffer name (one per forcing buffer, each of the buffer's shape,
    indexed by cell). It accumulates; which dataset record a buffer slot
    holds is the caller's bookkeeping."""
    jac = _prepared(obj, model_name, "forcing", forcing, "forcing")
    if not isinstance(g, dict):
        return vjp(g, jac, lam, states=states, params=params, t=t)
    import numpy as np

    for name, shape in jac.columns:
        acc = g.get(name)
        if acc is None:
            raise ValueError(f"no accumulator for forcing buffer `{name}`")
        if tuple(np.shape(acc)) != tuple(shape):
            raise ValueError(
                f"accumulator `{name}` has shape {tuple(np.shape(acc))}, the buffer {tuple(shape)}"
            )
    flat = vjp([0.0] * len(jac.colnames), jac, lam, states=states, params=params, t=t)
    k = 0
    for name, shape in jac.columns:
        acc = g[name]
        for cell in _product([(1, n) for n in shape]):
            idx = tuple(c - 1 for c in cell)
            acc[idx] += flat[k]
            k += 1
    return g
