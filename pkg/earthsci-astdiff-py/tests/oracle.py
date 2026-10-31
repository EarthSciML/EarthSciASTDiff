"""A finite-difference Jacobian oracle independent of the band calculus: the
model's state equations, observed inlined, evaluated by EarthSciAST's own
numpy tree-walk (``eval_expr``, which runs filtered and non-``+`` faq
nodes) and differenced cell by cell."""

from __future__ import annotations

import numpy as np
import pytest
from earthsci_ast.numpy_interpreter import EvalContext, eval_expr
from earthsci_astdiff import assemble_jacobian, inline_observed
from earthsci_astdiff.system import ctx_of, lhs_state, observed_defs, sysview, view_states


def _cells(name: str, shape) -> list[str]:
    if not shape:
        return [name]
    return [f"{name}[{','.join(str(c + 1) for c in cell)}]" for cell in np.ndindex(*shape)]


def rhs_function(file, model=None):
    """``(f, names, x0)``: ``f(x)`` the flat state derivative at the flat
    state ``x`` (states sorted, each one's cells row-major), ``names`` the
    cell names, ``x0`` the document defaults."""
    sv = sysview(file, model)
    ctx = ctx_of(sv)
    obs = observed_defs(sv)
    states = view_states(sv, obs)
    shapes = {n: tuple(ctx.shapes.get(n, [])) for n in states}
    layout, k = {}, 0
    for n in states:
        size = int(np.prod(shapes[n])) if shapes[n] else 1
        layout[n] = slice(k, k + size)
        k += size
    rhs = {}
    for eq in sv.equations:
        u = lhs_state(eq)
        if u is not None:
            rhs[u] = inline_observed(eq.rhs, obs)
    params = {}
    for n, var in sv.variables.items():
        if str(n) not in states and str(n) not in obs and getattr(var, "default", None) is not None:
            d = var.default
            if not isinstance(d, (list, tuple)):
                params[str(n)] = float(d)
    x0 = np.zeros(k)
    for n in states:
        d = getattr(sv.variables[n], "default", None)
        x0[layout[n]] = 0.0 if d is None else float(d)

    def f(x):
        c = EvalContext(
            state_layout=layout,
            state_shapes=shapes,
            param_values=params,
            observed_values={},
            y=np.asarray(x, dtype=float),
            t=0.0,
        )
        out = np.zeros(k)
        for n in states:
            if n in rhs:
                v = np.broadcast_to(np.asarray(eval_expr(rhs[n], c), dtype=float), shapes[n])
                out[layout[n]] = v.reshape(-1)
        return out

    names = [c for n in states for c in _cells(n, shapes[n])]
    return f, names, x0


def fd_jacobian(file, x, model=None, h=1e-6) -> dict:
    f, names, _ = rhs_function(file, model)
    x = np.asarray(x, dtype=float)
    out = {}
    for j in range(len(x)):
        xp, xm = x.copy(), x.copy()
        xp[j] += h
        xm[j] -= h
        col = (f(xp) - f(xm)) / (2 * h)
        for i, v in enumerate(col):
            out[(names[i], names[j])] = float(v)
    return out


def states_of(file, x, model=None) -> dict:
    """The flat state ``x`` as :func:`assemble_jacobian`'s ``states``."""
    sv = sysview(file, model)
    ctx = ctx_of(sv)
    states = view_states(sv)
    out, k = {}, 0
    for n in states:
        shape = ctx.shapes.get(n, [])
        size = int(np.prod(shape)) if shape else 1
        a = np.asarray(x[k : k + size], dtype=float)
        out[n] = a.reshape(shape).tolist() if shape else float(a[0])
        k += size
    return out


def check_against_fd(file, x, model=None, npat=None, tol=1e-7):
    """The assembled Jacobian at ``x`` against central differences: every
    value within ``tol``, and the pattern a superset of every nonzero."""
    got = assemble_jacobian(file, model, states=states_of(file, x, model))
    want = fd_jacobian(file, x, model)
    for key, v in want.items():
        if abs(v) > 1e-8:
            assert key in got, f"pattern misses {key} = {v}"
        assert got.get(key, 0.0) == pytest.approx(v, abs=tol), key
    assert set(got) <= set(want)
    if npat is not None:
        assert len(got) == npat
    return got
