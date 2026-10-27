"""Model-level driver (jacobian.jl): differentiate every ``D(u) = rhs``
equation w.r.t. every state/parameter that structurally occurs in its RHS —
or w.r.t. time itself (``wrt="time"``)."""
from __future__ import annotations

from dataclasses import dataclass

from earthsci_ast.expression import free_variables

from .bands import Band, BandError, bands, merge_bands, normalize_band
from .inline import inline_observed
from .system import (SysView, _var_type, ctx_of, inlinable_observed, is_derivative,
                     lhs_state, observed_defs, sysview, view_states)


@dataclass
class JacEntry:
    """One block of the Jacobian: row variable ``u``, column variable ``v``,
    and the :class:`Band` giving the rows it covers, the column-index
    expressions and the coefficient."""

    u: str
    v: str
    band: Band


def jacobian_bands(obj, model_name: str | None = None,
                   wrt: str = "states") -> list[JacEntry]:
    """Differentiate a model's right-hand side into a list of Jacobian bands.

    **Parameters**

    - ``obj`` — an ``EsmFile``, a flattened system or a :class:`SysView`.
    - ``model_name`` — which model of an ``EsmFile``; may be omitted when the
      file has exactly one.
    - ``wrt`` — ``"states"`` (default), ``"parameters"`` or ``"time"``.

    **Returns**

    One :class:`JacEntry` per (row variable, column variable) pair that occurs
    in the model. The derivative is symbolic, so the result does not depend on
    the point where it is later evaluated.
    """
    if wrt not in ("states", "parameters", "time"):
        raise ValueError(f"wrt must be states, parameters or time, got {wrt}")
    sv = obj if isinstance(obj, SysView) else sysview(obj, model_name)
    return _jacobian_bands(sv, ctx_of(sv), observed_defs(sv), wrt)


def _jacobian_bands(sv: SysView, ctx, obs: dict, wrt: str) -> list[JacEntry]:
    # The body of `jacobian_bands`, for callers (numeric assembly) that
    # already hold the view's context and observed definitions.
    if wrt == "time":
        targets = ["t"]
    elif wrt == "states":
        targets = view_states(sv, obs)
    else:
        targets = sorted(str(n) for n, var in sv.variables.items()
                         if _var_type(var) == "parameter")
    tset = set(targets)
    # Inline only the observed that can move with THIS wrt's targets; the
    # rest have an identically-zero derivative and stay opaque reads. Time is
    # not pruned: it also enters implicitly through time-varying data.
    if wrt != "time":
        obs = inlinable_observed(obs, sv.equations, tset)
    entries: list[JacEntry] = []
    obs_names: list = [None]
    for eq in sv.equations:
        u = lhs_state(eq)
        if u is None:
            if is_derivative(eq):
                raise BandError(
                    f"unsupported derivative LHS {eq.lhs!r} (a state's derivative "
                    "is `D(u)` or the lifted `faq{D(index(u, i…))}`)")
            continue
        shape_u = ctx.shapes.get(u, [])
        rhs = inline_observed(eq.rhs, obs, obs_names)
        # Structural occurrence gate, driven from the equation: intersect the
        # RHS's free variables with the target set (sorted = target order).
        for v in sorted(free_variables(rhs) & tset):
            for b in merge_bands(bands(rhs, v, ctx, shape_u=shape_u)):
                entries.append(JacEntry(u, v, normalize_band(b)))
    return entries
