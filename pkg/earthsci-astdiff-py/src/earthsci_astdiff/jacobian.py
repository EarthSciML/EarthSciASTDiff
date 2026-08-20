"""Model-level driver (jacobian.jl): differentiate every ``D(u) = rhs``
equation w.r.t. every state/parameter that structurally occurs in its RHS —
or w.r.t. time itself (``wrt="time"``)."""
from __future__ import annotations

from dataclasses import dataclass

from earthsci_ast.expression import free_variables

from .bands import Band, bands, merge_bands, normalize_band
from .inline import inline_observed
from .system import SysView, _var_type, ctx_of, lhs_state, observed_defs, sysview


@dataclass
class JacEntry:
    u: str
    v: str
    band: Band


def jacobian_bands(obj, model_name: str | None = None,
                   wrt: str = "states") -> list[JacEntry]:
    if wrt not in ("states", "parameters", "time"):
        raise ValueError(f"wrt must be states, parameters or time, got {wrt}")
    sv = obj if isinstance(obj, SysView) else sysview(obj, model_name)
    ctx = ctx_of(sv)
    if wrt == "time":
        targets = ["t"]
    else:
        want = "state" if wrt == "states" else "parameter"
        targets = sorted(str(n) for n, var in sv.variables.items()
                         if _var_type(var) == want)
    obs = observed_defs(sv)
    entries: list[JacEntry] = []
    for eq in sv.equations:
        u = lhs_state(eq)
        if u is None:
            continue
        shape_u = ctx.shapes.get(u, [])
        rhs = inline_observed(eq.rhs, obs)
        fv = free_variables(rhs)
        for v in targets:
            if v not in fv:
                continue                     # structural occurrence gate
            for b in merge_bands(bands(rhs, v, ctx, shape_u=shape_u)):
                entries.append(JacEntry(u, v, normalize_band(b)))
    return entries
