"""Model-level driver (jacobian.jl): differentiate every ``D(u) = rhs``
equation w.r.t. every state/parameter/forcing-buffer that structurally occurs
in its RHS — or w.r.t. time itself (``wrt="time"``)."""

from __future__ import annotations

import json
from dataclasses import dataclass

from earthsci_ast.expression import free_variables

from .bands import Band, BandError, bands, merge_bands, normalize_band
from .expr_helpers import ser_expr
from .inline import inline_observed
from .system import (
    SysView,
    _var_type,
    ctx_of,
    inlinable_observed,
    is_derivative,
    lhs_state,
    observed_defs,
    sysview,
    view_states,
)


@dataclass
class JacEntry:
    """One block of the Jacobian: row variable ``u``, column variable ``v``,
    and the :class:`Band` giving the rows it covers, the column-index
    expressions and the coefficient."""

    u: str
    v: str
    band: Band


_AXES = ("states", "parameters", "forcing", "time")


def _updates(var) -> list:
    u = getattr(var, "update", None)
    if u is None:
        return []
    return list(u) if isinstance(u, (list, tuple)) else [u]


def _is_parameter(var) -> bool:
    return _var_type(var) == "parameter"


def forcing_names(sv: SysView, forcing=None) -> list[str]:
    """The forcing set of a system, sorted by name: the parameters whose cells
    are the columns of the ``wrt="forcing"`` axis (esm-jacobian-spec.md §4.3).

    With ``forcing=None`` the set is inferred — every ARRAY-valued parameter
    with an ``update`` of kind ``"data"`` (a live buffer a data provider
    refills). An explicit list must name only parameters of the system: a
    name that resolves to no buffer raises ``ValueError``, never yields an
    empty axis.
    """
    if forcing is None:
        return sorted(
            str(n)
            for n, var in sv.variables.items()
            if _is_parameter(var)
            and getattr(var, "shape", None)
            and any(getattr(u, "kind", None) == "data" for u in _updates(var))
        )
    names = [str(n) for n in forcing]
    for n in names:
        var = sv.variables.get(n)
        if var is None or not _is_parameter(var):
            raise ValueError(
                f"forcing name `{n}` resolves to no buffer: it is not a parameter of the system"
            )
    return sorted(set(names))


def _targets(sv: SysView, wrt: str, forcing, obs) -> list[str]:
    # The column variables of one axis, in canonical (name-sorted) order.
    if wrt == "time":
        return ["t"]
    if wrt == "states":
        return view_states(sv, obs)
    fset = forcing_names(sv, forcing)
    if wrt == "forcing":
        return fset
    fs = set(fset)
    return sorted(
        str(n) for n, var in sv.variables.items() if _is_parameter(var) and str(n) not in fs
    )


def jacobian_columns(obj, model_name: str | None = None, wrt: str = "parameters", forcing=None):
    """The column space of a ``"parameters"`` or ``"forcing"`` block: each
    column variable with its integer shape (``[]`` for a scalar), sorted by
    name — the ``columns`` field of the emitted block (spec §4). The canonical
    column ordinal enumerates these variables in order, each one's cells
    column-major (first index fastest)."""
    if wrt not in ("parameters", "forcing"):
        raise ValueError(f"columns are defined for wrt = parameters or forcing, got {wrt}")
    sv = obj if isinstance(obj, SysView) else sysview(obj, model_name)
    ctx = ctx_of(sv)
    return [(n, list(ctx.shapes.get(n, []))) for n in _targets(sv, wrt, forcing, None)]


def jacobian_bands(
    obj, model_name: str | None = None, wrt: str = "states", forcing=None
) -> list[JacEntry]:
    """Differentiate a model's right-hand side into a list of Jacobian bands.

    **Parameters**

    - ``obj`` — an ``EsmFile``, a flattened system or a :class:`SysView`.
    - ``model_name`` — which model of an ``EsmFile``; may be omitted when the
      file has exactly one.
    - ``wrt`` — ``"states"`` (default), ``"parameters"`` (every parameter
      outside the forcing set, array-valued ones cell by cell), ``"forcing"``
      (the cells of the forcing buffers, record axis included) or ``"time"``.
    - ``forcing`` — the forcing set; ``None`` infers it (every array
      parameter with a ``data`` update, see :func:`forcing_names`).

    **Returns**

    One :class:`JacEntry` per (row variable, column variable) pair that occurs
    in the model. The derivative is symbolic, so the result does not depend on
    the point where it is later evaluated.

    On the ``"parameters"`` and ``"forcing"`` axes every column index must be
    closed at build time, and a ``"forcing"`` request whose right-hand sides
    read the forcing set but yield no band raises :class:`BandError`
    (spec §4.3).
    """
    if wrt not in _AXES:
        raise ValueError(f"wrt must be states, parameters, forcing or time, got {wrt}")
    sv = obj if isinstance(obj, SysView) else sysview(obj, model_name)
    return _jacobian_bands(sv, ctx_of(sv), observed_defs(sv), wrt, forcing)


def _jacobian_bands(sv: SysView, ctx, obs: dict, wrt: str, forcing=None) -> list[JacEntry]:
    # The body of `jacobian_bands`, for callers (numeric assembly) that
    # already hold the view's context and observed definitions.
    targets = _targets(sv, wrt, forcing, obs)
    tset = set(targets)
    # Inline only the observed that can move with THIS wrt's targets; the
    # rest have an identically-zero derivative and stay opaque reads. Time is
    # not pruned: it also enters implicitly through time-varying data.
    if wrt != "time":
        obs = inlinable_observed(obs, sv.equations, tset)
    entries: list[JacEntry] = []
    read_target = False
    obs_names: list = [None]
    for eq in sv.equations:
        u = lhs_state(eq)
        if u is None:
            if is_derivative(eq):
                raise BandError(
                    f"unsupported derivative LHS {eq.lhs!r} (a state's derivative "
                    "is `D(u)` or the lifted `faq{D(index(u, i…))}`)"
                )
            continue
        shape_u = ctx.shapes.get(u, [])
        rhs = inline_observed(eq.rhs, obs, obs_names)
        # Structural occurrence gate, driven from the equation: intersect the
        # RHS's free variables with the target set (sorted = target order).
        hits = sorted(free_variables(rhs) & tset)
        read_target = read_target or bool(hits)
        for v in hits:
            for b in merge_bands(bands(rhs, v, ctx, shape_u=shape_u)):
                entries.append(JacEntry(u, v, normalize_band(b)))
    if wrt in ("parameters", "forcing"):
        for en in entries:
            _check_closed_columns(en)
        if wrt == "forcing" and read_target and not entries:
            raise BandError(
                f"the right-hand sides read the forcing set {targets} but no band was "
                "produced; an empty forcing block for a document that reads forcing is "
                "refused rather than emitted"
            )
    return entries


def _check_closed_columns(en: JacEntry) -> None:
    # A column of the parameters / forcing axes is a cell of a buffer, so it
    # must be placeable at build time: its index expressions may read only
    # the entry's own row and contracted index names.
    b = en.band
    bound = {r for r in b.ridx if isinstance(r, str)}
    bound.update(n for n, _, _ in b.contracted)
    for c in b.cidx:
        for n in sorted(free_variables(c)):
            if n not in bound:
                raise BandError(
                    f"column index {json.dumps(ser_expr(c), separators=(',', ':'))} of "
                    f"`{en.v}` in the equation for `{en.u}` reads `{n}`, which is not "
                    "closed at build time (a parameter or forcing column may read only "
                    "literals, folded metaparameters, const gathers and its row or "
                    "contracted index names)"
                )
