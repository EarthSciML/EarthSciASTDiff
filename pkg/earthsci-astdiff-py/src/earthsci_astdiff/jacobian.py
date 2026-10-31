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
    empty axis. ``forcing`` may be one name (a ``str``) or a collection.
    """
    if forcing is None:
        return sorted(
            str(n)
            for n, var in sv.variables.items()
            if _is_parameter(var)
            and getattr(var, "shape", None)
            and any(getattr(u, "kind", None) == "data" for u in _updates(var))
        )
    # One buffer name is the natural call; iterating a str would split it
    # into characters.
    names = [forcing] if isinstance(forcing, str) else [str(n) for n in forcing]
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
    closed at build time; an unclosed one raises :class:`BandError` (spec
    §4.3). A model that reads the forcing set only where the derivative is
    identically zero (an ``ifelse`` condition, a piecewise-constant op) has an
    empty forcing list.
    """
    if wrt not in _AXES:
        raise ValueError(f"wrt must be states, parameters, forcing or time, got {wrt}")
    sv = obj if isinstance(obj, SysView) else sysview(obj, model_name)
    return _jacobian_bands(sv, ctx_of(sv), observed_defs(sv), wrt, forcing)


def _jacobian_bands(sv: SysView, ctx, obs: dict, wrt: str, forcing=None, seed=()) -> list[JacEntry]:
    # The body of `jacobian_bands`, for callers that
    # already hold the view's context and observed definitions. `seed` adds
    # names to the inlining seed without making them targets: the
    # second-order driver (second_order.py) differentiates these
    # coefficients again w.r.t. its second axis, so an observed that moves
    # only with that axis must be inlined too, or its sites stay hidden.
    targets = _targets(sv, wrt, forcing, obs)
    tset = set(targets)
    # Inline only the observed that can move with THIS wrt's targets; the
    # rest have an identically-zero derivative and stay opaque reads. Time is
    # not pruned: it also enters implicitly through time-varying data.
    if wrt != "time":
        obs = inlinable_observed(obs, sv.equations, tset | set(seed))
    entries: list[JacEntry] = []
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
        for v in hits:
            for b in merge_bands(bands(rhs, v, ctx, shape_u=shape_u)):
                entries.append(JacEntry(u, v, normalize_band(b)))
    if wrt in ("parameters", "forcing"):
        for en in entries:
            _check_closed_columns(en)
    return entries


def _check_closed_columns(en: JacEntry) -> None:
    # A column of the parameters / forcing axes is a cell of a buffer, so it
    # must be placeable at build time: its index expressions may read only
    # the entry's own row and contracted index names. The second column of
    # a pair (second_order.py) is held to the same rule.
    _check_closed(
        en.band,
        en.band.cidx,
        lambda c: f"column index {_json(c)} of `{en.v}` in the equation for `{en.u}`",
    )


def _check_closed(b: Band, cols, what) -> None:
    bound = {r for r in b.ridx if isinstance(r, str)}
    bound.update(n for n, _, _ in b.contracted)
    for c in cols:
        for n in sorted(free_variables(c)):
            if n not in bound:
                raise BandError(
                    f"{what(c)} reads `{n}`, which is not closed at build time (a "
                    "parameter or forcing column may read only literals, folded "
                    "metaparameters, const gathers and its row or contracted index names)"
                )


def _json(c) -> str:
    return json.dumps(ser_expr(c), separators=(",", ":"))


class _FirstOrder:
    # The first-order passes of one emission (`jacobian_block`), and the
    # structure class of each. A pass depends only on its axis and the
    # inlining seed a second axis adds (`_bands2`); that seed changes
    # nothing on the time axis (never seeded) or when the second axis is the
    # first, so those share the plain pass's key: a block list asking for
    # "states" and ("states", "states") differentiates the system once.

    def __init__(self):
        self._bands: dict = {}
        self._structure: dict = {}

    def bands(self, sv: SysView, a1: str, a2, forcing, seed=()) -> list[JacEntry]:
        key = (a1, None if a1 == "time" or a2 == a1 else a2)
        if key not in self._bands:
            self._bands[key] = _jacobian_bands(
                sv, ctx_of(sv), observed_defs(sv), a1, forcing, seed=seed
            )
        return self._bands[key]

    def structure(self, e1: list[JacEntry], sv: SysView) -> str:
        from .structure import detect_structure

        # keyed by identity, the list kept alive beside its class
        hit = self._structure.get(id(e1))
        if hit is None or hit[0] is not e1:
            hit = self._structure[id(e1)] = (e1, detect_structure(e1, sv))
        return hit[1]
