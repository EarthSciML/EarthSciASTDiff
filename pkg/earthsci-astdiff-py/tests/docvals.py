"""Values from documents (port of the reference's test/docvals.jl): the
package evaluates nothing, so the tests get every number by running one of its
documents through EarthSciAST — ``J`` column by column from the ``JvpEval``
product document (spec §9.2), ``Jᵀ λ`` from ``VjpEval`` and the second-order
contraction from ``HvpEval`` (``HvpEvalSymmetric``).

Every point is named, not positional: ``states`` maps a source state cell
(``"u[3]"``, ``"S"``) to its value, a cell left out taking the document's
default; ``params`` maps a scalar parameter to its value (array parameters and
forcing buffers take their document defaults). The evaluator is EarthSciAST's
interpreter, which runs every coefficient form (filtered and non-``+`` faq,
equality-filtered scatters) the native compiler may refuse (spec §9.3).
"""

from __future__ import annotations

from dataclasses import dataclass

import earthsci_ast as ea
from earthsci_ast.simulation_array import evaluate_rhs
from earthsci_astdiff import (
    column_names,
    extended_bands,
    hvp_document,
    jacobian_columns,
    jacobian_pattern,
    jvp_document,
    stable_json,
    sysview,
    vjp_document,
)
from earthsci_astdiff.emit import _cellname, foreach_point
from earthsci_astdiff.second_order import axis_pair, jacobian_bands2
from earthsci_astdiff.system import SysView, ctx_of, view_states


def _doc_cell(io_map: dict, cell: str) -> str:
    """The document's name for a cell of a source variable: ``u[3]`` →
    ``<io_map["u"]>[3]``."""
    v, sep, rest = cell.partition("[")
    return io_map[v] + sep + rest


def state_cells(sv: SysView) -> list[str]:
    """The source's state layout: each state's cells, column-major."""
    shapes = ctx_of(sv).shapes
    return column_names([(n, shapes.get(n) or []) for n in view_states(sv)])


def _key(forcing):
    """A forcing set as a hashable cache key (a list becomes a tuple)."""
    return tuple(forcing) if isinstance(forcing, list) else forcing


def _colnames(sv: SysView, wrt: str, forcing, rows: list[str]) -> list[str]:
    if wrt == "states":
        return rows
    if wrt == "time":
        return ["t"]
    return column_names(jacobian_columns(sv, wrt=wrt, forcing=forcing))


_CACHE: dict = {}


def _document(build, obj, model_name, *args, **kw) -> tuple:
    """``(sv, _Doc, io)`` for ``build(sv, *args, **kw)``, built once per
    source object and arguments (the source is held, so its ``id`` stays
    its own)."""
    key = (build.__name__, id(obj), model_name, args, tuple(sorted(kw.items())))
    hit = _CACHE.get(key)
    if hit is None or hit[0] is not obj:
        sv = sysview(obj, model_name)
        doc, io = build(sv, *args, **kw)
        hit = (obj, sv, _Doc(doc, io["model"], state_cells(sv)), io)
        _CACHE[key] = hit
    return hit[1:]


class _Doc:
    """A derived document, loaded and flattened once, evaluated per call."""

    def __init__(self, doc: dict, model: str, rows: list[str]):
        self.flat = ea.flatten(ea.load_string(stable_json(doc)))
        self.pfx = f"{model}."
        self.rows = set(rows)

    def call(self, states, params, t: float, inputs: dict):
        """One right-hand-side call at the source point ``states`` with
        ``inputs`` (document name → value) written into their slots; a lookup
        of ``du`` by document name."""
        unknown = set(states or {}) - self.rows
        if unknown:
            raise ValueError(f"states names no source state cell: {sorted(unknown)}")
        x = dict(states or {})
        x.update(inputs)
        du = evaluate_rhs(self.flat, x, float(t), parameters=params, compiler="interpreter")
        return lambda n: du[self.pfx + n]


def _bound(d: dict, names: list[str], what: str) -> dict:
    """``d`` over exactly ``names``: a missing or misspelled name is an
    error, never a zero."""
    if set(d) != set(names):
        raise ValueError(f"{what} must bind exactly {names}, got {sorted(d)}")
    return d


@dataclass
class DocJacobian:
    """``J`` at a point: ``J`` maps every structural pattern pair ``(row,
    col)`` to its value (a zero-valued pattern entry included); ``pattern`` is
    :func:`jacobian_pattern`'s sorted pair list, ``entries`` its bands."""

    J: dict
    pattern: list
    rownames: list
    colnames: list
    entries: list


def doc_jacobian(
    obj,
    model_name: str | None = None,
    wrt: str = "states",
    states: dict | None = None,
    params: dict | None = None,
    t: float = 0.0,
    forcing=None,
    cse: bool = True,
) -> DocJacobian:
    """``J`` at ``(states, params, t)``: column ``c`` read off one
    ``JvpEval`` call at the unit direction ``e_c``, stored on the structural
    pattern (:func:`jacobian_pattern`); a nonzero off the pattern raises."""
    sv, d, io = _document(jvp_document, obj, model_name, wrt=wrt, forcing=_key(forcing), cse=cse)
    rows = state_cells(sv)
    cols = _colnames(sv, wrt, forcing, rows)
    pattern, entries = jacobian_pattern(sv, wrt=wrt, forcing=forcing)
    pat = set(pattern)
    J: dict = {}
    for c in cols:
        out = d.call(states, params, t, {_doc_cell(io["v"], c): 1.0})
        for r in rows:
            v = out(_doc_cell(io["jvp"], r))
            if (r, c) in pat:
                J[(r, c)] = v
            elif v != 0:
                raise AssertionError(f"J[{r}, {c}] = {v} lies off the structural pattern")
    assert len(J) == len(pat), "a pattern pair names no row or column of the axis"
    return DocJacobian(J, pattern, rows, cols, entries)


def doc_vjp(
    obj,
    lam: dict,
    model_name: str | None = None,
    wrt: str = "states",
    states: dict | None = None,
    params: dict | None = None,
    t: float = 0.0,
    forcing=None,
) -> dict:
    """``Jᵀ λ`` from one ``VjpEval`` call; ``lam`` binds every state cell.
    Returns ``{col: value}`` over the axis's columns, in canonical order."""
    sv, d, io = _document(vjp_document, obj, model_name, wrt=wrt, forcing=_key(forcing))
    rows = state_cells(sv)
    _bound(lam, rows, "λ")
    out = d.call(states, params, t, {_doc_cell(io["lam"], r): lam[r] for r in rows})
    return {c: out(_doc_cell(io["vjp"], c)) for c in _colnames(sv, wrt, forcing, rows)}


def doc_hvp(
    obj,
    pairs: list,
    model_name: str | None = None,
    wrt=("states", "states"),
    form: str = "contract",
    states: dict | None = None,
    params: dict | None = None,
    t: float = 0.0,
    forcing=None,
) -> dict:
    """The §4.4 contraction from one ``HvpEval`` (``HvpEvalSymmetric``) call.
    ``pairs`` is a list of stage pairs ``(μ, w)``: ``μ`` binds every state
    cell, ``w`` every first-axis column (a number on a time pair). Returns
    ``{col2: value}`` over every cell of the second axis, in canonical
    order."""
    a1, a2 = axis_pair(wrt)
    sv, d, io = _document(
        hvp_document, obj, model_name, len(pairs), wrt=(a1, a2), form=form, forcing=_key(forcing)
    )
    rows = state_cells(sv)
    inputs: dict = {}
    for s, (mu, w) in enumerate(pairs):
        for r, x in _bound(mu, rows, "μ").items():
            inputs[_doc_cell(io["mu"][s], r)] = x
        if a1 == "time":
            if not isinstance(w, (int, float)):
                raise ValueError("w is a number on a time pair")
            inputs[io["w"][s]["t"]] = w
        else:
            for c, x in _bound(w, rows, "w").items():
                inputs[_doc_cell(io["w"][s], c)] = x
    out = d.call(states, params, t, inputs)
    return {c: out(_doc_cell(io["hvp"], c)) for c in _colnames(sv, a2, forcing, rows)}


def reached2(obj, model_name: str | None = None, wrt=("states", "states"), forcing=None) -> set:
    """The second columns a point of the pair's bands reaches with both of
    its columns inside their extents (a ghost point reaches nothing):
    structural, read off the bands by a point walk — the key set of an
    ``*.hvals.json`` golden (scripts/regenerate-value-goldens.jl
    ``_reached2``)."""
    sv = sysview(obj, model_name)
    shapes = ctx_of(sv).shapes

    def inext(v, c):
        shape = shapes.get(v) or []
        return all(1 <= c[d] <= shape[d] for d in range(len(c)))

    a1, a2 = axis_pair(wrt)
    # through a recurrence (spec §4.5) a second column may be a recurrence,
    # which is no column of the axis, and the second axis's recurrence rows
    # carry the contraction on to their columns
    e2 = extended_bands(sv, wrt=a2, forcing=forcing, second=a1)
    out = set()
    for en in jacobian_bands2(sv, wrt=wrt, forcing=forcing):
        if e2.is_recurrence(en.w):
            continue
        for _, _, _, c in foreach_point(en.band):
            c1, c2 = c[: en.ncol], c[en.ncol :]
            if inext(en.v, c1) and inext(en.w, c2):
                out.add(_cellname(en.w, c2))
    for en in e2.rec_rows:
        if e2.is_recurrence(en.v):
            continue
        for _, _, _, c in foreach_point(en.band):
            if inext(en.v, c):
                out.add(_cellname(en.v, c))
    return out
