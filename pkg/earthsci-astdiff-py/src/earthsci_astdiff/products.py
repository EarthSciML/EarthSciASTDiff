"""Product documents (products.jl, esm-jacobian-spec.md §9): derived ESM
documents whose right-hand side, evaluated once by an unmodified EarthSciAST
evaluator, yields ``J v``, ``Jᵀ λ`` or the §4.4 second-order contraction.

Like the evaluation document (emit.py), they are ASTs only: this module
evaluates nothing at run time. The one numeric step is §9.1's build-time
decision of which points read a column inside its extent, over closed column
indices. Byte-gated by the ``*.{jvp,vjp,hvp,hvp-sym}.evaldoc.json`` goldens.
"""

from __future__ import annotations

from dataclasses import replace
from functools import reduce

from earthsci_ast.esm_types import ExprNode
from earthsci_ast.expression import free_variables

from .bands import Band, BandError
from .emit import (
    _base_document,
    _derived_version,
    _eval_guard,
    _hoist,
    _zero_source_derivatives,
    foreach_point,
)
from .expr_helpers import ser_expr
from .jacobian import JacEntry, jacobian_bands, jacobian_columns
from .second_order import JacEntry2, axis_pair, jacobian_bands2
from .system import SysView, ctx_of, sysview, view_states


def _sop(o: str, *args) -> dict:
    return {"op": o, "args": list(args)}


def _sread(v: str, idx: list):
    """``index(v, idx…)``, or the bare name for a scalar read."""
    return _sop("index", v, *idx) if idx else v


def _sfold(o: str, xs: list):
    """An ``o`` over terms as a left fold, so the association is fixed (§9.1)."""
    return reduce(lambda a, b: _sop(o, a, b), xs)


def _sand(cs: list):
    return cs[0] if len(cs) == 1 else _sop("and", *cs)


def _fresh(taken: set[str], base: str) -> str:
    """The first of ``base``, ``base_``, ``base__``, … not in ``taken``, then
    taken."""
    nm = base
    while nm in taken:
        nm += "_"
    taken.add(nm)
    return nm


def _doc_names(doc: dict) -> set[str]:
    """Every name of a derived document (§9.1): the variable and index-set
    names, every ``ranges`` key, and every string its equations hold other
    than an ``op``."""
    taken = set(doc["index_sets"])
    for m in doc["models"].values():
        taken |= set(m["variables"])
        _collect_names(taken, m["equations"])
    return taken


def _collect_names(s: set[str], x) -> None:
    if isinstance(x, str):
        s.add(x)
    elif isinstance(x, (list, tuple)):
        for v in x:
            _collect_names(s, v)
    elif isinstance(x, dict):
        for k, v in x.items():
            if k == "op":
                continue
            if k == "ranges":
                s |= {str(n) for n in v}
            _collect_names(s, v)


def _column_vars(sv: SysView, wrt: str, forcing) -> list[str]:
    """The column variables of one first-order axis, in canonical order."""
    if wrt == "time":
        return ["t"]
    if wrt == "states":
        return view_states(sv)
    return [n for n, _ in jacobian_columns(sv, wrt=wrt, forcing=forcing)]


def _declare(m: dict, sv: SysView, ctx, name: str, like: str, rhs) -> None:
    """Declare an input or output of the product document: an unknown shaped
    like ``like`` (``"t"``: a scalar) with default 0 and the equation
    ``D(name) = rhs``, ``rhs is None`` meaning zero."""
    d: dict = {"type": "unknown", "default": 0.0}
    shape = ctx.shapes.get(like) or []
    if shape:
        d["shape"] = list(sv.variables[like].shape)
    m["variables"][name] = d
    if rhs is None:
        rhs = (
            0.0
            if not shape
            else {
                "op": "makearray",
                "args": [],
                "regions": [[[1, n] for n in shape]],
                "values": [0.0],
            }
        )
    m["equations"].append({"lhs": {"op": "D", "args": [name], "wrt": "t"}, "rhs": rhs})


def _product_base(sv: SysView, name: str, entries: list[JacEntry], cse: bool, cse_min_nodes: int):
    """Base of every product document (§9.1): the evaluation document's base,
    with the observed kept that the coefficients, column indices and
    ``where`` guards read, and the coefficients optionally hoisted. Returns
    ``(doc, model, ctx, coefficients)``."""
    doc = _base_document(sv, name)
    m = doc["models"][name]
    ctx = ctx_of(sv)
    roots: list = []
    for en in entries:
        roots.append(en.band.coef)
        roots.extend(en.band.cidx)
        if en.band.guard is not None:
            roots.append(en.band.guard)
    _zero_source_derivatives(m, sv, ctx, roots)
    return doc, m, ctx, _hoist(doc, m, sv, entries, cse, cse_min_nodes)


def _walk_band(b: Band, ci: list, guard) -> Band:
    """The point walk of ``b`` reading ``ci``, under ``guard``."""
    return replace(b, cidx=list(ci), guard=guard)


def _index_box(b: Band) -> dict[str, tuple[int, int]]:
    """A band's index names (row names, then contracted names) and their
    ranges."""
    box = {r: tuple(b.rows[k]) for k, r in enumerate(b.ridx) if isinstance(r, str)}
    for n, lo, hi in b.contracted:
        box[n] = (lo, hi)
    return box


def _index_only(e, box) -> bool:
    """Whether ``e`` reads only the band's index names."""
    return set(free_variables(e)) <= box.keys()


def _index_bounds(e, box):
    """Bounds ``(lo, hi, exact, names read)`` of an index expression over
    ``box``, by interval arithmetic over the integer ``+``, ``-``, ``*``
    core; ``None`` outside it (a ``const`` gather, a division, …). The bounds
    always enclose every value; ``exact`` when they are also attained at
    corners of the box: an affine expression reading each name at most
    once."""
    if isinstance(e, bool):
        return None
    if isinstance(e, int):
        return (e, e, True, [])
    if isinstance(e, float):
        return (int(e), int(e), True, []) if e.is_integer() else None
    if isinstance(e, str):
        if e not in box:
            return None
        lo, hi = box[e]
        return (lo, hi, True, [e])
    if isinstance(e, ExprNode) and e.op in ("+", "-", "*") and e.args:
        parts = [_index_bounds(a, box) for a in e.args]
        if any(p is None for p in parts):
            return None
        lo, hi, ex, ns = parts[0]
        ns = list(ns)
        if e.op == "-" and len(parts) == 1:
            return (-hi, -lo, ex, ns)
        for plo, phi, pex, pns in parts[1:]:
            if e.op == "+":
                lo, hi = lo + plo, hi + phi
            elif e.op == "-":
                lo, hi = lo - phi, hi - plo
            else:
                # affine only while one factor is a constant
                ex = ex and (not ns or not pns)
                c = (lo * plo, lo * phi, hi * plo, hi * phi)
                lo, hi = min(c), max(c)
            ex = ex and pex
            ns.extend(pns)
        return (lo, hi, ex and len(set(ns)) == len(ns), ns)
    return None


def _in_extent(b: Band, ci: list, shape: list[int]):
    """Whether every point of ``b`` reads ``ci`` inside ``shape``; ``None``
    when that has no build-time answer: an index not closed (a gather through
    a table that is not ``const``), or a ``where`` guard not closed that a
    point outside the extent depends on. Decided from the indices' bounds
    where they settle it (§9.1's affine breakpoints), by walking the band's
    points otherwise. An evaluation failure of that walk propagates, never
    taken for "not closed"."""
    if not ci:
        return True
    if len(ci) > len(shape):
        return None
    box = _index_box(b)
    if any(lo > hi for lo, hi in box.values()):
        return True  # no points
    if not all(_index_only(e, box) for e in ci):
        return None
    bounds = [_index_bounds(e, box) for e in ci]
    if all(x is not None for x in bounds):
        if all(1 <= bounds[d][0] and bounds[d][1] <= shape[d] for d in range(len(ci))):
            return True
        # a corner outside the extent is a point of the band unless a guard
        # may drop it
        if b.guard is None and all(x[2] for x in bounds):
            return False

    def inside(walk):
        return all(
            all(1 <= c[d] <= shape[d] for d in range(len(c))) for _, _, _, c in foreach_point(walk)
        )

    if b.guard is not None and not _index_only(b.guard, box):
        # a guard not closed cannot drop a point: only an all-inside box decides
        return True if inside(_walk_band(b, ci, None)) else None
    return inside(_walk_band(b, ci, b.guard))


def _check_rank(v: str, ci: list, shape: list[int]) -> None:
    """Refuse a read of ``v`` with more indices than its shape has
    dimensions."""
    if len(ci) > len(shape):
        raise BandError(
            f"a band reads `{v}` with {len(ci)} indices, but it has {len(shape)} dimensions"
        )


def _forcing_ghost(v: str, ok) -> BandError:
    """The forcing axis has no ghost read (§9.1): a point outside its buffer
    is refused rather than dropped."""
    why = (
        " or where that is not decidable at build time (an index or `where` guard not closed)"
        if ok is None
        else ""
    )
    return BandError(
        f"a point of a band reads forcing buffer `{v}` outside its extent{why}; "
        "a product document would drop it (esm-jacobian-spec.md §9.1)"
    )


def _ghost_conjuncts(b: Band, v: str, ci: list, shape: list[int], wrt: str) -> list:
    """§9.1's ghost rule for a read of column variable ``v`` at ``ci`` on axis
    ``wrt``: the in-extent conjuncts ``G`` to filter by, as ``(conjunct,
    names)`` pairs; none when every point is in extent (the region split
    then leaves the box whole)."""
    _check_rank(v, ci, shape)
    ok = _in_extent(b, ci, shape)
    if ok is True:
        return []
    if wrt == "forcing":
        raise _forcing_ghost(v, ok)
    out = []
    for d, e in enumerate(ci):
        ns = set(free_variables(e))
        out.append((_sop("<=", 1, ser_expr(e)), ns))
        out.append((_sop("<=", ser_expr(e), shape[d]), ns))
    return out


def _point_in_extent(b: Band, v: str, ci: list, shape: list[int], wrt: str) -> bool:
    """A band with no index names has one point (none when its closed
    ``where`` fails), and §9.2 has its column indices closed: whether that
    point exists and reads ``v`` inside its extent. A ghost is dropped,
    except on the forcing axis, where it is refused. Decided from the
    literal indices."""
    _check_rank(v, ci, shape)
    if not all(not free_variables(e) for e in ci):
        raise BandError(
            f"a band with no index names reads `{v}` at an index not closed at build "
            "time (esm-jacobian-spec.md §9.2)"
        )
    if not _closed_point(b):
        return False
    ok = _in_extent(_walk_band(b, ci, None), ci, shape) is True
    if not ok and wrt == "forcing":
        raise _forcing_ghost(v, False)
    return ok


def _row_names(b: Band) -> set[str]:
    return {r for r in b.ridx if isinstance(r, str)}


def _has_names(b: Band) -> bool:
    return any(isinstance(r, str) for r in b.ridx) or bool(b.contracted)


def _where_conj(b: Band) -> list:
    return [] if b.guard is None else [(ser_expr(b.guard), set(free_variables(b.guard)))]


def _names_of(es) -> set[str]:
    return {n for e in es for n in free_variables(e)}


def _closed_point(b: Band) -> bool:
    """Whether a band with no index names has its one point: its ``where``
    guard, which §9.2 has closed, holds."""
    if b.guard is None:
        return True
    if free_variables(b.guard):
        raise BandError(
            "a band with no index names has a `where` guard not closed at build "
            "time (esm-jacobian-spec.md §9.2)"
        )
    return _eval_guard(b.guard, {})


def _rho(b: Band) -> list:
    return list(b.ridx)


def _jnames(taken: set[str], b: Band, coef, n: int) -> list[str]:
    """Fresh output-index names ``_j1 … _jn``, clear of the entry's and the
    document's names (§9.1). The document's names are only looked up, never
    copied: a per-entry copy of the document-wide set is quadratic."""
    local = _row_names(b) | {c[0] for c in b.contracted} | _names_of((coef, *b.cidx))
    if b.guard is not None:
        local |= free_variables(b.guard)
    out = []
    for d in range(n):
        nm = f"_j{d + 1}"
        while nm in taken or nm in local:
            nm += "_"
        local.add(nm)
        out.append(nm)
    return out


def _term_faq(b: Band, output_idx: list, outranges, conj: list, body, bodynames: set[str]):
    """The ``faq`` of one entry's term: ``ranges`` (the output names first,
    then the row names, then the contracted names), the ``filter``
    conjunction, and ``body`` under §9.1's ``ifelse`` guard when a reduced
    name would not occur in it."""
    ranges = {n: [lo, hi] for n, lo, hi in outranges}
    for k, r in enumerate(b.ridx):
        if isinstance(r, str):
            ranges[r] = [b.rows[k][0], b.rows[k][1]]
    for n, lo, hi in b.contracted:
        ranges[n] = [lo, hi]
    outnames = {x for x in output_idx if isinstance(x, str)}
    miss = [n for n in ranges if n not in outnames and n not in bodynames]
    expr = body
    if miss:
        phi = [c for c, ns in conj if any(n in ns for n in miss)]
        if not all(any(n in ns for _, ns in conj) for n in miss):
            raise BandError(
                f"the reduced index name(s) {miss} occur neither in the product's body "
                "nor in its filter (esm-jacobian-spec.md §9.1)"
            )
        expr = _sop("ifelse", _sand(phi), body, 0.0)
    f = {"op": "faq", "args": [], "output_idx": list(output_idx), "ranges": ranges}
    if conj:
        f["filter"] = _sand([c for c, _ in conj])
    f["expr"] = expr
    return f


# ── Jv (§9.2) ────────────────────────────────────────────────────────────────


def _jvp_term(en: JacEntry, coef, vin: str, ctx, wrt: str):
    b = en.band
    shape_v = ctx.shapes.get(en.v) or []
    body = _sop("*", ser_expr(coef), _sread(vin, [ser_expr(e) for e in b.cidx]))
    shape_u = ctx.shapes.get(en.u) or []
    if _has_names(b):
        conj = _where_conj(b) + _ghost_conjuncts(b, en.v, b.cidx, shape_v, wrt)
        bodynames = _names_of((coef, *b.cidx)) | _row_names(b)
        out = [] if not shape_u else [r if isinstance(r, str) else 1 for r in b.ridx]
        val = _term_faq(b, out, (), conj, body, bodynames)
    else:
        # no index names: `where` and the column are closed, decided here
        if not _point_in_extent(b, en.v, b.cidx, shape_v, wrt):
            return None
        val = body
    if not shape_u:
        return val
    return {
        "op": "makearray",
        "args": [],
        "regions": [[[1, n] for n in shape_u], [[lo, hi] for lo, hi in b.rows]],
        "values": [0.0, val],
    }


# ── the equality-filter scatter (§9.3, §9.4) ─────────────────────────────────


def _scatter_term(
    b: Band,
    coef,
    out: str,
    ci: list,
    shape_o: list[int],
    wrt: str,
    conj: list,
    body,
    bodynames: set[str],
    taken: set[str],
):
    """One entry's term landing in output variable ``out`` (shape
    ``shape_o``) at column index ``ci``: Σ over the points with ``ci == j``
    of ``body``, under the extra filter conjuncts ``conj``."""
    # the equality drops every ghost, but on the forcing axis a dropped
    # point is a lost gradient
    if wrt == "forcing":
        _ghost_conjuncts(b, out, ci, shape_o, wrt)
    if not _has_names(b) and not shape_o:
        return body if _closed_point(b) else None
    j = _jnames(taken, b, coef, len(shape_o))
    eqs = [(_sop("==", ser_expr(e), j[d]), {*free_variables(e), j[d]}) for d, e in enumerate(ci)]
    return _term_faq(
        b,
        j,
        [(j[d], 1, shape_o[d]) for d in range(len(j))],
        _where_conj(b) + conj + eqs,
        body,
        bodynames,
    )


def _vjp_term(en: JacEntry, coef, lam: str, ctx, wrt: str, taken: set[str]):
    return _vjp_term_read(en, coef, _sread(lam, _rho(en.band)), ctx, wrt, taken)


def _vjp_term_read(en: JacEntry, coef, lam, ctx, wrt: str, taken: set[str]):
    # `_vjp_term` with the cotangent read at the row point given: a plain
    # input, or a reversed adjoint sweep (§9.7).
    b = en.band
    body = _sop("*", ser_expr(coef), lam)
    return _scatter_term(
        b,
        coef,
        en.v,
        b.cidx,
        ctx.shapes.get(en.v) or [],
        wrt,
        [],
        body,
        _names_of((coef,)) | _row_names(b),
        taken,
    )


# ── second-order contraction (§9.4) ──────────────────────────────────────────


def _hvp_term(en: JacEntry2, coef, mu, w, ctx, a1: str, a2: str, taken: set[str]):
    # `form = "contract"`: scatter by `col2_idx`, M summed over the stages.
    b = en.band
    ci, c2 = en.col_idx, en.col2_idx
    m = _sfold(
        "+",
        [
            _sop("*", _sread(mu[i][en.u], _rho(b)), _sread(w[i][en.v], [ser_expr(e) for e in ci]))
            for i in range(len(mu))
        ],
    )
    shape1 = ctx.shapes.get(en.v) or []
    if _has_names(b):
        g = _ghost_conjuncts(b, en.v, ci, shape1, a1)
    elif _point_in_extent(b, en.v, ci, shape1, a1):
        g = []
    else:
        return None
    return _scatter_term(
        b,
        coef,
        en.w,
        c2,
        ctx.shapes.get(en.w) or [],
        a2,
        g,
        _sop("*", ser_expr(coef), m),
        _names_of((coef, *ci)) | _row_names(b),
        taken,
    )


def _hvp_sym_term(en: JacEntry2, coef, mu, w, ctx, taken: set[str]):
    # `form = "symmetric"` on (states, states): scatter by the FIRST column.
    b = en.band
    ci, c2 = en.col_idx, en.col2_idx
    body = _sfold(
        "+",
        [
            _sop(
                "*",
                _sread(mu[i][en.u], _rho(b)),
                ser_expr(coef),
                _sread(w[i][en.w], [ser_expr(e) for e in c2]),
            )
            for i in range(len(mu))
        ],
    )
    shape2 = ctx.shapes.get(en.w) or []
    if _has_names(b):
        g = _ghost_conjuncts(b, en.w, c2, shape2, "states")
    elif _point_in_extent(b, en.w, c2, shape2, "states"):
        g = []
    else:
        return None
    return _scatter_term(
        b,
        coef,
        en.v,
        ci,
        ctx.shapes.get(en.v) or [],
        "states",
        g,
        body,
        _names_of((coef, *c2)) | _row_names(b),
        taken,
    )


# ── the documents ────────────────────────────────────────────────────────────


def _outputs(m: dict, sv: SysView, ctx, outs: dict[str, str], term, owner: list) -> None:
    """Each output's equation: the left fold of its entries' terms (§9.1).
    ``owner`` lists ``(k, the output variable entry k lands in)``."""
    by_owner: dict[str, list[int]] = {}
    for k, o in owner:  # one pass, not one per output; entry order kept
        by_owner.setdefault(o, []).append(k)
    for c, nm in outs.items():
        terms = []
        for k in by_owner.get(c, ()):
            t = term(k)
            if t is not None:
                terms.append(t)
        _declare(m, sv, ctx, nm, c, _sfold("+", terms) if terms else None)


def _jvp_document(
    sv: SysView,
    entries: list[JacEntry],
    wrt: str,
    forcing,
    cse: bool = True,
    cse_min_nodes: int = 12,
):
    ext = _extended_for(sv, wrt, forcing)
    if ext.recurrences:
        return _jvp_sweeps(sv, ext, wrt, forcing, cse, cse_min_nodes)
    doc, m, ctx, coefs = _product_base(sv, "JvpEval", entries, cse, cse_min_nodes)
    taken = _doc_names(doc)
    vin = {c: _fresh(taken, f"_v_{c}") for c in _column_vars(sv, wrt, forcing)}
    out = {r: _fresh(taken, f"_jvp_{r}") for r in view_states(sv)}
    for c, nm in vin.items():
        _declare(m, sv, ctx, nm, c, None)
    _outputs(
        m,
        sv,
        ctx,
        out,
        lambda k: _jvp_term(entries[k], coefs[k], vin[entries[k].v], ctx, wrt),
        [(k, en.u) for k, en in enumerate(entries)],
    )
    return _derived_version(doc), {"model": "JvpEval", "v": vin, "jvp": out}


def _vjp_document(
    sv: SysView,
    entries: list[JacEntry],
    wrt: str,
    forcing,
    cse: bool = True,
    cse_min_nodes: int = 12,
):
    ext = _extended_for(sv, wrt, forcing)
    if ext.recurrences:
        return _vjp_sweeps(sv, ext, wrt, forcing, cse, cse_min_nodes)
    doc, m, ctx, coefs = _product_base(sv, "VjpEval", entries, cse, cse_min_nodes)
    taken = _doc_names(doc)
    lam = {r: _fresh(taken, f"_lam_{r}") for r in view_states(sv)}
    out = {c: _fresh(taken, f"_vjp_{c}") for c in _column_vars(sv, wrt, forcing)}
    for r, nm in lam.items():
        _declare(m, sv, ctx, nm, r, None)
    _outputs(
        m,
        sv,
        ctx,
        out,
        lambda k: _vjp_term(entries[k], coefs[k], lam[entries[k].u], ctx, wrt, taken),
        [(k, en.v) for k, en in enumerate(entries)],
    )
    return _derived_version(doc), {"model": "VjpEval", "lam": lam, "vjp": out}


def _hvp_document(
    sv: SysView,
    entries: list[JacEntry2],
    wrt: tuple[str, str],
    forcing,
    stages: int,
    form: str = "contract",
    cse: bool = True,
    cse_min_nodes: int = 12,
):
    a1, a2 = wrt
    if form not in ("contract", "symmetric"):
        raise ValueError(f"form must be 'contract' or 'symmetric', got {form!r}")
    if form == "symmetric" and (a1, a2) != ("states", "states"):
        raise ValueError(f"form='symmetric' needs a ('states', 'states') pair, got ({a1}, {a2})")
    if stages < 1:
        raise ValueError(f"a contraction needs at least one stage, got {stages}")
    model = "HvpEval" if form == "contract" else "HvpEvalSymmetric"
    pe = _pair_extended(sv, a1, a2, forcing)
    if pe is not None:
        return _hvp_sweeps(
            sv, entries, pe, (a1, a2), forcing, stages, form, model, cse, cse_min_nodes
        )
    # the coefficients only: a second-order entry rides through the base and
    # the hoisting as a first-order one (its `cidx` holds both columns)
    doc, m, ctx, coefs = _product_base(
        sv, model, [JacEntry(en.u, en.v, en.band) for en in entries], cse, cse_min_nodes
    )
    taken = _doc_names(doc)
    states = view_states(sv)
    cols1 = ["t"] if a1 == "time" else states
    mu: list[dict[str, str]] = []
    w: list[dict[str, str]] = []
    for i in range(1, stages + 1):
        mu.append({r: _fresh(taken, f"_mu{i}_{r}") for r in states})
        w.append({c: _fresh(taken, f"_w{i}_{c}") for c in cols1})
    cols2 = _column_vars(sv, a2, forcing) if form == "contract" else states
    out = {c: _fresh(taken, f"_hvp_{c}") for c in cols2}
    for i in range(stages):
        for r, nm in mu[i].items():
            _declare(m, sv, ctx, nm, r, None)
        for c, nm in w[i].items():
            _declare(m, sv, ctx, nm, c, None)
    if form == "contract":
        _outputs(
            m,
            sv,
            ctx,
            out,
            lambda k: _hvp_term(entries[k], coefs[k], mu, w, ctx, a1, a2, taken),
            [(k, en.w) for k, en in enumerate(entries)],
        )
    else:
        _outputs(
            m,
            sv,
            ctx,
            out,
            lambda k: _hvp_sym_term(entries[k], coefs[k], mu, w, ctx, taken),
            [(k, en.v) for k, en in enumerate(entries)],
        )
    return _derived_version(doc), {"model": model, "mu": mu, "w": w, "hvp": out}


# ── recurrences: tangent and adjoint sweeps (§9.7) ───────────────────────────


def _extended_for(sv: SysView, wrt: str, forcing):
    # The extended system of axis `wrt` (spec §4.5); empty (no pass run)
    # when the view has no recurrence at all.
    from .jacobian import _extended_pass
    from .recurrence import ExtendedBands, recurrences
    from .system import observed_defs

    ctx = ctx_of(sv)
    obs = observed_defs(sv)
    if not recurrences(sv, obs, ctx):
        return ExtendedBands()
    return _extended_pass(sv, ctx, obs, wrt, forcing, [])


def _pair_extended(sv: SysView, a1: str, a2: str, forcing):
    # The two extended systems of a pair with an active recurrence: the first
    # axis's (seeded with the second axis's targets, as `jacobian_bands2`
    # differentiates it) and the second axis's (its recurrence rows carry the
    # adjoint back to the second axis's columns), both over the recurrences
    # that move with either axis. None without one.
    from .jacobian import _extended_pass, _targets
    from .recurrence import recurrences
    from .system import observed_defs

    ctx = ctx_of(sv)
    obs = observed_defs(sv)
    if not recurrences(sv, obs, ctx):
        return None
    t1 = _targets(sv, a1, forcing, obs)
    t2 = _targets(sv, a2, forcing, obs)
    e1 = _extended_pass(sv, ctx, obs, a1, forcing, t2)
    if not e1.recurrences:
        return None
    e2 = e1 if a1 == a2 else _extended_pass(sv, ctx, obs, a2, forcing, t1)
    return e1, e2


def _ghost_exprs(b: Band, v: str, ci: list, ci_sub: list, shape: list[int], wrt: str) -> list:
    # §9.1's ghost rule for a read of `v` at `ci` from band `b`, as expression
    # conjuncts over the substituted indices (the forcing axis refuses).
    from .expr_helpers import op

    _check_rank(v, ci, shape)
    if _in_extent(b, ci, shape) is True:
        return []
    _ghost_conjuncts(b, v, ci, shape, wrt)  # refuses on the forcing axis
    out = []
    for e, n in zip(ci_sub, shape):
        out.append(op(">=", e, 1))
        out.append(op("<=", e, n))
    return out


def _observe(m: dict, sv: SysView, name: str, like: str, rhs) -> None:
    # Declare an intermediate of the product document (§9.7): an observed
    # unknown shaped like recurrence `like`, defined by `name ~ rhs`.
    m["variables"][name] = {
        "type": "unknown",
        "shape": [str(x) for x in (sv.variables[like].shape or [])],
    }
    m["equations"].append({"lhs": name, "rhs": rhs})


def _tangent(rec, rows: list, coefs: list, target: str, dir: dict, sweeps: dict, ctx, wrt: str):
    # The forward sweep variable of `rec` (`target`): its own bands among
    # `rows` (with their hoisted coefficients), reading a target column
    # through `dir` and another recurrence's sweep through `sweeps`.
    from .recurrence import _index, forward_sweep

    own = []
    cols = []
    for en, c in zip(rows, coefs):
        if en.u == rec.name:
            own.append(replace(en.band, coef=c))
            cols.append(en.v)

    def col(b: Band, c: str, ci: list):
        shape = ctx.shapes.get(c) or []
        if c in sweeps:
            nm, w = sweeps[c], "states"
        else:
            nm, w = dir[c], wrt
        return _index(nm, ci), _ghost_exprs(b, c, b.cidx, ci, shape, w)

    return ser_expr(forward_sweep(rec, own, cols, target, col, 0.0))


def _adjoint(rec, rows: list, coefs: list, target: str, src: str):
    # The adjoint sweep variable of `rec` (`target`, reversed axis) fed by
    # the cotangent source `src`: `rec`'s self bands among `rows`.
    from .recurrence import adjoint_sweep, reversed_read

    own = [
        replace(en.band, coef=c)
        for en, c in zip(rows, coefs)
        if en.u == rec.name and en.v == rec.name
    ]
    src_read = reversed_read(rec, src, list(rec.frame))
    return ser_expr(adjoint_sweep(rec, own, target, src_read, 0.0))


def _cot_read(b: Band, u: str, plain: dict, rev: dict, ext):
    # A cotangent read at the row point of `b`: a plain input (a state row)
    # or a reversed adjoint sweep (a recurrence row).
    from .recurrence import reversed_read

    rec = ext.recurrence(u)
    if rec is not None and u in rev:
        return ser_expr(reversed_read(rec, rev[u], list(b.ridx)))
    return _sread(plain[u], _rho(b))


def _cot_source(v: str, entries: list, coefs: list, lam: dict, rev: dict, ext, ctx, taken):
    # The cotangent source of recurrence `v`: Σ over the bands reading `v`
    # (other than `v`'s own) of `coef · λ[row]`, scattered onto `v`'s cells
    # (an equality-filter scatter, §9.3); zero when none reads it.
    from .emit import _zeros

    terms = []
    for en, c in zip(entries, coefs):
        if en.v != v or en.u == v:
            continue
        t = _vjp_term_read(en, c, _cot_read(en.band, en.u, lam, rev, ext), ctx, "states", taken)
        if t is not None:
            terms.append(t)
    return _sfold("+", terms) if terms else _zeros(ctx, v)


def _rec_names(taken: set[str], ext, tag: str) -> dict:
    return {r.name: _fresh(taken, f"_{tag}_{r.name}") for r in ext.recurrences}


def _jvp_sweeps(sv: SysView, ext, wrt: str, forcing, cse: bool, cse_min_nodes: int):
    every = ext.rows + ext.rec_rows
    doc, m, ctx, coefs = _product_base(sv, "JvpEval", every, cse, cse_min_nodes)
    taken = _doc_names(doc)
    vin = {c: _fresh(taken, f"_v_{c}") for c in _column_vars(sv, wrt, forcing)}
    vdot = _rec_names(taken, ext, "vdot")
    out = {r: _fresh(taken, f"_jvp_{r}") for r in view_states(sv)}
    for c, nm in vin.items():
        _declare(m, sv, ctx, nm, c, None)
    nrow = len(ext.rows)
    for rec in ext.recurrences:
        t = _tangent(rec, every[nrow:], coefs[nrow:], vdot[rec.name], vin, vdot, ctx, wrt)
        _observe(m, sv, vdot[rec.name], rec.name, t)

    def term(k):
        en = ext.rows[k]
        if en.v in vdot:
            return _jvp_term(en, coefs[k], vdot[en.v], ctx, "states")
        return _jvp_term(en, coefs[k], vin[en.v], ctx, wrt)

    _outputs(m, sv, ctx, out, term, [(k, en.u) for k, en in enumerate(ext.rows)])
    return _derived_version(doc), {"model": "JvpEval", "v": vin, "jvp": out}


def _vjp_sweeps(sv: SysView, ext, wrt: str, forcing, cse: bool, cse_min_nodes: int):
    every = ext.rows + ext.rec_rows
    doc, m, ctx, coefs = _product_base(sv, "VjpEval", every, cse, cse_min_nodes)
    taken = _doc_names(doc)
    lam = {r: _fresh(taken, f"_lam_{r}") for r in view_states(sv)}
    vbar = _rec_names(taken, ext, "vbar")
    vrev = _rec_names(taken, ext, "vrev")
    out = {c: _fresh(taken, f"_vjp_{c}") for c in _column_vars(sv, wrt, forcing)}
    for r, nm in lam.items():
        _declare(m, sv, ctx, nm, r, None)
    nrow = len(ext.rows)
    for rec in ext.recurrences:
        v = rec.name
        _observe(m, sv, vbar[v], v, _cot_source(v, every, coefs, lam, vrev, ext, ctx, taken))
        _observe(m, sv, vrev[v], v, _adjoint(rec, every[nrow:], coefs[nrow:], vrev[v], vbar[v]))

    def term(k):
        en = every[k]
        lr = _cot_read(en.band, en.u, lam, vrev, ext)
        return _vjp_term_read(en, coefs[k], lr, ctx, wrt, taken)

    _outputs(m, sv, ctx, out, term, [(k, en.v) for k, en in enumerate(every)])
    return _derived_version(doc), {"model": "VjpEval", "lam": lam, "vjp": out}


def _hvp_sweeps(
    sv: SysView, entries, pe, wrt, forcing, stages: int, form: str, model: str, cse, cse_min
):
    # The §4.4 contraction through recurrences (§9.7): per stage i, the
    # tangent sweep of w_i and the adjoint sweep of μ_i over the first axis's
    # extended system make the contraction's M read a recurrence row's
    # cotangent and a recurrence column's direction; the second-order bands
    # scattered onto a recurrence column feed one more adjoint sweep Ĥ, which
    # the second axis's recurrence rows carry back to its columns.
    e1, e2 = pe
    a1, a2 = wrt
    first = e1.rows + e1.rec_rows
    back = list(e2.rec_rows)
    n2 = len(entries)
    doc, m, ctx, coefs = _product_base(
        sv, model, [JacEntry(en.u, en.v, en.band) for en in entries] + first + back, cse, cse_min
    )
    c2s, c1s, cbs = coefs[:n2], coefs[n2 : n2 + len(first)], coefs[n2 + len(first) :]
    taken = _doc_names(doc)
    states = view_states(sv)
    cols1 = ["t"] if a1 == "time" else states
    mu: list[dict[str, str]] = []
    w: list[dict[str, str]] = []
    for i in range(1, stages + 1):
        mu.append({r: _fresh(taken, f"_mu{i}_{r}") for r in states})
        w.append({c: _fresh(taken, f"_w{i}_{c}") for c in cols1})
    vdot, vbar, vrev = [], [], []
    for i in range(1, stages + 1):
        vdot.append(_rec_names(taken, e1, f"vdot{i}"))
        vbar.append(_rec_names(taken, e1, f"vbar{i}"))
        vrev.append(_rec_names(taken, e1, f"vrev{i}"))
    hbar = _rec_names(taken, e1, "hbar")
    hrev = _rec_names(taken, e1, "hrev")
    cols2 = _column_vars(sv, a2, forcing) if form == "contract" else states
    out = {c: _fresh(taken, f"_hvp_{c}") for c in cols2}
    for i in range(stages):
        for r, nm in mu[i].items():
            _declare(m, sv, ctx, nm, r, None)
        for c, nm in w[i].items():
            _declare(m, sv, ctx, nm, c, None)
    nrow = len(e1.rows)
    for i in range(stages):
        for rec in e1.recurrences:
            v = rec.name
            t = _tangent(rec, first[nrow:], c1s[nrow:], vdot[i][v], w[i], vdot[i], ctx, a1)
            _observe(m, sv, vdot[i][v], v, t)
            src = _cot_source(v, first, c1s, mu[i], vrev[i], e1, ctx, taken)
            _observe(m, sv, vbar[i][v], v, src)
            _observe(
                m,
                sv,
                vrev[i][v],
                v,
                _adjoint(rec, first[nrow:], c1s[nrow:], vrev[i][v], vbar[i][v]),
            )

    def cot(i, b, u):
        return _cot_read(b, u, mu[i], vrev[i], e1)

    def dirr(i, v, ci):
        nm = vdot[i][v] if v in vdot[i] else w[i][v]
        return _sread(nm, [ser_expr(e) for e in ci])

    def axis(v, a):
        return "states" if e1.is_recurrence(v) else a

    def term2(k):
        en = entries[k]
        b = en.band
        coef = c2s[k]
        ci, c2 = en.col_idx, en.col2_idx
        bodynames = _names_of((coef, *ci, *c2)) | _row_names(b)
        if form == "contract":
            mm = _sfold("+", [_sop("*", cot(i, b, en.u), dirr(i, en.v, ci)) for i in range(stages)])
            shape1 = ctx.shapes.get(en.v) or []
            a = axis(en.v, a1)
            if _has_names(b):
                g = _ghost_conjuncts(b, en.v, ci, shape1, a)
            elif _point_in_extent(b, en.v, ci, shape1, a):
                g = []
            else:
                return None
            return _scatter_term(
                b,
                coef,
                en.w,
                c2,
                ctx.shapes.get(en.w) or [],
                axis(en.w, a2),
                g,
                _sop("*", ser_expr(coef), mm),
                bodynames,
                taken,
            )
        body = _sfold(
            "+",
            [_sop("*", cot(i, b, en.u), ser_expr(coef), dirr(i, en.w, c2)) for i in range(stages)],
        )
        shape2 = ctx.shapes.get(en.w) or []
        if _has_names(b):
            g = _ghost_conjuncts(b, en.w, c2, shape2, "states")
        elif _point_in_extent(b, en.w, c2, shape2, "states"):
            g = []
        else:
            return None
        return _scatter_term(
            b, coef, en.v, ci, ctx.shapes.get(en.v) or [], "states", g, body, bodynames, taken
        )

    owner2 = [en.w if form == "contract" else en.v for en in entries]

    def back_term(k):
        en = back[k]
        lr = _cot_read(en.band, en.u, {}, hrev, e2)
        return _vjp_term_read(en, cbs[k], lr, ctx, axis(en.v, a2), taken)

    for rec in e1.recurrences:
        v = rec.name
        terms = []
        for k in range(n2):
            if owner2[k] == v:
                t = term2(k)
                if t is not None:
                    terms.append(t)
        for k, en in enumerate(back):
            if en.v == v and en.u != v:
                t = back_term(k)
                if t is not None:
                    terms.append(t)
        _observe(m, sv, hbar[v], v, _sfold("+", terms) if terms else _zeros_of(ctx, v))
        _observe(m, sv, hrev[v], v, _adjoint(rec, back, cbs, hrev[v], hbar[v]))
    owner = [(k, o) for k, o in enumerate(owner2)] + [(n2 + k, en.v) for k, en in enumerate(back)]
    _outputs(m, sv, ctx, out, lambda k: term2(k) if k < n2 else back_term(k - n2), owner)
    return _derived_version(doc), {"model": model, "mu": mu, "w": w, "hvp": out}


def _zeros_of(ctx, v: str):
    from .emit import _zeros

    return _zeros(ctx, v)


def jvp_document(
    obj,
    model_name: str | None = None,
    wrt: str = "states",
    forcing=None,
    cse: bool = True,
    cse_min_nodes: int = 12,
):
    """The ``J v`` product document of spec §9.2 for one first-order axis.

    A derived ESM document (model ``"JvpEval"``) whose right-hand side,
    evaluated by an unmodified EarthSciAST evaluator at ``(u, inputs, p,
    t)``, leaves ``J v`` in the derivative slots of its outputs. ``obj`` is
    an ``EsmFile``, a flattened system or a :class:`SysView`, as for
    :func:`~earthsci_astdiff.jacobian.jacobian_bands`.

    Returns ``(doc, io)``: ``io["model"]`` is the model name,
    ``io["v"]`` maps each column variable to its direction input and
    ``io["jvp"]`` each state to its output. The consumer writes the inputs
    into the evaluator's state slots beside the source states and reads the
    outputs from ``du`` through the evaluator's variable map."""
    sv = sysview(obj, model_name)
    entries = jacobian_bands(sv, wrt=wrt, forcing=forcing)
    return _jvp_document(sv, entries, wrt, forcing, cse=cse, cse_min_nodes=cse_min_nodes)


def vjp_document(
    obj,
    model_name: str | None = None,
    wrt: str = "states",
    forcing=None,
    cse: bool = True,
    cse_min_nodes: int = 12,
):
    """The ``Jᵀ λ`` product document of spec §9.3 for one first-order axis
    (model ``"VjpEval"``), as :func:`jvp_document`: ``io["lam"]`` maps each
    state to its adjoint input and ``io["vjp"]`` each column variable to its
    output.

    A ``VjpEval`` document is an equality-filtered ``faq`` scatter: correct
    under any evaluator, linear-time only under one that implements §9.3's
    equality-filter scatter."""
    sv = sysview(obj, model_name)
    entries = jacobian_bands(sv, wrt=wrt, forcing=forcing)
    return _vjp_document(sv, entries, wrt, forcing, cse=cse, cse_min_nodes=cse_min_nodes)


def hvp_document(
    obj,
    stages: int,
    model_name: str | None = None,
    wrt=("states", "states"),
    form: str = "contract",
    forcing=None,
    cse: bool = True,
    cse_min_nodes: int = 12,
):
    """The second-order product document of spec §9.4 for the axis pair
    ``wrt`` and ``stages`` stage pairs: its right-hand side leaves
    ``g = ∇_{a2} Σᵢ μᵢᵀ J(u) wᵢ`` (``∇_{a2} (Σᵢ wᵢ μᵢ)ᵀ f_t`` on a time pair)
    in the derivative slots of its outputs.

    Returns ``(doc, io)``: ``io["mu"][i]`` / ``io["w"][i]`` map each state
    (each first-axis column variable; ``"t"`` on a time pair) to stage
    ``i``'s inputs and ``io["hvp"]`` each second-axis column variable to its
    output. ``form="symmetric"`` (a ``("states", "states")`` pair only)
    scatters by the first column instead (model ``"HvpEvalSymmetric"``);
    both forms give the same ``g``."""
    a1, a2 = axis_pair(wrt)
    sv = sysview(obj, model_name)
    entries = jacobian_bands2(sv, wrt=(a1, a2), forcing=forcing)
    return _hvp_document(
        sv, entries, (a1, a2), forcing, stages, form=form, cse=cse, cse_min_nodes=cse_min_nodes
    )
