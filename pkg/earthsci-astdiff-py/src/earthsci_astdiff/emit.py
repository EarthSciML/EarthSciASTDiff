"""§4 band-entry serialization, the structural pattern and the derived
evaluation document (emit.jl).

``stable_json(serialize_band(e) for e in entries)`` and ``stable_json`` of
:func:`evaluation_document` are cross-language conformance surfaces:
byte-identical to the Julia reference's goldens."""

from __future__ import annotations

from earthsci_ast.numpy_interpreter import evaluate
from earthsci_ast.parse import _parse_expression as parse_expression

from .bands import Band
from .expr_helpers import ser_expr
from .jacobian import JacEntry, jacobian_bands, jacobian_columns
from .second_order import JacEntry2, axis_pair, jacobian_block2, parse_band2, serialize_band2
from .structure import column_lines
from .system import sysview


def serialize_band(en: JacEntry | JacEntry2) -> dict:
    """One entry of the ``jacobians`` block, as a JSON-ready dict (spec §4;
    a :class:`JacEntry2` adds ``col2`` / ``col2_idx``, §4.4)."""
    if isinstance(en, JacEntry2):
        return serialize_band2(en)
    b = en.band
    d = {
        "row": en.u,
        "col": en.v,
        "rows": [[lo, hi] for lo, hi in b.rows],
        "row_idx": list(b.ridx),
        "col_idx": [ser_expr(c) for c in b.cidx],
        "coef": ser_expr(b.coef),
    }
    if b.contracted:
        d["contracted"] = [[n, lo, hi] for n, lo, hi in b.contracted]
    if b.guard is not None:
        d["where"] = ser_expr(b.guard)
    return d


def parse_band(d: dict) -> JacEntry:
    band = Band(
        [(int(r[0]), int(r[1])) for r in d["rows"]],
        [x if isinstance(x, str) else int(x) for x in d["row_idx"]],
        [parse_expression(c) for c in d["col_idx"]],
        parse_expression(d["coef"]),
        [(str(c[0]), int(c[1]), int(c[2])) for c in d.get("contracted", [])],
        parse_expression(d["where"]) if "where" in d else None,
    )
    return JacEntry(str(d["row"]), str(d["col"]), band)


def _cellname(v: str, cell) -> str:
    return v if not cell else f"{v}[{','.join(str(c) for c in cell)}]"


def _fold_const_gathers(e, bind: dict[str, float]):
    from earthsci_ast.esm_types import ExprNode
    from earthsci_ast.expr_walk import map_children

    if not isinstance(e, ExprNode):
        return e
    if e.op == "index" and isinstance(e.args[0], ExprNode) and e.args[0].op == "const":
        v = e.args[0].value
        for x in e.args[1:]:
            k = _eval_cidx(x, bind)
            if k < 1:  # wire tables are 1-based; never wrap to the end
                raise IndexError(f"const gather index {k} out of range")
            v = v[k - 1]
        return v
    return map_children(e, lambda x: _fold_const_gathers(x, bind))


def _eval_cidx(c, bind: dict[str, float]) -> int:
    return int(round(evaluate(_fold_const_gathers(c, bind), bind)))


def _eval_guard(g, bind: dict[str, float]) -> bool:
    return float(evaluate(_fold_const_gathers(g, bind), bind)) != 0


def foreach_point(b: Band):
    """Every point of one band (assemble.jl ``_foreach_point``): yields
    ``(cell, cc, bind, cidx)`` per (row cell, contracted cell), first index
    fastest, with ``bind`` the integer binding of the row and contracted
    names and ``cidx`` the column index evaluated there. A point the band's
    guard (``where``) excludes is not a point of the band."""
    for cell in _product(b.rows):
        for cc in _product([(lo, hi) for _, lo, hi in b.contracted]):
            bind = {r: cell[k] for k, r in enumerate(b.ridx) if isinstance(r, str)}
            for k, (n, _, _) in enumerate(b.contracted):
                bind[n] = cc[k]
            fbind = {k: float(v) for k, v in bind.items()}
            if b.guard is not None and not _eval_guard(b.guard, fbind):
                continue
            yield cell, cc, bind, [_eval_cidx(c, fbind) for c in b.cidx]


def _product(ranges):
    """Iterate integer boxes with the FIRST dimension fastest (Julia order)."""
    if not ranges:
        yield []
        return
    from itertools import product as _p

    for cell_r in _p(*[range(lo, hi + 1) for lo, hi in reversed(ranges)]):
        yield list(reversed(cell_r))


def scatter_pairs(sv, entries: list[JacEntry]):
    """Every (band k, row name, col name) triple — build-time, like the
    Julia `_scatter_pairs` (without the derived-model slot names)."""
    pairs = []
    for k, en in enumerate(entries):
        for cell, _, _, cidx in foreach_point(en.band):
            pairs.append((k, _cellname(en.u, cell), _cellname(en.v, cidx)))
    return pairs


def jacobian_pattern(obj, model_name: str | None = None, wrt: str = "states", forcing=None):
    """The structural sparsity pattern as sorted (rowname, colname) pairs
    plus the entries — dependency-light (no scipy). On the ``"parameters"``
    and ``"forcing"`` axes the columns are cells of the canonical column
    space (:func:`jacobian_columns`): a forcing column must resolve to one,
    and a parameter column outside its variable's extent is a ghost read."""
    sv = sysview(obj, model_name)
    entries = jacobian_bands(sv, wrt=wrt, forcing=forcing)
    if wrt != "states":
        triples = scatter_pairs(sv, entries)
    if wrt in ("parameters", "forcing"):
        cols = set(column_names(jacobian_columns(sv, wrt=wrt, forcing=forcing)))
        triples = [(k, r, c) for k, r, c in triples if check_column(c in cols, c, r, wrt)]
    elif wrt == "states":
        from .system import ctx_of, view_states

        states = set(view_states(sv))
        shapes = ctx_of(sv).shapes
        triples = [
            (k, _cellname(en.u, cell), _cellname(en.v, cidx))
            for k, en in enumerate(entries)
            for cell, _, _, cidx in foreach_point(en.band)
            if check_state_column(
                states, shapes, en.v, cidx, _cellname(en.v, cidx), _cellname(en.u, cell)
            )
        ]
    pairs = sorted({(r, c) for _, r, c in triples})
    return pairs, entries


def column_names(cols) -> list[str]:
    """The canonical column names of a :func:`jacobian_columns` space, in
    ordinal order: variables in order, each one's cells column-major."""
    names: list[str] = []
    for v, shape in cols:
        if not shape:
            names.append(v)
        else:
            names.extend(_cellname(v, c) for c in _product([(1, n) for n in shape]))
    return names


def check_column(ok: bool, colname: str, rowname: str, wrt: str) -> bool:
    """Whether a column resolves (``ok``), raising where it must. On the
    forcing axis a column is a cell of a live buffer, and the ghost-read skip
    that is legitimate for a stencil reading past the domain is not: an
    unresolvable column there would silently drop a gradient, so it is an
    error (spec §4). The parameters axis follows the state axis: an array
    parameter read past its extent is a ghost read, and ``False`` says to
    skip it."""
    from .bands import BandError

    if not ok and wrt == "forcing":
        raise BandError(
            f"column `{colname}` (row `{rowname}`) is outside the declared extent of its "
            f"variable; on the {wrt} axis every column must resolve (no ghost skip)"
        )
    return ok


def check_state_column(
    states: set, shapes: dict, v: str, cidx: list, colname: str, rowname: str
) -> bool:
    """Whether a state column resolves (assemble.jl ``_colmap`` /
    ``_colslot``): a cell past a state's extent is a ghost read (``False``,
    skip it), and a column variable that is not a state of the layout at all
    — an observed the band calculus took for a state, say — resolves nowhere
    and is refused rather than silently dropped."""
    from .bands import BandError

    if v not in states:
        raise BandError(
            f"column `{colname}` (row `{rowname}`) names no state of the layout; a band "
            "column must resolve to a state slot or read past a state's extent"
        )
    shape = shapes.get(v) or []
    return len(cidx) == len(shape) and all(1 <= c <= n for c, n in zip(cidx, shape, strict=True))


def jacobian_block(
    obj,
    model_name: str | None = None,
    wrt="states",
    cse: bool = True,
    cse_min_nodes: int = 12,
    forcing=None,
    column_axis: str | None = None,
):
    """The §4 `jacobians` block for one model: entries (template-CSE'd by
    default) and the block-local `expression_templates`; on the state and
    time axes the structure classification plus the `factorization` plan
    where one exists for the detected structure (per-cell block LU with
    Markowitz ordering), on the parameters and forcing axes the column space
    (`columns`, :func:`jacobian_columns`) instead.

    ``wrt`` may also be a list of distinct axes: the result is then the list
    of blocks, one per axis in the given order (a model's `jacobians` entry
    in its array form, spec §4). Inside such a list a ``tuple`` ``(a1, a2)``
    (or a two-name list) is an axis PAIR, a block of second-order entries
    (:func:`jacobian_bands2`, §4.4): ``["states", ("states", "states")]``,
    and ``[("states", "states")]`` for a pair block alone. A bare ``tuple``
    ``wrt`` is refused: it reads both as a pair and as a list of axes, so
    neither is assumed.

    ``column_axis`` (an index-set name) lets the state block report the
    ``"column_block"`` structure class with its line-by-line factorization
    plan when every coupling stays within one line along that axis
    (:func:`~earthsci_astdiff.structure.column_lines`)."""
    from .jacobian import _FirstOrder
    from .system import sysview

    if isinstance(wrt, tuple):
        raise ValueError(
            f"a tuple wrt {wrt!r} is ambiguous: write a list of axes as a list "
            '(["states", "forcing"]) and a second-order pair inside one '
            '([("states", "forcing")])'
        )
    sv = sysview(obj, model_name)
    memo = _FirstOrder()
    # One axis is a str (iterating it would split it into characters).
    if isinstance(wrt, str):
        return _block_of(sv, wrt, forcing, cse, cse_min_nodes, memo, column_axis)
    axes = [_axis_key(w) for w in wrt]
    if not axes or len(set(axes)) != len(axes):
        raise ValueError(f"wrt must name one or more distinct axes, got {wrt}")
    return [_block_of(sv, w, forcing, cse, cse_min_nodes, memo, column_axis) for w in axes]


def _axis_key(w):
    return axis_pair(w) if isinstance(w, (tuple, list)) else str(w)


def _block_of(sv, w, forcing, cse: bool, cse_min_nodes: int, memo, column_axis=None) -> dict:
    if isinstance(w, tuple):
        # a pair block has no structure class to refine
        return jacobian_block2(sv, w, forcing, cse, cse_min_nodes, memo)
    return _jacobian_block(sv, w, forcing, cse, cse_min_nodes, memo, column_axis)


def _jacobian_block(
    sv, wrt: str, forcing, cse: bool, cse_min_nodes: int, memo, column_axis=None
) -> dict:
    from .jacobian import _AXES

    if wrt not in _AXES:
        raise ValueError(f"wrt must be states, parameters, forcing or time, got {wrt}")
    entries = memo.bands(sv, wrt, None, forcing)
    # A parameter or forcing block describes a rectangular matrix, which the
    # structure classes and the factorization plan do not: it carries its
    # column space instead.
    rect = wrt in ("parameters", "forcing")
    structure = None if rect else memo.structure(entries, sv)
    # Line-by-line structure is a property of the square state matrix.
    # The one point walk both the class and its plan read.
    cax = column_axis if wrt == "states" else None
    cl = (
        None
        if cax is None or structure in ("empty", "block_diagonal")
        else column_lines(entries, sv, str(cax))
    )
    if cl is not None:
        structure = "column_block"
    block = _block(
        wrt,
        entries,
        serialize_band,
        _columns_field(sv, wrt, forcing) if rect else None,
        structure,
        cse,
        cse_min_nodes,
    )
    from .recurrence import auxiliary
    from .system import ctx_of, observed_defs

    aux = auxiliary(sv, observed_defs(sv), ctx_of(sv), entries)
    if aux:
        block["auxiliary"] = _auxiliary_field(aux)
    if not rect:
        from .factorization import _column_block_plan, plan_factorization, serialize_plan

        plan = (
            plan_factorization(entries, sv, structure=structure)
            if cl is None
            else _column_block_plan(cl)
        )
        if plan is not None:
            block["factorization"] = serialize_plan(plan)
    return block


def _block(wrt, entries, serialize, columns, structure, cse: bool, cse_min_nodes: int) -> dict:
    # The fields every block shares, in their emitted order: `wrt`, the
    # entries (coefficients template-CSE'd when `cse`), the `columns` of a
    # parameter or forcing axis and the `structure` class (each when given),
    # and the block-local `expression_templates`.
    from dataclasses import replace

    from .cse import cse_templates

    ser_entries = entries
    templates: dict = {}
    if cse:
        templates, rewritten = cse_templates(
            [en.band.coef for en in entries], min_nodes=cse_min_nodes
        )
        if templates:
            ser_entries = [
                replace(en, band=replace(en.band, coef=rewritten[k]))
                for k, en in enumerate(entries)
            ]
    block: dict = {"wrt": wrt, "entries": [serialize(en) for en in ser_entries]}
    if columns is not None:
        block["columns"] = columns
    if structure is not None:
        block["structure"] = structure
    if templates:
        block["expression_templates"] = {
            n: {"params": [], "body": ser_expr(b)} for n, b in templates.items()
        }
    return block


def _columns_field(sv, wrt: str, forcing) -> list:
    return [
        {"name": n, "shape": list(shape)}
        for n, shape in jacobian_columns(sv, wrt=wrt, forcing=forcing)
    ]


def parse_jacobian_block(block: dict):
    """Inverse of the block emission: expands `apply_expression_template`
    references back, so the returned entries are always closed. A block
    whose ``wrt`` is a pair holds second-order entries (§4.4): it returns
    ``((a1, a2), [JacEntry2, ...])``."""
    from dataclasses import replace

    from .cse import expand_templates

    pair = isinstance(block["wrt"], list)
    wrt = axis_pair(block["wrt"]) if pair else block["wrt"]
    parse = parse_band2 if pair else parse_band
    entries = [parse(d) for d in block["entries"]]
    if "expression_templates" in block:
        reg = {
            str(n): parse_expression(t["body"]) for n, t in block["expression_templates"].items()
        }
        entries = [
            replace(
                en,
                band=replace(
                    en.band,
                    cidx=[expand_templates(c, reg) for c in en.band.cidx],
                    coef=expand_templates(en.band.coef, reg),
                ),
            )
            for en in entries
        ]
    return wrt, entries


def parse_jacobian_blocks(x) -> list:
    """:func:`parse_jacobian_block` over a model's `jacobians` entry, which
    is either one block or a list of blocks with distinct `wrt` (spec §4);
    returns one ``(wrt, entries)`` per block, in order (``wrt`` a str, or a
    tuple for a second-order pair)."""
    blocks = [x] if isinstance(x, dict) else list(x)
    out = [parse_jacobian_block(b) for b in blocks]
    ws = [w for w, _ in out]
    if len(set(ws)) != len(ws):
        raise ValueError("a model's jacobian blocks must have distinct `wrt`")
    return out


# ── the evaluation model ─────────────────────────────────────────────────────
#
# Evaluation is EarthSciAST's: emit a derived single-model document in which
# each band's coefficient field is a STATE `J_k` whose RHS is the band field —
#
#   D(J_k) = makearray( zero default, band region ← faq(coef) )
#
# so one RHS call of an unmodified EarthSciAST evaluator leaves every band
# value in `du`. The product documents (products.py) share its base.


def _inline_value(x):
    """A variable's value in the document's inline form: an array value is
    row-major nested JSON (esm-spec §6.3), and a number is a float, as the
    reference's loader holds every default."""
    if hasattr(x, "tolist"):
        x = x.tolist()
    if isinstance(x, (list, tuple)):
        return [_inline_value(v) for v in x]
    if isinstance(x, int) and not isinstance(x, bool):
        return float(x)
    return x


def _base_document(sv, name: str) -> dict:
    """The source model as a derived document named ``name``: every variable
    (``unknown`` or ``parameter``, shape and default kept) and every equation
    verbatim. esm 1.0.0 has two declared types; whether an unknown is a state
    or an observed is made by the equations, which are copied whole."""
    from .system import _iset_size, _var_type

    variables: dict = {}
    for n in sorted(sv.variables):
        var = sv.variables[n]
        d: dict = {"type": "parameter" if _var_type(var) == "parameter" else "unknown"}
        shape = getattr(var, "shape", None)
        if shape:
            d["shape"] = list(shape)
        default = getattr(var, "default", None)
        if default is not None:
            d["default"] = _inline_value(default)
        variables[str(n)] = d
    equations = [{"lhs": ser_expr(eq.lhs), "rhs": ser_expr(eq.rhs)} for eq in sv.equations]
    isets = {}
    for n, s in sv.index_sets.items():
        sz = _iset_size(s)
        if sz is not None:
            isets[str(n)] = {"kind": "interval", "size": int(sz)}
    return {
        # Provisional: `_derived_version` settles it once the document is
        # complete.
        "esm": "1.1.0",
        "metadata": {"name": name},
        "index_sets": isets,
        "models": {name: {"variables": variables, "equations": equations}},
    }


def _derived_version(doc: dict) -> dict:
    """Settle the ``esm`` version a finished derived document declares: the
    lowest that admits what it holds. 1.1.0 for its ``faq`` nodes; 1.2.0
    when any expression node carries ``units`` (a unit-bearing ``const``
    copied from a 1.2.0 source). Read off the content, not the source."""
    doc["esm"] = "1.2.0" if _has_units(doc["models"]) else "1.1.0"
    return doc


def _has_units(x) -> bool:
    if isinstance(x, dict):
        return ("units" in x and "op" in x) or any(_has_units(v) for v in x.values())
    if isinstance(x, (list, tuple)):
        return any(_has_units(v) for v in x)
    return False


def _axis_name(doc: dict, sz: int) -> str:
    """A literal-size axis index set of the derived document (the
    hoisted-observed axes' ``jcax<size>`` convention, reused when present)."""
    from .eval_cse import jcax_name

    isets = doc["index_sets"]
    nm = jcax_name(lambda n: n in isets and isets[n].get("size") != sz, sz)
    if nm not in isets:
        isets[nm] = {"kind": "interval", "size": sz}
    return nm


def _defined_name(eq) -> str | None:
    # The variable a non-derivative equation defines: a bare LHS (`y ~ …`) or
    # an indexed one (`y[i…] ~ …`, lifted or pinned); None otherwise.
    from .system import _indexed_definition, lhs_definition

    n = lhs_definition(eq)
    if n is not None:
        return str(n)
    d = _indexed_definition(eq)
    return None if d is None else str(d[0])


def _zero_source_derivatives(m: dict, sv, ctx, roots: list) -> dict:
    """Trim the source's ODE right-hand side out of the derived model ``m``:
    every ``D(u)`` equation of the source becomes ``D(u) = 0`` (a zero
    ``makearray`` over an array state's shape), in place of the first one and
    with any further ones dropped. The states keep their solver slots — the
    consumer copies ``u`` into them — so the observed definitions and band
    fields read them unchanged, but each call no longer recomputes ``f``.
    Observed that ``roots`` (the expressions the derived model evaluates: the
    coefficients, and for a product document also the column indices and
    ``where`` guards, spec §9) do not read, directly or through other
    observed, are dropped with their declarations: they fed only ``f``.

    Zero derivatives rather than turning the states into parameters: an
    evaluator may treat observed that read only parameters as constant and
    cache them, so writing a new ``u`` in place would leave them stale."""
    import json

    from earthsci_ast.expression import free_variables

    from .bands import BandError
    from .system import is_derivative, lhs_state

    states = {u for u in (lhs_state(eq) for eq in sv.equations) if u is not None}
    defined = [_defined_name(eq) if lhs_state(eq) is None else None for eq in sv.equations]
    reads: dict[str, list] = {}  # observed → its definitions' RHS
    for eq, n in zip(sv.equations, defined, strict=True):
        if n is not None and n not in states:
            reads.setdefault(n, []).append(eq.rhs)
    live: set[str] = set()
    stack = [v for r in roots for v in free_variables(r)]
    while stack:
        n = stack.pop()
        if n not in reads or n in live:
            continue
        live.add(n)
        for r in reads[n]:
            stack.extend(free_variables(r))
    done: set[str] = set()
    eqs: list = []
    for eq, ser, n in zip(sv.equations, m["equations"], defined, strict=False):
        u = lhs_state(eq)
        if u is None:
            if is_derivative(eq):
                raise BandError(
                    f"unsupported derivative LHS {json.dumps(ser_expr(eq.lhs))} (a state's "
                    "derivative is `D(u)` or the lifted `faq{D(index(u, i…))}`)"
                )
            if n is not None and n in reads and n not in live:
                m["variables"].pop(n, None)
                continue
            eqs.append(ser)
        elif u not in done:
            done.add(u)
            eqs.append({"lhs": {"op": "D", "args": [u], "wrt": "t"}, "rhs": _zeros(ctx, u)})
    m["equations"] = eqs
    return m


def _zeros(ctx, name: str):
    """The zero value shaped like variable ``name``: ``0.0``, or a zero
    ``makearray`` over its shape."""
    shape = ctx.shapes.get(name) or []
    if not shape:
        return 0.0
    return {"op": "makearray", "args": [], "regions": [[[1, n] for n in shape]], "values": [0.0]}


def _hoist(doc: dict, m: dict, sv, entries: list, cse: bool, cse_min_nodes: int) -> list:
    """With ``cse``, hoist repeated coefficient subtrees into observed of the
    derived model ``m`` (:func:`~earthsci_astdiff.eval_cse.hoist_observed`),
    so the evaluator computes each once per cell per call. Returns the
    coefficients the model reads."""
    from .eval_cse import hoist_observed

    h = hoist_observed(entries, sv, min_nodes=cse_min_nodes) if cse else None
    if h is None:
        return [en.band.coef for en in entries]
    for nm, sz in h.axes.items():
        doc["index_sets"][nm] = {"kind": "interval", "size": sz}
    m["variables"].update(h.obs)
    # A hoisted observed is an unknown plus a bare-LHS equation, pushed
    # BEFORE the derived model's own derivative equations.
    for nm, rhs in h.obs_eqs:
        m["equations"].append({"lhs": nm, "rhs": rhs})
    return h.coefs


def _add_auxiliary(m: dict, aux: list) -> None:
    """The ``auxiliary`` definitions (spec §4.5) as observed of a derived
    model: each a declared unknown of its shape defined by a bare-LHS
    equation, after the source's own equations."""
    for n, shape, _ in aux:
        m["variables"][n] = {"type": "unknown", "shape": list(shape)}
    for n, _, d in aux:
        m["equations"].append({"lhs": n, "rhs": ser_expr(d)})


def _auxiliary_field(aux: list) -> list:
    """The block's ``auxiliary`` field (spec §4.5): ``[{name, shape, rhs}]``
    in recurrence order."""
    return [{"name": n, "shape": list(shape), "rhs": ser_expr(d)} for n, shape, d in aux]


def evaluation_document(
    obj,
    model_name: str | None = None,
    wrt: str = "states",
    forcing=None,
    cse: bool = True,
    cse_min_nodes: int = 12,
):
    """The derived evaluation document of the first-order bands on axis
    ``wrt`` (spec §8, model ``"JacobianEval"``): the source's observed the
    coefficients read, every source derivative zeroed, and one band-field
    state ``J_k`` per entry whose ``D`` equation is that band's coefficient
    field. One right-hand-side call of an unmodified EarthSciAST evaluator at
    ``(u, p, t)`` leaves every band value in ``du``.

    Returns ``(doc, fields)``, ``fields[k]`` naming entry ``k``'s field. With
    ``cse``, repeated coefficient subtrees are hoisted into observed arrays
    (:func:`~earthsci_astdiff.eval_cse.hoist_observed`)."""
    sv = sysview(obj, model_name)
    entries = jacobian_bands(sv, wrt=wrt, forcing=forcing)
    return _evaluation_document(sv, entries, cse=cse, cse_min_nodes=cse_min_nodes)


def _evaluation_document(sv, entries: list, cse: bool = True, cse_min_nodes: int = 12):
    # The source's observed definitions that the coefficients read, its state
    # derivatives zeroed, then one state per band whose RHS is the band's
    # coefficient field, after the optional hoisting. Returns (document,
    # band-state names).
    from .recurrence import auxiliary
    from .system import ctx_of, observed_defs

    doc = _base_document(sv, "JacobianEval")
    m = doc["models"]["JacobianEval"]
    ctx = ctx_of(sv)
    aux = auxiliary(sv, observed_defs(sv), ctx, entries)
    roots = [en.band.coef for en in entries] + [d for _, _, d in aux]
    _zero_source_derivatives(m, sv, ctx, roots)
    _add_auxiliary(m, aux)
    coefs = _hoist(doc, m, sv, entries, cse, cse_min_nodes)
    names: list[str] = []
    for k, en in enumerate(entries):
        b = en.band
        nm = f"J_{k + 1}"
        names.append(nm)
        shape_u = ctx.shapes.get(en.u) or []
        cr = b.contracted
        if not shape_u and not cr:
            m["variables"][nm] = {"type": "unknown", "default": 0.0}
            rhs = ser_expr(coefs[k])
        else:
            # The band field spans the row dims PLUS any free contracted
            # column dims — the scatter reads one slot per (row, contracted)
            # cell and ACCUMULATES into the column each cell gathers.
            ushape = list(sv.variables[en.u].shape) if shape_u else []
            m["variables"][nm] = {
                "type": "unknown",
                "default": 0.0,
                "shape": ushape + [_axis_name(doc, hi) for _, _, hi in cr],
            }
            free = [(k2, r) for k2, r in enumerate(b.ridx) if isinstance(r, str)]
            aggnames = [r for _, r in free] + [n for n, _, _ in cr]
            if not aggnames:
                val = ser_expr(coefs[k])
            else:
                rngs = {r: [b.rows[k2][0], b.rows[k2][1]] for k2, r in free}
                for n, lo, hi in cr:
                    rngs[n] = [lo, hi]
                val = {
                    "op": "faq",
                    "args": [],
                    "output_idx": aggnames,
                    "ranges": rngs,
                    "expr": ser_expr(coefs[k]),
                }
            rhs = {
                "op": "makearray",
                "args": [],
                "regions": [
                    [[1, n] for n in shape_u] + [[1, hi] for _, _, hi in cr],
                    [[lo, hi] for lo, hi in b.rows] + [[lo, hi] for _, lo, hi in cr],
                ],
                "values": [0.0, val],
            }
        m["equations"].append({"lhs": {"op": "D", "args": [nm], "wrt": "t"}, "rhs": rhs})
    _component_tree(doc, "JacobianEval")
    return _derived_version(doc), names


# ── the component tree (spec §8.1) ───────────────────────────────────────────
#
# A flattened system's variables carry scoped names (`SimpleOzone.O3`).
# Declared in the one derived model, EarthSciAST reads such a name as a
# reference into a component `SimpleOzone` that does not exist, and refuses
# the document. So each scoped name `P.v` is declared as `v` in the model at
# component path `P` (the first segment a top-level model, each further one a
# `subsystems` entry), and each equation goes to the model of its subject.
# Every reference stays as written, fully scoped, which EarthSciAST resolves
# from any model, so loading and flattening the document gives back the
# source's flattened names. The band fields, hoisted observed and recurrence
# auxiliaries keep their unscoped names in the derived model, so a source with
# no scoped name (one model of a document) gives the document it always did.


def _is_scoped(n) -> bool:
    return isinstance(n, str) and "." in n


def _lhs_subject(lhs):
    """The variable an equation is about, from its serialized left-hand side:
    the ``D`` or ``ic`` argument, a bare name, or the variable an ``index``
    (inside a lifted ``faq``) writes; None for anything else."""
    if isinstance(lhs, str):
        return lhs
    if not isinstance(lhs, dict):
        return None
    op = lhs.get("op")
    if op == "faq":
        return _lhs_subject(lhs.get("expr"))
    if op in ("D", "ic", "index"):
        args = lhs.get("args") or []
        return _lhs_subject(args[0]) if args else None
    return None


def _component_node(models: dict, path: str, derived: str) -> dict:
    """The model at component ``path``, created (with its parents) when
    missing."""
    from .bands import BandError

    segs = path.split(".")
    if segs[0] == derived:
        raise BandError(
            f"the source has a component named `{derived}`, which is the derived model's name"
        )
    node = models.setdefault(segs[0], {"variables": {}, "equations": []})
    for s in segs[1:]:
        node = node.setdefault("subsystems", {}).setdefault(s, {"variables": {}, "equations": []})
    return node


def _component_tree(doc: dict, derived: str) -> dict:
    """Move the scoped declarations and the equations whose subject is scoped
    out of the derived model ``derived`` into the component tree, in order."""
    models = doc["models"]
    m = models[derived]
    for n in sorted(k for k in m["variables"] if _is_scoped(k)):
        path, _, v = n.rpartition(".")
        _component_node(models, path, derived)["variables"][v] = m["variables"].pop(n)
    kept = []
    for eq in m["equations"]:
        s = _lhs_subject(eq["lhs"])
        if _is_scoped(s):
            _component_node(models, s.rpartition(".")[0], derived)["equations"].append(eq)
        else:
            kept.append(eq)
    m["equations"] = kept
    return doc
