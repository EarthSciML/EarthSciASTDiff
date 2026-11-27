"""Matrix-structure classification from the bands' column-index expressions
(structure.jl) — never probed numerically."""

from __future__ import annotations

from earthsci_ast.esm_types import Expr, ExprNode

from .expr_helpers import mul, op, ser_expr
from .jacobian import JacEntry
from .simplify_branches import canon_lits
from .system import SysView, ctx_of


def _is_lit(x: Expr) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _offset_class(c: Expr, r) -> str:
    if isinstance(r, str):
        if isinstance(c, str) and c == r:
            return "diag"
        if isinstance(c, ExprNode) and c.op in ("+", "-") and len(c.args) == 2:
            a, b = c.args
            if isinstance(a, str) and a == r and _is_lit(b):
                return "affine"
            if c.op == "+" and isinstance(b, str) and b == r and _is_lit(a):
                return "affine"
        if _is_lit(c):
            return "fixed"
        return "other"
    if _is_lit(c):
        return "diag" if int(round(float(c))) == r else "fixed"
    return "other"


def detect_structure(
    entries: list[JacEntry],
    sv: SysView,
    column_axis: str | None = None,
    cell_axes: list[str] | None = None,
) -> str:
    """Classify the band list: ``"block_diagonal"``, ``"banded"``, ``"general"``
    (or ``"empty"`` / ``"other"``), read off the column-index expressions
    without evaluating anything. Use it to choose a matrix type.

    Given ``column_axis`` (an index-set name, from the caller's locality
    pass), ``"column_block"`` replaces ``"banded"`` / ``"general"`` when
    every coupling stays within one LINE along that axis
    (:func:`column_lines`), so a solver can factorize line by line.

    Given ``cell_axes`` (index-set names, from the caller), ``"cell_block"``
    replaces ``"banded"`` / ``"general"``, before ``"column_block"``, when
    every coupling stays inside one CELL of those axes (:func:`cell_blocks`),
    whose plan pairs the cell axes by index-set name. ``"block_diagonal"``
    and ``"banded"`` are positional (their consumers pair row and column
    cells by storage position): a band whose column and row have different
    ranks, whose column is over an index set its row is not, or whose column
    is over the row's index sets in another order is ``"general"`` (by
    position when either variable's names are not known). A lone ``str``
    ``cell_axes`` is one axis."""
    st = _detect_structure(entries, sv)
    if st in ("empty", "block_diagonal"):
        return st
    cell_axes = cell_axes_list(cell_axes)
    if cell_axes is not None and cell_blocks(entries, sv, cell_axes) is not None:
        return "cell_block"
    if column_axis is None:
        return st
    return st if column_lines(entries, sv, str(column_axis)) is None else "column_block"


def cell_axes_list(cell_axes) -> list[str] | None:
    """The caller's cell axes as index-set names: a lone name (a ``str``) is
    one axis, not a sequence of one-character names (structure.jl
    ``_cell_axes_list``)."""
    if cell_axes is None:
        return None
    if isinstance(cell_axes, str):
        return [cell_axes]
    return [str(a) for a in cell_axes]


def cell_blocks(entries: list[JacEntry], sv: SysView, cell_axes) -> dict | None:
    """The per-cell block structure of a state Jacobian whose couplings all
    stay inside one cell of the index sets ``cell_axes``, or ``None`` when one
    does not (structure.jl ``cell_blocks``): every row and column variable is
    an array whose shape holds each cell axis exactly once, and at every band
    point the column's index on each cell axis (paired by index-set name)
    equals the row's; a point whose column falls outside its variable's
    extent is a ghost read. Returns a dict with ``cell_axes``,
    ``block_vars`` (sorted), ``elements`` (the block's rows in natural order:
    each variable in name order, its elements along its other axes
    column-major, as ``(variable, [1-based indices])``) and ``pattern`` (the
    union over cells of the block pattern, diagonal included)."""
    from .emit import foreach_point

    axes = cell_axes_list(cell_axes)
    if not entries or not axes or len(set(axes)) != len(axes):
        return None
    sizes = ctx_of(sv).index_sets
    vars = sorted({en.u for en in entries} | {en.v for en in entries})
    info = {}
    elements: list = []
    for v in vars:
        var = sv.variables.get(v)
        s = None if var is None else getattr(var, "shape", None)
        if not s:
            return None
        sh = [str(x) for x in s]
        pos = []
        for a in axes:
            k = [d for d, x in enumerate(sh) if x == a]
            if len(k) != 1:
                return None
            pos.append(k[0])
        other = [d for d in range(len(sh)) if d not in pos]
        ext = [sizes[sh[d]] for d in other]
        info[v] = {
            "pos": pos,
            "other": other,
            "ext": ext,
            "full": [sizes[x] for x in sh],
            "off": len(elements),
        }
        n = 1
        for x in ext:
            n *= x
        for k in range(n):
            rem, idx = k, []
            for x in ext:
                idx.append(rem % x + 1)
                rem //= x
            elements.append((v, idx))
    ns = len(elements)
    P = [[i == j for j in range(ns)] for i in range(ns)]

    def local(var, idx):
        k, stride = 0, 1
        for d, x in zip(var["other"], var["ext"], strict=True):
            k += (idx[d] - 1) * stride
            stride *= x
        return var["off"] + k

    for en in entries:
        iu, iv = info[en.u], info[en.v]
        for cell, _, _, cidx in foreach_point(en.band):
            if len(cidx) != len(iv["full"]):
                return None
            if any(not (1 <= cidx[d] <= iv["full"][d]) for d in range(len(cidx))):
                continue  # ghost read
            for k in range(len(axes)):
                if cidx[iv["pos"][k]] != cell[iu["pos"][k]]:
                    return None
            P[local(iu, cell)][local(iv, cidx)] = True
    return {"cell_axes": axes, "block_vars": vars, "elements": elements, "pattern": P}


def column_lines(entries: list[JacEntry], sv: SysView, axis: str) -> dict | None:
    """The per-line structure of a state Jacobian whose couplings all stay
    within one line along index set ``axis``, or ``None`` when one does not
    (structure.jl ``column_lines``): every band couples two array states of
    one common shape that holds ``axis`` once, and at every band point the
    column equals the row in every other dimension. A point reading past the
    axis is a ghost read and does not count. Returns a dict with ``axis``,
    ``dim`` (1-based), ``line_length``, ``block_vars``, ``kl`` / ``ku`` (the
    largest distances below / above the diagonal any point reaches along the
    axis) and ``pattern`` — ``"dense"``, ``"lower"``, ``"upper"``,
    ``"banded"`` or ``"diag_plus_rank1"`` — read off the exact band points,
    never probed, with ``rank1`` the rank-one terms of a
    ``"diag_plus_rank1"`` line block (spec §6; empty otherwise)."""
    from .emit import foreach_point

    if not entries:
        return None
    shape = None
    for en in entries:
        for n in (en.u, en.v):
            var = sv.variables.get(n)
            s = None if var is None else getattr(var, "shape", None)
            if not s:
                return None
            sh = [str(x) for x in s]
            if shape is None:
                shape = sh
            if sh != shape:
                return None
    dims = [k for k, x in enumerate(shape) if x == axis]
    if len(dims) != 1:
        return None
    dim = dims[0]
    N = ctx_of(sv).index_sets[axis]
    kl = ku = 0
    # per entry: whether every point sits at distance 0 on the axis
    diag = [True] * len(entries)
    for k, en in enumerate(entries):
        for cell, _, _, cidx in foreach_point(en.band):
            for d in range(len(cidx)):
                if d != dim and cidx[d] != cell[d]:
                    return None
            if not (1 <= cidx[dim] <= N):
                continue  # ghost read
            off = cidx[dim] - cell[dim]
            kl = max(kl, -off)
            ku = max(ku, off)
            diag[k] = diag[k] and off == 0
    # every entry at distance 0 or a rank-one term, and at least one of
    # those: D + Σ u vᵀ per line, for a Sherman–Morrison solve
    rank1 = []
    lowrank = (kl, ku) != (0, 0)
    for k, en in enumerate(entries):
        if diag[k]:
            continue
        t = _rank1_term(en.band, dim)
        if t is None:
            lowrank = False
        else:
            rank1.append(
                {"entry": k + 1, "u": ser_expr(canon_lits(t[0])), "v": ser_expr(canon_lits(t[1]))}
            )
    if not lowrank:
        rank1 = []
    if rank1:
        pattern = "diag_plus_rank1"
    elif kl == N - 1 and ku == N - 1:
        pattern = "dense"
    elif ku == 0 and kl == N - 1:
        pattern = "lower"
    elif kl == 0 and ku == N - 1:
        pattern = "upper"
    else:
        pattern = "banded"
    return {
        "axis": axis,
        "dim": dim + 1,
        "line_length": N,
        "block_vars": sorted({en.u for en in entries} | {en.v for en in entries}),
        "kl": kl,
        "ku": ku,
        "pattern": pattern,
        "rank1": rank1,
    }


def _factors(e, num: list, den: list) -> None:
    # The factors of a product: numerator and denominator factors, `*` and
    # `/` flattened and a unary minus taken out as -1.
    if isinstance(e, ExprNode):
        if e.op == "*":
            for a in e.args:
                _factors(a, num, den)
            return
        if e.op == "/" and len(e.args) == 2:
            _factors(e.args[0], num, den)
            _factors(e.args[1], den, num)
            return
        if e.op == "-" and len(e.args) == 1:
            num.append(-1)
            _factors(e.args[0], num, den)
            return
    num.append(e)


def _split_rank1(coef, kap: str, rows: set):
    # `coef = u · v` with `u` free of the contracted name `kap` and `v` free
    # of every row name, or None when a factor reads both.
    from .recurrence import free_vars

    num: list = []
    den: list = []
    _factors(coef, num, den)
    un, ud, vn, vd = [], [], [], []
    for fs, is_num in ((num, True), (den, False)):
        for f in fs:
            fv = free_vars(f)
            k, r = kap in fv, bool(fv & rows)
            if k and r:
                return None
            if k:
                (vn if is_num else vd).append(f)
            else:
                (un if is_num else ud).append(f)

    def build(n, d):
        n = mul(*n)
        return n if not d else op("/", n, mul(*d))

    return build(un, ud), build(vn, vd)


def _rank1_term(b, dim: int):
    # A band whose every point reads one other cell of its row's line — the
    # line position being its one contracted name, under no `where` — with
    # a coefficient that splits as `u(row) · v(κ)`: a rank-one term.
    # A structure-only band (coef None, spec §2.1) has no coefficient to
    # split, so it is no rank-one term.
    if b.coef is None or b.guard is not None or len(b.contracted) != 1:
        return None
    kap = b.contracted[0][0]
    if not (dim < len(b.cidx) and isinstance(b.cidx[dim], str) and b.cidx[dim] == kap):
        return None
    return _split_rank1(b.coef, kap, {r for r in b.ridx if isinstance(r, str)})


def _positional(ctx, u: str, v: str) -> bool:
    """Whether column variable ``v`` lies over row variable ``u``'s index
    sets in the same order, so that pairing their dimensions by position
    (what ``"block_diagonal"`` and ``"banded"``, and the lane-block scatter
    plan and the block-diagonal LU plan that read them, do) is pairing them
    by index-set name; ``True`` when either variable's index-set names are
    not known (``ctx.axes``) (structure.jl ``_positional``)."""
    axu, axv = ctx.axes.get(u), ctx.axes.get(v)
    return axu is None or axv is None or list(axu) == list(axv)


def _detect_structure(entries: list[JacEntry], sv: SysView) -> str:
    if not entries:
        return "empty"
    ctx = ctx_of(sv)
    seen_affine = False
    for en in entries:
        rowarr = en.u in ctx.shapes
        colarr = en.v in ctx.shapes
        if rowarr != colarr:
            return "general"
        b = en.band
        if not b.cidx:
            continue
        # a column of another rank than its row is not the row in every
        # dimension, whatever the shared dimensions say
        if len(b.cidx) != len(b.ridx):
            return "general"
        # the classes are positional (their consumers pair row and column
        # cells by storage position): a column over other index sets than
        # its row, or over them in another order, is not the row
        if not _positional(ctx, en.u, en.v):
            return "general"
        per = [_offset_class(c, r) for c, r in zip(b.cidx, b.ridx, strict=True)]
        if any(p in ("other", "fixed") for p in per):
            return "general"
        if any(p == "affine" for p in per):
            seen_affine = True
    return "banded" if seen_affine else "block_diagonal"
