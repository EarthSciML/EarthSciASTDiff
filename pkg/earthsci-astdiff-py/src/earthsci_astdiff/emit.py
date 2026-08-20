"""§4 band-entry serialization + the structural pattern (emit.jl subset).

``stable_json(serialize_band(e) for e in entries)`` is the cross-language
conformance surface: byte-identical to the Julia reference's goldens."""
from __future__ import annotations

from earthsci_ast.numpy_interpreter import evaluate
from earthsci_ast.parse import _parse_expression as parse_expression

from .bands import Band
from .expr_helpers import ser_expr, stable_json
from .jacobian import JacEntry, jacobian_bands
from .system import ctx_of, sysview


def serialize_band(en: JacEntry) -> dict:
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
    return d


def parse_band(d: dict) -> JacEntry:
    band = Band(
        [(int(r[0]), int(r[1])) for r in d["rows"]],
        [x if isinstance(x, str) else int(x) for x in d["row_idx"]],
        [parse_expression(c) for c in d["col_idx"]],
        parse_expression(d["coef"]),
        [(str(c[0]), int(c[1]), int(c[2])) for c in d.get("contracted", [])],
    )
    return JacEntry(str(d["row"]), str(d["col"]), band)


def _cellname(v: str, cell) -> str:
    return v if not cell else f"{v}[{','.join(str(c) for c in cell)}]"


def _fold_const_gathers(e, bind: dict[str, float]):
    from earthsci_ast.esm_types import ExprNode
    from earthsci_ast.expr_walk import map_children

    if not isinstance(e, ExprNode):
        return e
    if (e.op == "index" and isinstance(e.args[0], ExprNode)
            and e.args[0].op == "const"):
        v = e.args[0].value
        for x in e.args[1:]:
            v = v[_eval_cidx(x, bind) - 1]      # wire tables are 1-based
        return v
    return map_children(e, lambda x: _fold_const_gathers(x, bind))


def _eval_cidx(c, bind: dict[str, float]) -> int:
    return int(round(evaluate(_fold_const_gathers(c, bind), bind)))


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
        b = en.band
        for cell in _product(b.rows):
            for cc in _product([(lo, hi) for _, lo, hi in b.contracted]):
                bind = {r: float(cell[k2]) for k2, r in enumerate(b.ridx)
                        if isinstance(r, str)}
                for k2, (n, _, _) in enumerate(b.contracted):
                    bind[n] = float(cc[k2])
                cidx = [_eval_cidx(c, bind) for c in b.cidx]
                pairs.append((k, _cellname(en.u, cell),
                              _cellname(en.v, cidx)))
    return pairs


def jacobian_pattern(obj, model_name: str | None = None, wrt: str = "states"):
    """The structural sparsity pattern as sorted (rowname, colname) pairs
    plus the entries — dependency-light (no scipy)."""
    from .system import SysView

    sv = obj if isinstance(obj, SysView) else sysview(obj, model_name)
    entries = jacobian_bands(sv, wrt=wrt)
    pairs = sorted({(r, c) for _, r, c in scatter_pairs(sv, entries)})
    return pairs, entries
