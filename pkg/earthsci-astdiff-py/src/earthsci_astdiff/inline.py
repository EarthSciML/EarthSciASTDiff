"""Observed inlining and `index`-of-array-expression β-reduction (inline.jl)."""

from __future__ import annotations

import re
from dataclasses import replace

from earthsci_ast.esm_types import Expr, ExprNode
from earthsci_ast.expr_walk import map_children
from earthsci_ast.expression import free_variables

from .expr_helpers import add, op, region_bound, skey

MAX_INLINE_PASSES = 64


class InlineError(Exception):
    pass


def inline_observed(
    e: Expr,
    obs: dict[str, Expr],
    obs_names: list | None = None,
    shapes: dict[str, list[int]] | None = None,
    rank_obs: dict[str, Expr] | None = None,
    scalars: set[str] | frozenset[str] = frozenset(),
    axes: dict[str, list[str]] | None = None,
) -> Expr:
    """Inline observed reads, lower ``Pre(a)`` to ``a`` (spec §3.2),
    β-reduce `index`-of-`faq`, lower
    `index`-of-`makearray` to region membership `ifelse`. Fixpoint-iterated.
    Every substitution is capture-avoiding, and a faq node binding a name an
    enclosing faq already binds is renamed apart. ``obs_names`` caches the
    names ``obs``'s definitions use (a one-element list, ``[None]`` until
    filled); pass one to every call against the same ``obs``. ``shapes``
    (variable → declared extents, the band context's) gives the rank of an
    array variable used as a `makearray` region value; a variable it does not
    name (nor an observed definition) is read at the region's free rank.
    ``rank_obs`` (default ``obs``) holds every observed definition, including
    those a pruned ``obs`` leaves out, so that an elementwise observed read at
    a cell indexes each array operand, inlined or not. ``scalars`` names the
    variables declared without a shape that are not observed (scalar states
    and parameters); ``t`` is always scalar. Any other name in such a
    definition has no known rank and is refused. ``axes`` (variable → its
    declared index-set names) aligns the operands of an elementwise observed
    read at a cell by name (esm-spec §4.3.4): an operand over a subset of the
    observed's index sets is read at those of the cell's indices and repeats
    along the others."""
    if not obs and not _needs_inline(e):
        return e
    fresh = _Fresh(e, obs, [None] if obs_names is None else obs_names, shapes or {})
    # every observed definition, the pruned-away ones too: an operand left an
    # opaque read still has a rank (``_cell_rank``)
    fresh.rank_obs = obs if rank_obs is None else rank_obs
    fresh.scalars = scalars
    fresh.axes = axes or {}
    for _ in range(MAX_INLINE_PASSES):
        e2 = _inline_once(e, obs, fresh, frozenset())
        if skey(e2) == skey(e):
            return e2
        e = e2
    raise InlineError(f"no fixpoint after {MAX_INLINE_PASSES} passes (cyclic observed?)")


_GEN = re.compile(r"^_c\d+_")


class _Fresh:
    """Fresh names for renamed-apart binders (inline.jl ``_Fresh``):
    ``_c<n>_<name>``, the smallest ``n`` clear of every name the tree and the
    observed definitions use and of the caller's ``avoid`` set. Deterministic
    in (name, avoid), so copies of one sum renamed in the same scope stay
    structurally equal. The taken sets are collected on the first rename."""

    def __init__(
        self,
        root: Expr,
        obs: dict[str, Expr],
        obs_names: list,
        shapes: dict[str, list[int]] | None = None,
    ):
        self.root = root
        self.obs = obs
        self.obs_names = obs_names
        self.shapes = shapes or {}
        self.taken: set[str] | None = None
        self.ranks: dict[str, int] = {}  # observed name → its rank at a cell
        # id(node) → (node, its rank at a cell); the node is held so its id
        # is not reused
        self.op_ranks: dict[int, tuple] = {}
        self.rank_obs: dict[str, Expr] = obs
        self.scalars: set[str] | frozenset[str] = frozenset()

    def __call__(self, name: str, avoid) -> str:
        if self.taken is None:
            if self.obs_names[0] is None:
                t: set[str] = set()
                for d in self.obs.values():
                    all_names(d, t)
                self.obs_names[0] = t
            self.taken = all_names(self.root)
        base = _GEN.sub("", name)
        n = 0
        while True:
            n += 1
            c = f"_c{n}_{base}"
            if c not in self.taken and c not in self.obs_names[0] and c not in avoid:
                return c


def binders(e: ExprNode) -> list[str]:
    """The index names a faq node binds: its ranges (sorted, so renaming is
    deterministic across bindings) and its named output indices."""
    b = sorted(str(n) for n in (e.ranges or {}))
    for x in e.output_idx or []:
        if isinstance(x, str) and x not in b:
            b.append(x)
    return b


def all_names(e: Expr, acc: set[str] | None = None) -> set[str]:
    """Every variable name and every faq-bound index name in ``e``."""
    if acc is None:
        acc = set()
    if isinstance(e, str):
        acc.add(e)
    elif isinstance(e, ExprNode):
        if e.op == "faq":
            acc.update(binders(e))
        map_children(e, lambda x: (all_names(x, acc), x)[1])
    return acc


def _rename_binders(e: ExprNode, names, fresh: _Fresh, avoid) -> ExprNode:
    """Rename the binders ``names`` of faq node ``e`` to fresh ones, clear of
    ``avoid`` and of every name under ``e``."""
    if e.join is not None:
        raise InlineError("cannot rename the indices of a joined faq node (join keys name them)")
    av = all_names(e, set(avoid))
    ren: dict[str, Expr] = {}
    for n in names:
        f = fresh(n, av)
        av.add(f)
        ren[n] = f
    fv = set(ren.values())
    return replace(
        e,
        ranges=None
        if e.ranges is None
        else {ren.get(str(n), str(n)): r for n, r in e.ranges.items()},
        output_idx=None
        if e.output_idx is None
        else [ren.get(x, x) if isinstance(x, str) else x for x in e.output_idx],
        expr=_csubst(e.expr, ren, fv, fresh),
        filter=None if e.filter is None else _csubst(e.filter, ren, fv, fresh),
    )


def _csubst(e: Expr, m: dict, fv: set, fresh: _Fresh) -> Expr:
    """CAPTURE-AVOIDING substitution (``substitute`` is not): a faq node
    shadows the names it binds, and one binding a name free in a replacement
    (``fv``) is renamed apart before the substitution enters it."""
    if not m:
        return e
    if isinstance(e, str):
        return m.get(e, e)
    if not isinstance(e, ExprNode):
        return e
    if e.op == "faq":
        b = binders(e)
        if any(n in m for n in b):
            m = {k: v for k, v in m.items() if k not in b}
            if not m:
                return e
        clash = [n for n in b if n in fv]
        if clash:
            e = _rename_binders(e, clash, fresh, fv)
    return map_children(e, lambda x: _csubst(x, m, fv, fresh))


def _beta_faq(de: ExprNode, sub: list[Expr], fresh: _Fresh, scope) -> Expr:
    """β-reduce a read ``F[sub…]`` of a faq-defined array F (inline.jl
    ``_beta_faq``): output indices take the subscripts; any other index F's
    ranges bind is CONTRACTED, so the read is the scalar faq summing over it
    (``reduce``/``filter``/``semiring`` carried). Capture-avoiding: a
    contracted name the subscripts read, or one shadowing an index bound
    around the reading site (``scope``), is renamed apart."""
    oidx = [str(x) for x in (de.output_idx or [])]
    if len(sub) != len(oidx):
        raise InlineError(f"index arity {len(sub)} ≠ rank {len(oidx)} of a faq read")
    ranges = de.ranges or {}
    cnames = sorted(str(n) for n in ranges if str(n) not in oidx)
    subst: dict[str, Expr] = {n: sub[k] for k, n in enumerate(oidx)}
    fv: set[str] = set()
    for x in sub:
        fv |= free_variables(x)
    if not cnames and de.filter is None:
        return _csubst(de.expr, subst, fv, fresh)
    if de.join is not None:
        raise InlineError(
            "cannot inline a read of a joined faq definition (join keys name its indices)"
        )
    newranges = {}
    av = fv | set(scope) | set(cnames)
    for n in cnames:
        if n in fv or n in scope:
            f = fresh(n, av)
            av.add(f)
            subst[n] = f
            newranges[f] = ranges[n]
        else:
            newranges[n] = ranges[n]
    fv2 = fv | {x for x in subst.values() if isinstance(x, str)}
    return replace(
        de,
        output_idx=[],
        ranges=newranges,
        expr=_csubst(de.expr, subst, fv2, fresh),
        filter=None if de.filter is None else _csubst(de.filter, subst, fv2, fresh),
    )


def _needs_inline(e: Expr, scope=frozenset()) -> bool:
    """Whether a pass could change ``e`` with no observed to inline: a
    ``Pre`` to lower, an `index`-of-array-expression, or a faq node shadowing
    an enclosing index."""
    if not isinstance(e, ExprNode):
        return False
    if e.op == "Pre":
        return True
    if e.op == "index" and isinstance(e.args[0], ExprNode):
        return True
    if e.op == "faq":
        b = binders(e)
        if any(n in scope for n in b):
            return True
        scope = scope | set(b)
    found = False

    def visit(c: Expr) -> Expr:
        nonlocal found
        if not found:
            found = _needs_inline(c, scope)
        return c

    map_children(e, visit)
    return found


def _range_origin(r) -> int:
    """Where a faq range starts: an interval's lower bound, 1 for an
    index-set range."""
    if isinstance(r, (list, tuple)):
        return region_bound(r[0])
    return 1


def _value_rank(val: Expr, fresh: _Fresh, scope) -> tuple[int, list[int]] | None:
    """The rank of an array-valued `makearray` region value and, per dim it
    spans, the index at which the value starts (its ORIGIN): a faq node's own
    range start for each output index, 1 for any other array. ``None`` when
    the rank cannot be read here (an `index` value, a variable of no known
    shape): such a value is read at the region's free rank."""
    if isinstance(val, ExprNode) and val.op == "faq":
        vr = val.ranges or {}
        oidx = [str(n) for n in val.output_idx or []]
        return len(oidx), [_range_origin(vr[n]) if n in vr else 1 for n in oidx]
    if isinstance(val, ExprNode) and val.op == "makearray":
        return (len(val.regions[0]), [1] * len(val.regions[0])) if val.regions else None
    if isinstance(val, str) and val not in scope:
        if val in fresh.shapes:
            n = len(fresh.shapes[val])
            return n, [1] * n
        d = fresh.obs.get(val)
        if isinstance(d, ExprNode) and d.op == "faq":
            n = len(d.output_idx or [])
            return n, [1] * n
        if isinstance(d, ExprNode) and d.op == "makearray" and d.regions:
            n = len(d.regions[0])
            return n, [1] * n
    return None


def _index_makearray(
    ma: ExprNode, idx: list[Expr], obs: dict[str, Expr], fresh: _Fresh, scope
) -> Expr:
    """index(makearray(regions, values), e…) → nested region ifelse.
    Fold from the FIRST region outward so later regions override (§4.3.2)."""
    result: Expr = 0
    for region, val in zip(ma.regions, ma.values, strict=True):
        if len(region) != len(idx):
            raise InlineError(f"makearray region rank {len(region)} ≠ index arity {len(idx)}")
        conds: list[Expr] = []
        los: list[int] = []
        nonsing: list[int] = []
        for k, r in enumerate(region):
            conds.append(op(">=", idx[k], r[0]))
            conds.append(op("<=", idx[k], r[1]))
            los.append(region_bound(r[0]))
            if region_bound(r[1]) > los[-1]:
                nonsing.append(k)
        if isinstance(val, str) or (
            isinstance(val, ExprNode) and val.op in ("faq", "makearray", "index")
        ):
            # Two legal value shapes (esm-spec §4.3.2, the pair
            # `_bands_makearray` tells apart with `val_fullrank`): a FULL-RANK
            # value spans every region dim, singleton dims included (a 3-D
            # field's boundary face); a REDUCED-RANK one spans only the
            # non-singleton dims. The value fills the region by position: its
            # element at index `origin + p` lands on region index `lo + p`,
            # so a dim it spans is read at `idx − lo + origin`, the shift
            # `_bands_makearray` applies to the value's bands.
            rank, origins = _value_rank(val, fresh, scope) or (len(nonsing), [1] * len(nonsing))
            if rank == len(idx):
                dims = list(range(len(idx)))
            elif rank == len(nonsing):
                dims = nonsing
            else:
                raise InlineError(
                    f"makearray value rank {rank} matches neither the region rank "
                    f"{len(idx)} nor the region free rank {len(nonsing)}"
                )
            tgt = [add(idx[k], o - los[k]) for k, o in zip(dims, origins, strict=True)]
            if isinstance(val, ExprNode) and val.op == "faq":
                v = _beta_faq(val, tgt, fresh, scope)
            else:
                v = _inline_once(op("index", val, *tgt), obs, fresh, scope)
        else:
            v = val  # scalar broadcast over the region
        cond = conds[0] if len(conds) == 1 else op("and", *conds)
        result = op("ifelse", cond, v, result)
    return result


def _inline_once(e: Expr, obs: dict[str, Expr], fresh: _Fresh, scope: frozenset) -> Expr:
    """One pass; ``scope`` is the index names the enclosing faq nodes bind."""
    if isinstance(e, str):
        # A name an enclosing faq binds is that index, not an observed read.
        if e in scope:
            return e
        return obs.get(e, e)
    if not isinstance(e, ExprNode):
        return e
    if e.op == "Pre":
        # `Pre(a)` in a right-hand side is `a` (spec §3.2)
        return _inline_once(e.args[0], obs, fresh, scope)

    def args() -> list[Expr]:
        return [_inline_once(x, obs, fresh, scope) for x in e.args[1:]]

    if e.op == "index" and isinstance(e.args[0], ExprNode):
        t = e.args[0]
        if t.op == "makearray":
            return _index_makearray(t, args(), obs, fresh, scope)
        if t.op == "faq":
            return _inline_once(_beta_faq(t, args(), fresh, scope), obs, fresh, scope)
    if e.op == "faq":
        # A faq node binding a name already bound around it is renamed apart:
        # the band calculus reads a free index in a derivative by name.
        b = binders(e)
        clash = [n for n in b if n in scope]
        if clash:
            e = _rename_binders(e, clash, fresh, scope)
            b = binders(e)
        inner = scope | frozenset(b)
        return map_children(e, lambda x: _inline_once(x, obs, fresh, inner))
    if (
        e.op == "index"
        and isinstance(e.args[0], str)
        and e.args[0] in obs
        and e.args[0] not in scope
    ):
        name = e.args[0]
        de = obs[name]
        if isinstance(de, ExprNode) and de.op == "faq":
            oidx = de.output_idx or []
            if len(oidx) != len(e.args) - 1:
                raise InlineError(
                    f"index arity {len(e.args) - 1} ≠ rank {len(oidx)} of observed `{name}`"
                )
            return _beta_faq(de, args(), fresh, scope)
        if isinstance(de, ExprNode) and de.op == "makearray":
            return _index_makearray(de, args(), obs, fresh, scope)
        if isinstance(de, ExprNode) and de.op == "const":
            return op("index", de, *args())
        if isinstance(de, (str, ExprNode)):
            # An observed declared without a shape whose definition evaluates
            # to an array (esm-spec §4.3.4), or a shaped observed defined by
            # an elementwise expression of arrays: its element is the
            # definition with each array operand read at the same cell.
            return _gather_cell(de, args(), fresh, scope, name)
        return de  # a number: the same value at every cell
    return map_children(e, lambda x: _inline_once(x, obs, fresh, scope))


def subst_free(e: Expr, m: dict) -> Expr:
    """Substitution that does not descend into a faq for the names it binds
    (bands.jl ``_subst_free``)."""
    if isinstance(e, str):
        return m.get(e, e)
    if not isinstance(e, ExprNode):
        return e
    if e.op == "faq":
        bound = binders(e)
        if any(n in m for n in bound):
            m = {k: v for k, v in m.items() if k not in bound}
            if not m:
                return e
    return map_children(e, lambda x: subst_free(x, m))


# ── elementwise observeds read at a cell (inline.jl ``_gather_cell``) ────────
#
# ``index(x, i…)`` of an observed ``x`` whose definition is an elementwise
# expression (arithmetic, elementary, comparison, logical, ``ifelse``) of
# arrays and scalars: the definition with every array operand read at ``i…``
# and every scalar operand unchanged (esm-spec §4.3.4, §9.6.11). The array
# operands must all have the rank of the read. An array operand that is a
# variable, a ``faq`` or a ``makearray`` is read whole at the cell
# (``index(operand, i…)``, reduced by the next pass); an ``index`` inside the
# definition must select a scalar. Anything this cannot decide is refused,
# never guessed: a name whose rank is not known (neither declared, an
# observed, nor ``t``), an operand whose rank differs from the read's, a
# partial ``index``, a literal array or shape op as an operand, or a
# definition reading a name an index bound around the read site would
# capture. Ranks are computed on the closed definition (no enclosing scope),
# so they are memoized per observed name and per node. (This binding's
# loader expands operator references, so none reach here.)

_NOT_ELEMENTWISE = ("reshape", "transpose", "concat", "broadcast")


def _cell_rank(e: Expr, fresh: _Fresh, seen: set) -> int:
    """The rank of ``e`` in an elementwise position: 0 for a scalar."""
    if isinstance(e, str):
        if e == "t" or e in fresh.scalars:
            return 0
        if e in fresh.shapes:
            return len(fresh.shapes[e])
        if e in fresh.ranks:
            return fresh.ranks[e]
        d = fresh.rank_obs.get(e)
        if d is None:
            raise InlineError(
                f"`{e}`, read in an elementwise observed definition read at a cell, "
                "has no known rank (not a declared variable, nor an observed)"
            )
        if e in seen:
            raise InlineError(f"cyclic observed `{e}`")
        seen.add(e)
        r = _def_rank(d, fresh, seen, e)
        seen.discard(e)
        fresh.ranks[e] = r
        return r
    if isinstance(e, ExprNode):
        hit = fresh.op_ranks.get(id(e))
        if hit is not None and hit[0] is e:
            return hit[1]
        r = _def_rank(e, fresh, seen, None)
        fresh.op_ranks[id(e)] = (e, r)
        return r
    return 0


def _lit_rank(v) -> int:
    """The rank of a literal array value (its nesting depth)."""
    r = 0
    while isinstance(v, list):
        r += 1
        if not v:
            break
        v = v[0]
    return r


def _def_rank(d: Expr, fresh: _Fresh, seen: set, name) -> int:
    what = name if name is not None else "expression"
    if isinstance(d, str):
        return _cell_rank(d, fresh, seen)
    if not isinstance(d, ExprNode):
        return 0
    if d.op == "faq":
        return len(d.output_idx or [])
    if d.op == "makearray":
        if not d.regions:
            raise InlineError("makearray with no region read at a cell")
        return len(d.regions[0])
    if d.op == "index":
        # an `index` in an elementwise definition selects one element: its
        # arity is its operand's rank (a partial index is still an array, and
        # aligning what it leaves by position is not decidable here)
        a = d.args[0]
        if isinstance(a, ExprNode) and a.op == "const":
            r0 = _lit_rank(a.value)
        else:
            r0 = _cell_rank(a, fresh, seen)
        if r0 != len(d.args) - 1:
            raise InlineError(
                f"observed {what} reads `index` with {len(d.args) - 1} indices into an "
                f"operand of rank {r0} in an elementwise definition read at a cell"
            )
        return 0
    if d.op == "const" and not isinstance(d.value, list):
        return 0  # a scalar literal
    if d.op == "const" or d.op in _NOT_ELEMENTWISE:
        raise InlineError(
            f"observed {what} reads `{d.op}` in an elementwise definition read at a cell; "
            "not supported"
        )
    # Operands of different ranks are aligned by index-set name where the
    # definition is read (``_gather_cell``); a node's rank is then its widest
    # operand's. A named observed reaches here only when it is declared
    # without a shape, and its shape is the union of its operands' index sets
    # (esm-spec §4.3.4), which ranks alone cannot give: refused.
    r = 0
    for a in d.args:
        ra = _cell_rank(a, fresh, seen)
        if ra == 0:
            continue
        if r not in (0, ra) and name is not None:
            raise InlineError(
                f"observed {what}, declared without a shape, combines arrays of rank {r} "
                f"and {ra}; declare its shape so its operands can be aligned by index set"
            )
        r = max(r, ra)
    return r


def _aligned_idx(oax: list, tax: list, idx: list, name: str, operand: str) -> list:
    """The cell indices at which an operand over the index sets ``oax`` is
    read inside a read of observed ``name`` over ``tax`` at ``idx`` (esm-spec
    §4.3.4): the k-th dimension of the operand on index set X takes the index
    of the k-th dimension of the observed on X. An index set the observed does
    not have (or has fewer times) is refused."""
    out = []
    for d, x in enumerate(oax):
        k = oax[: d + 1].count(x)
        hits = [i for i, y in enumerate(tax) if y == x]
        if k > len(hits):
            raise InlineError(
                f"observed `{name}` over ({', '.join(tax)}) reads `{operand}` over "
                f"({', '.join(oax)}), whose index set `{x}` it is not shaped over"
            )
        out.append(idx[hits[k - 1]])
    return out


def _gather_cell(de: Expr, idx: list, fresh: _Fresh, scope, name: str) -> Expr:
    """``de`` read at the cell ``idx``, spliced under the indices ``scope``
    binds."""
    from .recurrence import free_vars

    # the definition is closed; a name it reads that an index around the read
    # site binds would be captured by it, and a global cannot be renamed
    cap = sorted(n for n in free_vars(de) if n in scope)
    if cap:
        names = ", ".join(f"`{n}`" for n in cap)
        raise InlineError(
            f"observed `{name}` reads {names}, which an index bound around its read at "
            "a cell would capture"
        )

    tax = getattr(fresh, "axes", {}).get(name)

    def g(e: Expr) -> Expr:
        if not isinstance(e, (str, ExprNode)):
            return e
        r = _cell_rank(e, fresh, set())
        if r == 0:
            return e
        if not (isinstance(e, str) or e.op in ("faq", "makearray")):
            return map_children(e, g)
        oax = getattr(fresh, "axes", {}).get(e) if isinstance(e, str) else None
        if tax is not None and oax is not None:
            return op("index", e, *_aligned_idx(oax, tax, idx, name, e))
        if r != len(idx):
            if r >= len(idx):
                why = ""
            elif tax is None:
                why = (
                    "; the observed is declared without a shape, so the operand cannot be "
                    "aligned by index set"
                )
            else:
                why = " whose index sets are not known, so it cannot be aligned by index set"
            raise InlineError(
                f"observed `{name}`, read with {len(idx)} indices, has an operand of rank {r}{why}"
            )
        return op("index", e, *idx)

    return g(de)
