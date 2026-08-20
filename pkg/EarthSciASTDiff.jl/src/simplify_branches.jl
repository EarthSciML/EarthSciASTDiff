# Branch-aware simplification — the `ifelse` half of the coefficient-size
# control for limiter-heavy schemes (README roadmap). `EarthSciAST.simplify`
# folds constants and algebraic identities but treats `ifelse` as an opaque
# operator, so two dominant swell sources survive it untouched:
#
#   1. the min/max left-fold derivative rule keeps the FULL condition tree
#      even when both branch derivatives agree (often both a literal, e.g.
#      the two branches of d(min(2u,3u))/dv for an unrelated v);
#   2. `index(makearray…)` region lowering nests region-membership `ifelse`
#      chains, and the product rule replicates them — along one path the
#      same condition then appears many times, each occurrence carrying a
#      dead (path-inconsistent) branch.
#
# Both are exact, point-independent rewrites: an ESM expression is pure, so
# within one evaluation a repeated condition subtree has one value, and an
# `ifelse` whose branches are structurally equal is that branch. Dropping a
# dead branch can only REMOVE structurally unreachable sites, so the global
# (branch-union) sparsity pattern remains valid and point-independent.

"""
    simplify_branches(e::ASTExpr) -> ASTExpr

Conditional-specific simplification, complementing `EarthSciAST.simplify`:

  * `ifelse(lit, a, b)` → `a` (lit ≠ 0) or `b` (lit == 0);
  * `ifelse(c, a, a)`   → `a` when the branches are structurally equal;
  * same-condition pruning along a path: inside the true branch of
    `ifelse(c, …)` any nested `ifelse(c, a₂, b₂)` collapses to `a₂`, inside
    the false branch to `b₂` (conditions compared structurally).

Bottom-up; nodes are rebuilt only where something changed.

STRUCTURAL EQUALITY, AND ITS COST. Conditions and branches are compared by
structure, and the pass asks for that comparison at every `ifelse` node it
meets. Doing it with [`skey`](@ref) — a full re-serialization of the subtree
— makes the pass Θ(n·depth): each of the `k` levels of an `index(makearray…)`
region chain re-serializes the whole chain below it. On a flattened document
that chain has one level per grid cell, so the pass does not finish. The
comparisons therefore go through a [`StructIndex`](@ref) (hashcons.jl):
bottom-up interning keys a node from its own shell plus its children's ids,
Θ(n) for the tree and O(1) per compare, and the id memo is keyed on node
identity so the identity-preserving rewrites below re-key nothing they did
not change. Same equality relation as `skey`, so the pass is unchanged.

The second Θ(n·depth) term is `_assume`, which re-descends a branch looking
for a condition that is usually not there at all. A one-pass census of how
many `ifelse` nodes carry each condition (`_count_conditions!`) decides that
in O(1) at the branch, so the chain costs Θ(n) instead of Θ(n·k); see `_sbr`.
"""
function simplify_branches(e::ASTExpr)::ASTExpr
    cx = _SbrCtx()
    _count_conditions!(cx, e)
    return _sbr(e, cx)
end

struct _SbrCtx
    ix::StructIndex
    counts::Dict{Int,Int}     # condition id → how many `ifelse` nodes carry it
end
_SbrCtx() = _SbrCtx(StructIndex(), Dict{Int,Int}())

_skc(e::ASTExpr, cx::_SbrCtx) = sid(cx.ix, e)

# One pass over the INPUT tree recording, per condition, how many `ifelse`
# nodes test it. See `_sbr` for what the count buys; `foreach_subexpr_once`
# both keeps the pass linear on a shared-subtree DAG and keeps the count
# SOUND for that use (a count of 1 means one such node exists anywhere, so
# no second one can be nested under it, however many parents reach it).
function _count_conditions!(cx::_SbrCtx, e::ASTExpr)
    EarthSciAST.foreach_subexpr_once(e) do x
        if x isa OpExpr && x.op == "ifelse" && length(x.args) == 3
            k = _skc(x.args[1], cx)
            cx.counts[k] = get(cx.counts, k, 0) + 1
        end
    end
    return cx
end

function _sbr(e::ASTExpr, cx::_SbrCtx)::ASTExpr
    e isa OpExpr || return e
    e2 = map_children(x -> _sbr(x, cx), e)
    (e2.op == "ifelse" && length(e2.args) == 3) || return e2
    c, a, b = e2.args
    if c isa NumExpr || c isa IntExpr
        return iszero_lit(c) ? b : a
    end
    ck = _skc(c, cx)
    # SAME-CONDITION PRUNING, SKIPPED WHEN THERE IS NOTHING TO PRUNE. `_assume`
    # re-descends a whole branch hunting for a nested test of this condition.
    # A `k`-deep `ifelse` chain therefore costs a full sub-chain walk at every
    # level — Θ(n·k) — and `index(makearray…)` lowering (inline.jl) builds
    # exactly that chain with ONE LEVEL PER REGION, i.e. one per grid cell on
    # a flattened document. The condition census says when the descent cannot
    # find anything: a condition carried by a single `ifelse` in the whole
    # input has no nested twin, and rewriting only ever DELETES `ifelse`
    # nodes, so a count of 1 on the input is still 1 here. Result identical,
    # the walk skipped. `2` is the "not in the census, assume shared" default.
    if get(cx.counts, _skc(e.args[1], cx), 2) == 1
        a2, b2 = a, b
    else
        a2 = _assume(a, ck, true, cx)
        b2 = _assume(b, ck, false, cx)
    end
    _skc(a2, cx) == _skc(b2, cx) && return a2
    (a2 === a && b2 === b) ? e2 : op("ifelse", c, a2, b2)
end

# Rewrite `e` under the assumption that the condition with id `ck` has the
# truth value `val`: every `ifelse` on that same condition collapses to the
# corresponding branch.
function _assume(e::ASTExpr, ck::Int, val::Bool, cx::_SbrCtx)::ASTExpr
    e isa OpExpr || return e
    if e.op == "ifelse" && length(e.args) == 3 && _skc(e.args[1], cx) == ck
        return _assume(e.args[val ? 2 : 3], ck, val, cx)
    end
    return map_children(x -> _assume(x, ck, val, cx), e)
end

# Wire-canonical literal normalization. On parse, EarthSciAST turns an
# integral Int64-representable float literal into an INTEGER literal
# (CONFORMANCE_SPEC §5.5.3.1 rule 1), while `simplify`'s mixed int/float
# constant folds produce float literals (`2 * 1.0` → `NumExpr(2.0)`). Keep
# in-memory coefficients in the parse-canonical form, or emit→parse
# round-trips (and cross-binding goldens) change the structural key.
_canon_lit(e::ASTExpr) = e
_canon_lit(e::NumExpr) =
    (isfinite(e.value) && isinteger(e.value) &&
     typemin(Int64) <= e.value <= typemax(Int64) &&
     Float64(Int64(e.value)) == e.value) ? IntExpr(Int64(e.value)) : e
function _canon_lits(e::ASTExpr)::ASTExpr
    e isa OpExpr && return map_children(_canon_lits, e)
    return _canon_lit(e)
end

# The coefficient simplifier used throughout the band calculus: algebraic
# pass, branch pass, a final algebraic pass to fold whatever the branch
# collapse exposed (e.g. an `ifelse` that became a literal inside a sum),
# then literal normalization to the wire-canonical form.
_simp(e::ASTExpr)::ASTExpr = _canon_lits(simplify(simplify_branches(simplify(e))))
