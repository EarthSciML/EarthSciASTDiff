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

Bottom-up; nodes are rebuilt only where something changed. Structural
equality is the full serialized form ([`skey`](@ref)), memoized on object
identity — derivative trees share subtrees heavily, so the memo makes the
pass near-linear in practice.
"""
simplify_branches(e::ASTExpr)::ASTExpr = _sbr(e, IdDict{Any,String}())

_skc(e::ASTExpr, cache::IdDict{Any,String}) = get!(() -> skey(e), cache, e)

function _sbr(e::ASTExpr, cache::IdDict{Any,String})::ASTExpr
    e isa OpExpr || return e
    e2 = map_children(x -> _sbr(x, cache), e)
    (e2.op == "ifelse" && length(e2.args) == 3) || return e2
    c, a, b = e2.args
    if c isa NumExpr || c isa IntExpr
        return iszero_lit(c) ? b : a
    end
    ck = _skc(c, cache)
    a2 = _assume(a, ck, true, cache)
    b2 = _assume(b, ck, false, cache)
    _skc(a2, cache) == _skc(b2, cache) && return a2
    (a2 === a && b2 === b) ? e2 : op("ifelse", c, a2, b2)
end

# Rewrite `e` under the assumption that the condition with key `ck` has the
# truth value `val`: every `ifelse` on that same condition collapses to the
# corresponding branch.
function _assume(e::ASTExpr, ck::String, val::Bool, cache)::ASTExpr
    e isa OpExpr || return e
    if e.op == "ifelse" && length(e.args) == 3 && _skc(e.args[1], cache) == ck
        return _assume(e.args[val ? 2 : 3], ck, val, cache)
    end
    return map_children(x -> _assume(x, ck, val, cache), e)
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
