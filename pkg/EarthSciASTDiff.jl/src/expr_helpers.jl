# Small expression constructors and the differentiation "site" abstraction.
#
# Everything here works on the public EarthSciAST AST (`NumExpr` / `IntExpr` /
# `VarExpr` / `OpExpr`) through `map_children`, the field-table-generated
# rewrite primitive — children living in `expr_body`, `values`, `ranges`,
# `lower`/`upper`, … are never missed.

lit(x::Integer) = IntExpr(Int64(x))
lit(x::Real) = NumExpr(Float64(x))
op(o, args...; kw...) = OpExpr(String(o), ASTExpr[args...]; kw...)

iszero_lit(e) = (e isa NumExpr && e.value == 0.0) || (e isa IntExpr && e.value == 0)
isone_lit(e) = (e isa NumExpr && e.value == 1.0) || (e isa IntExpr && e.value == 1)

"n-ary `+` that drops literal zeros and collapses to its single argument."
add(args...) = (
    a = filter(!iszero_lit, collect(ASTExpr, args));
    isempty(a) ? lit(0) : length(a) == 1 ? a[1] : op("+", a...)
)

"n-ary `*` that short-circuits on a literal zero and drops literal ones."
mul(args...) = any(iszero_lit, args) ? lit(0) :
    (
        a = filter(!isone_lit, collect(ASTExpr, args));
        isempty(a) ? lit(1) : length(a) == 1 ? a[1] : op("*", a...)
    )

neg(x) = iszero_lit(x) ? lit(0) : op("-", x)

# Term-identity key for SCALAR expressions (RFC §5.4 canonical form). Only
# valid on the scalar core — array ops carry fields `canonical_json` refuses.
ckey(e::ASTExpr) = canonical_json(e)

# Full-structure key for ANY expression (fixpoint checks over trees that may
# contain faq/makearray). Key order is made deterministic by
# `stable_json` (emit.jl), so equal trees always produce equal keys.
skey(e::ASTExpr) = stable_json(EarthSciAST.serialize_expression(e))

# A site's identity key: the canonical JSON, or — for an `index` node whose
# subscripts hold an array op the canonical form refuses (a level search
# `u[κ]` with κ a min-faq, §3.5) — `~` followed by its stable JSON, which
# sorts after every canonical key. The refusal is decided by a field walk
# first (`_canon_refuses`), so `is_site` — which keys every candidate node of
# such a site — raises no exception per node; the `catch` is the backstop
# for a refusal of another kind.
function _site_key(e::ASTExpr)::String
    _canon_refuses(e) && return "~" * skey(e)
    try
        return ckey(e)
    catch err
        err isa EarthSciAST.CanonicalizeError || rethrow()
        return "~" * skey(e)
    end
end

# Does `e` carry, at any node `canonical_json` emits, a field outside the
# canonical node encoding (an array op's `expr_body`, `ranges`, …)?
_canon_refuses(e::ASTExpr) = e isa OpExpr && (
    any(f -> getfield(e, f) !== nothing, EarthSciAST._NON_EMISSIBLE_FIELDS) ||
        any(_canon_refuses, e.args)
)

"""
    Site(expr)

The thing a scalar derivative is taken with respect to: a bare `VarExpr`
(scalar variable) or an `index(v, i…)` node (one symbolic cell of an array
variable). Two nodes denote the same site iff their canonical JSON agrees —
`d/d(u[i+1,j])` treats `index(u, +(i,1), j)` and `index(u, +(1,i), j)` as the
same site because canonicalization sorts commutative arguments.

`name`/`rank` are a PREFILTER cached off `expr`, not extra identity: the site
reads variable `name` at `rank` subscripts (`rank == -1` for a bare
`VarExpr`, `-2` for a shape this fast path does not describe, which then
falls back to the plain key compare). See [`is_site`](@ref) for why.
"""
struct Site
    expr::ASTExpr
    key::String
    name::String
    rank::Int
end
function Site(e::ASTExpr)
    if e isa VarExpr
        return Site(e, _site_key(e), e.name, -1)
    elseif e isa OpExpr && e.op == "index" && !isempty(e.args) &&
            e.args[1] isa VarExpr
        return Site(e, _site_key(e), e.args[1].name, length(e.args) - 1)
    end
    return Site(e, _site_key(e), "", -2)
end

"""
    is_site(e, s::Site) -> Bool

Is `e` the node the site denotes? Canonical-key equality, reached only
through a cheap structural prefilter.

SCALING. `ckey` serializes the whole node, and `occurs`/`dscalar` ask this
question once per node per site — on a coupled document that is millions of
serializations of subtrees whose variable does not even match. The
prefilter decides the answer from the node's own kind, variable name and
subscript arity, which are all O(1) reads, and can only reach `ckey` for a
node that reads the SAME variable at the SAME arity. It never changes the
answer: a `VarExpr`'s canonical JSON is its name and an `index` node's is an
object, so key equality already implies the prefilter's conditions.
"""
function is_site(e::ASTExpr, s::Site)::Bool
    if s.rank == -2                                 # unrecognized site shape:
        return (e isa VarExpr || (e isa OpExpr && e.op == "index")) &&
            _site_key(e) == s.key                # plain key compare
    end
    if e isa VarExpr
        return s.rank == -1 && e.name == s.name
    elseif e isa OpExpr && e.op == "index"
        s.rank >= 0 && length(e.args) == s.rank + 1 || return false
        a1 = e.args[1]
        (a1 isa VarExpr && a1.name == s.name) || return false
        return _site_key(e) == s.key
    end
    return false
end

"""
    occurs(e, s::Site) -> Bool

Structural occurrence test — THE sparsity primitive. Never a symbolic or
numeric zero test: `x - x` occurs, both `ifelse` branches occur. False means
the derivative is a structural zero and differentiation short-circuits
(UFL-style `Zero` propagation).
"""
occurs(e::ASTExpr, s::Site)::Bool = occurs(e, s, IdDict{OpExpr, Bool}())

"""
    occurs(e, s::Site, memo::IdDict{OpExpr,Bool}) -> Bool

As above, answering from `memo` for any `OpExpr` already decided.

SCALING. Occurrence of a fixed site in a fixed node is a property of that
node alone, so it is memoizable — and it MUST be, because the one caller
that matters ([`dscalar`](@ref)) asks it at every node on its way down. With
a per-call scan that is Θ(Σ_v |subtree(v)|) = Θ(n·depth): quadratic on the
deep trees an operator's lowered stencil produces, and half of the whole
ReSEACT system's `jacobian_bands` time. Threading ONE memo through a whole `dscalar`
call decides each node once, Θ(n) per site. Passing a fresh memo is exactly
the old behavior, only cached within the one scan.
"""
function occurs(e::ASTExpr, s::Site, memo::IdDict{OpExpr, Bool})::Bool
    is_site(e, s) && return true
    e isa OpExpr || return false
    cached = get(memo, e, nothing)
    cached === nothing || return cached
    found = false
    map_children(e) do c
        found || (found = occurs(c, s, memo))
        c
    end
    memo[e] = found
    return found
end

# ── every site of one variable at once ───────────────────────────────────────
#
# The band calculus differentiates a body once per site of `v`, and a faq
# body is clipped into one body per region first: on ReSEACT's lowered
# advection operator that is 343 regions × ~16 sites, each body ~4.5×10⁴
# nodes. A per-site occurrence memo walks the whole body once per site, and
# `is_site` re-serializes every `index` node of `v` it meets once per site.
# `_OccIndex` answers both for ALL sites of `v` from one walk: each node gets
# the set of site keys under it (a `BitSet` of key ids), and each `index`
# node of `v` its key, both memoized on node identity. Regions keep the
# subtrees clipping did not touch (`map_children` is identity-preserving),
# so one index serves every region of a body.
#
# Exact: a node's key set holds a site's key iff `occurs` would find it —
# `is_site` on a `VarExpr` compares the name, which for a site of `v` is the
# key compare; on an `index` node of `v` it compares `_site_key`, which is
# what the set records — and `_is_site` repeats `is_site`'s decision with
# the key read from the memo.

"""
    _OccIndex(v)

Occurrence of every site of variable `v`, memoized on node identity, for the
derivative scans of one band computation. See [`_SiteScan`](@ref).
"""
struct _OccIndex
    v::String
    keyid::Dict{String, Int}          # site key → id
    nodekey::IdDict{OpExpr, Int}      # `index` node of `v` → id of its key
    bits::IdDict{OpExpr, BitSet}      # node → ids of the site keys under it
    var::Base.RefValue{BitSet}        # the set of a bare `VarExpr(v)`, once keyed
    deriv::Dict{String, IdDict{OpExpr, ASTExpr}}  # site → node → its derivative
end
_OccIndex(v::String) = _OccIndex(
    v, Dict{String, Int}(), IdDict{OpExpr, Int}(), IdDict{OpExpr, BitSet}(), Ref(BitSet()),
    Dict{String, IdDict{OpExpr, ASTExpr}}()
)

const _NO_SITES = BitSet()            # shared, never mutated

_keyid!(ix::_OccIndex, k::String) = get!(ix.keyid, k, length(ix.keyid) + 1)

_is_v_index(ix::_OccIndex, e::OpExpr) = e.op == "index" && !isempty(e.args) &&
    e.args[1] isa VarExpr && e.args[1].name == ix.v

function _nodekey!(ix::_OccIndex, e::OpExpr)::Int
    k = get(ix.nodekey, e, 0)
    k == 0 || return k
    return ix.nodekey[e] = _keyid!(ix, _site_key(e))
end

# The ids of the site keys of `v` at or under `e`. A stored set is never
# mutated: a node whose sites all come from one child shares that child's set.
function _site_bits!(ix::_OccIndex, e::ASTExpr)::BitSet
    if e isa VarExpr
        e.name == ix.v || return _NO_SITES
        isempty(ix.var[]) && (ix.var[] = BitSet(_keyid!(ix, _site_key(e))))
        return ix.var[]
    end
    e isa OpExpr || return _NO_SITES
    cached = get(ix.bits, e, nothing)
    cached === nothing || return cached
    own = _is_v_index(ix, e)
    out = own ? BitSet(_nodekey!(ix, e)) : _NO_SITES
    mine = own                        # `out` is this node's own set
    map_children(e) do c
        b = _site_bits!(ix, c)
        if !isempty(b) && b !== out
            if isempty(out)
                out = b
            elseif !issubset(b, out)
                mine || (out = copy(out); mine = true)
                union!(out, b)
            end
        end
        c
    end
    ix.bits[e] = out
    return out
end

"""
    _SiteScan(ix::_OccIndex, s::Site)

The occurrence argument of one derivative scan ([`dscalar`](@ref)) for site
`s`: answered from the shared index `ix` when `s` is a site of `ix.v` (else
from a per-scan memo, as `occurs` does), with a memo of the derivatives
already taken, so a subtree shared by several parents is differentiated
once. For a site of `ix.v` that memo is the index's, shared by every scan of
the same site: the regions of one clipped body share every subtree clipping
left alone, and on ReSEACT's advection operator that is most of each body.
The derivative of a node with respect to a site is a function of the two
alone, and nothing rewrites an expression in place, so sharing it is
exact. The memo is keyed by the site's key and its own expression, so two
spellings of one site never share.
"""
struct _SiteScan
    ix::_OccIndex
    bit::Int                          # 0: `s` is not a site of `ix.v`
    memo::IdDict{OpExpr, Bool}        # the fallback occurrence memo
    d::IdDict{OpExpr, ASTExpr}        # node → its derivative in this scan
end
function _SiteScan(ix::_OccIndex, s::Site)
    ok = s.name == ix.v && s.rank >= -1
    ok || return _SiteScan(ix, 0, IdDict{OpExpr, Bool}(), IdDict{OpExpr, ASTExpr}())
    d = get!(() -> IdDict{OpExpr, ASTExpr}(), ix.deriv, s.key * "\0" * skey(s.expr))
    return _SiteScan(ix, _keyid!(ix, s.key), IdDict{OpExpr, Bool}(), d)
end
_SiteScan(s::Site) = _SiteScan(_OccIndex(s.name), s)

# `is_site(e, s)`, with an `index` node's key read from the index.
function _is_site(e::ASTExpr, s::Site, sc::_SiteScan)::Bool
    sc.bit == 0 && return is_site(e, s)
    if e isa VarExpr
        return s.rank == -1 && e.name == s.name
    elseif e isa OpExpr && e.op == "index"
        s.rank >= 0 && length(e.args) == s.rank + 1 || return false
        a1 = e.args[1]
        (a1 isa VarExpr && a1.name == s.name) || return false
        return _nodekey!(sc.ix, e) == sc.bit
    end
    return false
end
_is_site(e::ASTExpr, s::Site, ::IdDict{OpExpr, Bool}) = is_site(e, s)

function occurs(e::ASTExpr, s::Site, sc::_SiteScan)::Bool
    sc.bit == 0 && return occurs(e, s, sc.memo)
    e isa OpExpr || return _is_site(e, s, sc)
    return sc.bit in _site_bits!(sc.ix, e)
end

# What a derivative scan threads for occurrence: a per-site memo, or a scan
# over an `_OccIndex`.
const _Occ = Union{IdDict{OpExpr, Bool}, _SiteScan}

occurs_var(e::ASTExpr, v::String) = v in free_variables(e)

# A literal region/range bound, possibly still an (already metaparameter-
# folded, hence constant) expression. Post-load documents carry only
# closed-form bounds; anything else is an authoring error surfaced here.
_region_bound(b) = b isa ASTExpr ?
    Int(round(evaluate_expr(b, Dict{String, Float64}()))) : Int(b)
