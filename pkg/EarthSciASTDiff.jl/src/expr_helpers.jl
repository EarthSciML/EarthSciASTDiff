# Small expression constructors and the differentiation "site" abstraction.
#
# Everything here works on the public EarthSciAST AST (`NumExpr` / `IntExpr` /
# `VarExpr` / `OpExpr`) through `map_children`, the field-table-generated
# rewrite primitive — children living in `expr_body`, `values`, `ranges`,
# `lower`/`upper`, … are never missed.

lit(x::Integer) = IntExpr(Int64(x))
lit(x::Real)    = NumExpr(Float64(x))
op(o, args...; kw...) = OpExpr(String(o), ASTExpr[args...]; kw...)

iszero_lit(e) = (e isa NumExpr && e.value == 0.0) || (e isa IntExpr && e.value == 0)
isone_lit(e)  = (e isa NumExpr && e.value == 1.0) || (e isa IntExpr && e.value == 1)

"n-ary `+` that drops literal zeros and collapses to its single argument."
add(args...) = (a = filter(!iszero_lit, collect(ASTExpr, args));
                isempty(a) ? lit(0) : length(a) == 1 ? a[1] : op("+", a...))

"n-ary `*` that short-circuits on a literal zero and drops literal ones."
mul(args...) = any(iszero_lit, args) ? lit(0) :
               (a = filter(!isone_lit, collect(ASTExpr, args));
                isempty(a) ? lit(1) : length(a) == 1 ? a[1] : op("*", a...))

neg(x) = iszero_lit(x) ? lit(0) : op("-", x)

# Term-identity key for SCALAR expressions (RFC §5.4 canonical form). Only
# valid on the scalar core — array ops carry fields `canonical_json` refuses.
ckey(e::ASTExpr) = canonical_json(e)

# Full-structure key for ANY expression (fixpoint checks over trees that may
# contain aggregate/makearray). Key order is made deterministic by
# `stable_json` (emit.jl), so equal trees always produce equal keys.
skey(e::ASTExpr) = stable_json(EarthSciAST.serialize_expression(e))

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
        return Site(e, ckey(e), e.name, -1)
    elseif e isa OpExpr && e.op == "index" && !isempty(e.args) &&
           e.args[1] isa VarExpr
        return Site(e, ckey(e), e.args[1].name, length(e.args) - 1)
    end
    return Site(e, ckey(e), "", -2)
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
               ckey(e) == s.key                     # plain key compare
    end
    if e isa VarExpr
        return s.rank == -1 && e.name == s.name
    elseif e isa OpExpr && e.op == "index"
        s.rank >= 0 && length(e.args) == s.rank + 1 || return false
        a1 = e.args[1]
        (a1 isa VarExpr && a1.name == s.name) || return false
        return ckey(e) == s.key
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
occurs(e::ASTExpr, s::Site)::Bool = occurs(e, s, IdDict{OpExpr,Bool}())

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
function occurs(e::ASTExpr, s::Site, memo::IdDict{OpExpr,Bool})::Bool
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

occurs_var(e::ASTExpr, v::String) = v in free_variables(e)

# A literal region/range bound, possibly still an (already metaparameter-
# folded, hence constant) expression. Post-load documents carry only
# closed-form bounds; anything else is an authoring error surfaced here.
_region_bound(b) = b isa ASTExpr ?
    Int(round(evaluate_expr(b, Dict{String,Float64}()))) : Int(b)
