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
"""
struct Site
    expr::ASTExpr
    key::String
end
Site(e::ASTExpr) = Site(e, ckey(e))

is_site(e::ASTExpr, s::Site) =
    (e isa VarExpr || (e isa OpExpr && e.op == "index")) && ckey(e) == s.key

"""
    occurs(e, s::Site) -> Bool

Structural occurrence test — THE sparsity primitive. Never a symbolic or
numeric zero test: `x - x` occurs, both `ifelse` branches occur. False means
the derivative is a structural zero and differentiation short-circuits
(UFL-style `Zero` propagation).
"""
function occurs(e::ASTExpr, s::Site)::Bool
    is_site(e, s) && return true
    e isa OpExpr || return false
    found = false
    map_children(e) do c
        found || (found = occurs(c, s))
        c
    end
    return found
end

occurs_var(e::ASTExpr, v::String) = v in free_variables(e)

# A literal region/range bound, possibly still an (already metaparameter-
# folded, hence constant) expression. Post-load documents carry only
# closed-form bounds; anything else is an authoring error surfaced here.
_region_bound(b) = b isa ASTExpr ?
    Int(round(evaluate_expr(b, Dict{String,Float64}()))) : Int(b)
