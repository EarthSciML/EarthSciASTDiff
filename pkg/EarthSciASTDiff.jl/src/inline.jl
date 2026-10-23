# Chain rule through algebraic intermediates: observed-variable inlining and
# `index`-of-array-expression β-reduction.
#
# Differentiation needs a closed expression in states/parameters. Two
# constructs stand between the flattened RHS and that form:
#
#   1. OBSERVED variables (algebraic intermediates). Scalar observed inline by
#      plain substitution; an array observed defined by a `faq` node inlines
#      at each read site by β-reduction (`index(F, e…)` → F's body with output
#      indices := e). A CONTRACTING definition (ranges binding names beyond
#      output_idx) keeps its sum: the read becomes a scalar `faq` over the
#      contracted names (see `_beta_faq`). Every substitution is
#      capture-avoiding, and a faq node binding a name an enclosing faq
#      already binds is renamed apart (`_inline_once`).
#   2. `index(makearray(…), i, j)` — the form the pointwise lift produces when
#      an operator's `makearray` is read per cell. Lowered to a nested
#      region-membership `ifelse`: later regions override earlier ones, which
#      is exactly the spec's §4.3.2 overwrite semantics, folded from the first
#      region outward.
#
# Iterated to a bounded fixpoint (observed may reference observed), mirroring
# the template engine's MAX_REWRITE_PASSES discipline.

const MAX_INLINE_PASSES = 64

struct InlineError <: Exception
    msg::String
end
Base.showerror(io::IO, e::InlineError) = print(io, "InlineError: ", e.msg)

"""
    inline_observed(e, obs::Dict{String,ASTExpr}) -> ASTExpr

Inline every observed-variable read in `e` (`obs` maps observed name →
defining expression), β-reduce `index`-of-`faq`, and lower
`index`-of-`makearray` to region `ifelse`. Fixpoint-iterated; throws
[`InlineError`](@ref) on non-convergence (cyclic observed definitions).
`obs_names` caches the names used by `obs`'s definitions; a caller inlining
many expressions against one `obs` passes one `Ref` to all of them.
"""
function inline_observed(e::ASTExpr, obs::Dict{String,ASTExpr};
                         obs_names = Ref{Union{Nothing,Set{String}}}(nothing))::ASTExpr
    isempty(obs) && !_needs_inline(e) && return e
    # FIXPOINT TEST, AT A COST THAT DOES NOT GROW WITH THE TREE. The test is
    # "did this pass change anything?", and `skey(e2) == skey(e)` answers it
    # by serializing the whole tree TWICE PER PASS — that tree is the
    # post-inlining RHS of one equation, the largest object in the calculus
    # (measured on ReSEACT: 4.2×10⁵ nodes per equation), and the
    # serialization is redone from scratch every pass. Two cheaper tests in
    # order, same relation: `_inline_once` routes untouched nodes through the
    # identity-preserving `map_children`, so at the fixpoint it returns the
    # SAME object — a pointer compare; and otherwise interning
    # ([`sid`](@ref)) re-keys only the subtrees the pass rebuilt, because the
    # index's memo is carried across passes.
    ix = StructIndex()
    fresh = _Fresh(e, obs, obs_names, nothing)
    for _ in 1:MAX_INLINE_PASSES
        e2 = _inline_once(e, obs, fresh, Set{String}())
        e2 === e && return e2
        sid(ix, e2) == sid(ix, e) && return e2
        e = e2
    end
    throw(InlineError("no fixpoint after $MAX_INLINE_PASSES passes (cyclic observed?)"))
end

# Fresh names for renamed-apart binders: `_c<n>_<name>`, the smallest `n`
# clear of every name the document can put in this tree (the tree's own and
# every observed definition's, which later passes splice in) and of the
# caller's `avoid` set. Deterministic in (`name`, `avoid`): two copies of one
# sum renamed in the same scope get the same name, so they stay structurally
# equal and interning, template CSE and `hoist_observed` still share them.
# The taken set is collected on the first rename only (and `obs_names`, when
# given, is filled once and shared across the equations of one document), so
# a tree needing no rename pays nothing for it.
mutable struct _Fresh
    root::ASTExpr
    obs::Dict{String,ASTExpr}
    obs_names::Base.RefValue{Union{Nothing,Set{String}}}
    taken::Union{Set{String},Nothing}
end

function (f::_Fresh)(name::String, avoid)::String
    if f.taken === nothing
        if f.obs_names[] === nothing
            t = Set{String}()
            for d in values(f.obs)
                _all_names!(t, d)
            end
            f.obs_names[] = t
        end
        f.taken = _all_names!(Set{String}(), f.root)
    end
    base = replace(name, r"^_c\d+_" => "")
    n = 0
    while true
        n += 1
        c = "_c$(n)_$base"
        c in f.taken || c in f.obs_names[] || c in avoid || return c
    end
end

# Every variable name and every faq-bound index name in `e`.
function _all_names!(acc::Set{String}, e::ASTExpr)::Set{String}
    if e isa VarExpr
        push!(acc, e.name)
    elseif e isa OpExpr
        e.op == "faq" && union!(acc, _binders(e))
        map_children(x -> (_all_names!(acc, x); x), e)
    end
    return acc
end

# The index names a faq node binds: its ranges (sorted, so renaming is
# deterministic across bindings) and its named output indices.
function _binders(e::OpExpr)::Vector{String}
    b = sort!(String[String(n) for n in keys(something(e.ranges, Dict{String,Any}()))])
    for x in something(e.output_idx, Any[])
        x isa AbstractString && !(String(x) in b) && push!(b, String(x))
    end
    return b
end

# Rename the binders `names` of faq node `e` to fresh ones (clear of `avoid`
# and of every name under `e`, so no inner binder can capture them).
function _rename_binders(e::OpExpr, names, fresh::_Fresh, avoid)::OpExpr
    e.join === nothing || throw(InlineError(
        "cannot rename the indices of a joined faq node (join keys name them)"))
    av = _all_names!(Set{String}(avoid), e)
    ren = Dict{String,ASTExpr}()
    for n in names
        f = fresh(n, av)
        push!(av, f)
        ren[n] = VarExpr(f)
    end
    rn(n) = haskey(ren, String(n)) ? ren[String(n)].name : n
    fv = Set{String}(x.name for x in values(ren))
    return EarthSciAST.reconstruct(e;
        ranges = e.ranges === nothing ? nothing :
                 Dict{String,Any}(rn(n) => r for (n, r) in e.ranges),
        output_idx = e.output_idx === nothing ? nothing :
                     Any[x isa AbstractString ? rn(x) : x for x in e.output_idx],
        expr_body = _csubst(e.expr_body, ren, fv, fresh),
        filter = e.filter === nothing ? nothing : _csubst(e.filter, ren, fv, fresh))
end

# CAPTURE-AVOIDING substitution (upstream `substitute` is not): a faq node
# shadows the names it binds, and one binding a name free in a replacement
# (`fv`) is renamed apart before the substitution enters it.
function _csubst(e::ASTExpr, m::Dict{String,ASTExpr}, fv::Set{String},
                 fresh::_Fresh)::ASTExpr
    isempty(m) && return e
    e isa VarExpr && return get(m, e.name, e)
    e isa OpExpr || return e
    if e.op == "faq"
        b = _binders(e)
        if any(n -> haskey(m, n), b)
            m = Dict{String,ASTExpr}(k => v for (k, v) in m if !(k in b))
            isempty(m) && return e
        end
        clash = [n for n in b if n in fv]
        isempty(clash) || (e = _rename_binders(e, clash, fresh, fv))
    end
    return map_children(x -> _csubst(x, m, fv, fresh), e)
end

# β-reduce a read `F[sub…]` of a faq-defined array F. Output indices take the
# subscripts. Any other index F's ranges bind is CONTRACTED: the read is the
# scalar faq summing over it (F's own `reduce`, `filter` and `semiring`
# carried), never the bare body, which would leave that index free — or
# captured by a faq around the reading site that binds the same name, which
# turns the sum into one term with no error. The substitution is
# capture-avoiding: a contracted name, or a name bound inside the body, that
# the subscripts read freely is renamed apart first. A contracted name that
# shadows an index bound around the reading site (`scope`) is renamed too,
# since the band calculus reads a free index in a derivative by name. A
# filter on a non-contracting F is carried the same way (a sum over the one
# point it gates); the band calculus then refuses it, as it refuses a filter
# written inline.
function _beta_faq(def::OpExpr, sub::Vector{ASTExpr}, fresh::_Fresh, scope)::ASTExpr
    oidx = [String(x) for x in something(def.output_idx, Any[])]
    length(sub) == length(oidx) || throw(InlineError(
        "index arity $(length(sub)) ≠ rank $(length(oidx)) of a faq read"))
    ranges = def.ranges === nothing ? Dict{String,Any}() : def.ranges
    cnames = sort!([String(n) for n in keys(ranges) if !(String(n) in oidx)])
    subst = Dict{String,ASTExpr}(n => sub[k] for (k, n) in enumerate(oidx))
    fv = Set{String}()
    for x in sub
        union!(fv, free_variables(x))
    end
    isempty(cnames) && def.filter === nothing &&
        return _csubst(def.expr_body, subst, fv, fresh)
    def.join === nothing || throw(InlineError(
        "cannot inline a read of a joined faq definition (join keys name its indices)"))
    newranges = Dict{String,Any}()
    av = union(fv, scope, cnames)
    for n in cnames
        if n in fv || n in scope
            f = fresh(n, av)
            push!(av, f)
            subst[n] = VarExpr(f)
            newranges[f] = ranges[n]
        else
            newranges[n] = ranges[n]
        end
    end
    fv2 = copy(fv)
    for x in values(subst)
        x isa VarExpr && push!(fv2, x.name)
    end
    return EarthSciAST.reconstruct(def; output_idx = Any[], ranges = newranges,
                                   expr_body = _csubst(def.expr_body, subst, fv2, fresh),
                                   filter = def.filter === nothing ? nothing :
                                            _csubst(def.filter, subst, fv2, fresh))
end

# Whether a pass could change `e` with no observed to inline: an
# `index`-of-array-expression to reduce, or a faq node shadowing an index
# bound around it (renamed apart by `_inline_once`).
function _needs_inline(e::ASTExpr, scope = Set{String}())::Bool
    e isa OpExpr || return false
    e.op == "index" && e.args[1] isa OpExpr && return true
    if e.op == "faq"
        b = _binders(e)
        any(n -> n in scope, b) && return true
        scope = union(scope, b)
    end
    found = false
    map_children(e) do c
        found || (found = _needs_inline(c, scope))
        c
    end
    return found
end

# index(makearray(regions, values), e…) → nested ifelse on region membership.
# Fold from the FIRST region outward so later regions override (§4.3.2).
function _index_makearray(ma::OpExpr, idx::Vector{ASTExpr}, obs, fresh, scope)::ASTExpr
    result = lit(0)
    for (region, val) in zip(ma.regions, ma.values)
        conds = ASTExpr[]; sub_idx = ASTExpr[]
        for (k, r) in enumerate(region)
            lo = r[1] isa ASTExpr ? r[1] : lit(r[1])
            hi = r[2] isa ASTExpr ? r[2] : lit(r[2])
            push!(conds, op(">=", idx[k], lo))
            push!(conds, op("<=", idx[k], hi))
            _region_bound(r[2]) > _region_bound(r[1]) && push!(sub_idx, idx[k])
        end
        v = if val isa OpExpr && val.op == "faq"
            oidx = [String(x) for x in val.output_idx]
            # TWO legal value shapes, the same pair `_bands_makearray!`
            # distinguishes with `val_fullrank` (bands.jl): a FULL-RANK
            # faq node names every region dim, singleton dims included; a
            # REDUCED-RANK one names only the non-singleton dims. Bind each
            # output index to the read index of the dim it actually names. Only
            # the reduced-rank shape was accepted here, so a full-rank value
            # over a region with a pinned dim — the shape a 3-D field's
            # boundary face takes — was rejected outright.
            #
            # Inherited, and NOT changed here: the binding assumes the value
            # faq node's own range for a dim STARTS where the region does, so
            # no origin shift is applied. `_bands_makearray!` does apply one
            # (`shift = lo - origins[kk]`). No fixture in this suite exercises
            # a shifted value range on this path, so the two are consistent
            # only where the shift is zero.
            tgt = length(oidx) == length(idx)     ? idx :
                  length(oidx) == length(sub_idx) ? sub_idx :
                  throw(InlineError(
                      "makearray value rank $(length(oidx)) matches neither the " *
                      "region rank $(length(idx)) nor the region free rank " *
                      "$(length(sub_idx))"))
            _beta_faq(val, tgt, fresh, scope)
        elseif val isa VarExpr || (val isa OpExpr && val.op in ("makearray", "index"))
            _inline_once(op("index", val, sub_idx...), obs, fresh, scope)
        else
            val   # scalar broadcast over the region
        end
        cond = length(conds) == 1 ? conds[1] : op("and", conds...)
        result = op("ifelse", cond, v, result)
    end
    return result
end

# `scope`: the index names bound by the faq nodes enclosing `e`.
function _inline_once(e::ASTExpr, obs, fresh::_Fresh, scope::Set{String})::ASTExpr
    if e isa VarExpr
        # A name an enclosing faq binds is that index, not an observed read.
        (e.name in scope || !haskey(obs, e.name)) && return e
        return obs[e.name]
    elseif e isa OpExpr
        args = () -> ASTExpr[_inline_once(x, obs, fresh, scope) for x in e.args[2:end]]
        if e.op == "index" && e.args[1] isa OpExpr && e.args[1].op == "makearray"
            return _index_makearray(e.args[1], args(), obs, fresh, scope)
        elseif e.op == "index" && e.args[1] isa OpExpr && e.args[1].op == "faq"
            return _inline_once(_beta_faq(e.args[1], args(), fresh, scope), obs, fresh, scope)
        elseif e.op == "faq"
            # A faq node binding a name already bound around it — a spliced
            # definition's own sum (`t = 2·Σ_k u[k]` read under a faq over k)
            # or a hand-written one — is renamed apart: the band calculus
            # reads a free index in a derivative by name, so a shadowed one
            # would conflate the two indices (spec §3.5).
            b = _binders(e)
            clash = [n for n in b if n in scope]
            if !isempty(clash)
                e = _rename_binders(e, clash, fresh, scope)
                b = _binders(e)
            end
            inner = union(scope, b)
            return map_children(x -> _inline_once(x, obs, fresh, inner), e)
        end
        if e.op == "index" && e.args[1] isa VarExpr && haskey(obs, e.args[1].name) &&
                !(e.args[1].name in scope)
            def = obs[e.args[1].name]
            if def isa OpExpr && def.op == "faq"
                oidx = something(def.output_idx, Any[])
                length(oidx) == length(e.args) - 1 || throw(InlineError(
                    "index arity $(length(e.args)-1) ≠ rank $(length(oidx)) of observed `$(e.args[1].name)`"))
                return _beta_faq(def, args(), fresh, scope)
            elseif def isa OpExpr && def.op == "makearray"
                # A named flux-style field: same lowering as an anonymous
                # `index(makearray…)` read (region-membership ifelse, later
                # clipped exactly by clip.jl where the guards are decidable).
                return _index_makearray(def, args(), obs, fresh, scope)
            elseif def isa OpExpr && def.op == "const"
                # A literal connectivity/weight table (`conn[i,k]`): keep the
                # gather, inline the table itself so downstream consumers
                # (site column evaluation, the derived evaluation model) see
                # a closed `index(const, …)` form.
                return op("index", def, args()...)
            else
                throw(InlineError(
                    "index into observed `$(e.args[1].name)` whose definition is not a faq, makearray or const"))
            end
        end
        return map_children(x -> _inline_once(x, obs, fresh, scope), e)
    end
    return e
end
