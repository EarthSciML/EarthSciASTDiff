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
#      contracted names, renamed apart first (see `_beta_faq`).
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
"""
function inline_observed(e::ASTExpr, obs::Dict{String,ASTExpr})::ASTExpr
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
    fresh = _Fresh(e)
    for _ in 1:MAX_INLINE_PASSES
        e2 = _inline_once(e, obs, fresh)
        e2 === e && return e2
        sid(ix, e2) == sid(ix, e) && return e2
        e = e2
    end
    throw(InlineError("no fixpoint after $MAX_INLINE_PASSES passes (cyclic observed?)"))
end

# Fresh names for renamed-apart contraction indices: `_c<n>_<name>`, skipping
# every name the tree being inlined already uses. The taken set is collected
# on the first request only, so a document with no contracting observed pays
# nothing for it.
mutable struct _Fresh
    root::ASTExpr
    taken::Union{Set{String},Nothing}
    n::Int
end
_Fresh(e::ASTExpr) = _Fresh(e, nothing, 0)

function (f::_Fresh)(base::String)::String
    f.taken === nothing && (f.taken = _all_names!(Set{String}(), f.root))
    while true
        f.n += 1
        name = "_c$(f.n)_$base"
        name in f.taken || (push!(f.taken, name); return name)
    end
end

# Every variable name and every faq-bound index name in `e`.
function _all_names!(acc::Set{String}, e::ASTExpr)::Set{String}
    if e isa VarExpr
        push!(acc, e.name)
    elseif e isa OpExpr
        if e.op == "faq"
            e.ranges === nothing || union!(acc, String.(keys(e.ranges)))
            e.output_idx === nothing ||
                union!(acc, String(x) for x in e.output_idx if x isa AbstractString)
        end
        map_children(x -> (_all_names!(acc, x); x), e)
    end
    return acc
end

# β-reduce a read `F[sub…]` of a faq-defined array F. Output indices take the
# subscripts. Any other index F's ranges bind is CONTRACTED: the read is the
# scalar faq summing over it (F's own `reduce`, `filter` and `semiring`
# carried), never the bare body, which would leave that index free — or
# captured by a faq around the reading site that binds the same name, which
# turns the sum into one term with no error. `substitute` is not
# capture-avoiding, so the contracted names are renamed to fresh ones in the
# same pass that substitutes the subscripts (which may themselves use those
# names). A filter on a non-contracting F is carried the same way (a sum over
# the one point it gates); the band calculus then refuses it, as it refuses a
# filter written inline.
function _beta_faq(def::OpExpr, sub::Vector{ASTExpr}, fresh::_Fresh)::ASTExpr
    oidx = [String(x) for x in def.output_idx]
    length(sub) == length(oidx) || throw(InlineError(
        "index arity $(length(sub)) ≠ rank $(length(oidx)) of a faq read"))
    ranges = def.ranges === nothing ? Dict{String,Any}() : def.ranges
    cnames = sort!([String(n) for n in keys(ranges) if !(String(n) in oidx)])
    subst = Dict{String,ASTExpr}(n => sub[k] for (k, n) in enumerate(oidx))
    isempty(cnames) && def.filter === nothing &&
        return substitute(def.expr_body, subst)
    def.join === nothing || throw(InlineError(
        "cannot inline a read of a joined faq definition (join keys name its indices)"))
    newranges = Dict{String,Any}()
    for n in cnames
        f = fresh(n)
        subst[n] = VarExpr(f)
        newranges[f] = ranges[n]
    end
    return EarthSciAST.reconstruct(def; output_idx = Any[], ranges = newranges,
                                   expr_body = substitute(def.expr_body, subst),
                                   filter = def.filter === nothing ? nothing :
                                            substitute(def.filter, subst))
end

function _needs_inline(e::ASTExpr)::Bool
    e isa OpExpr || return false
    e.op == "index" && e.args[1] isa OpExpr && return true
    found = false
    map_children(e) do c
        found || (found = _needs_inline(c))
        c
    end
    return found
end

# index(makearray(regions, values), e…) → nested ifelse on region membership.
# Fold from the FIRST region outward so later regions override (§4.3.2).
function _index_makearray(ma::OpExpr, idx::Vector{ASTExpr}, obs, fresh)::ASTExpr
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
            _beta_faq(val, tgt, fresh)
        elseif val isa VarExpr || (val isa OpExpr && val.op in ("makearray", "index"))
            _inline_once(op("index", val, sub_idx...), obs, fresh)
        else
            val   # scalar broadcast over the region
        end
        cond = length(conds) == 1 ? conds[1] : op("and", conds...)
        result = op("ifelse", cond, v, result)
    end
    return result
end

function _inline_once(e::ASTExpr, obs, fresh::_Fresh)::ASTExpr
    if e isa VarExpr
        haskey(obs, e.name) || return e
        def = obs[e.name]
        # A scalar sum renamed apart too: spliced under a faq binding the
        # same index, its sites would be the outer ones, structurally.
        def isa OpExpr && def.op == "faq" && isempty(something(def.output_idx, Any[])) &&
            return _beta_faq(def, ASTExpr[], fresh)
        return def
    elseif e isa OpExpr
        if e.op == "index" && e.args[1] isa OpExpr && e.args[1].op == "makearray"
            return _index_makearray(e.args[1],
                                    ASTExpr[_inline_once(x, obs, fresh) for x in e.args[2:end]],
                                    obs, fresh)
        elseif e.op == "index" && e.args[1] isa OpExpr && e.args[1].op == "faq"
            sub = ASTExpr[_inline_once(x, obs, fresh) for x in e.args[2:end]]
            return _inline_once(_beta_faq(e.args[1], sub, fresh), obs, fresh)
        end
        if e.op == "index" && e.args[1] isa VarExpr && haskey(obs, e.args[1].name)
            def = obs[e.args[1].name]
            if def isa OpExpr && def.op == "faq"
                oidx = [String(x) for x in def.output_idx]
                length(oidx) == length(e.args) - 1 || throw(InlineError(
                    "index arity $(length(e.args)-1) ≠ rank $(length(oidx)) of observed `$(e.args[1].name)`"))
                return _beta_faq(
                    def, ASTExpr[_inline_once(x, obs, fresh) for x in e.args[2:end]], fresh)
            elseif def isa OpExpr && def.op == "makearray"
                # A named flux-style field: same lowering as an anonymous
                # `index(makearray…)` read (region-membership ifelse, later
                # clipped exactly by clip.jl where the guards are decidable).
                return _index_makearray(
                    def, ASTExpr[_inline_once(x, obs, fresh) for x in e.args[2:end]],
                    obs, fresh)
            elseif def isa OpExpr && def.op == "const"
                # A literal connectivity/weight table (`conn[i,k]`): keep the
                # gather, inline the table itself so downstream consumers
                # (site column evaluation, the derived evaluation model) see
                # a closed `index(const, …)` form.
                return op("index", def,
                          ASTExpr[_inline_once(x, obs, fresh) for x in e.args[2:end]]...)
            else
                throw(InlineError(
                    "index into observed `$(e.args[1].name)` whose definition is not a faq, makearray or const"))
            end
        end
        return map_children(x -> _inline_once(x, obs, fresh), e)
    end
    return e
end
