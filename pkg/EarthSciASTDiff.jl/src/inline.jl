# Chain rule through algebraic intermediates: observed-variable inlining and
# `index`-of-array-expression β-reduction.
#
# Differentiation needs a closed expression in states/parameters. Two
# constructs stand between the flattened RHS and that form:
#
#   1. OBSERVED variables (algebraic intermediates). Scalar observed inline by
#      plain substitution; an array observed defined by an `aggregate` inlines
#      at each read site by β-reduction (`index(F, e…)` → F's body with output
#      indices := e).
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
defining expression), β-reduce `index`-of-`aggregate`, and lower
`index`-of-`makearray` to region `ifelse`. Fixpoint-iterated; throws
[`InlineError`](@ref) on non-convergence (cyclic observed definitions).
"""
function inline_observed(e::ASTExpr, obs::Dict{String,ASTExpr})::ASTExpr
    isempty(obs) && !_needs_inline(e) && return e
    for _ in 1:MAX_INLINE_PASSES
        e2 = _inline_once(e, obs)
        skey(e2) == skey(e) && return e2
        e = e2
    end
    throw(InlineError("no fixpoint after $MAX_INLINE_PASSES passes (cyclic observed?)"))
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
function _index_makearray(ma::OpExpr, idx::Vector{ASTExpr}, obs)::ASTExpr
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
        v = if val isa OpExpr && val.op == "aggregate"
            oidx = [String(x) for x in val.output_idx]
            # TWO legal value shapes, the same pair `_bands_makearray!`
            # distinguishes with `val_fullrank` (bands.jl): a FULL-RANK
            # aggregate names every region dim, singleton dims included; a
            # REDUCED-RANK one names only the non-singleton dims. Bind each
            # output index to the read index of the dim it actually names. Only
            # the reduced-rank shape was accepted here, so a full-rank value
            # over a region with a pinned dim — the shape a 3-D field's
            # boundary face takes — was rejected outright.
            #
            # Inherited, and NOT changed here: the binding assumes the value
            # aggregate's own range for a dim STARTS where the region does, so
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
            substitute(val.expr_body,
                       Dict{String,ASTExpr}(n => tgt[k] for (k, n) in enumerate(oidx)))
        elseif val isa VarExpr || (val isa OpExpr && val.op in ("makearray", "index"))
            _inline_once(op("index", val, sub_idx...), obs)
        else
            val   # scalar broadcast over the region
        end
        cond = length(conds) == 1 ? conds[1] : op("and", conds...)
        result = op("ifelse", cond, v, result)
    end
    return result
end

function _inline_once(e::ASTExpr, obs)::ASTExpr
    if e isa VarExpr
        return haskey(obs, e.name) ? obs[e.name] : e
    elseif e isa OpExpr
        if e.op == "index" && e.args[1] isa OpExpr && e.args[1].op == "makearray"
            return _index_makearray(e.args[1],
                                    ASTExpr[_inline_once(x, obs) for x in e.args[2:end]], obs)
        elseif e.op == "index" && e.args[1] isa OpExpr && e.args[1].op == "aggregate"
            def = e.args[1]
            oidx = [String(x) for x in def.output_idx]
            subst = Dict{String,ASTExpr}(
                n => _inline_once(e.args[k+1], obs) for (k, n) in enumerate(oidx))
            return _inline_once(substitute(def.expr_body, subst), obs)
        end
        if e.op == "index" && e.args[1] isa VarExpr && haskey(obs, e.args[1].name)
            def = obs[e.args[1].name]
            if def isa OpExpr && def.op == "aggregate"
                oidx = [String(x) for x in def.output_idx]
                length(oidx) == length(e.args) - 1 || throw(InlineError(
                    "index arity $(length(e.args)-1) ≠ rank $(length(oidx)) of observed `$(e.args[1].name)`"))
                subst = Dict{String,ASTExpr}(
                    n => _inline_once(e.args[k+1], obs) for (k, n) in enumerate(oidx))
                return substitute(def.expr_body, subst)
            elseif def isa OpExpr && def.op == "makearray"
                # A named flux-style field: same lowering as an anonymous
                # `index(makearray…)` read (region-membership ifelse, later
                # clipped exactly by clip.jl where the guards are decidable).
                return _index_makearray(
                    def, ASTExpr[_inline_once(x, obs) for x in e.args[2:end]], obs)
            elseif def isa OpExpr && def.op == "const"
                # A literal connectivity/weight table (`conn[i,k]`): keep the
                # gather, inline the table itself so downstream consumers
                # (site column evaluation, the derived evaluation model) see
                # a closed `index(const, …)` form.
                return op("index", def,
                          ASTExpr[_inline_once(x, obs) for x in e.args[2:end]]...)
            else
                throw(InlineError(
                    "index into observed `$(e.args[1].name)` whose definition is not an aggregate, makearray or const"))
            end
        end
        return map_children(x -> _inline_once(x, obs), e)
    end
    return e
end
