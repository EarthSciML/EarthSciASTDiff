# A uniform view over the two differentiable inputs — a single `Model` inside
# an `EsmFile`, and a coupled document's `FlattenedSystem` — plus the
# expansion step that guarantees we differentiate the SAME tree the tree-walk
# evaluator compiles.

"""
    SysView

Uniform system view: `variables` (name → `ModelVariable`), `equations`, and
the document's `index_sets`. Construct with [`sysview`](@ref).
"""
struct SysView
    variables::Dict{String,Any}
    equations::Vector
    index_sets::Dict{String,Any}
end

# `expanded_model` is EarthSciAST's public template-expansion seam
# (EarthSciAST ≥ 0.9.1): the typed model with Option-B
# `apply_expression_template` references expanded — the same step
# `build_evaluator` performs before compiling. Imported (not redefined) so
# `EarthSciASTDiff.expanded_model === EarthSciAST.expanded_model` and the two
# packages' exports never collide; re-exported from this module for
# convenience.
#
# `expand_flattened_refs` is the SAME seam for the other input shape — a
# coupled document's `FlattenedSystem`, whose surviving references resolve
# against the flattener's MERGED `template_registry` (esm-spec §9.6.4 rule 7)
# rather than a per-model `component_templates` entry. `expanded_model` takes
# an `EsmFile` + model name and cannot serve here; this is EarthSciAST's
# documented "Expand at your boundary" utility for flattened consumers (the
# MTK `System`/`PDESystem` constructors call it at their entries for exactly
# this reason). Not exported: it is EarthSciAST-internal, unlike
# `expanded_model`.
import EarthSciAST: expanded_model, expand_flattened_refs

"""
    sysview(file::EsmFile, model_name) -> SysView
    sysview(flat::FlattenedSystem)     -> SysView
"""
function sysview(file::EarthSciAST.EsmFile, model_name)
    model = expanded_model(file, model_name === nothing ? nothing : String(model_name))
    SysView(Dict{String,Any}(String(k) => v for (k, v) in model.variables),
            model.equations,
            Dict{String,Any}(String(k) => v for (k, v) in file.index_sets))
end

function sysview(flat::EarthSciAST.FlattenedSystem)
    # Expand FIRST, for the same reason the `EsmFile` method does. `flatten`
    # ALWAYS hands its consumers reference-preserving expressions, so without
    # this the two methods disagree: an `EsmFile` reaches the band calculus
    # Option-A expanded, a `FlattenedSystem` reaches it with
    # `apply_expression_template` nodes intact — nodes the calculus has no
    # rule for and the inliner rejects outright ("definition is not an
    # aggregate, makearray or const"). Expanding here restores the invariant
    # this file exists to hold: we differentiate the SAME tree the tree-walk
    # evaluator compiles, whichever input shape it arrived in. A no-op when no
    # references survived (empty registry), so single-model and
    # `ESS_TEMPLATE_REF_DISABLE=1` documents are byte-identical to before.
    flat = expand_flattened_refs(flat)
    vars = Dict{String,Any}()
    for d in (flat.state_variables, flat.parameters, flat.observed_variables),
        (k, v) in d
        vars[String(k)] = v
    end
    SysView(vars, flat.equations,
            Dict{String,Any}(String(k) => v for (k, v) in flat.index_sets))
end

function _ctx(sv::SysView)
    isets = Dict{String,Int}()
    for (n, s) in sv.index_sets
        s.size === nothing || (isets[String(n)] = s.size)
    end
    shapes = Dict{String,Vector{Int}}(); params = Set{String}()
    for (n, var) in sv.variables
        if var.shape !== nothing && !isempty(var.shape)
            shapes[String(n)] =
                [x isa Integer ? Int(x) : isets[String(x)] for x in var.shape]
        end
        var.type == EarthSciAST.ParameterVariable && push!(params, String(n))
    end
    return Ctx(isets, shapes, params)
end

"""
    lhs_state(eq) -> Union{String,Nothing}

The state variable an equation integrates, accepting both the plain `D(u)`
LHS and the pointwise-lifted `aggregate{ expr: D(index(u, i, j)) }` form the
flatten pipeline produces. `nothing` for `ic`/observed/other equations.
"""
function lhs_state(eq)
    l = eq.lhs
    l isa OpExpr || return nothing
    if l.op == "aggregate" && l.expr_body isa OpExpr && l.expr_body.op == "D"
        a = l.expr_body.args[1]
        a isa OpExpr && a.op == "index" && a.args[1] isa VarExpr &&
            return a.args[1].name
        return nothing
    end
    l.op == "D" && l.args[1] isa VarExpr && return l.args[1].name
    return nothing
end

"""
    lhs_definition(eq) -> Union{String,Nothing}

The variable an equation DEFINES by a bare-variable LHS (`y ~ f(...)`), or
`nothing`. The 1.0.0 counterpart of [`lhs_state`](@ref): esm 1.0.0 removed the
`observed` declared type and the `variable.expression` field, and an observed
unknown is now exactly an unknown some equation names bare on the left
(esm-spec 6.3.1).
"""
lhs_definition(eq) = eq.lhs isa VarExpr ? eq.lhs.name : nothing

# Observed-variable definitions of a view, for inlining.
#
# The 0.x form of this read `variables[n].expression` for every
# `ObservedVariable`. Both are gone in esm 1.0.0 -- `StateVariable` and
# `ObservedVariable` were removed from the enum with no deprecation path, and a
# definition moved out of the variable into the model's `equations`. So the same
# set is now derived: a bare-variable LHS naming an unknown that no D(.) equation
# also targets. An unknown a `D` equation DOES target is an ODE state, and a
# bare-LHS equation on it would be a constraint on the state rather than a
# definition of it -- hence the exclusion, which mirrors
# `EarthSciAST.observed_definitions`. First definition wins, as there.
function _observed_defs(sv::SysView)
    states = Set{String}()
    for eq in sv.equations
        n = lhs_state(eq); n === nothing || push!(states, String(n))
    end
    defs = Dict{String,ASTExpr}()
    for eq in sv.equations
        n = lhs_definition(eq); n === nothing && continue
        n = String(n)
        n in states && continue
        var = get(sv.variables, n, nothing)
        (var !== nothing && var.type == EarthSciAST.UnknownVariable) || continue
        haskey(defs, n) || (defs[n] = eq.rhs)
    end
    return defs
end

"""
    view_states(sv::SysView) -> Vector{String}

The unknowns occupying a solver slot: every unknown MINUS the observed ones.
The 1.0.0 replacement for `var.type == StateVariable`, and deliberately the
complement rather than the `D(.)`-targeted set, so an unknown constrained only
implicitly (an algebraic unknown) still counts as a slot instead of vanishing
-- which is what the 0.x declared type `state` meant.
"""
function view_states(sv::SysView)
    obs = keys(_observed_defs(sv))
    return sort!(String[String(n) for (n, var) in sv.variables
                        if var.type == EarthSciAST.UnknownVariable && !(String(n) in obs)])
end

# ── inline pruning: which observed the calculus must descend into ─────────────
#
# WHY. The chain rule only has to descend into an algebraic intermediate whose
# VALUE can move when a differentiation target moves. An observed that
# transitively references none of the targets is, at every point of the state
# space, a CONSTANT with respect to them: its derivative is identically zero,
# not merely small. Such a subtree therefore contributes nothing to any band
# and may be left un-inlined — kept as an opaque `v` / `index(v, …)` read.
#
# This is exact, not an approximation, and it is not a change of value either:
# `_base_document` (emit.jl) serializes EVERY variable of the view, observed
# definitions included, so a coefficient that still names such an observed is
# evaluated by the ordinary tree-walk from the SAME definition it would have
# been inlined from.
#
# WHAT IT BUYS. Inlining is the step that demands the differentiator understand
# every array construct on the way down (rank-reducing spatial-join gathers,
# regrid overlap templates, connectivity tables). Constant-w.r.t.-target
# subtrees carry exactly those constructs — emissions regridding, geometry,
# static maps — and the calculus never needed them. Pruning them is what lets a
# coupled document be differentiated at all, and it also removes them from the
# expression swell.
#
# WHY IT IS CONDITIONED ON `wrt`, AND MUST BE. The pruned set is computed from
# the ACTUAL target set, never hard-coded to states. The two axes genuinely
# disagree: an emissions field is state-independent (`∂E/∂u ≡ 0`) but is a
# perfectly ordinary function of its parameters, so pruning it for
# `wrt = :parameters` would silently return a WRONG (zero) parameter Jacobian —
# precisely the failure mode the adjoint needs `∂f/∂p` for. Same code, two
# different seeds; the seed is what makes each case right.
#
# `wrt = :time` is deliberately NOT pruned (see `jacobian_bands`): time enters
# an expression both by the explicit `t` site AND implicitly, through
# externally-provided time-varying data, so a free-variable seed of `{"t"}` is
# not a sound over-approximation of time-dependence.

# An expression whose true dependencies `free_variables` CANNOT see. A
# surviving `apply_expression_template` node keeps its body in the registry and
# exposes only its bindings, so its body's free references are invisible
# (EarthSciAST `expression_graph` guards the same way, graph.jl:320). `sysview`
# expands references at both entries, so this should never fire; it is here so
# that if an unexpanded reference ever does reach the calculus, the observed is
# treated as target-DEPENDENT (kept, hence inlined, hence loudly rejected by
# the inliner) rather than silently pruned to zero.
function _opaque_deps(e::ASTExpr)::Bool
    found = false
    EarthSciAST.foreach_subexpr(e) do x
        found |= x isa OpExpr && x.op == "apply_expression_template"
    end
    return found
end

# Names occurring in an index-ARGUMENT position — `args[2:end]` of an `index`
# node, i.e. a subscript rather than a value. `args[1]` (the indexed object) is
# an ordinary value position and is walked normally.
function _index_arg_names!(acc::Set{String}, e::ASTExpr)
    e isa OpExpr || return acc
    if e.op == "index"
        _index_arg_names!(acc, e.args[1])
        for k in 2:length(e.args)
            union!(acc, free_variables(e.args[k]))   # everything under a subscript
        end
        return acc
    end
    map_children(c -> (_index_arg_names!(acc, c); c), e)
    return acc
end

"""
    inlinable_observed(obs, equations, targets) -> Dict{String,ASTExpr}

The subset of the observed definitions `obs` that the band calculus actually
has to inline. Two independent reasons to keep one, unioned to a joint
fixpoint:

 1. **It can move with a target.** Reflexive-transitive closure over the
    observed→observed reference graph, seeded from the observed that name a
    `targets` member directly. Everything outside this closure is a constant
    with respect to every target, so its derivative is identically zero and
    inlining it could not have produced a band. See the rationale above for
    why the seed must track `wrt`.

 2. **A column index needs its value at build time.** A band's `cidx` is
    evaluated ONCE per row cell during assembly (`_eval_cidx`, assemble.jl),
    with only the row/contracted index names bound — so an observed reached
    through a subscript (`u[conn[i,k]]`, a connectivity table) must be a
    closed `index(const, …)` gather by then, and is inlined however
    target-independent it is. Dropping this clause is not a wrong VALUE, it is
    a build-time failure to evaluate the column, which is why it is a separate
    clause and not folded into (1).

Inlining a body splices it in at its use site, so a body kept for either
reason can expose fresh subscript occurrences: the scan is iterated to a
fixpoint over the equations plus the currently-kept bodies.
"""
function inlinable_observed(obs::Dict{String,ASTExpr}, equations,
                            targets::AbstractSet{String})::Dict{String,ASTExpr}
    fvs = Dict{String,Set{String}}(n => free_variables(e) for (n, e) in obs)
    readers = Dict{String,Vector{String}}()      # name → observed that read it
    work = String[]
    for (n, fv) in fvs
        (isdisjoint(fv, targets) && !_opaque_deps(obs[n])) || push!(work, n)
        for m in fv
            haskey(obs, m) && push!(get!(readers, m, String[]), n)
        end
    end
    keep = Set{String}(work)
    while !isempty(work)                          # (1): closure over the READERS
        for n in get(readers, pop!(work), ())     # of a target-dependent name
            n in keep || (push!(keep, n); push!(work, n))
        end
    end
    while true                                    # (2): fixpoint over the BODIES
        acc = Set{String}()                       # kept so far — the other
        for eq in equations                       # direction; a subscripted name
            _index_arg_names!(acc, eq.rhs)        # does NOT pull in its readers,
        end                                       # only what its own body needs
        for n in keep
            _index_arg_names!(acc, obs[n])
        end
        fresh = String[n for n in acc if haskey(obs, n) && !(n in keep)]
        isempty(fresh) && break
        union!(keep, fresh)
    end
    return Dict{String,ASTExpr}(n => obs[n] for n in keep)
end
