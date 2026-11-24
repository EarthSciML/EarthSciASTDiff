# A uniform view over the two differentiable inputs — a single `Model` inside
# an `EsmFile`, and a coupled document's `FlattenedSystem` — plus the
# expansion step that guarantees we differentiate the SAME tree the tree-walk
# evaluator compiles.

"""
    SysView

Uniform system view: `variables` (name → `ModelVariable`), `equations`, the
document's `index_sets`, its `domain` (`nothing` when it has none; the
evaluation document carries its time origin, esm-jacobian-spec.md §8.1) and
its `data_sources` (`nothing` when unknown: a `FlattenedSystem` does not
carry them; the derived documents copy the entries a data-fed parameter
names, §8.1). Construct with [`sysview`](@ref).
"""
struct SysView
    variables::Dict{String, Any}
    equations::Vector
    index_sets::Dict{String, Any}
    domain::Any
    data_sources::Any
    refs::Any       # RefLayer of the system's operator definitions, or nothing
end
SysView(variables, equations, index_sets) = SysView(variables, equations, index_sets, nothing)
SysView(variables, equations, index_sets, domain) =
    SysView(variables, equations, index_sets, domain, nothing)
SysView(variables, equations, index_sets, domain, data_sources) =
    SysView(variables, equations, index_sets, domain, data_sources, nothing)

# `expanded_model` is EarthSciAST's explicit expansion utility, re-exported
# for convenience as before. No analysis here calls it: references are kept
# (esm-spec §9.6.4 rule 2, operator_refs.jl).
import EarthSciAST: expanded_model

"""
    sysview(file::EsmFile, model_name; data_sources = file.data_sources) -> SysView
    sysview(flat::FlattenedSystem; data_sources = nothing)              -> SysView

`data_sources` (a document's `data_sources`, name → `DataSource`) is what
the derived documents copy for each data-fed parameter (esm-jacobian-spec.md
§8.1). A `FlattenedSystem` does not carry them, so pass the flattened
document's own when its parameters read data.

Operator references (`apply_operator`, esm-spec §9.6) are kept as they are:
never expanded. The view carries the definitions they reach — a flattened
system's merged registry (bodies closed by `flatten`, §9.6.4 rule 7), or a
model's own `operators` block, closed here the same way — and every analysis
differentiates each definition once (operator_refs.jl).
"""
function sysview(file::EarthSciAST.EsmFile, model_name; data_sources = file.data_sources)
    name = model_name === nothing ? nothing : String(model_name)
    model = _select_model(file, name)
    isets = Dict{String, Any}(String(k) => v for (k, v) in file.index_sets)
    raw = _model_registry(file, name)
    eqs = model.equations
    refs = nothing
    if raw !== nothing && !isempty(raw)
        eqs, refs = _closed_refs(raw, eqs, isets)
    end
    return SysView(
        Dict{String, Any}(String(k) => v for (k, v) in model.variables),
        eqs, isets, file.domain, data_sources, refs
    )
end

# The model `name` of `file` (the only one when `name` is nothing).
function _select_model(file::EarthSciAST.EsmFile, name)
    models = file.models
    (models === nothing || isempty(models)) && throw(ArgumentError("the document has no model"))
    if name === nothing
        length(models) == 1 ||
            throw(ArgumentError("document has $(length(models)) models; pass model_name"))
        return first(values(models))
    end
    haskey(models, name) || throw(ArgumentError("the document has no model `$name`"))
    return models[name]
end

# The model's own `operators` block (esm-spec §9.6.1) as the loader keeps it.
function _model_registry(file::EarthSciAST.EsmFile, name)
    ct = file.component_templates
    ct === nothing && return nothing
    n = name === nothing ? String(first(keys(file.models))) : name
    return get(ct, "models.$n", nothing)
end

_iset_sizes(isets) = Dict{String, Int}(
    String(n) => s.size for (n, s) in isets if s.size !== nothing
)

function sysview(flat::EarthSciAST.FlattenedSystem; data_sources = nothing)
    # A pointwise-lifted state's `shape` is the one `flatten` records: the
    # lift's index sets (esm-libraries-spec §4.7.5, "Pointwise lift: index
    # sets"; Julia's `flatten` has recorded one since before 0.6, by that
    # section's rules from 0.8). Nothing here derives it, so the view, its
    # context, the bands and every derived document name a lifted state's
    # dimensions exactly as EarthSciAST does. It is checked against
    # `lifted_shapes` (`_check_lifted_shape`), which also declares the
    # synthetic `_liftdim<d>_<n>` that `flatten` before 0.8 names in the
    # shape without adding to `index_sets`.
    vars = Dict{String, Any}()
    for d in (flat.state_variables, flat.parameters, flat.observed_variables),
            (k, v) in d
        vars[String(k)] = v
    end
    isets = Dict{String, Any}(String(k) => v for (k, v) in flat.index_sets)
    for (name, extents) in flat.lifted_shapes
        var = get(vars, String(name), nothing)
        var === nothing && continue
        _check_lifted_shape(String(name), var.shape, extents, isets)
    end
    # The merged registry `flatten` carries (esm-spec §9.6.4 rule 7): every
    # body closed, every reference binding all of its definition's params.
    # It is closed here all the same (a no-op on closed bodies): a body that
    # still read a name freely would otherwise contribute no derivative with
    # respect to it, a structurally nonzero entry silently dropped.
    reg = flat.template_registry
    eqs, refs = isempty(reg) ? (flat.equations, nothing) :
        _closed_refs(reg, flat.equations, isets)
    return SysView(vars, eqs, isets, flat.domain, data_sources, refs)
end

# The equations and the reference layer of a registry closed as `flatten`
# closes it (`_close_registry`): every reference binds the free names of the
# definitions it reaches.
function _closed_refs(raw, eqs, isets)
    reg, closing = _close_registry(raw)
    eqs = isempty(closing) ? eqs : [_close_equation(eq, closing) for eq in eqs]
    return eqs, RefLayer(reg, _iset_sizes(isets))
end

# Check the shape `flatten` recorded on lifted state `name` against its
# `lifted_shapes` extents (esm-libraries-spec §4.7.5): the same rank, every
# entry an index set of the flattened system, of that dimension's extent.
# The one entry that may be missing is the synthetic `_liftdim<d>_<n>` of its
# own dimension, which `flatten` before EarthSciAST 0.8 names without adding;
# it is added to `isets` here.
function _check_lifted_shape(name::String, shape, extents, isets::Dict{String, Any})
    shape = shape === nothing ? Any[] : shape
    length(shape) == length(extents) || throw(
        ArgumentError(
            "lifted state `$name` has shape $(collect(shape)) of rank " *
                "$(length(shape)), but its lifted_shapes entry $(collect(extents)) " *
                "has rank $(length(extents))"
        )
    )
    for (d, (axis, n)) in enumerate(zip(shape, extents))
        axis = String(axis)
        if !haskey(isets, axis)
            axis == "_liftdim$(d)_$(n)" || throw(
                ArgumentError(
                    "lifted state `$name`: dimension $d names index set " *
                        "`$axis`, which the flattened system does not declare"
                )
            )
            isets[axis] = EarthSciAST.IndexSet("interval"; size = Int(n))
        end
        size = isets[axis].size
        size == n || throw(
            ArgumentError(
                "lifted state `$name`: dimension $d names index set `$axis` " *
                    "of size $size, but its lifted_shapes extent is $n"
            )
        )
    end
    return nothing
end

function _ctx(sv::SysView)
    isets = Dict{String, Int}()
    for (n, s) in sv.index_sets
        s.size === nothing || (isets[String(n)] = s.size)
    end
    shapes = Dict{String, Vector{Int}}(); params = Set{String}()
    for (n, var) in sv.variables
        if var.shape !== nothing && !isempty(var.shape)
            shapes[String(n)] =
                [x isa Integer ? Int(x) : isets[String(x)] for x in var.shape]
        end
        var.type == EarthSciAST.ParameterVariable && push!(params, String(n))
    end
    return Ctx(isets, shapes, params, sv.refs)
end

"""
    lhs_state(eq) -> Union{String,Nothing}

The state variable an equation integrates, accepting both the plain `D(u)`
LHS and the pointwise-lifted `faq{ expr: D(index(u, i, j)) }` form the
flatten pipeline produces. `nothing` for `ic`/observed/other equations.

The `faq` (Functional Aggregate Query) node was spelled `aggregate` before esm
1.1.0, and esm 2.0.0 removed that spelling (EarthSciAST's parser rejects it,
`removed_op`), so only `faq` can reach this package.
"""
function lhs_state(eq)
    l = eq.lhs
    l isa OpExpr || return nothing
    if l.op == "faq" && l.expr_body isa OpExpr && l.expr_body.op == "D"
        a = l.expr_body.args[1]
        a isa OpExpr && a.op == "index" && a.args[1] isa VarExpr &&
            return a.args[1].name
        return nothing
    end
    l.op == "D" && l.args[1] isa VarExpr && return l.args[1].name
    return nothing
end

# Whether an equation's LHS is a time derivative of any form — including ones
# `lhs_state` does not recognize (`D(index(u, 1))`), which no pass supports
# and none may silently skip.
_is_derivative(eq) = eq.lhs isa OpExpr && (
    eq.lhs.op == "D" ||
        (eq.lhs.op == "faq" && eq.lhs.expr_body isa OpExpr && eq.lhs.expr_body.op == "D")
)

"""
    lhs_definition(eq) -> Union{String,Nothing}

The variable an equation DEFINES by a bare-variable LHS (`y ~ f(...)`), or
`nothing`. The 1.0.0 counterpart of [`lhs_state`](@ref): esm 1.0.0 removed the
`observed` declared type and the `variable.expression` field, and an observed
unknown is now exactly an unknown some equation names bare on the left
(esm-spec 6.3.1).
"""
lhs_definition(eq) = eq.lhs isa VarExpr ? eq.lhs.name : nothing

# An arrayed-observed definition `faq{expr: index(y, i…)} ~ f(i…)` (`y[i…] ~
# f(i…)` as documents spell it), rewritten to the whole-array form
# `y ~ faq{output_idx: [i…], expr: f(i…)}` that `inline_observed` β-reduces.
# The indexed half of `EarthSciAST.observed_definitions`, whose LHS unwrap
# credits this form to `y`. `nothing` when the LHS is not an indexed read of a
# bare name; `(y, nothing)` when it is one whose subscripts are not exactly the
# output indices — a partial or pinned definition (`y[1] ~ …`), which has no
# whole-array form.
function _indexed_definition(eq)
    l = eq.lhs
    l isa OpExpr || return nothing
    f = nothing
    if l.op == "faq"
        (l.expr_body isa OpExpr && l.expr_body.op == "index") || return nothing
        f, l = l, l.expr_body
    end
    (l.op == "index" && !isempty(l.args) && l.args[1] isa VarExpr) || return nothing
    name = l.args[1].name
    sub = l.args[2:end]
    (f !== nothing && all(x -> x isa VarExpr, sub)) || return (name, nothing)
    names = String[x.name for x in sub]
    (String[String(x) for x in something(f.output_idx, Any[])] == names && allunique(names)) ||
        return (name, nothing)
    return (
        name, OpExpr(
            "faq", ASTExpr[]; output_idx = Any[names...],
            expr_body = eq.rhs, ranges = f.ranges
        ),
    )
end

# The variable a non-derivative equation defines: a bare LHS (`y ~ …`) or an
# indexed one (`y[i…] ~ …`, lifted or pinned); `nothing` otherwise.
function _defined_name(eq)
    n = lhs_definition(eq)
    n === nothing || return String(n)
    d = _indexed_definition(eq)
    return d === nothing ? nothing : String(d[1])
end

# Observed-variable definitions of a view, for inlining.
#
# The 0.x form of this read `variables[n].expression` for every
# `ObservedVariable`. Both are gone in esm 1.0.0 -- `StateVariable` and
# `ObservedVariable` were removed from the enum with no deprecation path, and a
# definition moved out of the variable into the model's `equations`. So the same
# set is now derived: an LHS naming an unknown -- bare, or indexed (an ARRAYED
# observed, rewritten by `_indexed_definition`) -- that no D(.) equation also
# targets. An unknown a `D` equation DOES target is an ODE state, and a
# bare-LHS equation on it would be a constraint on the state rather than a
# definition of it -- hence the exclusion, which mirrors
# `EarthSciAST.observed_definitions`. First definition wins, as there.
#
# The indexed form is not optional: left out, an arrayed observed is counted
# as a solver slot by `view_states` -- a spurious column target -- and is never
# inlined, so the chain rule through it is silently lost. An arrayed observed
# whose definition cannot be rewritten (partial, pinned, or split over several
# equations) is therefore an error rather than a fallback to that.
#
# A faq definition whose `output_idx` is absent (`nothing`: the schema
# requires the field on the wire, the typed AST does not) is read as `[]`, a
# scalar sum, so a bare read splices the same tree either way.
function _observed_defs(sv::SysView)
    states = Set{String}()
    for eq in sv.equations
        n = lhs_state(eq); n === nothing || push!(states, String(n))
    end
    defs = Dict{String, ASTExpr}()
    indexed = Set{String}()
    for eq in sv.equations
        bare = lhs_definition(eq)
        nd = bare === nothing ? _indexed_definition(eq) : (bare, eq.rhs)
        nd === nothing && continue
        n, d = String(nd[1]), nd[2]
        n in states && continue
        var = get(sv.variables, n, nothing)
        (var !== nothing && var.type == EarthSciAST.UnknownVariable) || continue
        d === nothing && throw(
            BandError(
                "arrayed observed `$n`: defining LHS subscripts must be the " *
                    "distinct output indices (partial definitions are not supported)"
            )
        )
        if haskey(defs, n)
            (n in indexed || bare === nothing) && throw(
                BandError(
                    "arrayed observed `$n` is defined by more than one equation " *
                        "(piecewise definitions are not supported)"
                )
            )
            continue
        end
        defs[n] = _oidx_present(d)
        bare === nothing && push!(indexed, n)
    end
    return defs
end

_oidx_present(d) = d isa OpExpr && d.op == "faq" && d.output_idx === nothing ?
    EarthSciAST.reconstruct(d; output_idx = Any[]) : d

"""
    view_states(sv::SysView, obs = _observed_defs(sv)) -> Vector{String}

The unknowns occupying a solver slot: every unknown MINUS the observed ones.
The 1.0.0 replacement for `var.type == StateVariable`, and deliberately the
complement rather than the `D(.)`-targeted set, so an unknown constrained only
implicitly (an algebraic unknown) still counts as a slot instead of vanishing
-- which is what the 0.x declared type `state` meant. Pass `obs` when the
caller already holds it.
"""
function view_states(sv::SysView, obs = _observed_defs(sv))
    return sort!(
        String[
            String(n) for (n, var) in sv.variables
                if var.type == EarthSciAST.UnknownVariable && !haskey(obs, String(n))
        ]
    )
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
function inlinable_observed(
        obs::Dict{String, ASTExpr}, equations,
        targets::AbstractSet{String}
    )::Dict{String, ASTExpr}
    fvs = Dict{String, Set{String}}(n => free_variables(e) for (n, e) in obs)
    readers = Dict{String, Vector{String}}()      # name → observed that read it
    work = String[]
    for (n, fv) in fvs
        isdisjoint(fv, targets) || push!(work, n)
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
    # (2): fixpoint over the BODIES kept so far — the other direction; a
    # subscripted name does NOT pull in its readers, only what its own body
    # needs. The equations are scanned once and each pass scans only the
    # bodies it just added: nothing else can change between passes, and
    # re-walking every equation per pass is O(chain length × document).
    acc = Set{String}()
    for eq in equations
        _index_arg_names!(acc, eq.rhs)
    end
    added = collect(keep)
    while true
        for n in added
            _index_arg_names!(acc, obs[n])
        end
        added = String[n for n in acc if haskey(obs, n) && !(n in keep)]
        isempty(added) && break
        union!(keep, added)
    end
    return Dict{String, ASTExpr}(n => obs[n] for n in keep)
end

# ── data-fed parameters (esm-jacobian-spec.md §4.3, §8.1) ───────────────────

# The `data` rules of a parameter's `update`, in their order (none for an
# unknown). The one test of "data-fed": the inferred forcing axis
# ([`forcing_names`](@ref)) and the rules the derived documents carry agree.
_data_rules(var) = var.type == EarthSciAST.ParameterVariable ?
    [r for r in something(var.update, ()) if r.kind == "data"] : Any[]

# The names a kept `data` rule reads: its `expression`, an expression
# `from.unit_conversion`, and a handler's `read_vars` / `read_params`. The
# derived documents keep the observed these name, so a carried rule never
# references a variable they drop.
function _rule_reads(r)
    names = String[]
    r.expression === nothing || append!(names, free_variables(r.expression))
    f = r.from
    f !== nothing && f.unit_conversion isa ASTExpr &&
        append!(names, free_variables(f.unit_conversion))
    h = r.handler
    if h !== nothing
        append!(names, something(h.read_vars, String[]))
        append!(names, something(h.read_params, String[]))
    end
    return names
end
