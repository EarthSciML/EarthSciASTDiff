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

# Observed-variable definitions of a view, for inlining.
function _observed_defs(sv::SysView)
    Dict{String,ASTExpr}(
        String(n) => var.expression for (n, var) in sv.variables
        if var.type == EarthSciAST.ObservedVariable && var.expression !== nothing)
end
