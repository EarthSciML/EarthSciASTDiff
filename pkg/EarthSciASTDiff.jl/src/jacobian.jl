# Model-level driver: differentiate every `D(u) = rhs` equation w.r.t. every
# state (or parameter) that structurally occurs in its RHS.

"""
    JacEntry

One band of the model Jacobian: `∂(equation for state `u`)/∂`v`` with its
[`Band`](@ref).
"""
struct JacEntry
    u::String
    v::String
    band::Band
end

"""
    jacobian_bands(file::EsmFile, model_name; wrt = :states) -> Vector{JacEntry}
    jacobian_bands(flat::FlattenedSystem;    wrt = :states) -> Vector{JacEntry}
    jacobian_bands(sv::SysView;              wrt = :states) -> Vector{JacEntry}

Differentiate every differential equation of the system with respect to every
state (`wrt = :states`) or every parameter (`wrt = :parameters`) that
structurally occurs in its (observed-inlined) RHS.

The pattern implied by the result is GLOBAL: `ifelse`/`min`/`max` branches
are unioned, so it is valid at every point and is a superset of any local
(point-wise) pattern.
"""
jacobian_bands(file::EarthSciAST.EsmFile, model_name; kw...) =
    jacobian_bands(sysview(file, model_name); kw...)
jacobian_bands(flat::EarthSciAST.FlattenedSystem; kw...) =
    jacobian_bands(sysview(flat); kw...)

function jacobian_bands(sv::SysView; wrt::Symbol = :states)
    wrt in (:states, :parameters) ||
        throw(ArgumentError("wrt must be :states or :parameters, got $wrt"))
    ctx = _ctx(sv)
    want = wrt == :states ? EarthSciAST.StateVariable : EarthSciAST.ParameterVariable
    targets = sort!([String(n) for (n, var) in sv.variables if var.type == want])
    obs = _observed_defs(sv)
    entries = JacEntry[]
    for eq in sv.equations
        u = lhs_state(eq)
        u === nothing && continue
        shape_u = get(ctx.shapes, u, Int[])
        rhs = inline_observed(eq.rhs, obs)
        fv = free_variables(rhs)
        for v in targets
            v in fv || continue                     # structural occurrence gate
            for b in merge_bands(bands(rhs, v, ctx; shape_u = shape_u))
                push!(entries, JacEntry(u, v, normalize_band(b)))
            end
        end
    end
    return entries
end
