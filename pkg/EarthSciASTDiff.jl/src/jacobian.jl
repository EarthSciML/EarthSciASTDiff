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
structurally occurs in its (observed-inlined) RHS — or with respect to time
itself (`wrt = :time`, the single scalar site `t`; `datetime.*` reads are
piecewise-constant and differentiate to 0, `interp.linear`/`interp.bilinear`
of `t` to the active-segment slope).

The pattern implied by the result is GLOBAL: `ifelse`/`min`/`max` branches
are unioned, so it is valid at every point and is a superset of any local
(point-wise) pattern.
"""
jacobian_bands(file::EarthSciAST.EsmFile, model_name; kw...) =
    jacobian_bands(sysview(file, model_name); kw...)
jacobian_bands(flat::EarthSciAST.FlattenedSystem; kw...) =
    jacobian_bands(sysview(flat); kw...)

function jacobian_bands(sv::SysView; wrt::Symbol = :states)
    wrt in (:states, :parameters, :time) ||
        throw(ArgumentError("wrt must be :states, :parameters or :time, got $wrt"))
    ctx = _ctx(sv)
    targets = if wrt == :time
        ["t"]
    else
        want = wrt == :states ? EarthSciAST.StateVariable : EarthSciAST.ParameterVariable
        sort!([String(n) for (n, var) in sv.variables if var.type == want])
    end
    obs = _observed_defs(sv)
    # Only the observed that can actually move with a target are inlined; the
    # rest have an identically-zero derivative and stay opaque reads that the
    # derived evaluation model resolves from their own definitions. Exact, and
    # seeded from THIS `wrt`'s targets — see `inlinable_observed`
    # (system.jl) for why the state and parameter axes must not share a seed.
    # `wrt = :time` is excluded: time also enters implicitly, through
    # externally-provided time-varying data, so `{"t"}` is not a sound
    # over-approximation of the set of time-dependent names.
    wrt == :time || (obs = inlinable_observed(obs, sv.equations, Set{String}(targets)))
    entries = JacEntry[]
    # The structural occurrence gate, driven from the EQUATION rather than
    # from the target list: scanning `targets` per equation is O(|equations| ×
    # |targets|) even when each RHS names a handful of them. `targets` is one
    # entry per state VARIABLE, so this is invisible on a document whose
    # states stay array-valued (ReSEACT: 13 either way) and quadratic on one
    # lowered to scalar cells, where a variable is a cell — a cost driven by
    # the flattening style rather than by how coupled the system is.
    # Intersecting the RHS's own free variables with the target SET is
    # O(|free variables|). Order is unchanged: `targets` is sorted by name, so
    # re-sorting the hits by name reproduces the same per-equation order, and
    # the `entries` order is what downstream pattern/scatter construction
    # (assemble.jl) reads.
    tset = Set{String}(targets)
    for eq in sv.equations
        u = lhs_state(eq)
        u === nothing && continue
        shape_u = get(ctx.shapes, u, Int[])
        rhs = inline_observed(eq.rhs, obs)
        hits = sort!(String[v for v in free_variables(rhs) if v in tset])
        for v in hits
            for b in merge_bands(bands(rhs, v, ctx; shape_u = shape_u))
                push!(entries, JacEntry(u, v, normalize_band(b)))
            end
        end
    end
    return entries
end
