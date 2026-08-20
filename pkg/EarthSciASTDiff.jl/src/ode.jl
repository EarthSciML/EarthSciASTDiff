# ODE-solver wiring. Split per EarthSciAST's `[[library-exposes-rhs-not-solver]]`
# convention: the solver-free half (`ode_components`) lives here in the core,
# and the two SciMLBase constructors (`odefunction` / `odeproblem`) dispatch
# into the `EarthSciASTDiffSciMLBaseExt` package extension, which is loaded
# automatically whenever SciMLBase is in the session (any OrdinaryDiffEq*
# solver package brings it). Without it, they throw a pointer at the fix
# instead of a MethodError.

"""
    ode_components(input; model_name = nothing, build_kwargs = NamedTuple())
        -> (f!, jac!, jac_prototype, tgrad!, u0, p, var_map, jacobian)

Everything a stiff ODE solver needs, with no solver dependency:

  - `f!` — the tree-walk RHS, `f!(du, u, p, t)`.
  - `jac!` — the prepared analytical Jacobian, `jac!(J, u, p, t)`, filling
    the nonzeros of a matrix shaped like `jac_prototype` in place.
  - `jac_prototype::SparseMatrixCSC{Float64}` — the fixed structural
    (point-independent) pattern with stored zeros, ready to hand to a solver.
  - `tgrad!` — the analytical time gradient, `tgrad!(dT, u, p, t)` filling
    `∂f/∂t` in place (`wrt = :time` through the same band calculus). For a
    system whose RHS never reads `t` this is a zero fill with no second
    evaluator built.
  - `u0`, `p` — the document's initial state and parameter values.
  - `var_map` — state-element name → index in `u`.
  - `jacobian::JacobianEvaluator` — the underlying prepared evaluator
    (pattern, band entries, detected `structure`, …).

`input` is an `EsmFile` (with `model_name` when it holds several models) or
a `FlattenedSystem`. Always `wrt = :states` — that is the Jacobian an ODE
solver consumes. Wrap the pieces yourself for any solver interface, or load
SciMLBase and call [`odefunction`](@ref) / [`odeproblem`](@ref).
"""
function ode_components(input; model_name = nothing, build_kwargs = NamedTuple())
    f!, u0, p0, _, vm = _build_eval(_sv_src(input, model_name)[2], model_name;
                                    build_kwargs...)
    jac = prepare_jacobian(input; wrt = :states, model_name = model_name,
                           build_kwargs = build_kwargs)
    jac! = (J, u, p, t) -> jac(J, u, p, t)
    # Time gradient: an autonomous RHS (no structural `t` occurrence) gets a
    # zero fill instead of a second compiled evaluator.
    tgrad! = if isempty(jacobian_bands(_sv_src(input, model_name)[1]; wrt = :time))
        (dT, u, p, t) -> fill!(dT, 0.0)
    else
        tg = prepare_jacobian(input; wrt = :time, model_name = model_name,
                              build_kwargs = build_kwargs)
        Tbuf = copy(tg.prototype)
        (dT, u, p, t) -> begin
            tg(Tbuf, u, p, t)
            fill!(dT, 0.0)
            rows = rowvals(Tbuf); vals = nonzeros(Tbuf)
            for k in eachindex(rows)
                dT[rows[k]] += vals[k]        # single `t` column
            end
            dT
        end
    end
    return (f! = f!, jac! = jac!, jac_prototype = copy(jac.prototype),
            tgrad! = tgrad!, u0 = u0, p = p0, var_map = vm, jacobian = jac)
end

const _SCIML_HINT = "requires the SciMLBase glue: load SciMLBase (any \
OrdinaryDiffEq* solver package brings it, e.g. `using \
OrdinaryDiffEqRosenbrock`) so the EarthSciASTDiffSciMLBaseExt extension \
activates, or use `ode_components` directly."

"""
    odefunction(input; model_name = nothing, build_kwargs = NamedTuple())
        -> SciMLBase.ODEFunction

The model's RHS as a `SciMLBase.ODEFunction` with the analytical Jacobian
attached (`jac = jac!`, `jac_prototype` = the structural sparse pattern), so
stiff solvers factorize the true sparse Jacobian instead of a dense
finite-difference one. Requires SciMLBase in the session (see
[`ode_components`](@ref) for the solver-free pieces).
"""
function odefunction(input; kwargs...)
    ext = Base.get_extension(@__MODULE__, :EarthSciASTDiffSciMLBaseExt)
    ext === nothing && throw(ArgumentError("odefunction " * _SCIML_HINT))
    return ext._odefunction(input; kwargs...)
end

"""
    odeproblem(input, tspan; u0 = nothing, p = nothing,
               model_name = nothing, build_kwargs = NamedTuple())
        -> SciMLBase.ODEProblem

An `ODEProblem` over [`odefunction`](@ref)'s analytical-Jacobian RHS.
`u0` / `p` default to the document's initial state and parameters
(`p` must keep the tree-walk RHS's parameter NamedTuple shape — reorder or
rescale values, don't rename). Solve with any stiff OrdinaryDiffEq algorithm:

    using OrdinaryDiffEqRosenbrock
    sol = solve(odeproblem(file, (0.0, 1.0)), Rodas5P())
"""
function odeproblem(input, tspan; kwargs...)
    ext = Base.get_extension(@__MODULE__, :EarthSciASTDiffSciMLBaseExt)
    ext === nothing && throw(ArgumentError("odeproblem " * _SCIML_HINT))
    return ext._odeproblem(input, tspan; kwargs...)
end
