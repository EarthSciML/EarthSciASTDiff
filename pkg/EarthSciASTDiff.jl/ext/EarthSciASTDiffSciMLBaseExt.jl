"""
    EarthSciASTDiffSciMLBaseExt

The SciMLBase half of the ODE wiring, loaded automatically when SciMLBase is
in the session. Supplies the two constructors the core `odefunction` /
`odeproblem` entry points dispatch into (via `Base.get_extension` — the core
never depends on a solver stack, mirroring EarthSciAST's
`[[library-exposes-rhs-not-solver]]` SimulateExt pattern). All the actual
work — differentiation, band assembly, the scatter map, the sparse
prototype — happens in the solver-free `ode_components`; this module only
wraps the pieces in SciMLBase types.
"""
module EarthSciASTDiffSciMLBaseExt

using EarthSciASTDiff: ode_components
import SciMLBase

function _odefunction(input; model_name = nothing, build_kwargs = NamedTuple())
    c = ode_components(input; model_name = model_name, build_kwargs = build_kwargs)
    return SciMLBase.ODEFunction(c.f!; jac = c.jac!, jac_prototype = c.jac_prototype)
end

function _odeproblem(input, tspan; u0 = nothing, p = nothing,
                     model_name = nothing, build_kwargs = NamedTuple())
    c = ode_components(input; model_name = model_name, build_kwargs = build_kwargs)
    f = SciMLBase.ODEFunction(c.f!; jac = c.jac!, jac_prototype = c.jac_prototype)
    return SciMLBase.ODEProblem(f, u0 === nothing ? c.u0 : u0,
                                (Float64(tspan[1]), Float64(tspan[2])),
                                p === nothing ? c.p : p)
end

end # module
