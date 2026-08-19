# ODE-solve wiring: `ode_components` (solver-free) and the SciMLBase
# extension (`odefunction` / `odeproblem`). The end-to-end claim under test:
# a stiff OrdinaryDiffEq solver consumes the ANALYTICAL sparse Jacobian —
# it is called during the solve, its values match the ForwardDiff oracle,
# and the trajectory matches a no-jac reference solve of the same RHS.

using SciMLBase
using OrdinaryDiffEqRosenbrock: Rodas5P

@testset "ode_components" begin
    file = fixture("bd_chem.esm")
    c = ode_components(file)
    @test c.jac_prototype isa SparseMatrixCSC{Float64}
    @test size(c.jac_prototype) == (length(c.u0), length(c.u0))
    @test c.jacobian.structure == :block_diagonal

    # jac! at a non-initial point matches AD through the tree-walk RHS.
    u = c.u0 .+ 0.3 .* (1:length(c.u0)) ./ length(c.u0)
    J = copy(c.jac_prototype)
    c.jac!(J, u, c.p, 0.7)
    @test Matrix(J) ≈ Matrix(ad_jacobian(c.f!, u, c.p, 0.7)) rtol = 1e-12
end

@testset "odefunction / odeproblem via SciMLBase ext" begin
    file = fixture("bd_chem.esm")
    f = odefunction(file)
    @test f isa SciMLBase.ODEFunction
    @test f.jac_prototype isa SparseMatrixCSC{Float64}

    c = ode_components(file)
    u = c.u0 .* 1.1 .+ 0.05
    J = copy(f.jac_prototype)
    f.jac(J, u, c.p, 0.0)
    @test Matrix(J) ≈ Matrix(ad_jacobian(c.f!, u, c.p, 0.0)) rtol = 1e-12

    prob = odeproblem(file, (0.0, 25.0))
    @test prob isa SciMLBase.ODEProblem
    @test prob.u0 == c.u0

    # Stiff solve on the analytical sparse Jacobian…
    sol = SciMLBase.solve(prob, Rodas5P(); reltol = 1e-8, abstol = 1e-10)
    @test SciMLBase.successful_retcode(sol)

    # …against the same RHS with NO jac (internal finite differences).
    ref = SciMLBase.solve(SciMLBase.ODEProblem(c.f!, c.u0, (0.0, 25.0), c.p),
                          Rodas5P(); reltol = 1e-8, abstol = 1e-10)
    @test SciMLBase.successful_retcode(ref)
    @test sol.u[end] ≈ ref.u[end] rtol = 1e-6

    # The analytical jac is genuinely CALLED by the solver (not silently
    # bypassed): count invocations through a wrapping ODEFunction.
    calls = Ref(0)
    counting = SciMLBase.ODEFunction(c.f!;
        jac = (J, u, p, t) -> (calls[] += 1; c.jac!(J, u, p, t)),
        jac_prototype = copy(c.jac_prototype))
    sol2 = SciMLBase.solve(SciMLBase.ODEProblem(counting, c.u0, (0.0, 25.0), c.p),
                           Rodas5P(); reltol = 1e-8, abstol = 1e-10)
    @test SciMLBase.successful_retcode(sol2)
    @test calls[] > 0
    @test sol2.u[end] ≈ ref.u[end] rtol = 1e-6
end

@testset "coupled document through flatten solves too" begin
    # The flattened chemistry ⊕ advection document: general structure, and
    # the same end-to-end contract on a PDE-shaped system.
    flat = EarthSciAST.flatten(fixture("coupled_chem_advection.esm"))
    c = ode_components(flat)
    prob = odeproblem(flat, (0.0, 0.5))
    sol = SciMLBase.solve(prob, Rodas5P(); reltol = 1e-8, abstol = 1e-10)
    ref = SciMLBase.solve(SciMLBase.ODEProblem(c.f!, c.u0, (0.0, 0.5), c.p),
                          Rodas5P(); reltol = 1e-8, abstol = 1e-10)
    @test SciMLBase.successful_retcode(sol) && SciMLBase.successful_retcode(ref)
    @test sol.u[end] ≈ ref.u[end] rtol = 1e-6
end
