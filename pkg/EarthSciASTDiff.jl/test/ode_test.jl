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

@testset "tgrad: analytical ∂f/∂t (wrt = :time)" begin
    # Non-autonomous model: D(w) = sin(t)·w + c·t (∂/∂t = cos(t)·w + c),
    # D(v) = 2 v w (autonomous row → 0).
    doc = Dict{String,Any}("esm" => "0.8.0",
        "metadata" => Dict{String,Any}("name" => "TGrad"),
        "models" => Dict{String,Any}("M" => Dict{String,Any}(
            "variables" => Dict{String,Any}(
                "w" => Dict{String,Any}("type" => "state", "default" => 0.8),
                "v" => Dict{String,Any}("type" => "state", "default" => 1.1),
                "c" => Dict{String,Any}("type" => "parameter", "default" => 3.0)),
            "equations" => Any[
                Dict{String,Any}(
                    "lhs" => Dict{String,Any}("op" => "D", "args" => Any["w"], "wrt" => "t"),
                    "rhs" => Dict{String,Any}("op" => "+", "args" => Any[
                        Dict{String,Any}("op" => "*", "args" => Any[
                            Dict{String,Any}("op" => "sin", "args" => Any["t"]), "w"]),
                        Dict{String,Any}("op" => "*", "args" => Any["c", "t"])])),
                Dict{String,Any}(
                    "lhs" => Dict{String,Any}("op" => "D", "args" => Any["v"], "wrt" => "t"),
                    "rhs" => Dict{String,Any}("op" => "*",
                                              "args" => Any[2.0, "v", "w"]))])))
    file = EarthSciAST.load(doc)
    entries = jacobian_bands(sysview(file, "M"); wrt = :time)
    @test length(entries) == 1 && entries[1].v == "t"

    c = ode_components(file)
    τ = 0.7
    u = copy(c.u0)
    dT = similar(u)
    c.tgrad!(dT, u, c.p, τ)
    iw = c.var_map["w"]; iv = c.var_map["v"]
    @test dT[iw] ≈ cos(τ) * u[iw] + 3.0 rtol = 1e-12
    @test dT[iv] == 0.0
    # against a central finite difference in t through the tree-walk RHS
    h = 1e-6
    dup = similar(u); dum = similar(u)
    c.f!(dup, u, c.p, τ + h); c.f!(dum, u, c.p, τ - h)
    @test dT ≈ (dup .- dum) ./ 2h rtol = 1e-8
    # one-shot surface carries the same values as an n×1 matrix
    res = assemble_jacobian(file; wrt = :time, u = u, t = τ)
    @test res.colnames == ["t"] && size(res.J, 2) == 1
    @test Vector(res.J[:, 1]) ≈ dT rtol = 1e-12
    # SciMLBase wiring exposes it
    f = odefunction(file)
    dT2 = similar(u); f.tgrad(dT2, u, c.p, τ)
    @test dT2 ≈ dT

    # an autonomous system gets the zero fill
    ca = ode_components(fixture("bd_chem.esm"))
    dTa = fill(NaN, length(ca.u0))
    ca.tgrad!(dTa, ca.u0, ca.p, 0.3)
    @test all(dTa .== 0.0)
end
