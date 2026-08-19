# The band Jacobian against the ForwardDiff oracle (AD through the
# eltype-generic tree-walk RHS): values must agree to machine precision, and
# the structural (global) pattern must be a superset of the local AD pattern.

@testset "oracle: adv_react_2d (states)" begin
    file = fixture("adv_react_2d.esm")
    f!, u0, p, _, _ = build_evaluator(file)
    for _ in 1:3
        u = 0.3 .+ rand(length(u0))
        res = assemble_jacobian(file; u = u, p = p)
        Jad = ad_jacobian(f!, u, p, 0.0)
        @test maximum(abs.(Matrix(res.J) .- Matrix(Jad))) ≤ 1e-14
        @test all((Jad .!= 0) .<= res.pattern)      # global ⊇ local
    end
end

@testset "oracle: adv_react_2d (parameters)" begin
    file = fixture("adv_react_2d.esm")
    f!, u0, p, _, _ = build_evaluator(file)
    u = 0.3 .+ rand(length(u0))
    res = assemble_jacobian(file; wrt = :parameters, u = u, p = p)
    pv = collect(Float64, p)
    Jp_ad = ForwardDiff.jacobian(pv) do q
        pp = NamedTuple{keys(p)}(Tuple(q))
        du = similar(q, length(u)); f!(du, u, pp, 0.0); du
    end
    @test res.colnames == [String(k) for k in keys(p)]
    @test maximum(abs.(Matrix(res.J) .- Jp_ad)) ≤ 1e-12
end

@testset "oracle: prepared evaluator reuse" begin
    file = fixture("bd_chem.esm")
    f!, u0, p, _, _ = build_evaluator(file)
    jac = prepare_jacobian(file)
    J = jac_prototype(jac)
    for _ in 1:3                                   # same prototype, many points
        u = 0.2 .+ rand(length(u0))
        jac(J, u, p, 0.0)
        Jad = ad_jacobian(f!, u, p, 0.0)
        @test maximum(abs.(Matrix(J) .- Matrix(Jad))) ≤ 1e-14
    end
end

@testset "oracle: coupled document through flatten" begin
    file = fixture("coupled_chem_advection.esm")
    flat = flatten(file)
    f!, u0, p, _, _ = build_evaluator(flat)
    u = u0 .* (0.5 .+ rand(length(u0)))
    res = assemble_jacobian(flat; u = u, p = p)
    Jad = ad_jacobian(f!, u, p, 0.0)
    scale = maximum(abs.(Jad))
    @test maximum(abs.(Matrix(res.J) .- Matrix(Jad))) ≤ 1e-12 * max(scale, 1.0)
    @test all((Jad .!= 0) .<= res.pattern)
    @test res.structure == :general      # chemistry blocks ⊕ transport + faces
end

@testset "pattern is point-independent" begin
    file = fixture("adv_react_2d.esm")
    r1 = jacobian_pattern(file)
    r2 = jacobian_pattern(file)
    @test r1.pattern == r2.pattern
    @test nnz(r1.pattern) > 0
end
