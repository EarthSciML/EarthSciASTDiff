# The derived evaluation document (emit.jl `_evaluation_document`): the
# source's derivatives are ZEROED, so a `jac!` call evaluates the band fields
# and the observed they read, never the source RHS. Pinned byte-for-byte by
# tests/goldens/*.evaldoc.json, which the other bindings must reproduce.

using EarthSciASTDiff: _evaluation_document, lhs_state

@testset "evaluation document: goldens" begin
    golden_dir = joinpath(@__DIR__, "..", "..", "..", "tests", "goldens")
    for name in ("bd_chem", "adv_interior", "contracted_ops", "flux_form_adv", "minmod_adv")
        file = fixture("$name.esm")
        sv = sysview(file, first(keys(file.models)))
        doc, _ = _evaluation_document(sv, jacobian_bands(sv))
        @test stable_json(doc) == read(joinpath(golden_dir, "$name.evaldoc.json"), String)
    end
end

@testset "evaluation document: no source right-hand side survives" begin
    for name in ("bd_chem", "adv_interior", "adv_react_2d", "contracted_ops",
                 "flux_form_adv", "minmod_adv")
        file = fixture("$name.esm")
        sv = sysview(file, first(keys(file.models)))
        states = Set(String(u) for u in (lhs_state(eq) for eq in sv.equations) if u !== nothing)
        @test !isempty(states)
        doc, _ = _evaluation_document(sv, jacobian_bands(sv))
        eqs = doc["models"]["JacobianEval"]["equations"]
        for u in states
            mine = [eq for eq in eqs if eq["lhs"] isa AbstractDict &&
                    eq["lhs"]["op"] == "D" && eq["lhs"]["args"] == Any[u]]
            @test length(mine) == 1                   # exactly one, and zero
            rhs = only(mine)["rhs"]
            @test rhs == 0.0 || (rhs["op"] == "makearray" && rhs["values"] == Any[0.0])
        end
        # no lifted (faq-LHS) source derivative is left behind either
        @test !any(eq["lhs"] isa AbstractDict && eq["lhs"]["op"] == "faq" for eq in eqs)
    end
end

@testset "evaluation document: one prepared jac! tracks u across calls" begin
    # Moving u between calls on ONE prepared evaluator must give the same
    # Jacobian as a fresh evaluation at the new point — nothing derived from
    # u may be cached at build time (the hazard that ruled out carrying the
    # source states as parameters).
    for name in ("bd_chem", "minmod_adv", "flux_form_adv")
        file = fixture("$name.esm")
        f!, u0, p0, _, _ = build_evaluator(file)
        jac = prepare_jacobian(file)
        uA = 0.2 .+ 0.6 .* abs.(sin.((1:length(u0)) .* 1.3))
        uB = 0.2 .+ 0.6 .* abs.(cos.((1:length(u0)) .* 2.9))
        J = copy(jac.prototype)
        jac(J, uA, p0, 0.0)
        jac(J, uB, p0, 0.0)
        @test Matrix(J) ≈ Matrix(ad_jacobian(f!, uB, p0, 0.0)) rtol = 1e-10
    end
end
