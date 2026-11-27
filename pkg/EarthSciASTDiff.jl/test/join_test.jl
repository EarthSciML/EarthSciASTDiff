# Bands through a relational join (esm-jacobian-spec.md §3.5): a regrid whose
# only gate is a spatial overlap join. Byte-identical to the shared goldens;
# the evaluation document's values agree with central differences of the
# source, and the Jᵀλ document with Jᵀλ from them.

@testset "joins: a regrid's bands carry its overlap join" begin
    golden_dir = joinpath(@__DIR__, "..", "..", "..", "tests", "goldens")
    file = fixture("join_regrid.esm")
    sv = sysview(file, "Regrid")
    for (wrt, tag) in ((:states, ""), (:parameters, ".parameters"))
        entries = jacobian_bands(sv; wrt = wrt)
        @test stable_json(Any[EarthSciASTDiff.serialize_band(en) for en in entries]) ==
            read(joinpath(golden_dir, "join_regrid$tag.bands.json"), String)
        @test stable_json(jacobian_document(file, "Regrid"; wrt = wrt)["jacobians"]["Regrid"]) ==
            read(joinpath(golden_dir, "join_regrid$tag.block.json"), String)
        evdoc, _ = EarthSciASTDiff._evaluation_document(sv, entries)
        @test stable_json(evdoc) == read(joinpath(golden_dir, "join_regrid$tag.evaldoc.json"), String)
        for (kind, build) in (("jvp", jvp_document), ("vjp", vjp_document))
            doc, _ = build(file; wrt = wrt)
            @test stable_json(doc) == read(joinpath(golden_dir, "join_regrid$tag.$kind.evaldoc.json"), String)
        end
        # the joined band: contracted over the points, its join and ranges
        jb = only(en for en in entries if en.u == "c" && en.v == (wrt == :states ? "q" : "F"))
        @test jb.band.join !== nothing
        @test Dict(jb.band.join[2]) == Dict("i" => Dict("from" => "cells"), "r" => Dict("from" => "recs"))
        # a round trip keeps the join
        d = EarthSciASTDiff.serialize_band(jb)
        @test EarthSciASTDiff.serialize_band(EarthSciASTDiff.parse_band(d)) == d
    end
    # values: the state Jacobian against central differences of the source
    f!, u0, p, _, vm = tw_build(file)
    u = [0.2 + 0.6 * abs(sin(2.3 * i)) for i in eachindex(u0)]
    J = Matrix(doc_jacobian(file; u = u).J)
    h = 1.0e-6
    Jfd = zeros(length(u), length(u))
    for j in eachindex(u)
        up = copy(u); um = copy(u); up[j] += h; um[j] -= h
        a = similar(u); b = similar(u); f!(a, up, p, 0.0); f!(b, um, p, 0.0)
        Jfd[:, j] = (a .- b) ./ 2h
    end
    @test maximum(abs.(J .- Jfd)) < 1.0e-7
    @test count(!iszero, J) == 16           # 4 + 6 diagonal, 6 joined pairs
    # the F columns against central differences of the source with F
    # promoted to a state (a zero derivative)
    res = doc_jacobian(file; wrt = :parameters, u = u, p = p)
    d = EarthSciASTDiff.JSON3.read(read(joinpath(FIXTURES, "join_regrid.esm"), String), Dict{String, Any})
    m = d["models"]["Regrid"]; m["variables"]["F"]["type"] = "unknown"
    push!(
        m["equations"], Dict(
            "lhs" => Dict("op" => "D", "args" => ["F"], "wrt" => "t"),
            "rhs" => Dict("op" => "makearray", "args" => [], "regions" => [[[1, 6]]], "values" => [0.0])
        )
    )
    g!, x0, pp, _, vm2 = tw_build(EarthSciAST.load_string(EarthSciASTDiff.JSON3.write(d)))
    x = copy(x0)
    for nm in keys(vm)
        x[vm2[nm]] = u[vm[nm]]
    end
    Jp = Matrix(res.J)
    for (j, cn) in enumerate(res.colnames)
        startswith(cn, "F[") || continue
        xp = copy(x); xm = copy(x); xp[vm2[cn]] += h; xm[vm2[cn]] -= h
        a = similar(x); b = similar(x); g!(a, xp, pp, 0.0); g!(b, xm, pp, 0.0)
        for (nm, ri) in vm
            @test abs(Jp[ri, j] - (a[vm2[nm]] - b[vm2[nm]]) / 2h) < 1.0e-7
        end
    end
    # Jᵀλ from its document is Jᵀλ from the evaluation document
    λ = [cos(1.3 * i) for i in eachindex(u)]
    g, _ = doc_vjp(file, λ; u = u)
    @test g ≈ J' * λ rtol = 1.0e-12
end
