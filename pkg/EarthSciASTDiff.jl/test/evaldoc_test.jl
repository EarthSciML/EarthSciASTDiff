# The derived evaluation document (emit.jl `_evaluation_document`): the
# source's derivatives are ZEROED, so a `jac!` call evaluates the band fields
# and the observed they read, never the source RHS. Pinned byte-for-byte by
# tests/goldens/*.evaldoc.json, which the other bindings must reproduce.

using EarthSciASTDiff: _evaluation_document, lhs_state

@testset "evaluation document: goldens" begin
    golden_dir = joinpath(@__DIR__, "..", "..", "..", "tests", "goldens")
    for name in (
            "bd_chem", "adv_interior", "contracted_ops", "flux_form_adv", "minmod_adv",
            "contracting_observed", "tgrad_photolysis", "forcing_bracket",
            "filtered_scan", "semiring_reduce",
        )
        file = fixture("$name.esm")
        sv = sysview(file, first(keys(file.models)))
        doc, _ = _evaluation_document(sv, jacobian_bands(sv))
        @test stable_json(doc) == read(joinpath(golden_dir, "$name.evaldoc.json"), String)
        # without hoisting; no `-nocse` golden means nothing to hoist
        doc0, _ = _evaluation_document(sv, jacobian_bands(sv); cse = false)
        p0 = joinpath(golden_dir, "$name.evaldoc-nocse.json")
        @test stable_json(doc0) ==
            read(isfile(p0) ? p0 : joinpath(golden_dir, "$name.evaldoc.json"), String)
    end
end

@testset "evaluation document: no source right-hand side survives" begin
    for name in (
            "bd_chem", "adv_interior", "adv_react_2d", "contracted_ops",
            "flux_form_adv", "minmod_adv", "contracting_observed", "tgrad_photolysis", "forcing_bracket",
            "filtered_scan", "semiring_reduce",
        )
        file = fixture("$name.esm")
        sv = sysview(file, first(keys(file.models)))
        states = Set(String(u) for u in (lhs_state(eq) for eq in sv.equations) if u !== nothing)
        @test !isempty(states)
        doc, _ = _evaluation_document(sv, jacobian_bands(sv))
        eqs = doc["models"]["JacobianEval"]["equations"]
        for u in states
            mine = [
                eq for eq in eqs if eq["lhs"] isa AbstractDict &&
                    eq["lhs"]["op"] == "D" && eq["lhs"]["args"] == Any[u]
            ]
            @test length(mine) == 1                   # exactly one, and zero
            rhs = only(mine)["rhs"]
            @test rhs == 0.0 || (rhs["op"] == "makearray" && rhs["values"] == Any[0.0])
        end
        # no lifted (faq-LHS) source derivative is left behind either; an
        # indexed observed definition `faq{index(y, i)} ~ …` is not one
        @test !any(
            eq["lhs"] isa AbstractDict && eq["lhs"]["op"] == "faq" &&
                eq["lhs"]["expr"] isa AbstractDict && eq["lhs"]["expr"]["op"] == "D"
                for eq in eqs
        )
    end
end

@testset "evaluation document: only the observed the coefficients read" begin
    # flux_form_adv's flux F, contracting_observed's s and c and
    # contracted_ops's conn feed only f: inlined into every coefficient, so
    # nothing in the derived document reads them. minmod_adv's dx is read.
    for (name, gone, kept) in (
            ("flux_form_adv", ["F"], String[]),
            ("contracting_observed", ["s", "c"], String[]),
            ("contracted_ops", ["conn"], String[]),
            ("minmod_adv", String[], ["dx"]),
        )
        file = fixture("$name.esm")
        sv = sysview(file, first(keys(file.models)))
        doc, _ = _evaluation_document(sv, jacobian_bands(sv))
        m = doc["models"]["JacobianEval"]
        defs = Set(eq["lhs"] for eq in m["equations"] if eq["lhs"] isa String)
        for n in gone
            @test !haskey(m["variables"], n) && !(n in defs)
        end
        for n in kept
            @test haskey(m["variables"], n) && n in defs
        end
    end
end

@testset "evaluation document: an element derivative LHS is refused" begin
    # `D(index(u, 1)) = …` is no form lhs_state knows: dropping it would lose
    # its rows from the Jacobian and keep its RHS in the evaluation document.
    doc = EarthSciAST.load_document(
        Dict{String, Any}(
            "esm" => "1.1.0", "metadata" => Dict("name" => "Elem"),
            "index_sets" => Dict("x" => Dict("kind" => "interval", "size" => 3)),
            "models" => Dict(
                "Elem" => Dict(
                    "variables" => Dict(
                        "u" => Dict(
                            "type" => "unknown", "shape" => ["x"],
                            "default" => 1.0
                        )
                    ),
                    "equations" => [
                        Dict(
                            "lhs" => Dict(
                                "op" => "D", "wrt" => "t", "args" => [
                                    Dict("op" => "index", "args" => ["u", 1]),
                                ]
                            ),
                            "rhs" => Dict(
                                "op" => "-", "args" => [
                                    Dict("op" => "index", "args" => ["u", 1]),
                                ]
                            )
                        ),
                    ]
                )
            )
        )
    )
    sv = sysview(doc, "Elem")
    @test_throws EarthSciASTDiff.BandError jacobian_bands(sv)
    @test_throws EarthSciASTDiff.BandError _evaluation_document(sv, JacEntry[])
end

@testset "derived documents: one evaluator tracks u across calls" begin
    # Moving u between calls on ONE built evaluator of a derived document must
    # give the same product as a fresh evaluation at the new point — nothing
    # derived from u may be cached at build time (the hazard that ruled out
    # carrying the source states as parameters).
    for name in ("bd_chem", "minmod_adv", "flux_form_adv")
        file = fixture("$name.esm")
        f!, u0, p0, _, _ = build_evaluator(file)
        layout = _doc_source(file, nothing, NamedTuple())
        doc, io = jvp_document(file)
        ev = _doc_evaluator(doc, io.model)
        uA = 0.2 .+ 0.6 .* abs.(sin.((1:length(u0)) .* 1.3))
        uB = 0.2 .+ 0.6 .* abs.(cos.((1:length(u0)) .* 2.9))
        v = [sin(0.7 * k) for k in eachindex(u0)]
        inputs = Dict(_doc_cell(io.v, r) => v[i] for (i, r) in enumerate(layout.rownames))
        _doc_call(ev, layout, uA, p0, 0.0, inputs)
        out = _doc_call(ev, layout, uB, p0, 0.0, inputs)
        got = [out(_doc_cell(io.jvp, r)) for r in layout.rownames]
        @test got ≈ Matrix(ad_jacobian(f!, uB, p0, 0.0)) * v rtol = 1.0e-10
    end
end
