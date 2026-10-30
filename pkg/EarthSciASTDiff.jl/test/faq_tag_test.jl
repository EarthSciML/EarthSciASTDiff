# The `faq` node tag — the regression guard for a SILENT ZERO.
#
# EarthSciAST renamed the unified query node's wire tag from `aggregate` to
# `faq` (Functional Aggregate Query) at esm 1.1.0
# (`docs/content/rfcs/faq-node-rename.md`). `aggregate` survives as a
# deprecated alias that the load pipeline normalizes at the wire boundary,
# warning once per document, and is removed at esm 2.0.0.
#
# This package matched the OLD tag for four days after the rename. Nothing
# threw: `lhs_state` simply stopped recognising the pointwise-lifted
# `faq{ expr: D(index(u, …)) }` LHS that the flatten pipeline produces, so
# every array state equation was classified as non-differential and
# `prepare_jacobian` returned an EMPTY plan — zero entries, zero scatter — on
# models whose Jacobian is entirely array-valued. A downstream adjoint driver
# consuming that plan builds an all-zero Jacobian and reports success.
#
# So the assertions below are about NON-EMPTINESS as much as correctness:
# recognising the tag, and refusing to pass vacuously when nothing is found.

using EarthSciASTDiff: lhs_state, view_states

@testset "faq tag: pointwise-lifted LHS is an array state equation" begin
    O(kv...) = Dict{String, Any}(kv...)
    # The shape `flatten` emits: faq{ output_idx: [i], expr: D(index(u, i)) }.
    lifted = EarthSciAST.expression_from_json(
        O(
            "op" => "faq", "args" => Any[], "output_idx" => Any["i"],
            "ranges" => O("i" => Any[1, 4]),
            "expr" => O(
                "op" => "D", "wrt" => "t",
                "args" => Any[O("op" => "index", "args" => Any["u", "i"])]
            )
        )
    )
    @test lhs_state((lhs = lifted, rhs = nothing)) == "u"

    # The plain scalar LHS still works.
    plain = EarthSciAST.expression_from_json(
        O("op" => "D", "wrt" => "t", "args" => Any["u"])
    )
    @test lhs_state((lhs = plain, rhs = nothing)) == "u"

    # The pre-1.1.0 spelling reaching `expression_from_json` directly is
    # normalized to `faq` by EarthSciAST, so it is recognised too — the
    # deprecated-alias path, removed at esm 2.0.0.
    aliased = EarthSciAST.expression_from_json(
        O(
            "op" => "aggregate", "args" => Any[], "output_idx" => Any["i"],
            "ranges" => O("i" => Any[1, 4]),
            "expr" => O(
                "op" => "D", "wrt" => "t",
                "args" => Any[O("op" => "index", "args" => Any["u", "i"])]
            )
        )
    )
    @test aliased.op == "faq"
    @test lhs_state((lhs = aliased, rhs = nothing)) == "u"
end

@testset "faq tag: array models differentiate to a NON-EMPTY band set" begin
    # Every fixture whose states are arrays must produce bands. A tag
    # mismatch makes all of these silently empty.
    for name in (
            "adv_interior", "flux_form_adv", "minmod_adv", "contracted_ops",
            "adv_react_2d", "bd_chem", "contracting_observed", "tgrad_photolysis", "forcing_bracket",
        )
        file = fixture("$name.esm")
        model = first(keys(file.models))
        sv = sysview(file, model)
        @test !isempty(view_states(sv))
        entries = jacobian_bands(sv)
        @test !isempty(entries)                              # the silent zero
        @test any(!isempty(en.band.rows) for en in entries)
    end
end

@testset "faq tag: the deprecated alias differentiates identically" begin
    O(kv...) = Dict{String, Any}(kv...)
    body = O("op" => "*", "args" => Any[-0.5, O("op" => "index", "args" => Any["u", "i"])])
    mk(tag, version) = EarthSciAST.load_document(
        O(
            "esm" => version,
            "metadata" => O("name" => "FaqAlias"),
            "index_sets" => O("x" => O("kind" => "interval", "size" => 4)),
            "models" => O(
                "FaqAlias" => O(
                    "variables" => O(
                        "u" => O(
                            "type" => "unknown", "shape" => Any["x"],
                            "default" => 1.0
                        )
                    ),
                    "equations" => Any[
                        O(
                            "lhs" => O("op" => "D", "args" => Any["u"], "wrt" => "t"),
                            "rhs" => O(
                                "op" => tag, "args" => Any[], "output_idx" => Any["i"],
                                "ranges" => O("i" => Any[1, 4]), "expr" => body
                            )
                        ),
                    ]
                )
            )
        )
    )

    new_bands = jacobian_bands(mk("faq", "1.1.0"), "FaqAlias")
    # A 1.0.0 document spelling `aggregate` is LEGAL: that is the pre-1.1.0
    # spelling. EarthSciAST normalizes it and warns once.
    old_bands = @test_logs (:warn, r"E_DEPRECATED_OP_ALIAS") match_mode = :any begin
        jacobian_bands(mk("aggregate", "1.0.0"), "FaqAlias")
    end
    @test !isempty(new_bands)
    @test length(new_bands) == length(old_bands)
    for (a, b) in zip(new_bands, old_bands)
        @test a.u == b.u && a.v == b.v && a.band.rows == b.band.rows
        @test stable_json(EarthSciAST.serialize_expression(a.band.coef)) ==
            stable_json(EarthSciAST.serialize_expression(b.band.coef))
    end
end

@testset "faq tag: emitted documents declare the 1.1.0 floor and re-load" begin
    # The emitted band coefficients may be `faq` nodes, so the document has to
    # declare at least 1.1.0 or the loader rejects it (E_FAQ_VERSION_TOO_OLD).
    file = fixture("contracted_ops.esm")
    doc = jacobian_document(file, first(keys(file.models)))
    @test VersionNumber(doc["esm"]) >= v"1.1.0"
    @test occursin("\"faq\"", stable_json(doc["jacobians"]))
    @test !occursin("\"aggregate\"", stable_json(doc))
    # The draft `"jacobians"` block is not in `esm-schema.json`, so only the
    # model half is loadable; that half is what must survive the floor raise.
    model_half = Dict{String, Any}(k => v for (k, v) in doc if k != "jacobians")
    @test EarthSciAST.load_document(model_half) isa EarthSciAST.EsmFile

    # The evaluation model — the document `prepare_jacobian` actually loads —
    # carries `faq` coefficient fields and must load without the version gate
    # firing.
    ev, names = EarthSciASTDiff._evaluation_document(
        sysview(file, first(keys(file.models))),
        jacobian_bands(file, first(keys(file.models)))
    )
    @test VersionNumber(ev["esm"]) >= v"1.1.0"
    @test !isempty(names)
    @test EarthSciAST.load_document(ev) isa EarthSciAST.EsmFile
end
