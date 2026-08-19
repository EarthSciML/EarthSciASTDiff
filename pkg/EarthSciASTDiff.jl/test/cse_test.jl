# Coefficient-size control: branch-aware simplification (`simplify_branches`)
# and hash-consed template extraction (`cse_templates` / the
# `expression_templates` field of the emitted block). Regression target:
# `minmod_adv.esm`, a self-contained serialization of the EarthSciDiscretizations
# minmod-TVD periodic advection problem (N = 16) — the smallest fixture in
# the limiter family whose derivatives swell without these passes.

using EarthSciASTDiff: skey, _simp

@testset "simplify_branches" begin
    x = VarExpr("x"); y = VarExpr("y")
    c = OpExpr("<", ASTExpr[x, y])
    ife(cc, a, b) = OpExpr("ifelse", ASTExpr[cc, a, b])

    # equal branches collapse regardless of the condition
    @test simplify_branches(ife(c, x, x)) === x
    @test simplify_branches(ife(c, NumExpr(2.0), NumExpr(2.0))) isa NumExpr
    # equality is STRUCTURAL: a float 0 and an int 0 do not collapse
    r = simplify_branches(ife(c, NumExpr(0.0), IntExpr(0)))
    @test r isa OpExpr && r.op == "ifelse"

    # literal conditions select a branch
    @test simplify_branches(ife(NumExpr(1.0), x, y)) === x
    @test simplify_branches(ife(IntExpr(0), x, y)) === y

    # same-condition pruning along a path (both branch sides)
    e = ife(c, ife(c, x, y), y)                # inner true-branch ifelse on c
    @test skey(simplify_branches(e)) == skey(ife(c, x, y))
    e = ife(c, x, ife(c, x, y))                # inner false-branch ifelse on c
    @test skey(simplify_branches(e)) == skey(ife(c, x, y))
    # pruning can expose an equal-branch collapse
    e = ife(c, ife(c, x, y), x)
    @test simplify_branches(e) === x

    # non-ifelse trees pass through untouched (same structure)
    plain = OpExpr("*", ASTExpr[x, OpExpr("+", ASTExpr[y, NumExpr(2.0)])])
    @test skey(simplify_branches(plain)) == skey(plain)

    # via the derivative pipeline: d/dx of min(x + y, x + 2) has equal branch
    # derivatives (1 and 1), so the min's condition tree must not survive
    m = OpExpr("min", ASTExpr[OpExpr("+", ASTExpr[x, y]),
                              OpExpr("+", ASTExpr[x, NumExpr(2.0)])])
    d = _simp(dscalar(m, Site(x)))
    @test d isa Union{NumExpr,IntExpr}   # collapsed to a literal one
    @test EarthSciAST.evaluate_expr(d, Dict{String,Float64}()) == 1.0
end

@testset "wire-canonical literals" begin
    # `simplify`'s mixed int/float constant fold yields a float literal
    # (`2 + 0.5 + 0.5` → `NumExpr(3.0)`), but on parse EarthSciAST reads an
    # integral float literal back as an INTEGER literal (CONFORMANCE_SPEC
    # §5.5.3.1 rule 1). `_simp` must therefore land on the parse-canonical
    # form, or emitted coefficients change their structural key on
    # round-trip (the PPM regression that motivated `_canon_lits`).
    folded = _simp(OpExpr("+", ASTExpr[IntExpr(2), NumExpr(0.5), NumExpr(0.5)]))
    @test folded isa IntExpr && folded.value == 3
    rt = EarthSciAST.parse_expression(EarthSciAST.serialize_expression(folded))
    @test skey(rt) == skey(folded)
    # non-integral floats stay floats
    half = _simp(OpExpr("*", ASTExpr[NumExpr(0.5), IntExpr(3)]))
    @test half isa NumExpr && half.value == 1.5
    # normalization reaches nested positions
    nested = _simp(OpExpr("exp", ASTExpr[OpExpr("+", ASTExpr[
        VarExpr("x"), OpExpr("*", ASTExpr[IntExpr(4), NumExpr(0.5)])])]))
    @test skey(EarthSciAST.parse_expression(
        EarthSciAST.serialize_expression(nested))) == skey(nested)
end

@testset "cse_templates" begin
    x = VarExpr("x"); y = VarExpr("y")
    # a chunky repeated subtree (> min_nodes when threshold lowered)
    big = OpExpr("*", ASTExpr[OpExpr("exp", ASTExpr[OpExpr("+", ASTExpr[x, y])]),
                              OpExpr("log", ASTExpr[OpExpr("+", ASTExpr[x, NumExpr(2.0)])])])
    e1 = OpExpr("+", ASTExpr[big, VarExpr("a")])
    e2 = OpExpr("-", ASTExpr[big, VarExpr("b")])
    tmpl, rw = cse_templates(ASTExpr[e1, e2]; min_nodes = 4)
    @test !isempty(tmpl)
    reg = Dict{String,ASTExpr}(tmpl)
    @test skey(expand_templates(rw[1], reg)) == skey(e1)
    @test skey(expand_templates(rw[2], reg)) == skey(e2)
    # the rewritten forms actually reference a template
    @test occursin("apply_expression_template", stable_json(
        EarthSciAST.serialize_expression(rw[1])))

    # nothing to share → identity
    tmpl0, rw0 = cse_templates(ASTExpr[e1]; min_nodes = 50)
    @test isempty(tmpl0) && skey(rw0[1]) == skey(e1)

    # unknown reference errors on expansion
    bad = OpExpr("apply_expression_template", ASTExpr[]; name = "nope")
    @test_throws ArgumentError expand_templates(bad, Dict{String,ASTExpr}())
end

@testset "minmod limiter: oracle + swell control" begin
    file = fixture("minmod_adv.esm")
    mname = first(keys(file.models))
    f!, u0, p0, _, _ = build_evaluator(file)
    entries = jacobian_bands(file, mname)

    # analytical J matches AD at non-smooth-friendly points; pattern ⊇ AD
    for k in 1:2
        u = 0.2 .+ 0.6 .* abs.(sin.((1:length(u0)) .* (1.3 + k)))
        r = assemble_jacobian(file; u = u)
        Jad = ad_jacobian(f!, u, p0, 0.0)
        @test Matrix(r.J) ≈ Matrix(Jad) rtol = 1e-10
        @test all((Jad .!= 0) .<= r.pattern)
    end

    # swell regression: the summed coefficient text stays bounded
    # (measured 25,656 chars at introduction; slack for rule evolution)
    nchars = sum(length(stable_json(EarthSciAST.serialize_expression(e.band.coef)))
                 for e in entries)
    @test nchars < 60_000

    # CSE'd document: templates present, block smaller, round-trip exact
    d1 = jacobian_document(file, mname)
    d0 = jacobian_document(file, mname; cse = false)
    b1 = d1["jacobians"][String(mname)]
    @test haskey(b1, "expression_templates")
    @test length(stable_json(d1["jacobians"])) < length(stable_json(d0["jacobians"]))
    _, ents = parse_jacobian_block(b1)
    @test length(ents) == length(entries)
    @test all(skey(ents[k].band.coef) == skey(entries[k].band.coef)
              for k in eachindex(entries))
    # cse=false emits no templates and the same closed coefficients
    b0 = d0["jacobians"][String(mname)]
    @test !haskey(b0, "expression_templates")
    _, ents0 = parse_jacobian_block(b0)
    @test all(skey(ents0[k].band.coef) == skey(entries[k].band.coef)
              for k in eachindex(entries))
end
