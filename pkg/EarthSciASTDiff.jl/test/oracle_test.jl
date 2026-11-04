# The band Jacobian against the ForwardDiff oracle (AD through the
# eltype-generic tree-walk RHS): values must agree to machine precision, and
# the structural (global) pattern must be a superset of the local AD pattern.

@testset "oracle: adv_react_2d (states)" begin
    file = fixture("adv_react_2d.esm")
    f!, u0, p, _, _ = build_evaluator(file)
    for _ in 1:3
        u = 0.3 .+ rand(length(u0))
        res = doc_jacobian(file; u = u, p = p)
        Jad = ad_jacobian(f!, u, p, 0.0)
        @test maximum(abs.(Matrix(res.J) .- Matrix(Jad))) ≤ 1.0e-14
        @test all((Jad .!= 0) .<= res.pattern)      # global ⊇ local
    end
end

@testset "oracle: adv_react_2d (parameters)" begin
    file = fixture("adv_react_2d.esm")
    f!, u0, p, _, _ = build_evaluator(file)
    u = 0.3 .+ rand(length(u0))
    res = doc_jacobian(file; wrt = :parameters, u = u, p = p)
    pv = collect(Float64, p)
    Jp_ad = ForwardDiff.jacobian(pv) do q
        pp = NamedTuple{keys(p)}(Tuple(q))
        du = similar(q, length(u)); f!(du, u, pp, 0.0); du
    end
    @test res.colnames == [String(k) for k in keys(p)]
    @test maximum(abs.(Matrix(res.J) .- Jp_ad)) ≤ 1.0e-12
end

@testset "oracle: several points" begin
    file = fixture("bd_chem.esm")
    f!, u0, p, _, _ = build_evaluator(file)
    for _ in 1:3
        u = 0.2 .+ rand(length(u0))
        J = doc_jacobian(file; u = u, p = p).J
        Jad = ad_jacobian(f!, u, p, 0.0)
        @test maximum(abs.(Matrix(J) .- Matrix(Jad))) ≤ 1.0e-14
    end
end

@testset "oracle: coupled document through flatten" begin
    file = fixture("coupled_chem_advection.esm")
    flat = flatten(file)
    f!, u0, p, _, _ = build_evaluator(flat)
    u = u0 .* (0.5 .+ rand(length(u0)))
    res = doc_jacobian(flat; u = u, p = p)
    Jad = ad_jacobian(f!, u, p, 0.0)
    scale = maximum(abs.(Jad))
    @test maximum(abs.(Matrix(res.J) .- Matrix(Jad))) ≤ 1.0e-12 * max(scale, 1.0)
    @test all((Jad .!= 0) .<= res.pattern)
    @test res.structure == :general      # chemistry blocks ⊕ transport + faces
end

# A faq node's operand list `args` (esm-spec §4.3.1) names whole arrays but
# reads nothing: reseact's PBL-mixing observed carry one, and reading them
# used to fail as a whole-array reference inside a faq body.
@testset "oracle: faq operand lists read nothing" begin
    file = fixture("faq_operand_list.esm")
    bands(f) = stable_json(Any[EarthSciASTDiff.serialize_band(en) for en in jacobian_bands(f, "FaqOperandList")])
    d = EarthSciASTDiff.JSON3.read(read(joinpath(FIXTURES, "faq_operand_list.esm"), String), Dict{String, Any})
    strip!(x) = x isa AbstractDict ?
        (get(x, "op", "") == "faq" && (x["args"] = Any[]); foreach(strip!, values(x))) :
        x isa AbstractVector ? foreach(strip!, x) : nothing
    strip!(d)
    twin = joinpath(mktempdir(), "twin.esm")
    write(twin, EarthSciASTDiff.JSON3.write(d))
    @test bands(file) == bands(EarthSciAST.load_path(twin))
    f!, u0, p, _, _ = build_evaluator(file)
    u = 0.2 .+ rand(length(u0))
    res = doc_jacobian(file; u = u, p = p)
    Jad = ad_jacobian(f!, u, p, 0.0)
    @test maximum(abs.(Matrix(res.J) .- Matrix(Jad))) ≤ 1.0e-14
    @test all((Jad .!= 0) .<= res.pattern)
end

# `Pre(x)` in a right-hand side is the identity (spec §3.2): EarthSciAST
# evaluates it as `x`, so ForwardDiff sees the dependence and the bands must
# carry it. Before, the rule was `0` and these entries were missing. The
# fixture is nonlinear in its `Pre` reads (`Pre(b)·a[i]`, `exp(Pre(b))`), so
# the Jacobian varies with the state: checked at several pinned points, so a
# failure reproduces.
@testset "oracle: Pre is the identity" begin
    file = fixture("pre_operator.esm")
    f!, u0, p, _, _ = build_evaluator(file)
    for s in 1:3
        u = [0.2 + 0.6 * abs(sin(2.3 * i + 0.7 * s)) for i in eachindex(u0)]
        res = doc_jacobian(file; u = u, p = p)
        Jad = ad_jacobian(f!, u, p, 0.0)
        @test nnz(Jad) > 0
        @test maximum(abs.(Matrix(res.J) .- Matrix(Jad))) ≤ 1.0e-14
        @test all((Jad .!= 0) .<= res.pattern)
    end
end

# `Pre(a)` is lowered to `a` before differentiation (spec §3.2), so the
# document differentiates exactly as its twin with every `Pre` removed: the
# same bands, block and evaluation document (`Pre(x)` and `x` intern,
# simplify and share as one structure), and no emitted document, first- or
# second-order, asks its evaluator for a `Pre`.
@testset "Pre is lowered before differentiation" begin
    path = joinpath(FIXTURES, "pre_operator.esm")
    file = EarthSciAST.load_path(path)
    d = EarthSciASTDiff.JSON3.read(read(path, String), Dict{String, Any})
    unpre(x) = x isa AbstractDict ?
        (
            get(x, "op", "") == "Pre" ? unpre(x["args"][1]) :
            Dict{String, Any}(k => unpre(v) for (k, v) in x)
        ) :
        x isa AbstractVector ? Any[unpre(y) for y in x] : x
    twin_path = joinpath(mktempdir(), "twin.esm")
    write(twin_path, EarthSciASTDiff.JSON3.write(unpre(d)))
    twin = EarthSciAST.load_path(twin_path)
    @test !occursin("\"Pre\"", read(twin_path, String))
    @test occursin("\"Pre\"", read(path, String))
    m = "PreOperator"
    bands(f) = stable_json(Any[EarthSciASTDiff.serialize_band(en) for en in jacobian_bands(f, m)])
    @test bands(file) == bands(twin)
    @test stable_json(jacobian_document(file, m)["jacobians"]) ==
        stable_json(jacobian_document(twin, m)["jacobians"])
    sv, svt = sysview(file, m), sysview(twin, m)
    ev(s) = stable_json(first(EarthSciASTDiff._evaluation_document(s, jacobian_bands(s))))
    @test ev(sv) == ev(svt)
    nopre(doc) = !occursin("\"Pre\"", stable_json(doc))
    entries = jacobian_bands(sv)
    @test nopre(first(EarthSciASTDiff._evaluation_document(sv, entries)))
    @test nopre(first(EarthSciASTDiff._jvp_document(sv, entries, :states, nothing)))
    @test nopre(first(EarthSciASTDiff._vjp_document(sv, entries, :states, nothing)))
    e2 = jacobian_bands2(sv; wrt = (:states, :states))
    @test !isempty(e2)
    @test nopre(Any[EarthSciASTDiff.serialize_band(en) for en in e2])
    for form in (:contract, :symmetric)
        @test nopre(first(EarthSciASTDiff._hvp_document(sv, e2, (:states, :states), nothing, 2; form = form)))
    end
    # the inliner lowers `Pre` with no observed to inline, nested and under
    # an `index`
    js(e) = stable_json(EarthSciAST.serialize_expression(e))
    lower(e) = js(EarthSciASTDiff.inline_observed(e, Dict{String, ASTExpr}()))
    x = VarExpr("x")
    @test lower(OpExpr("Pre", ASTExpr[x])) == js(x)
    @test lower(OpExpr("*", ASTExpr[OpExpr("Pre", ASTExpr[OpExpr("Pre", ASTExpr[x])]), x])) ==
        js(OpExpr("*", ASTExpr[x, x]))
    @test lower(OpExpr("index", ASTExpr[VarExpr("a"), OpExpr("Pre", ASTExpr[VarExpr("j")])])) ==
        js(OpExpr("index", ASTExpr[VarExpr("a"), VarExpr("j")]))
end

@testset "pattern is point-independent" begin
    file = fixture("adv_react_2d.esm")
    r1 = jacobian_pattern(file)
    r2 = jacobian_pattern(file)
    @test r1.pattern == r2.pattern
    @test nnz(r1.pattern) > 0
end
