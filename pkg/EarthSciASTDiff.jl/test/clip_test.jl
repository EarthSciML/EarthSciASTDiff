# Static region clipping (clip.jl): decidable region-membership guards are
# decided exactly by splitting band row ranges at their affine breakpoints;
# equal-coefficient sub-bands re-merge. Regression fixture: flux_form_adv.esm
# — a named observed edge-flux field defined by a makearray and read per cell
# (`F[i]`, `F[i+1]`), the `index(makearray…)` lowering path.

using EarthSciASTDiff: clip_regions, merge_bands, skey

@testset "clip_regions" begin
    x = VarExpr("x"); i = VarExpr("i")
    ife(c, a, b) = OpExpr("ifelse", ASTExpr[c, a, b])
    ge(a, b) = OpExpr(">=", ASTExpr[a, b]); le(a, b) = OpExpr("<=", ASTExpr[a, b])
    land(a, b) = OpExpr("and", ASTExpr[a, b])

    # membership guard on i over (1,8): interior [2,8] vs boundary
    body = ife(land(ge(i, IntExpr(2)), le(i, IntExpr(8))), x, NumExpr(0.5))
    parts = clip_regions(body, ["i"], [(1, 8)])
    @test length(parts) == 2
    @test parts[1][1] == [(1, 1)] && parts[2][1] == [(2, 8)]
    @test skey(parts[1][2]) == skey(NumExpr(0.5))     # boundary: else branch
    @test skey(parts[2][2]) == skey(x)                # interior: value

    # offset read guard i+1 <= 8 splits at i = 8
    body2 = ife(le(OpExpr("+", ASTExpr[i, IntExpr(1)]), IntExpr(8)), x, NumExpr(0.5))
    parts2 = clip_regions(body2, ["i"], [(1, 8)])
    @test [p[1] for p in parts2] == [[(1, 7)], [(8, 8)]]

    # a state-dependent condition is untouched (no breakpoints)
    body3 = ife(OpExpr("<", ASTExpr[x, NumExpr(0.0)]), x, NumExpr(2.0))
    parts3 = clip_regions(body3, ["i"], [(1, 8)])
    @test length(parts3) == 1 && skey(parts3[1][2]) == skey(body3)

    # decided comparison used as a VALUE folds to its 0/1 literal
    body4 = OpExpr("*", ASTExpr[ge(i, IntExpr(1)), x])
    parts4 = clip_regions(body4, ["i"], [(1, 8)])
    @test length(parts4) == 1
    @test !occursin("\">=\"", stable_json(EarthSciAST.serialize_expression(parts4[1][2])))

    # no free names → identity
    @test length(clip_regions(body, Union{String, Int}[3], [(3, 3)])) == 1
end

@testset "merge_bands" begin
    c1 = VarExpr("a"); c2 = VarExpr("b"); ci = ASTExpr[VarExpr("i")]
    b(lo, hi, coef) = Band([(lo, hi)], Union{String, Int}["i"], ci, coef)
    # identical adjacent bands coalesce (products of splits re-merge fully)
    m = merge_bands([b(1, 3, c1), b(4, 4, c1), b(5, 8, c1)])
    @test length(m) == 1 && m[1].rows == [(1, 8)]
    # different coefficients stay apart
    m2 = merge_bands([b(1, 3, c1), b(4, 8, c2)])
    @test length(m2) == 2
    # non-adjacent identical bands stay apart
    m3 = merge_bands([b(1, 2, c1), b(4, 8, c1)])
    @test length(m3) == 2
end

@testset "flux-form fixture: exact pattern, guard-free coefficients" begin
    file = fixture("flux_form_adv.esm")
    entries = jacobian_bands(file, "FluxFormAdv")

    # every coefficient is guard-free — the membership ifelse was decided
    for en in entries
        @test !occursin(
            "ifelse",
            stable_json(EarthSciAST.serialize_expression(en.band.coef))
        )
    end

    # exact structure: -c/dx on the full diagonal (F[i+1] = c·u[i] for all i),
    # +c/dx on the sub-diagonal for rows 2..8 only (F[1] is the inflow
    # constant, so row 1 has no u[0] ghost entry — not even a unioned one)
    f!, u0, p0, _, _ = build_evaluator(file)
    u = 0.2 .+ 0.6 .* abs.(sin.((1:length(u0)) .* 2.7))
    r = doc_jacobian(file; u = u)
    Jad = ad_jacobian(f!, u, p0, 0.0)
    @test Matrix(r.J) ≈ Matrix(Jad) rtol = 1.0e-12
    @test r.pattern == (Jad .!= 0)     # EXACT, not merely a superset
    n = length(u0)
    @test count(r.pattern) == 2n - 1   # diagonal + sub-diagonal minus row 1

    # parameter Jacobian: only row 1 couples to `uin` (the inflow face)
    rp = jacobian_pattern(file; wrt = :parameters)
    col = findfirst(==("uin"), rp.colnames)
    @test findall(Vector(rp.pattern[:, col])) == [1]
end

# Each summand clipped on its own axes (clip.jl `clip_summands`,
# esm-jacobian-spec.md §2.2): a sum of an x-split and a y-split term divided
# by a state is split 3 + 3 ways instead of 3 × 3, and the bands of the
# overlapping pieces add to the ForwardDiff Jacobian. Fixture
# split_sum_2d.esm; its band golden pins the form for every binding.

@testset "clip_summands: each summand on its own axes" begin
    file = fixture("split_sum_2d.esm")
    sv = sysview(file, "S")
    gold = joinpath(@__DIR__, "..", "..", "..", "tests", "goldens", "split_sum_2d")
    @test stable_json(Any[EarthSciASTDiff.serialize_band(en) for en in jacobian_bands(sv)]) ==
        read("$gold.bands.json", String)
    eq = first(sv.equations)
    body = eq.rhs.expr_body
    @test length(EarthSciASTDiff.clip_regions(body, ["i", "j"], [(1, 6), (1, 5)])) == 9
    parts = EarthSciASTDiff.clip_summands(body, ["i", "j"], [(1, 6), (1, 5)])
    @test [p[1] for p in parts] == [
        [(1, 1), (1, 5)], [(2, 5), (1, 5)], [(6, 6), (1, 5)],
        [(1, 6), (1, 1)], [(1, 6), (2, 4)], [(1, 6), (5, 5)],
    ]
    # one summand group: exactly clip_regions
    xonly = body.args[1].args[1]
    @test [p[1] for p in EarthSciASTDiff.clip_summands(xonly, ["i", "j"], [(1, 6), (1, 5)])] ==
        [p[1] for p in EarthSciASTDiff.clip_regions(xonly, ["i", "j"], [(1, 6), (1, 5)])]
    # values: the overlapping bands add up to ForwardDiff's Jacobian
    f!, u0, p, _, _ = build_evaluator(file)
    for _ in 1:2
        u = 0.5 .+ rand(length(u0))
        res = doc_jacobian(file; u = u, p = p)
        Jad = ad_jacobian(f!, u, p, 0.0)
        @test maximum(abs.(Matrix(res.J) .- Matrix(Jad))) ≤ 1.0e-12
        @test all((Jad .!= 0) .<= res.pattern)
    end
    # structure-only points equal the full points
    full = jacobian_bands(sv); so = jacobian_bands(sv; coefficients = false)
    @test Set(EarthSciASTDiff._scatter_pairs(sv, full)) ==
        Set(EarthSciASTDiff._scatter_pairs(sv, so))
end
