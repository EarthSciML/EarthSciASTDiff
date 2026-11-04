# Structure-only entries (esm-jacobian-spec.md §2.1): no coefficient, the
# same clipping and sites, merged on structure. Their points include the full
# mode's, and equal them unless a full band's coefficient simplifies to zero.

using EarthSciASTDiff: _scatter_pairs, serialize_band, parse_band, drule_check,
    DerivativeRuleError, Site, _SiteScan, _OccIndex

const _SO_GOLDENS = joinpath(@__DIR__, "..", "..", "..", "tests", "goldens")

include("structure_fixtures.jl")

_so_view(name) = name == "coupled_chem_advection" ?
    sysview(EarthSciAST.flatten(fixture("$name.esm"))) :
    (f = fixture("$name.esm"); sysview(f, first(keys(f.models))))

@testset "structure-only entries: goldens and points" begin
    for name in STRUCTURE_FIXTURES
        sv = _so_view(name)
        full = jacobian_bands(sv)
        so = jacobian_bands(sv; coefficients = false)
        @test all(en -> en.band.coef === nothing, so)
        @test all(en -> !haskey(serialize_band(en), "coef"), so)
        pf = Set(_scatter_pairs(sv, full)); ps = Set(_scatter_pairs(sv, so))
        @test issubset(pf, ps)
        # no band of these fixtures has a coefficient that simplifies to zero
        @test pf == ps
        # merging on structure never yields more bands than merging on values
        @test length(so) <= length(full)
        text = stable_json(Any[serialize_band(en) for en in so])
        @test text == read(joinpath(_SO_GOLDENS, "$name.structure.bands.json"), String)
        # a structure-only entry reads back as it was written
        @test all(en -> serialize_band(parse_band(serialize_band(en))) == serialize_band(en), so)
    end
end

@testset "structure-only entries merge on structure alone" begin
    # `onehot_clipped`: the one-hot column of `c` that clipping splits into
    # rows 1..2 and 3..5 is two full bands (their one-hot selections differ)
    # and ONE structure-only band; `e`'s two equations keep their bands on `c`
    # apart in both modes.
    sv = _so_view("onehot_clipped")
    full = jacobian_bands(sv)
    so = jacobian_bands(sv; coefficients = false)
    rows(es, u, v, hot) = [
        en.band.rows for en in es if en.u == u && en.v == v && !isempty(en.band.contracted) == hot
    ]
    @test rows(full, "c", "c", true) == [[(1, 2)], [(3, 5)]]
    @test rows(so, "c", "c", true) == [[(1, 5)]]
    @test rows(full, "e", "c", false) == rows(so, "e", "c", false) == [[(1, 2)], [(3, 5)]]
end

@testset "structure-only entries: other axes" begin
    for (name, wrt) in (
            ("tgrad_photolysis", :time), ("forcing_bracket", :forcing),
            ("forcing_bracket", :parameters),
        )
        sv = _so_view(name)
        full = jacobian_bands(sv; wrt = wrt)
        so = jacobian_bands(sv; wrt = wrt, coefficients = false)
        @test issubset(Set(_scatter_pairs(sv, full)), Set(_scatter_pairs(sv, so)))
    end
end

@testset "structure-only entries keep a band whose coefficient is zero" begin
    # d/dx[i] of x[i] - x[i] simplifies to 0: the full mode drops the band,
    # the structure-only mode keeps it (a strict superset).
    ctx = EarthSciASTDiff.Ctx(Dict("i" => 4), Dict("x" => [4]), Set{String}())
    xi = OpExpr("index", ASTExpr[VarExpr("x"), VarExpr("i")])
    rhs = OpExpr(
        "faq", ASTExpr[]; output_idx = Any["i"],
        expr_body = OpExpr("-", ASTExpr[xi, xi]),
        ranges = Dict{String, Any}("i" => Any[1, 4])
    )
    @test isempty(bands(rhs, "x", ctx; shape_u = [4]))
    so = bands(rhs, "x", ctx; shape_u = [4], coefficients = false)
    @test length(so) == 1 && so[1].coef === nothing && so[1].rows == [(1, 4)]
end

@testset "structure-only pattern" begin
    file = fixture("adv_interior.esm")
    p = jacobian_pattern(file)
    q = jacobian_pattern(file; coefficients = false)
    @test q.pattern == p.pattern
    @test all(en -> en.band.coef === nothing, q.entries)
end

@testset "structure-only entries refuse what the full mode refuses" begin
    ctx = EarthSciASTDiff.Ctx(Dict("i" => 4), Dict("x" => [4]), Set{String}())
    xi = OpExpr("index", ASTExpr[VarExpr("x"), VarExpr("i")])
    over(body) = OpExpr(
        "faq", ASTExpr[]; output_idx = Any["i"], expr_body = body,
        ranges = Dict{String, Any}("i" => Any[1, 4])
    )
    # an op with no derivative rule, and a closed function outside §9.2
    for body in (
            OpExpr("D", ASTExpr[xi]; wrt = "lon"),
            OpExpr("fn", ASTExpr[xi]; name = "datetime.week"),
            OpExpr("*", ASTExpr[IntExpr(2), OpExpr("fn", ASTExpr[xi]; name = "datetime.week")]),
        )
        @test_throws DerivativeRuleError bands(over(body), "x", ctx; shape_u = [4])
        @test_throws DerivativeRuleError bands(
            over(body), "x", ctx; shape_u = [4], coefficients = false
        )
    end
end

@testset "drule_check agrees with dscalar" begin
    x = VarExpr("x"); y = VarExpr("y"); s = Site(x)
    sc() = _SiteScan(_OccIndex("x"), s)
    o(op, a...) = OpExpr(op, ASTExpr[a...])
    lin = OpExpr("const", ASTExpr[]; value = [0.0, 1.0, 4.0])
    ax = OpExpr("const", ASTExpr[]; value = [0.0, 1.0, 2.0])
    for e in (
            x, y, IntExpr(0), o("+", x, y), o("-", x), o("-", y, x), o("neg", y),
            o("*", x, y), o("*", IntExpr(0), x), o("*", o("sign", x), y),
            o("/", y, x), o("/", o("floor", x), y), o("^", x, IntExpr(0)),
            o("^", x, IntExpr(2)), o("^", IntExpr(2), x), o("^", o("sign", x), y),
            o("exp", x), o("log", o("sign", x)), o("sqrt", x), o("cos", o("ceil", x)),
            o("atan", y, x), o("abs", x), o("ifelse", o("<", x, y), y, IntExpr(1)),
            o("max", x, y), o("min", o("sign", x)), o("<", x, y),
            OpExpr("fn", ASTExpr[x, lin, ax]; name = "interp.searchsorted"),
            OpExpr("fn", ASTExpr[lin, ax, x]; name = "interp.linear"),
            OpExpr("fn", ASTExpr[o("*", IntExpr(3), x)]; name = "datetime.julian_day"),
            OpExpr(
                "faq", ASTExpr[]; output_idx = Any[], reduce = "max",
                ranges = Dict{String, Any}("k" => Any[1, 3]),
                expr_body = o("*", x, VarExpr("k"))
            ),
        )
        @test drule_check(e, s, sc()) == !EarthSciASTDiff.iszero_lit(dscalar(e, s, sc()))
    end
    # a max over two bound names refuses in both, unless the term is zero
    m2(body) = OpExpr(
        "faq", ASTExpr[]; output_idx = Any[], reduce = "max",
        ranges = Dict{String, Any}("j" => Any[1, 3], "k" => Any[1, 3]), expr_body = body
    )
    @test_throws DerivativeRuleError dscalar(m2(x), s, sc())
    @test_throws DerivativeRuleError drule_check(m2(x), s, sc())
    @test !drule_check(m2(o("sign", x)), s, sc())
    @test EarthSciASTDiff.iszero_lit(dscalar(m2(o("sign", x)), s, sc()))
end
