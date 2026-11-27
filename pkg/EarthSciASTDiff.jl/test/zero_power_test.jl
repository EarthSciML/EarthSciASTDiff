# The power rule at a zero base with a zero exponent (spec §3, `^`): in a
# product over ions of m^stoich with a parameter exponent, an absent ion
# (m = 0) whose exponent is 0 made x·b^(x−1) = 0·0^(−1) NaN, where the
# derivative of b^0 is 0. The shared fixture's derivative must be finite
# everywhere and equal the exact one: 0 at the singular point, and away from
# it the values central differences give (tests/goldens/
# zero_power_product.points.json, checked against them when written). The
# Rust and Python bindings check the same golden.

const _ZP_ROOT = joinpath(@__DIR__, "..", "..", "..")

@testset "power rule: zero base, zero exponent" begin
    file = EarthSciAST.load_path(joinpath(_ZP_ROOT, "tests", "valid", "zero_power_product.esm"))
    g = EarthSciASTDiff.JSON3.read(
        read(joinpath(_ZP_ROOT, "tests", "goldens", "zero_power_product.points.json"), String), Dict{String, Any}
    )
    # The native compiler declines a non-`+` reduction in a derived model.
    bkw = (compiler = :interpreter,)
    names = doc_jacobian(file; build_kwargs = bkw).rownames
    for pt in g["points"]
        u = Float64[pt["u"][n] for n in names]
        res = doc_jacobian(file; u = u, t = g["t"], build_kwargs = bkw)
        @test all(isfinite, nonzeros(res.J))
        for e in pt["entries"]
            r, c, v = e[1], e[2], e[3]
            got = res.J[findfirst(==(r), res.rownames), findfirst(==(c), res.colnames)]
            v == 0 ? (@test got == 0) : (@test isapprox(got, v; rtol = 1.0e-12))
        end
    end
    # The rule: a literal exponent 0 gives 0, another literal the usual form,
    # an exponent that is not a literal the guarded form, which is 0 at
    # b = 0, x = 0 and x·b^(x−1) elsewhere.
    d(x) = dscalar(EarthSciAST.expression_from_json(Dict("op" => "^", "args" => Any["b", x])), Site(VarExpr("b")))
    @test EarthSciASTDiff.iszero_lit(d(0))
    @test !occursin("ifelse", EarthSciASTDiff.skey(d(3)))
    @test occursin("ifelse", EarthSciASTDiff.skey(d("x")))
    @test evaluate_expr(d("x"), Dict("b" => 0.0, "x" => 0.0)) == 0.0
    @test evaluate_expr(d("x"), Dict("b" => 0.5, "x" => 1.5)) ≈ 1.5 * 0.5^0.5
end
