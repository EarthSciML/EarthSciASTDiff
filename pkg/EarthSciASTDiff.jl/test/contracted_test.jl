# Contracted-index bands: `reduce: "+"` aggregates whose ranges bind more
# names than output_idx. Three site classes, all in contracted_ops.esm:
#   * a gathered column (`u[j]`, j contracted)      → Band with `contracted`
#   * a column independent of the contraction        → plain band, coefficient
#     wrapped in a scalar `reduce: "+"` aggregate
#   * a mixed affine gather (`u[i+k]`, k contracted) → Band with `contracted`,
#     reached through a makearray region value

using EarthSciASTDiff: Band, skey

@testset "contracted-index bands" begin
    file = fixture("contracted_ops.esm")
    entries = jacobian_bands(file, "ContractedOps")

    yu = [en for en in entries if en.u == "y" && en.v == "u"]
    # the gathered matvec column: contracted over j ∈ [1,6], cidx == j
    gath = [en for en in yu if !isempty(en.band.contracted)]
    @test length(gath) == 1
    @test gath[1].band.contracted == [("j", 1, 6)]
    @test skey(gath[1].band.cidx[1]) == skey(VarExpr("j"))
    # the exp(0.1·u[i]) site: column i, coefficient a symbolic contraction sum
    diag = [en for en in yu if isempty(en.band.contracted)]
    @test length(diag) == 1
    @test skey(diag[1].band.cidx[1]) == skey(VarExpr("i"))
    @test occursin("\"aggregate\"", skey(diag[1].band.coef))
    @test occursin("\"reduce\":\"+\"", skey(diag[1].band.coef))

    qu = [en for en in entries if en.u == "q" && en.v == "u"]
    @test all(en -> en.band.contracted == [("k", 1, 2)], qu)

    # the conn[]-table gather: cidx is index(const, i, k) — inline const form
    zu = [en for en in entries if en.u == "z" && en.v == "u"]
    @test length(zu) == 1 && zu[1].band.contracted == [("k", 1, 2)]
    @test occursin("\"const\"", skey(zu[1].band.cidx[1]))

    # values + exact pattern against the AD oracle (the scatter must
    # ACCUMULATE contracted entries landing on one column)
    f!, u0, p0, _, _ = build_evaluator(file)
    u = 0.2 .+ 0.6 .* abs.(sin.((1:length(u0)) .* 1.7))
    res = assemble_jacobian(file; u = u)
    Jad = ad_jacobian(f!, u, p0, 0.0)
    @test Matrix(res.J) ≈ Matrix(Jad) rtol = 1e-12
    @test res.pattern == (Jad .!= 0)
    # row 3 of z gathers column 5 twice: the two contracted points accumulate
    vm = Dict(n => i for (i, n) in enumerate(res.rownames))
    @test res.J[vm["z[3]"], vm["u[5]"]] ≈ (0.3 + 0.1 * 1) + (0.3 + 0.1 * 2) rtol = 1e-14
    @test res.structure == :general

    # eval-CSE on/off agree (hoisting must bind contracted names too)
    jp = prepare_jacobian(file; eval_cse = false)
    Jp = copy(jp.prototype); jp(Jp, u, p0, 0.0)
    @test Matrix(res.J) ≈ Matrix(Jp) rtol = 1e-14

    # serialize → parse round-trip carries the contracted dims
    doc = jacobian_document(file, "ContractedOps")
    _, back = parse_jacobian_block(doc["jacobians"]["ContractedOps"])
    @test length(back) == length(entries)
    for (a, b) in zip(entries, back)
        @test a.band.contracted == b.band.contracted
        @test skey(a.band.coef) == skey(b.band.coef)
    end
end

@testset "contracted-index guards" begin
    mk(rhs) = EarthSciAST.load(Dict{String,Any}("esm" => "0.8.0",
        "metadata" => Dict{String,Any}("name" => "G"),
        "index_sets" => Dict{String,Any}(
            "x" => Dict{String,Any}("kind" => "interval", "size" => 3)),
        "models" => Dict{String,Any}("G" => Dict{String,Any}(
            "variables" => Dict{String,Any}(
                "u" => Dict{String,Any}("type" => "state", "shape" => Any["x"],
                                        "default" => 0.5)),
            "equations" => Any[Dict{String,Any}(
                "lhs" => Dict{String,Any}("op" => "D", "args" => Any["u"], "wrt" => "t"),
                "rhs" => rhs)]))))
    agg(; kw...) = Dict{String,Any}("op" => "aggregate", "args" => Any[],
        "output_idx" => Any["i"],
        "ranges" => Dict{String,Any}("i" => Any[1, 3], "j" => Any[1, 3]),
        "expr" => Dict{String,Any}("op" => "index", "args" => Any["u", "j"]),
        (String(k) => v for (k, v) in kw)...)
    # non-smooth semiring reductions have no bands
    @test_throws EarthSciASTDiff.BandError jacobian_bands(
        mk(agg(reduce = "min")), "G")
    # a filter gate cannot be silently dropped from the derivative
    @test_throws EarthSciASTDiff.BandError jacobian_bands(
        mk(agg(reduce = "+",
               filter = Dict{String,Any}("op" => "<", "args" => Any["j", 3]))), "G")
end
