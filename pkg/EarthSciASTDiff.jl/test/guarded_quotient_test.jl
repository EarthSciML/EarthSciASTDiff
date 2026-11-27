# A guard holds over a composed derivative (spec §3.6, named steps): the
# derivative of g = ifelse(s > 0, 10^lg, 1) through the observed lg = x/s is
# composed as (∂g/∂lg)·_du_lg with ∂g/∂lg = ifelse(s > 0, …, 0), and where
# s = 0 the inner derivative _du_lg is NaN; the product is written
# ifelse(s > 0, …·_du_lg, 0), so the guard that keeps g finite keeps its
# derivative finite too, as MOSAIC's MTEM sulfate-poor log gamma needs. The
# shared golden (tests/goldens/guarded_quotient.points.json) holds the
# exact derivatives with the guard holding and open (matching central
# differences there); the Rust and Python bindings check it too.

const _GQ_ROOT = joinpath(@__DIR__, "..", "..", "..")

@testset "a guard holds over a composed derivative" begin
    path = joinpath(_GQ_ROOT, "tests", "valid", "guarded_quotient.esm")
    g = EarthSciASTDiff.JSON3.read(
        read(joinpath(_GQ_ROOT, "tests", "goldens", "guarded_quotient.points.json"), String), Dict{String, Any}
    )
    for pt in g["points"]
        d = EarthSciASTDiff.JSON3.read(read(path, String), Dict{String, Any})
        d["models"]["Gq"]["variables"]["c"]["default"] = pt["params"]["c"]
        file = EarthSciAST.load_string(EarthSciASTDiff.JSON3.write(d))
        names = doc_jacobian(file).rownames
        res = doc_jacobian(file; u = Float64[pt["u"][n] for n in names], t = g["t"])
        @test all(isfinite, nonzeros(res.J))
        for e in pt["entries"]
            r, c, v = e[1], e[2], e[3]
            got = res.J[findfirst(==(r), res.rownames), findfirst(==(c), res.colnames)]
            v == 0 ? (@test got == 0) : (@test isapprox(got, v; rtol = 1.0e-12))
        end
    end
end
