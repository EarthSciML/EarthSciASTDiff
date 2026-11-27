# A makearray whose region values are cell reads (scalars), read at a cell
# through an observed: each value is read as itself, never wrapped in an
# `index` with no subscripts. Byte-identical to the shared golden, and the
# values agree with ForwardDiff.

@testset "makearray of cell reads: a scalar region value is read as itself" begin
    golden = joinpath(@__DIR__, "..", "..", "..", "tests", "goldens", "makearray_of_reads.bands.json")
    file = fixture("makearray_of_reads.esm")
    entries = jacobian_bands(sysview(file, "M"))
    @test stable_json(Any[EarthSciASTDiff.serialize_band(en) for en in entries]) == read(golden, String)
    @test [(en.u, en.v) for en in entries] == [("u", "u"), ("u", "v"), ("v", "v")]
    # structure-only entries are structural (spec §2.1): a superset
    so = jacobian_bands(sysview(file, "M"); coefficients = false)
    @test issubset(Set((en.u, en.v) for en in entries), Set((en.u, en.v) for en in so))
    f!, u0, p, _, _ = tw_build(file)
    for _ in 1:3
        u = 0.3 .+ rand(length(u0))
        res = doc_jacobian(file; u = u, p = p)
        Jad = ad_jacobian(f!, u, p, 0.0)
        @test maximum(abs.(Matrix(res.J) .- Matrix(Jad))) ≤ 1.0e-14
    end
end
