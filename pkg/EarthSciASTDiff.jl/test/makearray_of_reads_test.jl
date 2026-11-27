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

# makearray_of_reads with the region values replaced by `vals` (and `edit!`
# applied to the model).
function _mar_variant(vals; edit! = m -> nothing)
    d = EarthSciASTDiff.JSON3.read(read(joinpath(FIXTURES, "makearray_of_reads.esm"), String), Dict{String, Any})
    m = d["models"]["M"]
    m["equations"][1]["rhs"]["expr"]["args"][1]["values"] = vals
    edit!(m)
    return EarthSciAST.load_string(EarthSciASTDiff.JSON3.write(d))
end
_ix(a...) = Dict("op" => "index", "args" => Any[a...])

@testset "makearray of cell reads: a base of no declared shape is ranked, or refused" begin
    # a cell read of an observed declared without a shape (`w = 2u`,
    # elementwise): ranked as an operand of an elementwise definition, so it
    # is a scalar, read as itself
    file = _mar_variant(
        Any[_ix("w", "k"), _ix("v", "k")];
        edit! = m -> begin
            m["variables"]["w"] = Dict{String, Any}("type" => "unknown")
            push!(m["equations"], Dict("lhs" => "w", "rhs" => Dict("op" => "*", "args" => Any[2, "u"])))
        end
    )
    entries = jacobian_bands(sysview(file, "M"))
    @test [(en.u, en.v) for en in entries] == [("u", "u"), ("u", "v"), ("v", "v")]
    f!, u0, p, _, _ = tw_build(file)
    u = 0.3 .+ rand(length(u0))
    @test maximum(abs.(Matrix(doc_jacobian(file; u = u, p = p).J) .- Matrix(ad_jacobian(f!, u, p, 0.0)))) ≤ 1.0e-14
    IE = EarthSciASTDiff.InlineError
    # more subscripts than the array has dimensions
    @test_throws IE jacobian_bands(sysview(_mar_variant(Any[_ix("u", "k", "k"), _ix("v", "k")]), "M"))
    # a partial read of an array whose ranges do not all start at 1
    shifted = Dict(
        "op" => "faq", "args" => Any[], "output_idx" => Any["a", "b"],
        "ranges" => Dict("a" => Dict("from" => "x"), "b" => Any[2, 3]),
        "expr" => _ix("u", "a")
    )
    @test_throws IE jacobian_bands(sysview(_mar_variant(Any[_ix("u", "k"), _ix(shifted, "k")]), "M"))
end
