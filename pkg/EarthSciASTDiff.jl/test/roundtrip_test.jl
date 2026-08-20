# The "jacobians" document block: emission, parse round-trip, and golden
# stability (the cross-language conformance surface).

@testset "document block round-trip" begin
    file = fixture("bd_chem.esm")
    doc = jacobian_document(file, "Chem")
    @test haskey(doc, "jacobians")
    block = doc["jacobians"]["Chem"]
    @test block["wrt"] == "states"
    @test block["structure"] == "block_diagonal"
    @test haskey(block, "factorization")

    wrt, entries = parse_jacobian_block(block)
    @test wrt == :states
    original = jacobian_bands(file, "Chem")
    @test length(entries) == length(original)
    for (a, b) in zip(entries, original)
        @test a.u == b.u && a.v == b.v
        @test a.band.rows == b.band.rows
        @test a.band.ridx == b.band.ridx
        @test stable_json(EarthSciAST.serialize_expression(a.band.coef)) ==
              stable_json(EarthSciAST.serialize_expression(b.band.coef))
    end
end

@testset "stable_json determinism" begin
    d1 = Dict("b" => 1, "a" => Dict("z" => [1, 2], "y" => "s"))
    d2 = Dict("a" => Dict("y" => "s", "z" => [1, 2]), "b" => 1)
    @test stable_json(d1) == stable_json(d2)
    @test stable_json(d1) == "{\"a\":{\"y\":\"s\",\"z\":[1,2]},\"b\":1}"
end

@testset "goldens" begin
    # Golden = stable_json of the serialized entries. Regenerate with
    # scripts/regenerate-goldens.jl after an INTENDED rule/format change.
    golden_dir = joinpath(@__DIR__, "..", "..", "..", "tests", "goldens")
    for name in ("bd_chem", "adv_interior", "contracted_ops", "flux_form_adv", "minmod_adv")
        file = fixture("$name.esm")
        entries = jacobian_bands(file, first(keys(file.models)))
        text = stable_json(Any[EarthSciASTDiff.serialize_band(en) for en in entries])
        gpath = joinpath(golden_dir, "$name.bands.json")
        if isfile(gpath)
            @test text == read(gpath, String)
        else
            write(gpath, text)                 # first run seeds the golden
            @test isfile(gpath)
        end
    end
end
