# An elementwise observed read at a cell, its array operands aligned by
# index-set name (esm-spec §4.3.4): byte-identical to the shared golden.

@testset "elementwise observed read at a cell: operands aligned by index-set name" begin
    golden = joinpath(@__DIR__, "..", "..", "..", "tests", "goldens", "elementwise_aligned.bands.json")
    entries = jacobian_bands(sysview(fixture("elementwise_aligned.esm"), "Box"))
    @test stable_json(Any[EarthSciASTDiff.serialize_band(en) for en in entries]) == read(golden, String)
    # row c reads q at its own y index and r with its axes swapped
    byc = Dict(en.v => EarthSciASTDiff.serialize_band(en) for en in entries if en.u == "c")
    @test byc["q"]["col_idx"] == ["j"]
    @test byc["r"]["col_idx"] == ["j", "i"]
end

# elementwise_aligned with `edit!(document_dict)` applied.
function _aligned_variant(edit!)
    path = joinpath(FIXTURES, "elementwise_aligned.esm")
    d = EarthSciASTDiff.JSON3.read(read(path, String), Dict{String, Any})
    edit!(d)
    return EarthSciAST.load_string(EarthSciASTDiff.JSON3.write(d))
end

@testset "elementwise observed read at a cell: what cannot be aligned is refused" begin
    # an operand over an index set the observed is not shaped over
    foreign = _aligned_variant() do d
        d["index_sets"]["z"] = Dict{String, Any}("kind" => "interval", "size" => 2)
        m = d["models"]["Box"]
        m["variables"]["p"] = Dict{String, Any}("type" => "parameter", "shape" => Any["z"], "default" => 0.1)
        push!(m["equations"][1]["rhs"]["args"], "p")
    end
    err = try
        jacobian_bands(sysview(foreign, "Box")); nothing
    catch e
        e
    end
    @test err isa EarthSciASTDiff.InlineError && occursin("index set `z`", sprint(showerror, err))
    # an observed declared without a shape that mixes ranks: its shape (the
    # union of its operands' index sets) is not known from ranks alone
    undeclared = _aligned_variant() do d
        delete!(d["models"]["Box"]["variables"]["w"], "shape")
    end
    err = try
        jacobian_bands(sysview(undeclared, "Box")); nothing
    catch e
        e
    end
    @test err isa EarthSciASTDiff.InlineError && occursin("declared without a shape", sprint(showerror, err))
end
