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
    # row s, a whole-array right-hand side, aligns its operands the same way
    bys = [EarthSciASTDiff.serialize_band(en) for en in entries if en.u == "s"]
    @test Set(b["col_idx"] for b in bys if b["col"] == "q") == Set([["_r2"]])
    @test only(b["col_idx"] for b in bys if b["col"] == "r") == ["_r2", "_r1"]
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
    # an observed declared without a shape spans the union of its named
    # operands' index sets, in order of first appearance (esm-spec §4.3.4):
    # `w = c·q + k + r` spans [x, y], as declared, and has the same bands
    undeclared = _aligned_variant() do d
        delete!(d["models"]["Box"]["variables"]["w"], "shape")
    end
    golden = joinpath(@__DIR__, "..", "..", "..", "tests", "goldens", "elementwise_aligned.bands.json")
    @test stable_json(
        Any[EarthSciASTDiff.serialize_band(en) for en in jacobian_bands(sysview(undeclared, "Box"))]
    ) == read(golden, String)
end

# what the refusal throws, or `nothing`
function _refusal(f)
    return try
        jacobian_bands(sysview(f, "Box")); nothing
    catch e
        e
    end
end

@testset "elementwise observed read at a cell: arity, mixed ranks, whole arrays" begin
    # a read whose arity is not the observed's rank
    for ix in (Any["w", "i"], Any["w", "i", "j", "i"])
        bad = _aligned_variant() do d
            d["models"]["Box"]["equations"][2]["rhs"]["expr"]["args"][1]["args"] = ix
        end
        err = _refusal(bad)
        @test err isa EarthSciASTDiff.InlineError &&
            occursin("of rank 2 is read at a cell with $(length(ix) - 1) indices", sprint(showerror, err))
    end
    # an observed declared without a shape mixing ranks below its top node
    nested = _aligned_variant() do d
        d["index_sets"]["z"] = Dict{String, Any}("kind" => "interval", "size" => 2)
        m = d["models"]["Box"]
        m["variables"]["p"] = Dict{String, Any}("type" => "parameter", "shape" => Any["z"], "default" => 0.1)
        m["variables"]["u"] = Dict{String, Any}("type" => "unknown")
        push!(
            m["equations"], Dict{String, Any}(
                "lhs" => "u", "rhs" => Dict{String, Any}(
                    "op" => "-", "args" => Any[Dict{String, Any}("op" => "*", "args" => Any["c", "p"])]
                )
            )
        )
        push!(m["equations"][1]["rhs"]["args"], "u")
    end
    err = _refusal(nested)
    # `u = -(c·p)` spans [x, y, z] (§4.3.4), and `w` over [x, y] is not
    # shaped over `z`
    @test err isa EarthSciASTDiff.InlineError &&
        occursin("reads `u` over (x, y, z), whose index set `z`", sprint(showerror, err))
    # a whole-array operand over an index set the row is not shaped over
    foreign = _aligned_variant() do d
        d["index_sets"]["z"] = Dict{String, Any}("kind" => "interval", "size" => 2)
        m = d["models"]["Box"]
        m["variables"]["p"] = Dict{String, Any}("type" => "parameter", "shape" => Any["z"], "default" => 0.1)
        push!(m["equations"][end]["rhs"]["args"][1]["args"], "p")
    end
    err = _refusal(foreign)
    @test err isa EarthSciASTDiff.BandError &&
        occursin("reads `p` over (z) at array level, whose index set `z`", sprint(showerror, err))
end

@testset "elementwise observed read at a cell: a synthetic lift index set aligns by position" begin
    # `g` over the pointwise lift's synthetic `_liftdim1_3` (an anonymous
    # dimension of x's extent), read by `h` over [x] at a cell
    lifted = _aligned_variant() do d
        d["index_sets"]["_liftdim1_3"] = Dict{String, Any}("kind" => "interval", "size" => 3)
        m = d["models"]["Box"]
        V = m["variables"]
        V["g"] = Dict{String, Any}("type" => "unknown", "shape" => Any["_liftdim1_3"], "default" => 1.0)
        V["h"] = Dict{String, Any}("type" => "unknown", "shape" => Any["x"])
        V["e"] = Dict{String, Any}("type" => "unknown", "shape" => Any["x"], "default" => 1.0)
        dt(n) = Dict{String, Any}("op" => "D", "args" => Any[n], "wrt" => "t")
        cell(n, X) = Dict{String, Any}(
            "op" => "faq", "output_idx" => Any["i"], "args" => Any[],
            "ranges" => Dict{String, Any}("i" => Dict{String, Any}("from" => X)),
            "expr" => Dict{String, Any}(
                "op" => "-", "args" => Any[Dict{String, Any}("op" => "index", "args" => Any[n, "i"])]
            )
        )
        push!(m["equations"], Dict{String, Any}("lhs" => "h", "rhs" => Dict{String, Any}("op" => "*", "args" => Any[2, "g"])))
        push!(m["equations"], Dict{String, Any}("lhs" => dt("g"), "rhs" => cell("g", "_liftdim1_3")))
        push!(m["equations"], Dict{String, Any}("lhs" => dt("e"), "rhs" => cell("h", "x")))
    end
    sv = sysview(lifted, "Box")
    @test !haskey(EarthSciASTDiff._ctx(sv).axes, "g")
    entries = jacobian_bands(sv)
    eg = only(en for en in entries if en.u == "e" && en.v == "g")
    @test EarthSciASTDiff.serialize_band(eg)["col_idx"] == ["i"]
end
