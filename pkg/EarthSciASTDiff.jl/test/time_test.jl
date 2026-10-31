# Time-derivative bands (`wrt = :time`, spec §4.2): one scalar column `t`.
# Pinned byte-for-byte by tests/goldens/<fixture>.time.{bands,block,evaldoc}.json
# and by value in <fixture>.time.jvals.json, which the other bindings must
# reproduce. Kept in step with scripts/regenerate-goldens.jl.

using EarthSciASTDiff: _evaluation_document

const TIME_FIXTURES = ("tgrad_photolysis",)

@testset "time bands: goldens" begin
    golden_dir = joinpath(@__DIR__, "..", "..", "..", "tests", "goldens")
    gold(name, kind) = joinpath(golden_dir, "$name.time.$kind.json")
    for name in TIME_FIXTURES
        file = fixture("$name.esm")
        mname = first(keys(file.models))
        sv = sysview(file, mname)
        entries = jacobian_bands(sv; wrt = :time)
        @test !isempty(entries)
        @test all(
            en.v == "t" && isempty(en.band.cidx) && isempty(en.band.contracted)
                for en in entries
        )
        @test stable_json(Any[EarthSciASTDiff.serialize_band(en) for en in entries]) ==
            read(gold(name, "bands"), String)
        block = jacobian_document(file, mname; wrt = :time)["jacobians"][mname]
        @test block["wrt"] == "time"
        @test stable_json(block) == read(gold(name, "block"), String)
        doc, _ = _evaluation_document(sv, entries)
        @test stable_json(doc) == read(gold(name, "evaldoc"), String)
        doc0, _ = _evaluation_document(sv, entries; cse = false)
        p0 = isfile(gold(name, "evaldoc-nocse")) ? gold(name, "evaldoc-nocse") :
            gold(name, "evaldoc")
        @test stable_json(doc0) == read(p0, String)
        # the block parses back to the same bands
        wrt, back = parse_jacobian_block(block)
        @test wrt == :time && length(back) == length(entries)
    end
end

@testset "time bands: empty list for an autonomous model (f_t ≡ 0)" begin
    for name in (
            "bd_chem", "adv_interior", "flux_form_adv", "minmod_adv",
            "contracted_ops", "contracting_observed", "filtered_scan", "semiring_reduce",
        )
        file = fixture("$name.esm")
        @test isempty(jacobian_bands(file, first(keys(file.models)); wrt = :time))
    end
end

# ∂f/∂t as the n×1 Jacobian on the time axis, from the JvpEval document.
function _tgrad(file, u, τ)
    res = doc_jacobian(file; wrt = :time, u = u, t = τ)
    @test res.colnames == ["t"] && size(res.J, 2) == 1
    return Vector(res.J[:, 1])
end

@testset "time bands: ∂f/∂t against a central difference in t" begin
    # tgrad_photolysis covers max(0, cos(ω t + φᵢ)) through an array observed,
    # interp.linear of t through a scalar observed, a datetime.hour factor
    # (a.e. zero) and a datetime.julian_day source. That source is O(1e-10)
    # here, below a central difference's resolution at JD ≈ 2.4e6; but the O3
    # row is the NO row plus it, so their difference pins it — to 1e-3, since
    # binary64 resolves JD to ~5e-10 days, coarse against the phase τ/86400.
    file = fixture("tgrad_photolysis.esm")
    f!, u0, p, _, vm = build_evaluator(file)
    u = [0.2 + 0.6 * abs(sin(2.3 * i)) for i in eachindex(u0)]
    for τ in (0.3, 0.7, 1.4)
        dT = _tgrad(file, u, τ)
        h = 1.0e-6; up = similar(u); um = similar(u)
        f!(up, u, p, τ + h); f!(um, u, p, τ - h)
        @test dT ≈ (up .- um) ./ 2h rtol = 1.0e-7
        jd = -0.05 * sin(2π * τ / 86400) * 2π / 86400      # d/dt e0 cos(2π (JD - JD₀))
        for i in 1:4
            @test dT[vm["O3[$i]"]] - dT[vm["NO[$i]"]] ≈ jd rtol = 1.0e-3
        end
        @test dT[vm["S"]] == 0.0                     # autonomous row
    end
end

@testset "time bands: analytical ∂f/∂t of a small non-autonomous model" begin
    # D(w) = sin(t)·w + c·t (∂/∂t = cos(t)·w + c), D(v) = 2 v w (autonomous
    # row → 0).
    doc = Dict{String, Any}(
        "esm" => "1.0.0",
        "metadata" => Dict{String, Any}("name" => "TGrad"),
        "models" => Dict{String, Any}(
            "M" => Dict{String, Any}(
                "variables" => Dict{String, Any}(
                    "w" => Dict{String, Any}("type" => "unknown", "default" => 0.8),
                    "v" => Dict{String, Any}("type" => "unknown", "default" => 1.1),
                    "c" => Dict{String, Any}("type" => "parameter", "default" => 3.0)
                ),
                "equations" => Any[
                    Dict{String, Any}(
                        "lhs" => Dict{String, Any}("op" => "D", "args" => Any["w"], "wrt" => "t"),
                        "rhs" => Dict{String, Any}(
                            "op" => "+", "args" => Any[
                                Dict{String, Any}(
                                    "op" => "*", "args" => Any[
                                        Dict{String, Any}("op" => "sin", "args" => Any["t"]), "w",
                                    ]
                                ),
                                Dict{String, Any}("op" => "*", "args" => Any["c", "t"]),
                            ]
                        )
                    ),
                    Dict{String, Any}(
                        "lhs" => Dict{String, Any}("op" => "D", "args" => Any["v"], "wrt" => "t"),
                        "rhs" => Dict{String, Any}(
                            "op" => "*",
                            "args" => Any[2.0, "v", "w"]
                        )
                    ),
                ]
            )
        )
    )
    file = EarthSciAST.load_document(doc)
    entries = jacobian_bands(sysview(file, "M"); wrt = :time)
    @test length(entries) == 1 && entries[1].v == "t"

    f!, u0, p, _, vm = build_evaluator(file)
    τ = 0.7
    u = copy(u0)
    dT = _tgrad(file, u, τ)
    iw = vm["w"]; iv = vm["v"]
    @test dT[iw] ≈ cos(τ) * u[iw] + 3.0 rtol = 1.0e-12
    @test dT[iv] == 0.0
    # against a central finite difference in t through the tree-walk RHS
    h = 1.0e-6
    dup = similar(u); dum = similar(u)
    f!(dup, u, p, τ + h); f!(dum, u, p, τ - h)
    @test dT ≈ (dup .- dum) ./ 2h rtol = 1.0e-8

    # an autonomous system has an empty time column
    resa = doc_jacobian(fixture("bd_chem.esm"); wrt = :time, t = 0.3)
    @test nnz(resa.pattern) == 0 && all(iszero, resa.J)
end

@testset "time bands: value golden reproduces" begin
    golden_dir = joinpath(@__DIR__, "..", "..", "..", "tests", "goldens")
    for name in TIME_FIXTURES
        g = EarthSciASTDiff.JSON3.read(read(joinpath(golden_dir, "$name.time.jvals.json"), String))
        file = fixture("$name.esm")
        res0 = doc_jacobian(file; wrt = :time, u = nothing)
        u = [Float64(g.u[Symbol(n)]) for n in res0.rownames]
        res = doc_jacobian(file; wrt = :time, u = u, t = Float64(g.t))
        @test res.colnames == ["t"]
        got = Dict((res.rownames[i], "t") => v for (i, _, v) in zip(findnz(res.J)...))
        want = Dict((String(e[1]), String(e[2])) => Float64(e[3]) for e in g.entries)
        @test keys(got) == keys(want)
        @test all(isapprox(got[k], want[k]; rtol = 1.0e-12, atol = 1.0e-15) for k in keys(want))
    end
end
