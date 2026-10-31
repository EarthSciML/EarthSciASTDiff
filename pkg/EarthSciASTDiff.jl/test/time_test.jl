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

@testset "time bands: tgrad! against a central difference in t" begin
    # tgrad_photolysis covers max(0, cos(ω t + φᵢ)) through an array observed,
    # interp.linear of t through a scalar observed, a datetime.hour factor
    # (a.e. zero) and a datetime.julian_day source. That source is O(1e-10)
    # here, below a central difference's resolution at JD ≈ 2.4e6; but the O3
    # row is the NO row plus it, so their difference pins it — to 1e-3, since
    # binary64 resolves JD to ~5e-10 days, coarse against the phase τ/86400.
    file = fixture("tgrad_photolysis.esm")
    c = ode_components(file)
    u = [0.2 + 0.6 * abs(sin(2.3 * i)) for i in eachindex(c.u0)]
    for τ in (0.3, 0.7, 1.4)
        dT = similar(u); c.tgrad!(dT, u, c.p, τ)
        h = 1.0e-6; up = similar(u); um = similar(u)
        c.f!(up, u, c.p, τ + h); c.f!(um, u, c.p, τ - h)
        @test dT ≈ (up .- um) ./ 2h rtol = 1.0e-7
        jd = -0.05 * sin(2π * τ / 86400) * 2π / 86400      # d/dt e0 cos(2π (JD - JD₀))
        for i in 1:4
            @test dT[c.var_map["O3[$i]"]] - dT[c.var_map["NO[$i]"]] ≈ jd rtol = 1.0e-3
        end
        @test dT[c.var_map["S"]] == 0.0                     # autonomous row
    end
end

@testset "time bands: value golden reproduces" begin
    golden_dir = joinpath(@__DIR__, "..", "..", "..", "tests", "goldens")
    for name in TIME_FIXTURES
        g = EarthSciASTDiff.JSON3.read(read(joinpath(golden_dir, "$name.time.jvals.json"), String))
        file = fixture("$name.esm")
        res0 = assemble_jacobian(file; wrt = :time, u = nothing)
        u = [Float64(g.u[Symbol(n)]) for n in res0.rownames]
        res = assemble_jacobian(file; wrt = :time, u = u, t = Float64(g.t))
        @test res.colnames == ["t"]
        got = Dict((res.rownames[i], "t") => v for (i, _, v) in zip(findnz(res.J)...))
        want = Dict((String(e[1]), String(e[2])) => Float64(e[3]) for e in g.entries)
        @test keys(got) == keys(want)
        @test all(isapprox(got[k], want[k]; rtol = 1.0e-12, atol = 1.0e-15) for k in keys(want))
    end
end
