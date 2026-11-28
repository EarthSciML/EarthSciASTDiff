# Time-derivative bands (`wrt = :time`, spec §4.2): one scalar column `t`.
# Pinned byte-for-byte by tests/goldens/<fixture>.time.{bands,block,evaldoc}.json
# and by value in <fixture>.time.jvals.json, which the other bindings must
# reproduce. Kept in step with scripts/regenerate-goldens.jl.

using EarthSciASTDiff: _evaluation_document

const TIME_FIXTURES = ("tgrad_photolysis", "tgrad_named_steps", "tgrad_surface")

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

@testset "time bands: data held fixed, elementwise observed read at a cell" begin
    # tgrad_data_fixed: only observed that move with `t` are followed (spec
    # §4.2), so `src` (parameters only, operands of mixed rank) is never read
    # at a cell; `jr`, declared without a shape, is (`index(jr, i, j)`).
    golden = joinpath(@__DIR__, "..", "..", "..", "tests", "goldens", "tgrad_data_fixed.time.bands.json")
    file = fixture("tgrad_data_fixed.esm")
    entries = jacobian_bands(sysview(file, "Box"); wrt = :time)
    @test length(entries) == 1 && entries[1].u == "c" && entries[1].v == "t"
    @test stable_json(Any[EarthSciASTDiff.serialize_band(en) for en in entries]) == read(golden, String)
end

@testset "time bands: companions keep the chain as named steps" begin
    # tgrad_named_steps (spec §4.2): w(t) → Tb (elementwise blend) → kr (a
    # rate of Tb) → c (at its cell) and s (under a sum over k). Each moving
    # observed gets one companion `_dt_<o>`, the bands read `_dt_kr`, and each
    # definition reads only the next companion down the chain. Eb (data and a
    # parameter weight only) does not move with t: no companion.
    file = fixture("tgrad_named_steps.esm")
    sv = sysview(file, "Column")
    entries = jacobian_bands(sv; wrt = :time)
    @test [(en.u, en.v) for en in entries] == [("c", "t"), ("s", "t")]
    reads(e) = sort!([n for n in EarthSciASTDiff.free_variables(e) if startswith(n, "_dt_")])
    @test reads(entries[1].band.coef) == ["_dt_kr"]
    @test entries[2].band.coef isa OpExpr && entries[2].band.coef.op == "faq"
    @test isempty(entries[2].band.coef.output_idx)
    comps = EarthSciASTDiff._time_companions(sv, entries)
    @test [(n, o) for (n, o, _) in comps] == [("_dt_Tb", "Tb"), ("_dt_kr", "kr"), ("_dt_w", "w")]
    defs = Dict(n => d for (n, _, d) in comps)
    @test reads(defs["_dt_kr"]) == ["_dt_Tb"]
    @test reads(defs["_dt_Tb"]) == ["_dt_w"]
    @test isempty(reads(defs["_dt_w"]))
    block = jacobian_document(file, "Column"; wrt = :time)["jacobians"]["Column"]
    @test [c["name"] for c in block["intermediates"]] == ["_dt_Tb", "_dt_kr", "_dt_w"]
    @test [c["of"] for c in block["intermediates"]] == ["Tb", "kr", "w"]
    # the evaluation document declares each like its observed
    doc, _ = _evaluation_document(sv, entries)
    vars = doc["models"]["JacobianEval"]["variables"]
    @test vars["_dt_Tb"]["shape"] == Any["x", "z"] && !haskey(vars["_dt_w"], "shape")
    @test !haskey(vars, "_dt_Eb")
    # a time pair takes `t` second (spec §4.4): the time bands of each state
    # entry's coefficient, through the same companions, never inlined
    e2 = jacobian_bands2(sv; wrt = :time => :states)
    @test !isempty(e2) && all(en -> en.v == "t" && !isempty(reads(en.band.coef)), e2)
    pblk = jacobian_document(file, "Column"; wrt = :time => :states)["jacobians"]["Column"]
    @test [c["name"] for c in pblk["intermediates"]] == ["_dt_Tb", "_dt_kr", "_dt_w"]
    # structure-only entries: the same points
    so = jacobian_bands(sv; wrt = :time, coefficients = false)
    @test [(en.u, en.band.rows) for en in so] == [(en.u, en.band.rows) for en in entries]
end

@testset "time bands: a companion name is clear of the source's names" begin
    d = EarthSciASTDiff.JSON3.read(read(joinpath(FIXTURES, "tgrad_named_steps.esm"), String), Dict{String, Any})
    d["models"]["Column"]["variables"]["_dt_w"] = Dict{String, Any}("type" => "parameter", "default" => 1.0)
    file = EarthSciAST.load_string(EarthSciASTDiff.JSON3.write(d))
    sv = sysview(file, "Column")
    comps = EarthSciASTDiff._time_companions(sv, jacobian_bands(sv; wrt = :time))
    @test [n for (n, _, _) in comps] == ["_dt_Tb", "_dt_kr", "_dt_w_"]
    @test EarthSciASTDiff._companion_names(sv, ["P.v", "w"]) == Dict("P.v" => "P._dt_v", "w" => "_dt_w_")
    # a component path is taken too: `P._dt_v` names a subsystem of a source
    # with a variable `P._dt_v.x`
    taken = EarthSciASTDiff._component_paths!(Set{String}(), ["P._dt_v.x", "w"])
    @test taken == Set(["P", "P._dt_v"])
end

@testset "time bands: companions against a central difference in t" begin
    # The composed f_t equals the derivative of the source right-hand side,
    # through the chain w → Tb → kr and the contraction over k, with the
    # data-fed P held at its value. Away from the floor kinks of w (dt = 3).
    file = fixture("tgrad_named_steps.esm")
    f!, u0, p, _, vm = tw_build(file)
    for (s, τ) in enumerate((0.3, 1.7, 4.4))
        u = [0.2 + 0.6 * abs(sin(2.3 * i + s)) for i in eachindex(u0)]
        dT = Vector(doc_jacobian(file; wrt = :time, u = u, t = τ).J[:, 1])
        h = 1.0e-5; up = similar(u); um = similar(u)
        f!(up, u, p, τ + h); f!(um, u, p, τ - h)
        @test dT ≈ (up .- um) ./ 2h rtol = 1.0e-6
        @test all(!iszero, dT)
    end
end

@testset "time bands: a surface-only companion is a region, not a gate" begin
    # emis exists only at lev = 1 and reads the observed delP there: its
    # companion is one faq over the whole shape reading a makearray whose
    # second region is the surface (spec §4.2), and f_t is the derivative of
    # the source right-hand side
    file = fixture("tgrad_surface.esm")
    blk = jacobian_document(file, "Box"; wrt = :time)["jacobians"]["Box"]
    c = only(x for x in blk["intermediates"] if x["name"] == "_dt_emis")
    rhs = c["rhs"]
    @test rhs["op"] == "faq" && rhs["expr"]["op"] == "index"
    ma = rhs["expr"]["args"][1]
    @test ma["op"] == "makearray" && ma["regions"] == Any[Any[Any[1, 3], Any[1, 4]], Any[Any[1, 3], Any[1, 1]]]
    @test ma["values"][1] == 0
    f!, u0, p, _, _ = tw_build(file)
    for τ in (700.0, 2900.0)
        u = [0.2 + 0.6 * abs(sin(2.3 * i)) for i in eachindex(u0)]
        dT = Vector(doc_jacobian(file; wrt = :time, u = u, t = τ).J[:, 1])
        h = 1.0e-3; up = similar(u); um = similar(u)
        f!(up, u, p, τ + h); f!(um, u, p, τ - h)
        @test dT ≈ (up .- um) ./ 2h rtol = 1.0e-7
        @test count(!iszero, dT) == 3
    end
end

# tgrad_data_fixed with `edit!(model_dict)` applied.
function _data_fixed_variant(edit!)
    path = joinpath(FIXTURES, "tgrad_data_fixed.esm")
    d = EarthSciASTDiff.JSON3.read(read(path, String), Dict{String, Any})
    edit!(d["models"]["Box"])
    return EarthSciAST.load_string(EarthSciASTDiff.JSON3.write(d))
end
_jrdef!(m, rhs) = (m["equations"][1]["rhs"] = rhs; m)
const _JR_T = Dict("op" => "*", "args" => Any["J0", Dict("op" => "cos", "args" => Any[Dict("op" => "*", "args" => Any["omega", "t"])])])

@testset "time bands: a time pair follows what its second axis seeds" begin
    # kc = c·c moves only with the states: the plain time pass leaves
    # `index(kc, i, j)` an opaque read, the (time, states) pair follows it.
    # Each pair keeps its own pass, so the order of emission is immaterial.
    file = _data_fixed_variant() do m
        m["variables"]["kc"] = Dict{String, Any}("type" => "unknown")
        push!(m["equations"], Dict("lhs" => "kc", "rhs" => Dict("op" => "*", "args" => Any["c", "c"])))
        body = m["equations"][4]["rhs"]["expr"]
        m["equations"][4]["rhs"]["expr"] = Dict(
            "op" => "+", "args" => Any[
                body, Dict(
                    "op" => "*", "args" => Any[
                        Dict("op" => "index", "args" => Any["kc", "i", "j"]),
                        Dict("op" => "cos", "args" => Any[Dict("op" => "*", "args" => Any["omega", "t"])]),
                    ]
                ),
            ]
        )
    end
    @test !isempty(jacobian_bands2(sysview(file, "Box"); wrt = :time => :states))
    pair = jacobian_document(file, "Box"; wrt = :time => :states)["jacobians"]["Box"]
    both = jacobian_document(file, "Box"; wrt = [:time, :time => :states])["jacobians"]["Box"]
    @test both[2] == pair
    @test both[1] == jacobian_document(file, "Box"; wrt = :time)["jacobians"]["Box"]
end

@testset "time bands: an elementwise observed read at a cell refuses what it cannot read" begin
    sv = sysview(fixture("tgrad_data_fixed.esm"), "Box")
    ctx = EarthSciASTDiff._ctx(sv)
    obs = EarthSciASTDiff._observed_defs(sv)
    rhs = only(eq.rhs for eq in sv.equations if eq.lhs isa OpExpr && eq.lhs.op == "D")
    # follow `jr` only (as the time pass does); a name of no known rank in its
    # definition (here: no declared scalar is named) is refused
    jr = Dict{String, ASTExpr}("jr" => obs["jr"])
    @test_throws EarthSciASTDiff.InlineError EarthSciASTDiff.inline_observed(rhs, jr; shapes = ctx.shapes, rank_obs = obs)
    sc = Set{String}(String(n) for (n, v) in sv.variables if (v.shape === nothing || isempty(v.shape)) && !haskey(obs, String(n)))
    @test !("jr" in EarthSciASTDiff.free_variables(EarthSciASTDiff.inline_observed(rhs, jr; shapes = ctx.shapes, rank_obs = obs, scalars = sc)))
    # a partial index (still an array) inside the definition
    partial = _data_fixed_variant() do m
        _jrdef!(m, Dict("op" => "+", "args" => Any[_JR_T, Dict("op" => "index", "args" => Any["Eb", 1])]))
    end
    @test_throws EarthSciASTDiff.BandError jacobian_bands(sysview(partial, "Box"); wrt = :time)
    # a global named like the read site's index `j`: the read is a column
    # (spec §4.2, companions), not inlined, and the companion row of `jr`
    # names its own indices clear of `j`, so nothing is captured
    capture = _data_fixed_variant() do m
        m["variables"]["j"] = Dict{String, Any}("type" => "parameter", "default" => 2.0)
        _jrdef!(m, Dict("op" => "*", "args" => Any[_JR_T, "j"]))
    end
    svc = sysview(capture, "Box")
    ec = jacobian_bands(svc; wrt = :time)
    @test length(ec) == 1
    (_, _, dc), = EarthSciASTDiff._time_companions(svc, ec)
    @test dc.output_idx == Any["_r1", "_r2"] && "j" in EarthSciASTDiff.free_variables(dc)
    # the full index of an array inside the definition is a scalar
    full = _data_fixed_variant() do m
        _jrdef!(m, Dict("op" => "*", "args" => Any[_JR_T, Dict("op" => "index", "args" => Any["Eb", 1, 2])]))
    end
    @test length(jacobian_bands(sysview(full, "Box"); wrt = :time)) == 1
end

@testset "time bands: a companion's extents are read or refused, never guessed" begin
    BE = EarthSciASTDiff.BandError
    # operands over different index sets: an undeclared shape is their union
    # by index-set name, in order of first appearance (esm-spec §4.3.4):
    # `jr = Ea·cos(…) + Eb` spans [x, y], and its companion is read at the
    # row's cell
    mixed = _data_fixed_variant() do m
        _jrdef!(m, Dict("op" => "+", "args" => Any[Dict("op" => "*", "args" => Any["Ea", _JR_T["args"][2]]), "Eb"]))
    end
    ms = stable_json(Any[EarthSciASTDiff.serialize_band(en) for en in jacobian_bands(sysview(mixed, "Box"); wrt = :time)])
    @test occursin("\"_dt_jr\",\"i\",\"j\"", ms)
    # a partial index (still an array) inside the definition
    partial = _data_fixed_variant() do m
        _jrdef!(m, Dict("op" => "+", "args" => Any[_JR_T, Dict("op" => "index", "args" => Any["Eb", 1])]))
    end
    @test_throws BE jacobian_bands(sysview(partial, "Box"); wrt = :time)
    # a band whose column index does not read every dimension of the observed
    b = EarthSciASTDiff.Band(
        [(1, 3)], Any["i"], ASTExpr[VarExpr("i")], EarthSciASTDiff.lit(1),
        Tuple{String, Int, Int}[], nothing
    )
    @test_throws BE EarthSciASTDiff._through_companion(b, "jr", "_dt_jr", [3, 2], true)
    # and in a time pair: the first `n` column indices are the pair's own
    @test_throws BE EarthSciASTDiff._pair_through_companion(b, 0, "jr", "_dt_jr", [3, 2])
    @test EarthSciASTDiff._pair_through_companion(b, 0, "jr", "_dt_jr", [3]).cidx == ASTExpr[]
end

@testset "time bands: moving observed that read each other are refused" begin
    loop = _data_fixed_variant() do m
        m["variables"]["jr"]["shape"] = Any["x", "y"]
        m["variables"]["q"] = Dict{String, Any}("type" => "unknown", "shape" => Any["x", "y"])
        cell(v) = Dict("op" => "index", "args" => Any[v, "i", "j"])
        each(e) = Dict(
            "op" => "faq", "args" => Any[], "output_idx" => Any["i", "j"],
            "ranges" => Dict("i" => Dict("from" => "x"), "j" => Dict("from" => "y")), "expr" => e
        )
        _jrdef!(m, each(Dict("op" => "+", "args" => Any[_JR_T["args"][2], cell("q")])))
        push!(m["equations"], Dict("lhs" => "q", "rhs" => each(Dict("op" => "*", "args" => Any[2, cell("jr")]))))
    end
    err = try
        jacobian_bands(sysview(loop, "Box"); wrt = :time)
        nothing
    catch e
        e
    end
    @test err isa EarthSciASTDiff.BandError && occursin("cyclic", err.msg)
end

@testset "time bands: an undeclared observed defined by an array reference" begin
    # g = sq(wt) (an array-valued operator reference, declared without a
    # shape) moves with t through wt: its extents are the reference body's,
    # and the band reads its companion
    d = EarthSciASTDiff.JSON3.read(read(joinpath(@__DIR__, "..", "..", "..", "tests", "valid", "two_shape_stencil.esm"), String), Dict{String, Any})
    m = d["models"]["M"]
    m["variables"]["wt"] = Dict{String, Any}("type" => "unknown", "shape" => Any["x"])
    m["variables"]["g"] = Dict{String, Any}("type" => "unknown")
    push!(
        m["equations"], Dict(
            "lhs" => "wt", "rhs" => Dict(
                "op" => "faq", "args" => Any[], "output_idx" => Any["i"], "ranges" => Dict("i" => Dict("from" => "x")),
                "expr" => Dict("op" => "+", "args" => Any[2, Dict("op" => "cos", "args" => Any[Dict("op" => "*", "args" => Any["k", "t", "i"])])])
            )
        )
    )
    push!(m["equations"], Dict("lhs" => "g", "rhs" => Dict("op" => "apply_operator", "args" => Any[], "name" => "sq", "bindings" => Dict("f" => "wt", "n" => 6))))
    m["equations"][1]["rhs"]["expr"] = Dict(
        "op" => "+", "args" => Any[m["equations"][1]["rhs"]["expr"], Dict("op" => "index", "args" => Any["g", "i"])]
    )
    file = EarthSciAST.load_string(EarthSciASTDiff.JSON3.write(d))
    sv = sysview(file, "M")
    entries = jacobian_bands(sv; wrt = :time)
    @test [en.u for en in entries] == ["a"]
    comps = EarthSciASTDiff._time_companions(sv, entries)
    @test [(n, o) for (n, o, _) in comps] == [("_dt_g", "g"), ("_dt_wt", "wt")]
    f!, u0, p, _, _ = tw_build(file)
    u = [0.2 + 0.1 * i for i in eachindex(u0)]
    for τ in (0.4, 1.3)
        dT = Vector(doc_jacobian(file; wrt = :time, u = u, t = τ).J[:, 1])
        h = 1.0e-5; up = similar(u); um = similar(u)
        f!(up, u, p, τ + h); f!(um, u, p, τ - h)
        @test dT ≈ (up .- um) ./ 2h rtol = 1.0e-6
    end
end

@testset "time bands: empty list for an autonomous model (f_t ≡ 0)" begin
    for name in (
            "bd_chem", "adv_interior", "flux_form_adv", "minmod_adv",
            "contracted_ops", "contracting_observed", "filtered_scan", "semiring_reduce",
            "pre_operator",
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
    # the esm:std linear template of t through a scalar observed, an hour
    # factor (a.e. zero) and a julian_day source. That source is O(1e-10)
    # here, below a central difference's resolution at JD ≈ 2.4e6; but the O3
    # row is the NO row plus it, so their difference pins it — to 1e-3, since
    # binary64 resolves JD to ~5e-10 days, coarse against the phase τ/86400.
    file = fixture("tgrad_photolysis.esm")
    f!, u0, p, _, vm = tw_build(file)
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

    f!, u0, p, _, vm = tw_build(file)
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
