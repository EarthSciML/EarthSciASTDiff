# The evaluation document of a flattened system (esm-jacobian-spec.md §8.1):
# its scoped names declared in the component tree, so EarthSciAST loads it,
# flattens it back to the source's names and builds its right-hand side, whose
# band fields give the Jacobian. Pinned byte-for-byte by
# tests/goldens/coupled_chem_advection.evaldoc*.json, which the other bindings
# must reproduce, and by value by coupled_chem_advection.jvals.json.

using EarthSciASTDiff: _evaluation_document

const _FLAT_GOLDENS = joinpath(@__DIR__, "..", "..", "..", "tests", "goldens")

@testset "flattened evaluation document: goldens and layout" begin
    flat = EarthSciAST.flatten(fixture("coupled_chem_advection.esm"))
    sv = sysview(flat)
    entries = jacobian_bands(sv)
    for (cse, tag) in ((true, "evaldoc"), (false, "evaldoc-nocse"))
        doc, fields = _evaluation_document(sv, entries; cse = cse)
        @test stable_json(doc) ==
            read(joinpath(_FLAT_GOLDENS, "coupled_chem_advection.$tag.json"), String)
        @test fields == ["J_$k" for k in eachindex(entries)]
        models = doc["models"]
        @test sort!(collect(keys(models))) == ["Advection", "JacobianEval", "SimpleOzone"]
        # no scoped name is declared anywhere; the derived model holds the
        # band fields (and the hoisted observed), the components the source
        @test !any(occursin('.', n) for m in values(models) for n in keys(m["variables"]))
        @test all(haskey(models["JacobianEval"]["variables"], f) for f in fields)
        @test sort!(collect(keys(models["SimpleOzone"]["variables"]))) ==
            ["M", "NO", "NO2", "O3", "T", "jNO2"]
        @test models["SimpleOzone"]["variables"]["O3"]["shape"] == ["lon", "lat"]
        # each state's zeroed derivative is in its component, fully scoped
        @test [eq["lhs"]["args"][1] for eq in models["SimpleOzone"]["equations"]] ==
            ["SimpleOzone.O3", "SimpleOzone.NO", "SimpleOzone.NO2"]
        # flattening the document gives back the source's names
        back = EarthSciAST.flatten(EarthSciAST.load_string(stable_json(doc)))
        for d in (flat.state_variables, flat.parameters)
            @test all(n -> haskey(back.state_variables, n) || haskey(back.parameters, n), keys(d))
        end
        @test all(f -> haskey(back.state_variables, "JacobianEval.$f"), fields)
    end
end

@testset "flattened evaluation document: values" begin
    file = fixture("coupled_chem_advection.esm")
    flat = EarthSciAST.flatten(file)
    g = EarthSciASTDiff.JSON3.read(
        read(joinpath(_FLAT_GOLDENS, "coupled_chem_advection.jvals.json"), String)
    )
    u = Dict{String, Float64}(String(k) => Float64(v) for (k, v) in pairs(g["u"]))
    @test sort!(collect(keys(u))) == sort!(flat_state_cells(flat))
    want = Dict{Tuple{String, String}, Float64}(
        (String(e[1]), String(e[2])) => Float64(e[3]) for e in g["entries"]
    )
    close(a, b) = isapprox(a, b; rtol = 1.0e-10, atol = 1.0e-13)
    for cse in (true, false), compiler in (:native, :interpreter)
        J, _ = flat_band_jacobian(flat; u = u, t = g["t"], cse = cse, compiler = compiler)
        @test Set(keys(J)) == Set(keys(want))
        @test all(close(J[k], want[k]) for k in keys(want))
    end
    # The golden is the source's Jacobian: ForwardDiff through EarthSciAST's
    # evaluator of the flattened source, every nonzero on the band pattern.
    prob = EarthSciAST.esm_problem(file, (0.0, 1.0))
    vm = Dict{String, Int}(String(k) => Int(v) for (k, v) in prob.var_map)
    x = zeros(length(prob.u0))
    for (c, v) in u
        x[vm[c]] = v
    end
    A = ad_jacobian(prob.f!, x, prob.p, Float64(g["t"]))
    names = Dict(v => k for (k, v) in vm)
    I, Jc, V = findnz(A)
    for k in eachindex(I)
        key = (names[I[k]], names[Jc[k]])
        V[k] == 0 || @test haskey(want, key)
    end
    @test all(close(A[vm[r], vm[c]], v) for ((r, c), v) in want)
end

@testset "flattened evaluation document: nested and coupled components" begin
    # Two coupled models, one with a nested subsystem: scoped names three
    # deep (`Chem.Inner.k`) go into `subsystems`, and a `variable_map`
    # couples the models' states.
    doc = EarthSciAST.load_string(
        """
        {"esm": "1.1.0", "metadata": {"name": "Nested"},
         "models": {
          "Chem": {
            "variables": {"c": {"type": "unknown", "default": 1.0},
                          "r": {"type": "parameter", "default": 0.5}},
            "equations": [{"lhs": {"op": "D", "args": ["c"], "wrt": "t"},
                           "rhs": {"op": "*", "args": [{"op": "-", "args": ["Inner.k"]}, "c", "c", "r"]}}],
            "subsystems": {"Inner": {
              "variables": {"k": {"type": "unknown", "default": 0.3}},
              "equations": [{"lhs": {"op": "D", "args": ["k"], "wrt": "t"},
                             "rhs": {"op": "*", "args": [-0.1, "k"]}}]}}},
          "Source": {
            "variables": {"s": {"type": "unknown", "default": 2.0},
                          "q": {"type": "parameter", "default": 0.1}},
            "equations": [{"lhs": {"op": "D", "args": ["s"], "wrt": "t"},
                           "rhs": {"op": "*", "args": [{"op": "-", "args": ["q"]}, "s"]}}]}},
         "coupling": [{"type": "variable_map", "from": "Source.s", "to": "Chem.r",
                       "transform": "param_to_var"}]}
        """
    )
    flat = EarthSciAST.flatten(doc)
    ev, fields = evaluation_document(flat; cse = false)
    @test haskey(ev["models"]["Chem"]["subsystems"]["Inner"]["variables"], "k")
    u = Dict(c => 0.4 + 0.1 * i for (i, c) in enumerate(flat_state_cells(flat)))
    J, _ = flat_band_jacobian(flat; u = u, cse = false)
    prob = EarthSciAST.esm_problem(doc, (0.0, 1.0))
    vm = Dict{String, Int}(String(k) => Int(v) for (k, v) in prob.var_map)
    x = zeros(length(prob.u0))
    for (c, v) in u
        x[vm[c]] = v
    end
    A = ad_jacobian(prob.f!, x, prob.p, 0.0)
    @test haskey(J, ("Chem.c", "Source.s")) && haskey(J, ("Chem.c", "Chem.Inner.k"))
    @test all(isapprox(A[vm[r], vm[c]], v; rtol = 1.0e-12) for ((r, c), v) in J)
    @test count(!iszero, A) == count(!iszero, values(J))
end

@testset "flattened evaluation document: a component named like the derived model" begin
    doc = EarthSciAST.load_string(
        """
        {"esm": "1.1.0", "metadata": {"name": "Clash"},
         "models": {"JacobianEval": {
           "variables": {"c": {"type": "unknown", "default": 1.0}},
           "equations": [{"lhs": {"op": "D", "args": ["c"], "wrt": "t"},
                          "rhs": {"op": "*", "args": [-1.0, "c", "c"]}}]}}}
        """
    )
    @test_throws EarthSciASTDiff.BandError evaluation_document(EarthSciAST.flatten(doc))
end

# The evaluation document carries the source's `domain` (esm-jacobian-spec.md
# §8.1), so it has the source's time origin: a runtime that turns timestamps
# into model time with `time_origin` gets the same `t` for the document as
# for its source. The right-hand side itself passes `t` to the calendar
# functions as UTC seconds whatever the origin, so a `datetime.hour`
# coefficient, `∂/∂c (-k hour(t) c) = -k hour(t)`, reads the same with or
# without the domain. The same in Python and Rust, written the same.
const _DIURNAL = """
{"esm": "1.1.0", "metadata": {"name": "Diurnal"},
 "domain": {"independent_variable": "t",
            "temporal": {"start": "2024-05-01T06:00:00Z", "end": "2024-05-02T06:00:00Z"}},
 "models": {"Chem": {
   "variables": {"c": {"type": "unknown", "default": 1.0},
                 "k": {"type": "parameter", "default": 0.5}},
   "equations": [{"lhs": {"op": "D", "args": ["c"], "wrt": "t"},
                  "rhs": {"op": "*", "args": [{"op": "-", "args": ["k"]},
                          {"op": "fn", "name": "datetime.hour", "args": ["t"]}, "c"]}}]}}}
"""

# J_1 of an evaluation document at state c = 0.3 and time t.
function _j1(doc, t)
    prob = EarthSciAST.esm_problem(
        EarthSciAST.load_string(stable_json(doc)), (0.0, 1.0); compiler = :interpreter
    )
    vm = Dict{String, Int}(String(k) => Int(v) for (k, v) in prob.var_map)
    du = zeros(length(prob.u0))
    prob.f!(du, fill(0.3, length(prob.u0)), prob.p, Float64(t))
    return du[vm["JacobianEval.J_1"]]
end

@testset "evaluation document: the source's domain" begin
    file = EarthSciAST.load_string(_DIURNAL)
    for src in (file, EarthSciAST.flatten(file))
        doc, fields = evaluation_document(src)
        @test fields == ["J_1"]
        @test stable_json(doc["domain"]) ==
            """{"temporal":{"end":"2024-05-02T06:00:00Z","start":"2024-05-01T06:00:00Z"}}"""
        @test doc["esm"] == "1.1.0"
        built = EarthSciAST.load_string(stable_json(doc))
        if isdefined(EarthSciAST, :time_origin)   # EarthSciAST ≥ 0.7
            @test EarthSciAST.time_origin(built) == EarthSciAST.time_origin(file) ==
                1714543200.0
        end
        # 13:00 UTC on 2024-05-01, in UTC seconds
        @test _j1(doc, 1714568400.0) == -6.5
        bare = copy(doc)
        delete!(bare, "domain")
        @test _j1(bare, 1714568400.0) == -6.5
    end
    # no domain, no `domain`
    doc, _ = evaluation_document(fixture("bd_chem.esm"))
    @test !haskey(doc, "domain")
end
