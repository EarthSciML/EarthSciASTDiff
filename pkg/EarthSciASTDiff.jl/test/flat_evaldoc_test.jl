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

# The product documents (esm-jacobian-spec.md §9) of a flattened system are
# laid out as its component tree too: the source's scoped names in their
# components, the inputs and outputs (`_v_SimpleOzone_O3`, the column's dots
# made `_`) unscoped in the derived model. Each loads and builds in
# EarthSciAST and gives `J v`, `Jᵀ λ` and the second-order contraction of the
# golden Jacobian `coupled_chem_advection.jvals.json`.
@testset "flattened product documents" begin
    flat = EarthSciAST.flatten(fixture("coupled_chem_advection.esm"))
    g = EarthSciASTDiff.JSON3.read(
        read(joinpath(_FLAT_GOLDENS, "coupled_chem_advection.jvals.json"), String)
    )
    u = Dict{String, Float64}(String(k) => Float64(v) for (k, v) in pairs(g["u"]))
    t = Float64(g["t"])
    want = Dict{Tuple{String, String}, Float64}(
        (String(e[1]), String(e[2])) => Float64(e[3]) for e in g["entries"]
    )
    cells = flat_state_cells(flat)
    v = Dict(c => sin(1.7 * i) for (i, c) in enumerate(cells))
    λ = Dict(c => cos(0.9 * i) for (i, c) in enumerate(cells))
    close(a, b) = isapprox(a, b; rtol = 1.0e-10, atol = 1.0e-13)
    layout_ok(doc, model) = begin
        ms = doc["models"]
        sort!(collect(keys(ms))) == sort!(["Advection", "SimpleOzone", model]) &&
            !any(occursin('.', n) for m in values(ms) for n in keys(m["variables"]))
    end
    for cse in (true, false)
        jv, io = jvp_document(flat; cse = cse)
        @test layout_ok(jv, "JvpEval")
        @test io.v["SimpleOzone.O3"] == "_v_SimpleOzone_O3"
        @test io.jvp["SimpleOzone.O3"] == "_jvp_SimpleOzone_O3"
        out = flat_doc_call(jv, io, u, Dict(_doc_cell(io.v, c) => v[c] for c in cells); t = t)
        for r in cells
            @test close(
                out(_doc_cell(io.jvp, r)),
                sum((x * v[c] for ((rr, c), x) in want if rr == r); init = 0.0)
            )
        end
        vj, io = vjp_document(flat; cse = cse)
        @test layout_ok(vj, "VjpEval")
        out = flat_doc_call(
            vj, io, u, Dict(_doc_cell(io.lam, r) => λ[r] for r in cells);
            t = t, compiler = :interpreter
        )
        for c in cells
            @test close(
                out(_doc_cell(io.vjp, c)),
                sum((λ[r] * x for ((r, cc), x) in want if cc == c); init = 0.0)
            )
        end
    end
    # ∇_u (λᵀ J(u) v), against a central difference of λᵀ J v through the
    # JvpEval document
    hv, io = hvp_document(flat; stages = 1)
    @test layout_ok(hv, "HvpEval")
    inputs = Dict{String, Float64}()
    for c in cells
        inputs[_doc_cell(io.mu[1], c)] = λ[c]
        inputs[_doc_cell(io.w[1], c)] = v[c]
    end
    out = flat_doc_call(hv, io, u, inputs; t = t, compiler = :interpreter)
    jv, jio = jvp_document(flat; cse = false)
    vin = Dict(_doc_cell(jio.v, c) => v[c] for c in cells)
    lJv(uu) = (o = flat_doc_call(jv, jio, uu, vin; t = t); sum(λ[r] * o(_doc_cell(jio.jvp, r)) for r in cells))
    h = 1.0e-5
    for k in cells
        up = copy(u); up[k] += h
        um = copy(u); um[k] -= h
        @test isapprox(out(_doc_cell(io.hvp, k)), (lJv(up) - lJv(um)) / (2h); rtol = 1.0e-6, atol = 1.0e-9)
    end
end

@testset "flattened documents: refused sources" begin
    # a flattened system mixing scoped and unscoped names: an equation moved
    # to `Chem` would read the unscoped `x` as `Chem.x`
    file = fixture("coupled_chem_advection.esm")
    sv = sysview(EarthSciAST.flatten(file))
    entries = jacobian_bands(sv)
    any_param = first(n for (n, x) in sv.variables if x.type == EarthSciAST.ParameterVariable)
    bad = EarthSciASTDiff.SysView(
        merge(sv.variables, Dict("x" => sv.variables[any_param])), sv.equations,
        sv.index_sets, sv.domain
    )
    err = try
        _evaluation_document(bad, entries)
        nothing
    catch e
        e
    end
    @test err isa EarthSciASTDiff.BandError && occursin("`x`", sprint(showerror, err))
    # a scoped name that is both a variable and a component
    st = first(n for n in keys(sv.variables) if startswith(n, "SimpleOzone."))
    bad = EarthSciASTDiff.SysView(
        merge(
            sv.variables, Dict(
                "Advection.Inner" => sv.variables[any_param],
                "Advection.Inner.q" => sv.variables[any_param]
            )
        ),
        sv.equations, sv.index_sets, sv.domain
    )
    err = try
        _evaluation_document(bad, entries)
        nothing
    catch e
        e
    end
    @test err isa EarthSciASTDiff.BandError &&
        occursin("`Advection.Inner` is both a variable and a component", sprint(showerror, err))
    # a source variable named like a band field
    clash = EarthSciAST.load_string(
        """
        {"esm": "1.1.0", "metadata": {"name": "Clash"},
         "models": {"JacobianEval": {
           "variables": {"J_1": {"type": "unknown", "default": 1.0}},
           "equations": [{"lhs": {"op": "D", "args": ["J_1"], "wrt": "t"},
                          "rhs": {"op": "*", "args": [-1.0, "J_1", "J_1"]}}]}}}
        """
    )
    @test_throws EarthSciASTDiff.BandError evaluation_document(clash)
end

@testset "product documents: the source's domain" begin
    file = EarthSciAST.load_string(_DIURNAL)
    for src in (file, EarthSciAST.flatten(file))
        want = """{"temporal":{"end":"2024-05-02T06:00:00Z","start":"2024-05-01T06:00:00Z"}}"""
        for (doc, _) in (jvp_document(src), vjp_document(src), hvp_document(src; stages = 1))
            @test stable_json(doc["domain"]) == want
        end
    end
    doc, _ = jvp_document(fixture("bd_chem.esm"))
    @test !haskey(doc, "domain")
end

# A lifted state on one of two same-size index sets
# (`lift_same_size_index_sets`, EarthSciAST's pointwise-lift conformance case
# of that name): `y` and `x` both have size 3, `y` declared first, and the
# operand `Wind.u`, declared on `[x, y]` and gathered at the lift's loops,
# names the grid. `flatten` records `Chem.c` on `[x, y]` (esm-libraries-spec
# §4.7.5), and the view, its context, the bands and the evaluation document
# take that shape as recorded. Pinned by tests/goldens/lift_same_size_index_sets.*,
# which Python and Rust reproduce from their own `flatten` (EarthSciAST ≥ 0.8).
@testset "flattened evaluation document: a lifted state's recorded shape" begin
    flat = EarthSciAST.flatten(fixture("lift_same_size_index_sets.esm"))
    @test flat.state_variables["Chem.c"].shape == ["x", "y"]
    sv = sysview(flat)
    @test sv.variables["Chem.c"].shape == ["x", "y"]
    @test EarthSciASTDiff._ctx(sv).shapes["Chem.c"] == [3, 3]
    @test !any(startswith(k, "_liftdim") for k in keys(sv.index_sets))
    entries = jacobian_bands(sv)
    # nothing to hoist: the unhoisted document is the same, so it has no
    # golden of its own
    @test !isfile(joinpath(_FLAT_GOLDENS, "lift_same_size_index_sets.evaldoc-nocse.json"))
    for cse in (true, false)
        doc, fields = _evaluation_document(sv, entries; cse = cse)
        @test stable_json(doc) ==
            read(joinpath(_FLAT_GOLDENS, "lift_same_size_index_sets.evaldoc.json"), String)
        @test doc["models"]["Chem"]["variables"]["c"]["shape"] == ["x", "y"]
        @test all(doc["models"]["JacobianEval"]["variables"][f]["shape"] == ["x", "y"] for f in fields)
    end
    g = EarthSciASTDiff.JSON3.read(
        read(joinpath(_FLAT_GOLDENS, "lift_same_size_index_sets.jvals.json"), String)
    )
    u = Dict{String, Float64}(String(k) => Float64(v) for (k, v) in pairs(g["u"]))
    @test sort!(collect(keys(u))) == sort!(flat_state_cells(flat))
    want = Dict{Tuple{String, String}, Float64}(
        (String(e[1]), String(e[2])) => Float64(e[3]) for e in g["entries"]
    )
    for compiler in (:native, :interpreter)
        J, _ = flat_band_jacobian(flat; u = u, t = g["t"], compiler = compiler)
        @test Set(keys(J)) == Set(keys(want))
        @test all(isapprox(J[k], want[k]; rtol = 1.0e-10, atol = 1.0e-13) for k in keys(want))
    end
    # the golden is the source's Jacobian: ForwardDiff of the flattened source
    prob = EarthSciAST.esm_problem(fixture("lift_same_size_index_sets.esm"), (0.0, 1.0))
    vm = Dict{String, Int}(String(k) => Int(v) for (k, v) in prob.var_map)
    x = zeros(length(prob.u0))
    for (c, v) in u
        x[vm[c]] = v
    end
    A = ad_jacobian(prob.f!, x, prob.p, Float64(g["t"]))
    @test count(!iszero, A) == length(want)
    @test all(isapprox(A[vm[r], vm[c]], v; rtol = 1.0e-12) for ((r, c), v) in want)
end

# A lift no operand names, onto two same-size index sets: `flatten` records
# the synthetic `_liftdim1_3` (esm-libraries-spec §4.7.5 rule 3). Before
# EarthSciAST 0.8, Julia's `flatten` names it in the shape without adding it
# to `index_sets`; the view adds it, so the context resolves either way.
const _AMBIGUOUS_LIFT = """
{"esm": "1.1.0", "metadata": {"name": "Lifted"},
 "index_sets": {"x": {"kind": "interval", "size": 3},
                "y": {"kind": "interval", "size": 3}},
 "models": {
  "Chem": {"variables": {"O3": {"type": "unknown", "default": 1.0},
                         "k1": {"type": "parameter", "default": 0.1}},
           "equations": [{"lhs": {"op": "D", "args": ["O3"], "wrt": "t"},
                          "rhs": {"op": "*", "args": [{"op": "-", "args": ["k1"]}, "O3", "O3"]}}]},
  "Adv": {"variables": {"u": {"type": "parameter", "default": 2.0},
                        "dx": {"type": "parameter", "default": 100.0}},
          "equations": [{"lhs": {"op": "D", "args": ["_var"], "wrt": "t"},
                         "rhs": {"op": "*", "args": [{"op": "-", "args": ["u"]},
                          {"op": "makearray", "args": [], "regions": [[[2, 3]], [[1, 1]]],
                           "values": [
                            {"op": "/", "args": [{"op": "-", "args": [
                              {"op": "index", "args": ["_var", "i"]},
                              {"op": "index", "args": ["_var", {"op": "-", "args": ["i", 1]}]}]}, "dx"]},
                            {"op": "/", "args": [{"op": "index", "args": ["_var", "i"]}, "dx"]}]}]}}]}},
 "coupling": [{"type": "operator_compose", "systems": ["Chem", "Adv"], "lifting": "pointwise"}]}
"""

@testset "sysview(flat): a recorded synthetic lift axis" begin
    flat = EarthSciAST.flatten(EarthSciAST.load_string(_AMBIGUOUS_LIFT))
    @test flat.lifted_shapes["Chem.O3"] == [3]
    @test collect(flat.state_variables["Chem.O3"].shape) == ["_liftdim1_3"]
    # as `flatten` before 0.8 leaves it: named, not declared
    delete!(flat.index_sets, "_liftdim1_3")
    sv = sysview(flat)
    @test sv.index_sets["_liftdim1_3"].size == 3
    @test EarthSciASTDiff._ctx(sv).shapes["Chem.O3"] == [3]
    @test !isempty(jacobian_bands(sv))
end

@testset "sysview(flat): a recorded lift shape is checked" begin
    check(shape, isets = Dict{String, Any}("x" => EarthSciAST.IndexSet("interval"; size = 3))) =
        EarthSciASTDiff._check_lifted_shape("Chem.O3", shape, [3], isets)
    @test check(["x"]) === nothing
    @test_throws ArgumentError check(["x", "x"])          # rank
    @test_throws ArgumentError check(String[])            # rank
    @test_throws ArgumentError check(["z"])               # undeclared
    @test_throws ArgumentError check(["_liftdim2_3"])     # not this dimension's
    @test_throws ArgumentError check(
        ["x"], Dict{String, Any}("x" => EarthSciAST.IndexSet("interval"; size = 4))
    )                                                     # size
    @test_throws ArgumentError check(
        ["_liftdim1_3"], Dict{String, Any}("_liftdim1_3" => EarthSciAST.IndexSet("interval"; size = 5))
    )
end
