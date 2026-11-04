# Data-fed parameters in the evaluation document (esm-jacobian-spec.md §8.1):
# a parameter whose `update` has a `data` rule keeps it, and the document
# carries the `data_sources` entries those rules name, so EarthSciAST builds
# the parameter as a forcing slot of `rhs_program`, not a constant. Pinned by
# tests/goldens/data_fed_flat.{evaldoc,bands}.json, which the other bindings
# must reproduce.

using EarthSciASTDiff: _evaluation_document, BandError

const _DF_GOLDENS = joinpath(@__DIR__, "..", "..", "..", "tests", "goldens")
const _DF_PATH = joinpath(FIXTURES, "data_fed_flat.esm")
const _DF_FED = ["Box.P", "Chem.E", "Chem.T"]

# The fixture without its data rules and `data_sources`: the same model with
# every data-fed parameter a constant at its default.
function _df_without_rules()
    d = copy(EarthSciASTDiff.JSON3.read(read(_DF_PATH, String)))
    delete!(d, :data_sources)
    for m in values(d[:models]), v in values(m[:variables])
        delete!(v, :update)
    end
    return EarthSciAST.load_string(EarthSciASTDiff.JSON3.write(d))
end

_df_program(doc) = EarthSciAST.rhs_program(
    EarthSciAST.esm_problem(
        EarthSciAST.load_string(stable_json(doc)), (0.0, 1.0); compiler = :native
    )
)

# One right-hand-side call of a program at its default state, with forcing `fs`.
function _df_rhs(prog, fs)
    u = zeros(EarthSciAST.n_states(prog))
    for b in EarthSciAST.state_layout(prog)
        u[b.offset:(b.offset + prod(b.shape; init = 1) - 1)] .= 1.0 + length(b.name)
    end
    p = EarthSciAST.parameter_vector(prog, Dict{String, Float64}())
    du = fill(NaN, length(u))
    EarthSciAST.rhs!(du, prog, EarthSciAST.rhs_workspace(prog), u, Float64[], p, 0.0, fs)
    return Dict(
        b.name => du[b.offset:(b.offset + prod(b.shape; init = 1) - 1)]
            for b in EarthSciAST.state_layout(prog)
    )
end

@testset "data-fed parameters: evaluation document" begin
    file = EarthSciAST.load_path(_DF_PATH)
    flat = EarthSciAST.flatten(file)
    sv = sysview(flat; data_sources = file.data_sources)
    entries = jacobian_bands(sv)
    for cse in (true, false)
        doc, _ = _evaluation_document(sv, entries; cse = cse)
        @test stable_json(doc) ==
            read(joinpath(_DF_GOLDENS, "data_fed_flat.evaldoc.json"), String)
    end
    doc, fields = evaluation_document(flat; data_sources = file.data_sources)
    @test stable_json(doc) == read(joinpath(_DF_GOLDENS, "data_fed_flat.evaldoc.json"), String)
    # each data-fed parameter keeps its rule, as the source writes it; the
    # sources the rules name are carried, the unread one is not
    vars = Dict(m => doc["models"][m]["variables"] for m in ("Chem", "Box"))
    @test vars["Chem"]["T"]["update"] ==
        Dict("kind" => "data", "source" => "MET", "from" => Dict("file_variable" => "T2M"))
    @test vars["Chem"]["E"]["update"]["from"] ==
        Dict("file_variable" => "NOX", "unit_conversion" => 0.001)
    @test vars["Box"]["P"]["update"]["source"] == "MET"
    @test !haskey(vars["Chem"]["k"], "update")
    @test sort!(collect(keys(doc["data_sources"]))) == ["EMIS", "MET"]
    @test doc["data_sources"]["MET"]["temporal"]["frequency"] == "PT1H"
    # a flattened system carries no `data_sources`: without them the
    # document is refused, never written with a parameter baked in
    @test_throws BandError evaluation_document(flat)
    @test_throws BandError jvp_document(flat)
    # an `EsmFile` gives its own
    @test haskey(first(evaluation_document(fixture("forcing_bracket.esm")))["data_sources"], "MET")
end

@testset "data-fed parameters: constants of the state Jacobian" begin
    gold = read(joinpath(_DF_GOLDENS, "data_fed_flat.bands.json"), String)
    for file in (EarthSciAST.load_path(_DF_PATH), _df_without_rules())
        entries = jacobian_bands(sysview(EarthSciAST.flatten(file)))
        @test stable_json(Any[EarthSciASTDiff.serialize_band(en) for en in entries]) == gold
    end
end

@testset "data-fed parameters: forcing slots of rhs_program" begin
    file = EarthSciAST.load_path(_DF_PATH)
    doc, fields = evaluation_document(
        EarthSciAST.flatten(file); cse = false, data_sources = file.data_sources
    )
    prog = _df_program(doc)
    slots = EarthSciAST.forcing_slots(prog)
    @test [(s.name, s.shape, s.kind) for s in slots] == [(n, [3], :forcing) for n in _DF_FED]
    @test [s.name for s in EarthSciAST.parameters(prog)] == ["Chem.k"]
    # Before: the same parameters were constants of the build, no slot.
    old, _ = evaluation_document(EarthSciAST.flatten(_df_without_rules()); cse = false)
    oprog = _df_program(old)
    @test isempty(EarthSciAST.forcing_slots(oprog))
    # At the default forcing (each parameter's default) the band fields are
    # those of the document with the parameters baked in; a new forcing set
    # moves exactly the fields that read it: `Chem.T` the chemistry band,
    # `Box.P` the box band, the additive `Chem.E` none.
    D = EarthSciAST.default_forcing_set(prog)
    base = _df_rhs(prog, D)
    @test base == _df_rhs(oprog, EarthSciAST.default_forcing_set(oprog))
    moved(k) = _df_rhs(prog, EarthSciAST.replace_forcing(D, k, 2 .* D.bufs[k]))
    changes(k) = sort!([n for (n, v) in moved(k) if v != base[n]])
    @test changes(findfirst(==("Chem.T"), _DF_FED)) == ["JacobianEval.J_1"]
    @test changes(findfirst(==("Box.P"), _DF_FED)) == ["JacobianEval.J_2"]
    @test isempty(changes(findfirst(==("Chem.E"), _DF_FED)))
end

@testset "data-fed parameters: flattened jacobian_document" begin
    file = EarthSciAST.load_path(_DF_PATH)
    flat = EarthSciAST.flatten(file)
    # the model keeps the data rules, so the document declares their sources
    doc = jacobian_document(flat; data_sources = file.data_sources)
    @test sort!(collect(keys(doc["data_sources"]))) == ["EMIS", "MET"]
    @test doc["models"]["Flattened"]["variables"]["Chem.T"]["update"]["source"] == "MET"
    # it loads (every source a rule names is declared) once the draft
    # `jacobians` block, which EarthSciAST does not read, is set aside
    @test EarthSciAST.load_string(stable_json(delete!(copy(doc), "jacobians"))) isa
        EarthSciAST.EsmFile
    # without them the document is refused, never written undeclared
    @test_throws BandError jacobian_document(flat)
end

# A kept rule's `unit_conversion` may be an expression reading an observed the
# coefficients do not: the derived documents keep that observed.
@testset "data-fed parameters: observed a kept rule reads" begin
    d = EarthSciASTDiff.JSON3.read(read(_DF_PATH, String), Dict{String, Any})
    chem = d["models"]["Chem"]
    chem["variables"]["f"] = Dict{String, Any}("type" => "unknown")
    push!(
        chem["equations"],
        Dict{String, Any}("lhs" => "f", "rhs" => Dict("op" => "*", "args" => Any[0.002, "k"]))
    )
    chem["variables"]["E"]["update"]["from"]["unit_conversion"] = "f"
    file = EarthSciAST.load_string(EarthSciASTDiff.JSON3.write(d))
    flat = EarthSciAST.flatten(file)
    for doc in (
            first(evaluation_document(flat; data_sources = file.data_sources)),
            first(jvp_document(flat; data_sources = file.data_sources)),
        )
        vars = doc["models"]["Chem"]["variables"]
        # the rule as the flattened system holds it (`f`, or `Chem.f` where
        # the flattener scopes it)
        @test endswith(vars["E"]["update"]["from"]["unit_conversion"], "f")
        @test haskey(vars, "f")
        @test EarthSciAST.load_string(stable_json(doc)) isa EarthSciAST.EsmFile
    end
end
