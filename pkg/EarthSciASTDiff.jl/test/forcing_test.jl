# Forcing-buffer and array-parameter columns (spec §4, §4.3): the
# `wrt = :forcing` axis over the cells of live data buffers (record axis
# included) and array parameters as real columns of `wrt = :parameters`.
# Pinned byte-for-byte by tests/goldens/forcing_bracket.{forcing,parameters}.*
# and by value in the matching `*.jvals.json`, which the other bindings must
# reproduce. Kept in step with scripts/regenerate-goldens.jl.

using EarthSciASTDiff: _evaluation_document, BandError

const FORCING_AXES = (:forcing, :parameters)

# The oracle: the same document with the named parameters promoted to states
# whose derivative is zero, so ForwardDiff through the tree-walk RHS with
# respect to the state vector differentiates with respect to the buffer
# contents too.
function _promoted(path, model, names)
    d = EarthSciASTDiff.JSON3.read(read(path, String), Dict{String, Any})
    m = d["models"][model]
    for n in names
        v = m["variables"][n]
        v["type"] = "unknown"
        delete!(v, "update")
        shape = [d["index_sets"][s]["size"] for s in v["shape"]]
        push!(
            m["equations"], Dict(
                "lhs" => Dict("op" => "D", "args" => [n], "wrt" => "t"),
                "rhs" => Dict(
                    "op" => "makearray", "args" => [],
                    "regions" => [[[1, k] for k in shape]], "values" => [0.0]
                )
            )
        )
    end
    return EarthSciAST.load_string(EarthSciASTDiff.JSON3.write(d))
end

const BRACKET = joinpath(FIXTURES, "forcing_bracket.esm")

# A one-model document around `F` (shape [2, 5]) and a state `c` (shape [4])
# with the given `D(c)` body over `i`; `extra` adds variables.
function _bracket_doc(body; extra = Dict{String, Any}())
    O(kv...) = Dict{String, Any}(kv...)
    vars = O(
        "c" => O("type" => "unknown", "shape" => Any["x"], "default" => 0.5),
        "F" => O("type" => "parameter", "shape" => Any["rec", "xf"], "default" => 1.0),
    )
    merge!(vars, extra)
    return EarthSciAST.load_document(
        O(
            "esm" => "1.1.0", "metadata" => O("name" => "B"),
            "index_sets" => O(
                "x" => O("kind" => "interval", "size" => 4),
                "xf" => O("kind" => "interval", "size" => 5),
                "rec" => O("kind" => "interval", "size" => 2)
            ),
            "models" => O(
                "B" => O(
                    "variables" => vars,
                    "equations" => Any[
                        O(
                            "lhs" => O("op" => "D", "args" => Any["c"], "wrt" => "t"),
                            "rhs" => O(
                                "op" => "faq", "args" => Any[], "output_idx" => Any["i"],
                                "ranges" => O("i" => O("from" => "x")), "expr" => body
                            )
                        ),
                    ]
                )
            )
        )
    )
end
_ix(v, i...) = Dict{String, Any}("op" => "index", "args" => Any[v, i...])
_op(o, a...) = Dict{String, Any}("op" => o, "args" => Any[a...])

@testset "forcing: the forcing set" begin
    sv = sysview(fixture("forcing_bracket.esm"), "Bracket")
    # inferred: the array parameters with a `data` update; E (no update) and
    # the scalars stay on the parameters axis
    @test forcing_names(sv) == ["F", "G"]
    @test jacobian_columns(sv; wrt = :forcing) == [("F", [2, 5]), ("G", [2, 3])]
    @test jacobian_columns(sv; wrt = :parameters) ==
        [("E", [4]), ("dt", Int[]), ("k", Int[]), ("t_ref", Int[])]
    # an explicit set (a planner that builds the providers itself)
    @test forcing_names(sv, ["G"]) == ["G"]
    @test first.(jacobian_columns(sv; wrt = :parameters, forcing = ["G"])) ==
        ["E", "F", "dt", "k", "t_ref"]
    # a name that resolves to no buffer throws, never an empty axis
    @test_throws ArgumentError forcing_names(sv, ["F", "nope"])
    @test_throws ArgumentError jacobian_bands(sv; wrt = :forcing, forcing = ["c"])
end

@testset "forcing: goldens" begin
    golden_dir = joinpath(@__DIR__, "..", "..", "..", "tests", "goldens")
    file = fixture("forcing_bracket.esm")
    sv = sysview(file, "Bracket")
    for wrt in FORCING_AXES
        gold(kind) = joinpath(golden_dir, "forcing_bracket.$wrt.$kind.json")
        entries = jacobian_bands(sv; wrt = wrt)
        @test !isempty(entries)
        @test stable_json(Any[EarthSciASTDiff.serialize_band(en) for en in entries]) ==
            read(gold("bands"), String)
        block = jacobian_document(file, "Bracket"; wrt = wrt)["jacobians"]["Bracket"]
        @test block["wrt"] == String(wrt)
        @test haskey(block, "columns") && !haskey(block, "structure")
        @test stable_json(block) == read(gold("block"), String)
        doc, _ = _evaluation_document(sv, entries)
        @test stable_json(doc) == read(gold("evaldoc"), String)
        doc0, _ = _evaluation_document(sv, entries; cse = false)
        p0 = isfile(gold("evaldoc-nocse")) ? gold("evaldoc-nocse") : gold("evaldoc")
        @test stable_json(doc0) == read(p0, String)
        w, back = parse_jacobian_block(block)
        @test w == wrt && length(back) == length(entries)
    end
    # the bracket: each record is its own column, the blend weight a factor
    # of the coefficient, and the regrid a contracted band over one record
    fb = jacobian_bands(sv; wrt = :forcing)
    cols = Set((en.u, en.v, stable_json(EarthSciASTDiff._ser_expr(en.band.cidx[1]))) for en in fb)
    @test ("c", "F", "1") in cols && ("c", "F", "2") in cols && ("S", "F", "1") in cols
    @test any(en -> en.v == "G" && !isempty(en.band.contracted), fb)
end

@testset "forcing: values against ForwardDiff through the RHS" begin
    file = fixture("forcing_bracket.esm")
    _, u00, _, _, vm0 = build_evaluator(file)
    rows = first.(sort(collect(vm0), by = last))
    u = [0.2 + 0.6 * abs(sin(2.3 * i)) for i in eachindex(u00)]
    for (wrt, names) in ((:forcing, ["F", "G"]), (:parameters, ["E"]))
        pf = _promoted(BRACKET, "Bracket", names)
        f!, uf, pp, _, vm = build_evaluator(pf)
        x = copy(uf)
        for (i, r) in enumerate(rows)
            x[vm[r]] = u[i]
        end
        # t = 0.37: w = 0.37; t = 2.0: w = 0 exactly (the record time)
        for t in (0.37, 2.0)
            Jad = ForwardDiff.jacobian((du, y) -> f!(du, y, pp, t), similar(x), x)
            res = assemble_jacobian(file; wrt = wrt, u = u, t = t)
            @test res.rownames == rows
            checked = 0
            for (j, cn) in enumerate(res.colnames)
                haskey(vm, cn) || continue
                for (i, r) in enumerate(rows)
                    ad = Jad[vm[r], vm[cn]]
                    @test res.J[i, j] ≈ ad atol = 1.0e-12 * max(1.0, abs(ad))
                    ad != 0 && @test res.pattern[i, j]   # global ⊇ local
                end
                checked += 1
            end
            shapes = Dict(jacobian_columns(sysview(file, "Bracket"); wrt = wrt))
            @test checked == sum(n -> prod(shapes[n]), names)   # every promoted cell
        end
    end
    # the scalar parameter columns against ForwardDiff over `p`
    f!, u0, p, _, _ = build_evaluator(file)
    res = assemble_jacobian(file; wrt = :parameters, u = u, p = p, t = 0.37)
    Jp = ForwardDiff.jacobian(collect(Float64, p)) do q
        du = similar(q, length(u)); f!(du, u, NamedTuple{keys(p)}(Tuple(q)), 0.37); du
    end
    for (k, n) in enumerate(keys(p))
        j = findfirst(==(String(n)), res.colnames)
        @test Vector(res.J[:, j]) ≈ Jp[:, k] atol = 1.0e-12
    end
end

@testset "forcing: value goldens reproduce" begin
    golden_dir = joinpath(@__DIR__, "..", "..", "..", "tests", "goldens")
    file = fixture("forcing_bracket.esm")
    for wrt in FORCING_AXES, sfx in ("", ".w0")
        g = EarthSciASTDiff.JSON3.read(
            read(joinpath(golden_dir, "forcing_bracket.$wrt$sfx.jvals.json"), String)
        )
        res0 = assemble_jacobian(file; wrt = wrt)
        u = [Float64(g.u[Symbol(n)]) for n in res0.rownames]
        res = assemble_jacobian(file; wrt = wrt, u = u, t = Float64(g.t))
        I, J, V = findnz(res.J)
        got = Dict((res.rownames[I[k]], res.colnames[J[k]]) => V[k] for k in eachindex(I))
        want = Dict((String(e[1]), String(e[2])) => Float64(e[3]) for e in g.entries)
        @test keys(got) == keys(want)
        @test all(isapprox(got[k], want[k]; rtol = 1.0e-10, atol = 1.0e-14) for k in keys(want))
    end
end

@testset "parameters: array-parameter columns are real (no silent drop)" begin
    file = fixture("forcing_bracket.esm")
    jac = prepare_jacobian(file; wrt = :parameters)
    J = copy(jac.prototype)
    _, u0, p, _, _ = build_evaluator(file)
    jac(J, u0, p, 0.37)
    for i in 1:4
        j = findfirst(==("E[$i]"), jac.colnames)
        @test j !== nothing
        @test J[findfirst(==("c[$i]"), jac.rownames), j] == 1.0
    end
    @test jac.columns == jacobian_columns(sysview(file, "Bracket"); wrt = :parameters)
end

@testset "forcing: vjp_forcing! is the transposed scatter" begin
    file = fixture("forcing_bracket.esm")
    jac = prepare_jacobian(file; wrt = :forcing)
    _, u0, p, _, _ = build_evaluator(file)
    u = u0 .* (0.5 .+ rand(length(u0)))
    λ = randn(length(u0))
    J = copy(jac.prototype)
    jac(J, u, p, 0.37)
    g = zeros(length(jac.colnames))
    vjp_forcing!(g, jac, λ, u, p, 0.37)
    @test g ≈ Matrix(J)' * λ atol = 1.0e-13
    vjp_forcing!(g, jac, λ, u, p, 0.37)                 # accumulates
    @test g ≈ 2 .* (Matrix(J)' * λ) atol = 1.0e-13
    # buffer-shaped accumulators, column-major like the canonical ordinal
    acc = Dict("F" => zeros(2, 5), "G" => zeros(2, 3))
    vjp_forcing!(acc, jac, λ, u, p, 0.37)
    @test vcat(vec(acc["F"]), vec(acc["G"])) ≈ Matrix(J)' * λ atol = 1.0e-13
    @test acc["F"][2, 3] ≈ (Matrix(J)' * λ)[findfirst(==("F[2,3]"), jac.colnames)]
    @test_throws ArgumentError vjp_forcing!(Dict("F" => zeros(2, 5)), jac, λ, u, p, 0.37)
    @test_throws DimensionMismatch vjp_forcing!(zeros(3), jac, λ, u, p, 0.37)
    # only on the forcing axis
    jp = prepare_jacobian(file; wrt = :parameters)
    @test_throws ArgumentError vjp_forcing!(zeros(length(jp.colnames)), jp, λ, u, p, 0.37)
    gp = vjp!(zeros(length(jp.colnames)), jp, λ, u, p, 0.37)
    Jp = copy(jp.prototype); jp(Jp, u, p, 0.37)
    @test gp ≈ Matrix(Jp)' * λ atol = 1.0e-13
end

@testset "forcing: live buffers through param_arrays" begin
    file = fixture("forcing_bracket.esm")
    F = [1.0 1.25 1.5 1.75 2.0; 1.5 1.25 2.0 1.0 0.75]
    G = [0.2 0.4 0.6; 0.3 0.5 0.1]
    bufs = Dict{String, Any}("F" => copy(F), "G" => copy(G))
    jac = prepare_jacobian(file; wrt = :forcing, build_kwargs = (param_arrays = bufs,))
    _, u0, p, _, _ = build_evaluator(file)
    J = copy(jac.prototype)
    jac(J, u0, p, 0.37)
    J0 = copy(J)
    @test J0 ≈ assemble_jacobian(file; wrt = :forcing, u = u0, t = 0.37).J
    # S's row reads F[1,1]²: refill the buffer in place, as a provider does
    bufs["F"][1, 1] = 3.0
    EarthSciAST.notify_forcing_refresh!(bufs["F"])
    jac(J, u0, p, 0.37)
    iS = findfirst(==("S"), jac.rownames); jF = findfirst(==("F[1,1]"), jac.colnames)
    @test J[iS, jF] ≈ -2 * 3.0 * u0[iS]
    @test J0[iS, jF] ≈ -2 * 1.0 * u0[iS]
end

@testset "forcing: several axes in one jacobians entry" begin
    file = fixture("forcing_bracket.esm")
    doc = jacobian_document(file, "Bracket"; wrt = [:states, :forcing])
    blocks = doc["jacobians"]["Bracket"]
    @test blocks isa AbstractVector && length(blocks) == 2
    @test [b["wrt"] for b in blocks] == ["states", "forcing"]
    golden_dir = joinpath(@__DIR__, "..", "..", "..", "tests", "goldens")
    @test stable_json(blocks[1]) == read(joinpath(golden_dir, "forcing_bracket.block.json"), String)
    @test stable_json(blocks[2]) ==
        read(joinpath(golden_dir, "forcing_bracket.forcing.block.json"), String)
    parsed = parse_jacobian_blocks(blocks)
    @test first.(parsed) == [:states, :forcing]
    @test first.(parse_jacobian_blocks(blocks[2])) == [:forcing]
    @test_throws ArgumentError jacobian_document(file, "Bracket"; wrt = [:states, :states])
    @test_throws ArgumentError parse_jacobian_blocks([blocks[2], blocks[2]])
end

@testset "forcing: refused rather than silently wrong" begin
    # a column index that reads a parameter is integer-valued, so piecewise
    # constant: its dimension becomes one-hot over the buffer's extent, every
    # point of which is a placeable column (spec §3.5)
    off = Dict{String, Any}("off" => Dict{String, Any}("type" => "parameter", "default" => 1))
    hot = _bracket_doc(_op("*", _ix("c", "i"), _ix("F", 1, _op("+", "i", "off"))); extra = off)
    en = only(jacobian_bands(hot, "B"; wrt = :forcing, forcing = ["F"]))
    @test [n for (n, _, _) in en.band.contracted] == ["_oh2"]
    @test en.band.cidx[2] == EarthSciAST.VarExpr("_oh2")
    # a literal offset is closed
    ok = _bracket_doc(_op("*", _ix("c", "i"), _ix("F", 2, _op("+", "i", 1))))
    @test length(jacobian_bands(ok, "B"; wrt = :forcing, forcing = ["F"])) == 1
    # a column outside the buffer's extent is an error, not a ghost skip.
    # EarthSciAST already refuses to build that RHS, so the assembly check is
    # reached with the (state-identical) layout of the in-range document.
    past = _bracket_doc(_op("*", _ix("c", "i"), _ix("F", 1, _op("+", "i", 2))))
    @test length(jacobian_bands(past, "B"; wrt = :forcing, forcing = ["F"])) == 1
    _, u0, _, _, vm = build_evaluator(ok)
    lay = (u0 = u0, var_map = vm)
    @test jacobian_pattern(ok; wrt = :forcing, forcing = ["F"], source_layout = lay).pattern |>
        nnz == 4
    @test_throws BandError jacobian_pattern(
        past; wrt = :forcing, forcing = ["F"], source_layout = lay
    )
    @test_throws BandError prepare_jacobian(
        past; wrt = :forcing, forcing = ["F"], source_layout = lay
    )
    # a model that does not read the forcing set at all has an empty block
    none = _bracket_doc(_op("*", -1, _ix("c", "i")))
    @test isempty(jacobian_bands(none, "B"; wrt = :forcing, forcing = ["F"]))
end

@testset "parameters: an array parameter read past its extent is a ghost read" begin
    ok = _bracket_doc(_op("*", _ix("c", "i"), _ix("F", 2, _op("+", "i", 1))))
    _, u0, _, _, vm = build_evaluator(ok)
    lay = (u0 = u0, var_map = vm)
    # an array parameter read past its extent (F is not data-fed here) is a
    # ghost read, as the same stencil on a state is: i = 4 reads F[1,6]
    past = _bracket_doc(_op("*", _ix("c", "i"), _ix("F", 1, _op("+", "i", 2))))
    @test nnz(jacobian_pattern(past; wrt = :parameters, source_layout = lay).pattern) == 3
    @test nnz(prepare_jacobian(past; wrt = :parameters, source_layout = lay).pattern) == 3
    ghost = _bracket_doc(_op("*", -1, _ix("c", _op("+", "i", 1))))
    @test nnz(jacobian_pattern(ghost; source_layout = lay).pattern) == 3
end

@testset "forcing: an identically-zero derivative is an empty block" begin
    # forcing read only where the derivative is 0 (inside a piecewise-
    # constant op, or as an `ifelse` condition) yields no band and no error
    flr = _bracket_doc(_op("*", _ix("c", "i"), _op("floor", _ix("F", 1, "i"))))
    @test isempty(jacobian_bands(flr, "B"; wrt = :forcing, forcing = ["F"]))
    cond = _bracket_doc(
        _op("ifelse", _op(">", _ix("F", 1, "i"), 0), _op("*", -1, _ix("c", "i")), 0)
    )
    @test isempty(jacobian_bands(cond, "B"; wrt = :forcing, forcing = ["F"]))
    doc = jacobian_document(cond, "B"; wrt = [:states, :forcing], forcing = ["F"])
    states, forcing = doc["jacobians"]["B"]
    @test length(states["entries"]) == 1 && isempty(forcing["entries"])
end

@testset "forcing: one name or axis as a String" begin
    ok = _bracket_doc(_op("*", _ix("c", "i"), _ix("F", 2, _op("+", "i", 1))))
    sv = sysview(ok, "B")
    @test forcing_names(sv, "F") == forcing_names(sv, :F) == ["F"]
    sb = EarthSciASTDiff.serialize_band
    @test sb.(jacobian_bands(ok, "B"; wrt = :forcing, forcing = "F")) ==
        sb.(jacobian_bands(ok, "B"; wrt = :forcing, forcing = ["F"]))
    @test jacobian_document(ok, "B"; wrt = "forcing", forcing = "F")["jacobians"]["B"] ==
        jacobian_document(ok, "B"; wrt = :forcing, forcing = ["F"])["jacobians"]["B"]
    @test jacobian_document(ok, "B"; wrt = "states")["jacobians"]["B"]["wrt"] == "states"
end

@testset "forcing: rectangular axes carry no structure class" begin
    file = fixture("forcing_bracket.esm")
    @test prepare_jacobian(file; wrt = :forcing).structure === nothing
    @test assemble_jacobian(file; wrt = :parameters).structure === nothing
    @test prepare_jacobian(file).structure isa Symbol
end

@testset "forcing: the evaluation document keeps array defaults row-major" begin
    # an array parameter's inline default is row-major nested JSON; written
    # flat (JSON3's column-major view of the loaded Array) it reloads rank-less
    file = fixture("forcing_bracket.esm")
    sv = sysview(file, "Bracket")
    doc, _ = _evaluation_document(sv, jacobian_bands(sv; wrt = :forcing))
    vars = doc["models"]["JacobianEval"]["variables"]
    @test vars["F"]["default"] == Any[Any[1.0, 1.25, 1.5, 1.75, 2.0], Any[1.5, 1.25, 2.0, 1.0, 0.75]]
    back = EarthSciAST.load_string(EarthSciASTDiff.JSON3.write(doc))
    for n in ("E", "F", "G")
        @test back.models["JacobianEval"].variables[n].default == sv.variables[n].default
    end
end
