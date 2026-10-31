# Product documents (spec §9): `J v`, `Jᵀ λ` and the §4.4 contraction as
# derived ESM documents. Pinned byte-for-byte by
# tests/goldens/<fixture>[.<axis>].{jvp,vjp}.evaldoc.json and
# <fixture>.<a1>-<a2>.{hvp,hvp-sym}.evaldoc.json, which the other bindings
# must reproduce, and checked by value: each document is evaluated by
# EarthSciAST and compared with the product formed from the value goldens
# themselves (`*.jvals.json`, `*.hvals.json`), so no evaluator of this
# package takes part. Kept in step with scripts/regenerate-goldens.jl.

using EarthSciASTDiff: BandError, _jvp_document, _vjp_document, _hvp_document

const _PRODUCT_DIR = joinpath(@__DIR__, "..", "..", "..", "tests", "goldens")

# Every axis with a value golden (spec §9.6), and the second-order pairs.
const PRODUCT_AXES = (
    ("bd_chem", :states), ("adv_interior", :states), ("contracted_ops", :states),
    ("flux_form_adv", :states), ("minmod_adv", :states),
    ("contracting_observed", :states), ("tgrad_photolysis", :states),
    ("forcing_bracket", :states), ("filtered_scan", :states),
    ("semiring_reduce", :states), ("tgrad_photolysis", :time),
    ("forcing_bracket", :forcing), ("forcing_bracket", :parameters),
)
_axistag(wrt::Symbol) = wrt == :states ? "" : ".$wrt"

# An equality-filtered `faq` is refused by EarthSciAST's native compiler until
# it implements §9.3's equality-filter scatter; the interpreter evaluates it.
function _eval_product(doc, model, u::AbstractDict, inputs::AbstractDict, t; params = nothing)
    f!, x0, p, _, vm = build_evaluator(
        EarthSciASTDiff.JSON3.read(stable_json(doc)); model_name = model,
        compiler = :interpreter
    )
    if params !== nothing
        for (k, v) in params
            @test getproperty(p, Symbol(k)) == v    # the goldens pin the defaults
        end
    end
    x = copy(x0)
    for (n, v) in u
        x[vm[String(n)]] = v
    end
    for (n, v) in inputs
        x[vm[n]] = v
    end
    du = zeros(length(x))
    f!(du, x, p, t)
    return n -> du[vm[n]]
end

_rowcells(g) = sort!([String(k) for k in keys(g.u)])
function _colcells(file, name, wrt, rows)
    wrt == :states && return rows
    wrt == :time && return ["t"]
    sv = sysview(file, first(keys(file.models)))
    return _column_names(jacobian_columns(sv; wrt = wrt))
end
_seed(n, a) = [sin(a * k + 0.4) for k in 1:n]

@testset "product documents: goldens" begin
    for (name, wrt) in PRODUCT_AXES
        file = fixture("$name.esm")
        sv = sysview(file, first(keys(file.models)))
        entries = jacobian_bands(sv; wrt = wrt)
        for (kind, build) in (("jvp", _jvp_document), ("vjp", _vjp_document))
            doc, _ = build(sv, entries, wrt, nothing)
            @test stable_json(doc) ==
                read(joinpath(_PRODUCT_DIR, "$name$(_axistag(wrt)).$kind.evaldoc.json"), String)
        end
    end
    for (name, wrt) in SECOND_ORDER
        file = fixture("$name.esm")
        sv = sysview(file, first(keys(file.models)))
        entries = jacobian_bands2(sv; wrt = wrt)
        forms = wrt == (:states, :states) ? (:contract, :symmetric) : (:contract,)
        for form in forms
            doc, _ = _hvp_document(sv, entries, wrt, nothing, 2; form = form)
            kind = form == :contract ? "hvp" : "hvp-sym"
            @test stable_json(doc) ==
                read(joinpath(_PRODUCT_DIR, "$name.$(_pairtag(wrt)).$kind.evaldoc.json"), String)
        end
    end
end

@testset "product documents: J v and Jᵀ λ against the value goldens" begin
    for (name, wrt) in PRODUCT_AXES, sfx in ("", ".w0")
        path = joinpath(_PRODUCT_DIR, "$name$(_axistag(wrt))$sfx.jvals.json")
        isfile(path) || continue
        g = EarthSciASTDiff.JSON3.read(read(path, String))
        file = fixture("$name.esm")
        rows = _rowcells(g)
        cols = _colcells(file, name, wrt, rows)
        J = zeros(length(rows), length(cols))
        ri = Dict(n => i for (i, n) in enumerate(rows))
        ci = Dict(n => i for (i, n) in enumerate(cols))
        for (r, c, v) in g.entries
            J[ri[String(r)], ci[String(c)]] += v
        end
        v = _seed(length(cols), 1.3)
        λ = _seed(length(rows), 0.7)
        doc, io = jvp_document(file; wrt = wrt)
        jv = _eval_product(
            doc, io.model, g.u, Dict(_doc_cell(io.v, c) => v[k] for (k, c) in enumerate(cols)),
            Float64(g.t); params = g.params
        )
        want = J * v
        for (k, r) in enumerate(rows)
            @test isapprox(jv(_doc_cell(io.jvp, r)), want[k]; rtol = 1.0e-10, atol = 1.0e-13)
        end
        doc, io = vjp_document(file; wrt = wrt)
        vj = _eval_product(
            doc, io.model, g.u, Dict(_doc_cell(io.lam, r) => λ[k] for (k, r) in enumerate(rows)),
            Float64(g.t); params = g.params
        )
        want = J' * λ
        for (k, c) in enumerate(cols)
            @test isapprox(vj(_doc_cell(io.vjp, c)), want[k]; rtol = 1.0e-10, atol = 1.0e-13)
        end
    end
end

@testset "product documents: the contraction against the value goldens" begin
    for (name, wrt) in SECOND_ORDER
        gd = EarthSciASTDiff.JSON3.read(
            read(joinpath(_PRODUCT_DIR, "$name.$(_pairtag(wrt)).hvals.json"), String),
            Dict{String, Any}
        )
        file = fixture("$name.esm")
        S = length(gd["pairs"])
        forms = wrt == (:states, :states) ? (:contract, :symmetric) : (:contract,)
        for form in forms
            doc, io = hvp_document(file; wrt = wrt, stages = S, form = form)
            inputs = Dict{String, Float64}()
            for (i, pr) in enumerate(gd["pairs"])
                for (r, x) in pr["mu"]
                    inputs[_doc_cell(io.mu[i], r)] = x
                end
                if pr["w"] isa Number
                    inputs[io.w[i]["t"]] = pr["w"]
                else
                    for (c, x) in pr["w"]
                        inputs[_doc_cell(io.w[i], c)] = x
                    end
                end
            end
            h = _eval_product(doc, io.model, gd["u"], inputs, Float64(gd["t"]))
            want = Dict{String, Float64}(String(n) => x for (n, x) in gd["g"])
            sv = sysview(file, first(keys(file.models)))
            a2 = wrt[2]
            cells = a2 == :states ? _column_names(
                    [(n, get(EarthSciASTDiff._ctx(sv).shapes, n, Int[])) for n in keys(io.hvp)]
                ) :
                _column_names(jacobian_columns(sv; wrt = a2))
            @test issubset(keys(want), cells)
            for c in cells
                @test isapprox(h(_doc_cell(io.hvp, c)), get(want, c, 0.0); rtol = 1.0e-10, atol = 1.0e-13)
            end
        end
    end
end

@testset "product documents: form" begin
    file = fixture("adv_interior.esm")
    doc, io = vjp_document(file)
    m = doc["models"]["VjpEval"]
    @test io == (model = "VjpEval", lam = Dict("u" => "_lam_u"), vjp = Dict("u" => "_vjp_u"))
    # the inputs are states with a zero derivative; the source's D(u) is zeroed
    for n in ("_lam_u", "_vjp_u")
        @test m["variables"][n] == Dict("type" => "unknown", "default" => 0.0, "shape" => ["x"])
    end
    rhs = only(eq["rhs"] for eq in m["equations"] if eq["lhs"] isa AbstractDict && eq["lhs"]["args"] == ["_vjp_u"])
    # two entries, a left fold of their scatters, each filtered by its column
    @test rhs["op"] == "+" && length(rhs["args"]) == 2
    f = rhs["args"][1]
    @test f["op"] == "faq" && f["output_idx"] == ["_j1"]
    @test f["ranges"] == Dict("_j1" => [1, 8], "i" => [2, 7])
    @test f["filter"]["op"] == "==" && f["filter"]["args"][2] == "_j1"
    # names already taken get `_` appended
    sv = sysview(file, "Adv")
    d2 = EarthSciASTDiff.JSON3.read(read(joinpath(FIXTURES, "adv_interior.esm"), String), Dict{String, Any})
    d2["models"]["Adv"]["variables"]["_v_u"] = Dict("type" => "parameter", "default" => 0.0)
    _, io2 = jvp_document(EarthSciAST.load_string(EarthSciASTDiff.JSON3.write(d2)))
    @test io2.v["u"] == "_v_u_" && io2.jvp["u"] == "_jvp_u"
end

@testset "product documents: the ifelse guard makes every reduced name occur" begin
    # contracting_observed's third entry reads its contracted name only in
    # its column index: Jᵀλ's body must still mention it
    file = fixture("contracting_observed.esm")
    doc, io = vjp_document(file)
    m = doc["models"][io.model]
    rhs = only(eq["rhs"] for eq in m["equations"] if eq["lhs"] isa AbstractDict && eq["lhs"]["args"] == [io.vjp["u"]])
    text = stable_json(rhs)
    @test occursin("\"ifelse\"", text)
end

@testset "product documents: refusals" begin
    file = fixture("bd_chem.esm")
    @test_throws ArgumentError hvp_document(file; stages = 2, wrt = (:states, :parameters), form = :symmetric)
    @test_throws ArgumentError hvp_document(file; stages = 0)
    @test_throws ArgumentError hvp_document(file; stages = 2, form = :dense)
end

@testset "product documents: build-time extent decisions" begin
    using EarthSciASTDiff: _in_extent, _point_in_extent, _all_points, _rebuild
    i, k = VarExpr("i"), VarExpr("k")
    add(a, b) = OpExpr("+", ASTExpr[a, b]); sub(a, b) = OpExpr("-", ASTExpr[a, b])
    mul(a, b) = OpExpr("*", ASTExpr[a, b])
    one = NumExpr(1.0)
    rows(lo, hi) = Band([(lo, hi)], Union{String, Int}["i"], ASTExpr[i], one)
    walk(b, ci, n) = _all_points((_, _, c) -> all(1 .<= c .<= n), _rebuild(b; cidx = ci))
    # the bounds decide what a walk of every point would, both ways
    b = Band([(2, 7)], Union{String, Int}["i"], ASTExpr[i], one, [("k", 1, 3)])
    for e in (
            i, add(i, IntExpr(1)), sub(i, IntExpr(2)), add(i, k), sub(i, k),
            mul(IntExpr(2), i), sub(mul(IntExpr(2), i), k), sub(i, i), add(i, i),
        )
        for n in (5, 8, 10, 13)
            @test _in_extent(b, ASTExpr[e], [n]) === walk(b, ASTExpr[e], n)
        end
    end
    # a guard that drops the out-of-extent corner: decided by the walk
    gb = _rebuild(rows(1, 8); guard = OpExpr(">=", ASTExpr[i, IntExpr(2)]))
    @test _in_extent(gb, ASTExpr[sub(i, IntExpr(1))], [8]) === true
    @test _in_extent(rows(1, 8), ASTExpr[sub(i, IntExpr(1))], [8]) === false
    # an index through a table that is not `const` has no build-time answer
    gather = OpExpr("index", ASTExpr[VarExpr("tbl"), i])
    @test _in_extent(rows(1, 8), ASTExpr[gather], [8]) === nothing
    # a guard that is not closed cannot drop a point: only an all-inside box decides
    ob = _rebuild(rows(1, 8); guard = OpExpr(">", ASTExpr[VarExpr("u"), NumExpr(0.0)]))
    @test _in_extent(ob, ASTExpr[i], [8]) === true
    @test _in_extent(ob, ASTExpr[add(i, IntExpr(1))], [8]) === nothing
    # a real evaluation failure is not taken for "not closed"
    tbl = OpExpr("const", ASTExpr[]; value = [1, 2, 3])
    @test_throws BoundsError _in_extent(rows(1, 8), ASTExpr[OpExpr("index", ASTExpr[tbl, i])], [8])
    # a band with no index names: an open column index is §9.2's refusal,
    # not a raw evaluation error
    nb = Band([(3, 3)], Union{String, Int}[3], ASTExpr[OpExpr("index", ASTExpr[VarExpr("tbl"), IntExpr(2)])], one)
    @test_throws BandError _point_in_extent(nb, "v", nb.cidx, [8], :states)
    lit = Band([(3, 3)], Union{String, Int}[3], ASTExpr[IntExpr(9)], one)
    @test _point_in_extent(lit, "v", lit.cidx, [8], :states) === false
    @test_throws BandError _point_in_extent(lit, "v", lit.cidx, [8], :forcing)
    shut = _rebuild(lit; guard = OpExpr("<", ASTExpr[IntExpr(2), IntExpr(1)]))
    @test _point_in_extent(shut, "v", shut.cidx, [8], :forcing) === false   # no point
end

@testset "jacobian_document of a flattened system declares its content's version" begin
    flat = flatten(fixture("minmod_adv.esm"))
    doc = jacobian_document(flat)
    @test EarthSciASTDiff._has_units((doc["models"], doc["jacobians"]))
    @test doc["esm"] == "1.2.0"
    @test jacobian_document(flatten(fixture("bd_chem.esm")))["esm"] == "1.1.0"
end

@testset "jac_prototype keeps its docstring" begin
    @test occursin("freshly zeroed sparse prototype", string(@doc jac_prototype))
end
