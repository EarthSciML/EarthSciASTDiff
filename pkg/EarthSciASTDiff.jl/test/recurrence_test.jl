# Causal recurrences (esm-jacobian-spec.md §4.5, §9.7), mirroring the Rust
# binding's tests/recurrence.rs and the Python binding's test_recurrence.py.
# The `recurrence_sweep` and `pbl_rank1` goldens are byte-compared, and their
# values checked against numbers no band of this package produced: central
# differences of the SOURCE right-hand side, evaluated by EarthSciAST, which
# executes the recurrences itself. The three derivations of `J` — the
# materialized bands through the transfer recurrence (the evaluation
# document), the tangent sweeps (`JvpEval`) and the adjoint sweeps
# (`VjpEval`) — agree to rounding, and the second-order contraction with
# central differences of `Σ μᵢᵀ J wᵢ`.

using EarthSciASTDiff: _evaluation_document, _cellname, _foreach_point, _ctx,
    _observed_defs, serialize_band, BandError, materialize

const _REC_GOLDENS = joinpath(@__DIR__, "..", "..", "..", "tests", "goldens")
const _REC_VALID = joinpath(@__DIR__, "..", "..", "..", "tests", "valid")
const _REC = "recurrence_sweep"

_rec_file(name = _REC) = EarthSciAST.load_path(joinpath(_REC_VALID, "$name.esm"))
_rec_gold(f) = read(joinpath(_REC_GOLDENS, f), String)
_rec_json(f) = EarthSciASTDiff.JSON3.read(_rec_gold(f), Dict{String, Any})

# The pinned point of a value golden, in the source's state order.
function _rec_point(file, g)
    layout = _doc_source(file, nothing, NamedTuple())
    u = Float64[g["u"][r] for r in layout.rownames]
    return layout, u
end

# `J` assembled from the evaluation document's band fields (the materialized
# bands through the transfer recurrence, §4.5), placed by each band point.
function _materialized(file, wrt, u, p, t)
    sv = sysview(file, first(keys(file.models)))
    shapes = _ctx(sv).shapes
    entries = jacobian_bands(sv; wrt = wrt)
    doc, names = _evaluation_document(sv, entries)
    layout = _doc_source(file, nothing, NamedTuple())
    out = _doc_call(_doc_evaluator(doc, "JacobianEval"), layout, u, p, t, Dict{String, Float64}())
    J = Dict{Tuple{String, String}, Float64}()
    for (en, nm) in zip(entries, names)
        shape = get(shapes, en.v, Int[])
        _foreach_point(en.band) do cell, cc, cidx
            all(d -> 1 <= cidx[d] <= shape[d], eachindex(shape)) || return
            key = (_cellname(en.u, cell), _cellname(en.v, cidx))
            J[key] = get(J, key, 0.0) + out(_cellname(nm, vcat(cell, cc)))
        end
    end
    return J
end

_sparse_dict(res) = (
    (I, J, V) = findnz(res.J);
    Dict((res.rownames[I[k]], res.colnames[J[k]]) => V[k] for k in eachindex(I))
)

# ── small documents ──────────────────────────────────────────────────────────

_O(kv...) = Dict{String, Any}(kv...)
_o(op, a...) = _O("op" => op, "args" => Any[a...])
_ix(v, a...) = _o("index", v, a...)
_faqk(body; idx = "k", range = _O("from" => "lev")) = _O(
    "op" => "faq", "args" => Any[], "output_idx" => Any[idx],
    "ranges" => _O(idx => range), "expr" => body
)

# One state x[lev], one recurrence V with body `body` (over k), D(x)[k] = V[k].
_recdoc(body) = _O(
    "esm" => "1.1.0", "metadata" => _O("name" => "R"),
    "index_sets" => _O("lev" => _O("kind" => "interval", "size" => 4)),
    "models" => _O(
        "R" => _O(
            "variables" => _O(
                "x" => _O("type" => "unknown", "shape" => Any["lev"], "default" => 0.5),
                "V" => _O("type" => "unknown", "shape" => Any["lev"]),
            ),
            "equations" => Any[
                _O("lhs" => "V", "rhs" => _faqk(body)),
                _O(
                    "lhs" => _O("op" => "D", "args" => Any["x"], "wrt" => "t"),
                    "rhs" => _faqk(_ix("V", "i"); idx = "i"),
                ),
            ],
        )
    ),
)

_model(variables, equations) = _O(
    "esm" => "1.1.0", "metadata" => _O("name" => "R"),
    "index_sets" => _O("lev" => _O("kind" => "interval", "size" => 4)),
    "models" => _O("R" => _O("variables" => variables, "equations" => equations)),
)

# `faq{D(u[i])} ~ rhs` over i in [lo, hi]: one equation of an array state.
_lifted(u, lo, hi, rhs) = _O(
    "lhs" => _O(
        "op" => "faq", "args" => Any[], "output_idx" => Any["i"],
        "ranges" => _O("i" => Any[lo, hi]),
        "expr" => _O("op" => "D", "args" => Any[_ix(u, "i")], "wrt" => "t"),
    ),
    "rhs" => rhs,
)

# `v[k] = src[k] + a·v[k − 1]` (src[1] at k = 1)
_sweep(v, src, a) = _O(
    "lhs" => v,
    "rhs" => _faqk(
        _o(
            "ifelse", _o("<=", "k", 1), _ix(src, 1),
            _o("+", _ix(src, "k"), _o("*", a, _ix(v, _o("-", "k", 1))))
        )
    ),
)

function _points(entries)
    out = Set{Tuple{String, Vector{Int}, String, Vector{Int}}}()
    for en in entries
        _foreach_point((cell, _, cidx) -> push!(out, (en.u, cell, en.v, cidx)), en.band)
    end
    return out
end

@testset "recurrences: goldens" begin
    for (name, axes, pairs, cax) in (
            (_REC, (:states, :parameters), ((:states, :states), (:states, :parameters)), "lev"),
            ("pbl_rank1", (:states,), (), "lev"),
        )
        file = _rec_file(name)
        mname = first(keys(file.models))
        sv = sysview(file, mname)
        for wrt in axes
            t = wrt == :states ? "" : ".$wrt"
            entries = jacobian_bands(sv; wrt = wrt)
            @test stable_json(Any[serialize_band(en) for en in entries]) ==
                _rec_gold("$name$t.bands.json")
            @test stable_json(jacobian_document(file, mname; wrt = wrt)["jacobians"][mname]) ==
                _rec_gold("$name$t.block.json")
            ev = stable_json(first(_evaluation_document(sv, entries)))
            @test ev == _rec_gold("$name$t.evaldoc.json")
            p0 = joinpath(_REC_GOLDENS, "$name$t.evaldoc-nocse.json")
            @test stable_json(first(_evaluation_document(sv, entries; cse = false))) ==
                (isfile(p0) ? read(p0, String) : ev)
            for (kind, build) in (("jvp", jvp_document), ("vjp", vjp_document))
                @test stable_json(first(build(file; wrt = wrt))) ==
                    _rec_gold("$name$t.$kind.evaldoc.json")
            end
            wrt == :states || continue
            so = jacobian_bands(sv; coefficients = false)
            @test all(en -> en.band.coef === nothing, so)
            @test stable_json(Any[serialize_band(en) for en in so]) ==
                _rec_gold("$name.structure.bands.json")
            @test stable_json(serialize_scatter_plan(scatter_plan(file, Csr()))) ==
                _rec_gold("$name.csr.plan.json")
            @test stable_json(
                jacobian_document(file, mname; column_axis = Symbol(cax))["jacobians"][mname]
            ) == _rec_gold("$name.column-$cax.block.json")
        end
        for wrt in pairs
            pt = "$(wrt[1])-$(wrt[2])"
            @test stable_json(Any[serialize_band(en) for en in jacobian_bands2(sv; wrt = wrt)]) ==
                _rec_gold("$name.$pt.bands.json")
            @test stable_json(jacobian_document(file, mname; wrt = Pair(wrt...))["jacobians"][mname]) ==
                _rec_gold("$name.$pt.block.json")
            for form in (wrt == (:states, :states) ? (:contract, :symmetric) : (:contract,))
                kind = form == :contract ? "hvp" : "hvp-sym"
                @test stable_json(first(hvp_document(file; stages = 2, wrt = wrt, form = form))) ==
                    _rec_gold("$name.$pt.$kind.evaldoc.json")
            end
        end
    end
end

@testset "recurrences: J matches central differences of the source" begin
    for (name, wrt, tag) in (
            (_REC, :states, ""), (_REC, :parameters, ".parameters"), ("pbl_rank1", :states, ""),
        )
        file = _rec_file(name)
        g = _rec_json("$name$tag.jvals.json")
        layout, u = _rec_point(file, g)
        want = Dict((String(e[1]), String(e[2])) => Float64(e[3]) for e in g["entries"])
        got = _sparse_dict(doc_jacobian(file; wrt = wrt, u = u, t = g["t"]))
        @test Set(keys(got)) == Set(keys(want))
        @test all(k -> isapprox(got[k], want[k]; rtol = 1.0e-12, atol = 1.0e-14), keys(want))
        name == _REC || continue
        # central differences of the SOURCE right-hand side, which
        # EarthSciAST evaluates by executing the recurrences itself
        f!, _, p0, _, vm0 = EarthSciASTDiff._build_eval(file, nothing)
        vm = Dict(String(k) => Int(v) for (k, v) in vm0)
        rhs(x, p) = (du = zeros(length(x)); f!(du, x, p, g["t"]); du)
        rows = layout.rownames
        h = 1.0e-6
        cols = wrt == :states ? rows : ["eps"]
        for c in cols
            if wrt == :states
                e = zeros(length(u)); e[vm[c]] = h
                up, dn = rhs(u .+ e, p0), rhs(u .- e, p0)
            else
                up = rhs(u, (; p0..., eps = p0.eps + h))
                dn = rhs(u, (; p0..., eps = p0.eps - h))
            end
            for r in rows
                fd = (up[vm[r]] - dn[vm[r]]) / (2h)
                if haskey(want, (r, c))
                    @test isapprox(want[(r, c)], fd; rtol = 1.0e-7, atol = 1.0e-7)
                else
                    @test abs(fd) < 1.0e-9          # off the pattern: exactly zero
                end
            end
        end
    end
end

@testset "recurrences: the three derivations agree" begin
    file = _rec_file()
    for (wrt, tag) in ((:states, ""), (:parameters, ".parameters"))
        g = _rec_json("$_REC$tag.jvals.json")
        layout, u = _rec_point(file, g)
        sweep = _sparse_dict(doc_jacobian(file; wrt = wrt, u = u, t = g["t"]))
        mat = _materialized(file, wrt, u, layout.p, g["t"])
        @test Set(keys(sweep)) == Set(keys(mat))
        @test all(k -> isapprox(mat[k], sweep[k]; rtol = 1.0e-12, atol = 1.0e-14), keys(sweep))
        # ⟨J v, λ⟩ = ⟨v, Jᵀ λ⟩, the tangent sweeps against the adjoint ones
        rows = layout.rownames
        λ = [sin(0.7 * (n - 1) + 0.3) for n in eachindex(rows)]
        jtl, cols = doc_vjp(file, λ; wrt = wrt, u = u, t = g["t"])
        v = [cos(0.4 * (n - 1) + 1.1) for n in eachindex(cols)]
        doc, io = jvp_document(file; wrt = wrt)
        out = _doc_call(
            _doc_evaluator(doc, io.model), layout, u, layout.p, g["t"],
            Dict(_doc_cell(io.v, c) => v[n] for (n, c) in enumerate(cols))
        )
        jv_lam = sum(out(_doc_cell(io.jvp, r)) * λ[n] for (n, r) in enumerate(rows))
        @test isapprox(jv_lam, sum(v .* jtl); rtol = 1.0e-12)
    end
end

@testset "recurrences: the contraction matches central differences" begin
    file = _rec_file()
    for wrt in ((:states, :states), (:states, :parameters))
        gd = _rec_json("$_REC.$(wrt[1])-$(wrt[2]).hvals.json")
        layout, u = _rec_point(file, gd)
        rows = layout.rownames
        pairs = [
            (Float64[p["mu"][r] for r in rows], Float64[p["w"][r] for r in rows])
                for p in gd["pairs"]
        ]
        doc, io = jvp_document(file)
        ev = _doc_evaluator(doc, io.model)
        function phi(x, p)
            tot = 0.0
            for (μ, w) in pairs
                out = _doc_call(
                    ev, layout, x, p, gd["t"],
                    Dict(_doc_cell(io.v, r) => w[n] for (n, r) in enumerate(rows))
                )
                tot += sum(μ[n] * out(_doc_cell(io.jvp, r)) for (n, r) in enumerate(rows))
            end
            return tot
        end
        g, gn = doc_hvp(file, pairs; wrt = wrt, u = u, t = gd["t"])
        h = 1.0e-5
        p0 = layout.p
        for (c, gv) in gd["g"]
            k = findfirst(==(c), gn)
            @test isapprox(g[k], gv; rtol = 1.0e-10, atol = 1.0e-12)
            fd = if wrt[2] == :states
                e = zeros(length(u)); e[findfirst(==(c), rows)] = h
                (phi(u .+ e, p0) - phi(u .- e, p0)) / (2h)
            else
                (phi(u, (; p0..., eps = p0.eps + h)) - phi(u, (; p0..., eps = p0.eps - h))) / (2h)
            end
            @test isapprox(gv, fd; rtol = 1.0e-7, atol = 1.0e-7)
        end
        if wrt == (:states, :states)
            gs, _ = doc_hvp(file, pairs; wrt = wrt, form = :symmetric, u = u, t = gd["t"])
            @test all(k -> isapprox(gs[k], g[k]; rtol = 1.0e-12, atol = 1.0e-14), eachindex(g))
        end
    end
end

@testset "recurrences: recognition and the rank-one plan" begin
    sv = sysview(_rec_file(), "RecurrenceSweep")
    recs = recurrences(sv)
    @test [(r.name, r.dim) for r in recs] ==
        [("bet", 2), ("uf", 2), ("yrev", 2), ("p", 2), ("phi", 2), ("r", 3)]
    block = _rec_json("$_REC.column-lev.block.json")
    @test block["structure"] == "column_block"
    @test block["factorization"]["pattern"] == "dense"

    file = _rec_file("pbl_rank1")
    mname = first(keys(file.models))
    b = jacobian_document(file, mname; column_axis = :lev)["jacobians"][mname]
    f = b["factorization"]
    @test f["pattern"] == "diag_plus_rank1"
    terms = [t["entry"] for t in f["rank1"]]
    @test length(terms) == 2
    for t in terms
        en = b["entries"][t]
        @test en["row"] == en["col"] && length(en["contracted"]) == 1
    end
    @test jacobian_document(file, mname)["jacobians"][mname]["structure"] == "general"
    # structure-only entries carry no coefficient to split: the rank-one
    # decision is left open, and no plan is made from them (spec §6)
    psv = sysview(file, mname)
    soe = jacobian_bands(psv; coefficients = false)
    so = column_lines(soe, psv, "lev")
    @test so.rank1_undecided
    @test so.pattern != "diag_plus_rank1"
    @test isempty(so.rank1)
    @test_throws ArgumentError plan_factorization(soe, psv; column_axis = :lev)
    # the full entries decide it
    full = column_lines(jacobian_bands(psv), psv, "lev")
    @test !full.rank1_undecided && full.pattern == "diag_plus_rank1"
end

@testset "recurrences: a running-max tie takes the earlier argument" begin
    # V[k] = max(V[k-1], x[k]): at an exact tie V[k-1] == x[k] the
    # derivative follows the earlier argument, the carried maximum (spec
    # §3.2's left fold)
    body = _o(
        "ifelse", _o("<=", "k", 1), _ix("x", 1),
        _o("max", _ix("V", _o("-", "k", 1)), _ix("x", "k"))
    )
    file = EarthSciAST.load_document(_recdoc(body))
    J = _sparse_dict(doc_jacobian(file; u = [0.7, 0.4, 0.7, 0.9]))
    # V = [0.7, 0.7, 0.7 (tie), 0.9]
    @test J[("x[3]", "x[1]")] == 1.0
    @test get(J, ("x[3]", "x[3]"), 0.0) == 0.0
    @test J[("x[4]", "x[4]")] == 1.0
    @test get(J, ("x[4]", "x[1]"), 0.0) == 0.0
end

@testset "recurrences: ill-founded ones are refused" begin
    # EarthSciAST refuses these documents itself; this package must refuse
    # them too, so the ill-founded body is put in place after a
    # well-founded document loads
    good = _o("ifelse", _o("<=", "k", 1), _ix("x", 1), _ix("V", _o("-", "k", 1)))
    sv0 = sysview(EarthSciAST.load_document(_recdoc(good)), "R")
    for (why, body) in (
            ("same cell", _o("+", _ix("x", "k"), _ix("V", "k"))),
            ("bare self-reference", _o("+", _ix("x", "k"), "V")),
            ("not along a frame axis", _o("+", _ix("x", "k"), _ix("V", 1, "k"))),
            # a read of a later cell: its lag is -1, never "≥ 1", so its
            # sweep term would be gated to a silent zero
            (
                "a later cell",
                _o("ifelse", _o(">=", "k", 4), _ix("x", "k"), _ix("V", _o("+", "k", 1))),
            ),
        )
        eqs = copy(sv0.equations)
        bad = EarthSciAST.expression_from_json(_faqk(body))
        eqs[1] = EarthSciAST.Equation(eqs[1].lhs, bad)
        sv = SysView(sv0.variables, eqs, sv0.index_sets, sv0.domain, sv0.data_sources)
        err = try
            jacobian_bands(sv)
            nothing
        catch e
            e
        end
        @test err isa BandError
        @test err isa BandError && occursin("recurrence `V`", err.msg)
    end
    # a read of the current cell next to an earlier one is a lag-0 self band:
    # refused as not strictly earlier, not as an unplaceable read
    eqs = copy(sv0.equations)
    body = _o(
        "ifelse", _o("<=", "k", 1), _ix("x", 1),
        _o("+", _o("*", 0.5, _ix("V", "k")), _ix("V", _o("-", "k", 1)))
    )
    eqs[1] = EarthSciAST.Equation(eqs[1].lhs, EarthSciAST.expression_from_json(_faqk(body)))
    sv = SysView(sv0.variables, eqs, sv0.index_sets, sv0.domain, sv0.data_sources)
    err = try
        jacobian_bands(sv)
        nothing
    catch e
        e
    end
    @test err isa BandError && occursin("lag 0", err.msg) &&
        occursin("not strictly earlier", err.msg)
end

@testset "recurrences: a coupled pair is refused" begin
    # V and W each read themselves and each other one level back: one
    # coupled system, which no transfer per recurrence expresses
    good = _o("ifelse", _o("<=", "k", 1), _ix("x", 1), _ix("V", _o("-", "k", 1)))
    sv0 = sysview(EarthSciAST.load_document(_recdoc(good)), "R")
    step(a, b) = _faqk(
        _o(
            "ifelse", _o("<=", "k", 1), _ix("x", 1),
            _o(
                "+", _ix("x", "k"), _o("*", 0.5, _ix(a, _o("-", "k", 1))),
                _o("*", 0.25, _ix(b, _o("-", "k", 1)))
            )
        )
    )
    vars = copy(sv0.variables)
    vars["W"] = vars["V"]
    eqs = copy(sv0.equations)
    eqs[1] = EarthSciAST.Equation(eqs[1].lhs, EarthSciAST.expression_from_json(step("V", "W")))
    insert!(
        eqs, 2,
        EarthSciAST.Equation(VarExpr("W"), EarthSciAST.expression_from_json(step("W", "V")))
    )
    sv = SysView(vars, eqs, sv0.index_sets, sv0.domain, sv0.data_sources)
    @test [r.name for r in recurrences(sv)] == ["V", "W"]
    err = try
        jacobian_bands(sv)
        nothing
    catch e
        e
    end
    @test err isa BandError && occursin("read each other", err.msg)
end

@testset "recurrences: a free name spelled like the transfer index" begin
    # V[k] = k_t·x[k] + 0.5·V[k − 1], with `k_t` a parameter: the transfer
    # band's contracted `κ` is also spelled `k_t`, and composing must rename
    # it apart rather than let the coefficient's `k_t` read the sum index
    lev = _O("type" => "unknown", "shape" => Any["lev"])
    doc = _model(
        _O(
            "x" => merge(lev, _O("default" => 0.5)), "V" => lev,
            "k_t" => _O("type" => "parameter", "default" => 2.0),
        ),
        Any[
            _O(
                "lhs" => "V",
                "rhs" => _faqk(
                    _o(
                        "ifelse", _o("<=", "k", 1), _o("*", "k_t", _ix("x", 1)),
                        _o(
                            "+", _o("*", "k_t", _ix("x", "k")),
                            _o("*", 0.5, _ix("V", _o("-", "k", 1)))
                        )
                    )
                ),
            ),
            _O(
                "lhs" => _O("op" => "D", "args" => Any["x"], "wrt" => "t"),
                "rhs" => _faqk(_ix("V", "i"); idx = "i"),
            ),
        ],
    )
    file = EarthSciAST.load_document(doc)
    entries = jacobian_bands(file, "R")
    reads = [en for en in entries if "k_t" in EarthSciAST.free_variables(en.band.coef)]
    @test !isempty(reads)
    @test all(en -> !("k_t" in first.(en.band.contracted)), reads)
    J = _materialized(file, :states, fill(0.5, 4), nothing, 0.0)
    for k in 1:4, j in 1:4
        want = j <= k ? 2.0 * 0.5^(k - j) : 0.0
        @test isapprox(get(J, ("x[$k]", "x[$j]"), 0.0), want; rtol = 1.0e-12)
    end
end

@testset "recurrences: composites obey the closed-column rule" begin
    # D(x)[i] = V[1 + jj, i] with `jj` a parameter. On the forcing axis (`p`)
    # the parameter-valued index is one-hot (spec §3.5), so the composites
    # through V's transfer have closed columns (§4.3, which `materialize`
    # checks on every composite); on the parameters axis `jj` is itself a
    # target, and the site-dependent read is refused before any composite
    isets = _O(
        "lev" => _O("kind" => "interval", "size" => 4),
        "col" => _O("kind" => "interval", "size" => 2),
    )
    body = _o(
        "ifelse", _o("<=", "k", 1), _ix("p", "c", 1),
        _o("+", _ix("p", "c", "k"), _o("*", 0.5, _ix("V", "c", _o("-", "k", 1))))
    )
    doc = _O(
        "esm" => "1.1.0", "metadata" => _O("name" => "R"), "index_sets" => isets,
        "models" => _O(
            "R" => _O(
                "variables" => _O(
                    "x" => _O("type" => "unknown", "shape" => Any["lev"], "default" => 0.5),
                    "V" => _O("type" => "unknown", "shape" => Any["col", "lev"]),
                    "p" => _O("type" => "parameter", "shape" => Any["col", "lev"], "default" => 1.0),
                    "jj" => _O("type" => "parameter", "default" => 1),
                ),
                "equations" => Any[
                    _O(
                        "lhs" => "V",
                        "rhs" => _O(
                            "op" => "faq", "args" => Any[], "output_idx" => Any["c", "k"],
                            "ranges" => _O("c" => _O("from" => "col"), "k" => _O("from" => "lev")),
                            "expr" => body,
                        ),
                    ),
                    _O(
                        "lhs" => _O("op" => "D", "args" => Any["x"], "wrt" => "t"),
                        "rhs" => _faqk(_ix("V", _o("+", 1, "jj"), "i"); idx = "i"),
                    ),
                ],
            )
        ),
    )
    sv = sysview(EarthSciAST.load_document(doc), "R")
    for co in (true, false)
        es = jacobian_bands(sv; wrt = :forcing, forcing = ["p"], coefficients = co)
        @test !isempty(es) && all(en -> en.v == "p", es)
        for en in es
            bound = Set{String}(r for r in en.band.ridx if r isa String)
            union!(bound, first.(en.band.contracted))
            @test all(c -> issubset(EarthSciAST.free_variables(c), bound), en.band.cidx)
        end
        @test_throws EarthSciASTDiff.DerivativeRuleError jacobian_bands(
            sv; wrt = :parameters, coefficients = co
        )
    end
end

@testset "recurrences: transfer names do not depend on activity" begin
    # `A` (inactive on the state axis: it reads only `p`) sorts before
    # `A_`, and the source already holds `_tr_A`: each recurrence's
    # transfer name is the same in the coefficients and in the
    # `auxiliary` field, so the evaluation document's J is A_'s chain
    # 0.5^(k − j), not A's 0.25^(k − j)
    lev = _O("type" => "unknown", "shape" => Any["lev"])
    doc = _model(
        _O(
            "x" => merge(lev, _O("default" => 0.5)),
            "p" => _O("type" => "parameter", "shape" => Any["lev"], "default" => 1.0),
            "_tr_A" => _O("type" => "parameter", "default" => 0.0),
            "A" => lev, "A_" => lev,
        ),
        Any[
            _sweep("A", "p", 0.25), _sweep("A_", "x", 0.5),
            _O(
                "lhs" => _O("op" => "D", "args" => Any["x"], "wrt" => "t"),
                "rhs" => _faqk(_o("+", _ix("A_", "i"), _ix("A", "i")); idx = "i"),
            ),
        ],
    )
    file = EarthSciAST.load_document(doc)
    block = jacobian_document(file, "R")["jacobians"]["R"]
    read = Set{String}()
    for en in block["entries"]
        union!(read, split(replace(EarthSciASTDiff.JSON3.write(en["coef"]), '"' => ' ')))
    end
    aux = [a["name"] for a in block["auxiliary"]]
    @test !isempty(aux) && issubset(aux, read)
    J = _materialized(file, :states, fill(0.5, 4), nothing, 0.0)
    for k in 1:4, j in 1:4
        want = j <= k ? 0.5^(k - j) : 0.0
        @test isapprox(get(J, ("x[$k]", "x[$j]"), 0.0), want; rtol = 1.0e-12)
    end
end

@testset "recurrences: entries keep each equation of a state apart" begin
    # two equations of `x` (its first level, then the rest): the second's
    # direct band on `x` must not move ahead of the first's band on `y`
    lev = _O("type" => "unknown", "shape" => Any["lev"], "default" => 0.5)
    doc = _model(
        _O("x" => lev, "y" => lev, "V" => _O("type" => "unknown", "shape" => Any["lev"])),
        Any[
            _sweep("V", "y", 0.5),
            _lifted("x", 1, 1, _ix("y", "i")),
            _lifted("x", 2, 4, _o("+", _o("-", _ix("x", "i")), _ix("V", "i"))),
        ],
    )
    cols = [en.v for en in jacobian_bands(EarthSciAST.load_document(doc), "R") if en.u == "x"]
    # the first equation's `y`, then the second's `x` and its composites
    # on `y` (merging the two would sort the second's `x` first)
    @test length(cols) > 2 && cols[1] == "y" && cols[2] == "x" && Set(cols[3:end]) == Set(["y"])
end

@testset "recurrences: structure-only keeps each equation of a state apart" begin
    # two equations of `x` (rows 1..2, then 3..4), each reading `y[i]`,
    # the second also the recurrence V (a sweep of `y`): the first
    # equation's run of column `y` ends where the second's begins, and the
    # structure-only re-merge through the recurrence (spec §2.1) must not
    # join them, as the full mode and the pass without a recurrence never do
    lev = _O("type" => "unknown", "shape" => Any["lev"], "default" => 0.5)
    f(lo, hi, body) = _faqk(body; idx = "i", range = Any[lo, hi])
    doc = _model(
        _O("x" => lev, "y" => lev, "V" => _O("type" => "unknown", "shape" => Any["lev"])),
        Any[
            _sweep("V", "y", 0.5),
            _lifted("x", 1, 2, f(1, 2, _ix("y", "i"))),
            _lifted("x", 3, 4, f(3, 4, _o("+", _ix("y", "i"), _ix("V", "i")))),
        ],
    )
    sv = sysview(EarthSciAST.load_document(doc), "R")
    full = jacobian_bands(sv)
    so = jacobian_bands(sv; coefficients = false)
    @test all(en -> en.band.coef === nothing, so)
    @test _points(so) == _points(full)
    xy = [en.band.rows[1] for en in so if (en.u, en.v) == ("x", "y")]
    @test (1, 2) in xy
    @test all(((lo, hi),) -> hi <= 2 || lo >= 3, xy)
end

@testset "recurrences: the extended system" begin
    sv = sysview(_rec_file(), "RecurrenceSweep")
    ext = extended_bands(sv)
    @test !isempty(ext.recurrences) && !isempty(ext.rec_rows)
    @test length(ext.row_eq) == length(ext.rows)
    # every recurrence of the view is named, active or not
    @test Set(keys(ext.transfer)) == Set(r.name for r in recurrences(sv))
    # materialized, the extended system is `jacobian_bands`'s list
    @test serialize_band.(materialize(sv, ext)) == serialize_band.(jacobian_bands(sv))
    # a pair's first pass is seeded with the second axis's targets: the
    # recurrences that move with either axis are active
    names(e) = Set(r.name for r in e.recurrences)
    seeded = names(extended_bands(sv; wrt = :parameters, second = :states))
    @test issubset(names(extended_bands(sv; wrt = :parameters)), seeded)
    @test issubset(names(ext), seeded)
end
