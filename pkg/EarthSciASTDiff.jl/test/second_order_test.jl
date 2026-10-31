# Second-order bands (spec §4.4): the band calculus applied to first-order
# coefficients, for the axis pairs a Rosenbrock adjoint contracts. Pinned
# byte-for-byte by tests/goldens/<fixture>.<a1>-<a2>.{bands,block}.json and by
# value in the matching `*.hvals.json`, which the other bindings must
# reproduce. Kept in step with scripts/regenerate-goldens.jl.

using EarthSciASTDiff: BandError, serialize_band

const SECOND_ORDER = (
    ("bd_chem", (:states, :states)),
    ("arrhenius_chem", (:states, :states)),
    ("arrhenius_chem", (:states, :parameters)),
    ("forcing_bracket", (:states, :forcing)),
    ("forcing_bracket", (:time, :states)),
    ("minmod_adv", (:states, :states)),
    ("filtered_scan", (:states, :states)),
    ("semiring_reduce", (:states, :states)),
)
_pairtag((a1, a2)) = "$a1-$a2"

# The oracle's document: the second axis's parameters (scalar or array)
# promoted to states with a zero derivative, so ForwardDiff with respect to
# the state vector also differentiates with respect to them.
function _promoted_any(path, model, names)
    d = EarthSciASTDiff.JSON3.read(read(path, String), Dict{String, Any})
    m = d["models"][model]
    for n in names
        v = m["variables"][n]
        v["type"] = "unknown"
        delete!(v, "update")
        shape = [d["index_sets"][s]["size"] for s in get(v, "shape", Any[])]
        rhs = isempty(shape) ? 0.0 : Dict(
                "op" => "makearray", "args" => [],
                "regions" => [[[1, k] for k in shape]], "values" => [0.0]
            )
        push!(m["equations"], Dict("lhs" => Dict("op" => "D", "args" => [n], "wrt" => "t"), "rhs" => rhs))
    end
    return EarthSciAST.load_string(EarthSciASTDiff.JSON3.write(d))
end

# The pinned point and stage pairs of the value goldens
# (scripts/regenerate-value-goldens.jl): two stages.
_hpoint(n) = [0.2 + 0.6 * abs(sin(2.3 * i)) for i in 1:n]
_hpairs(nrow, ncol, a1) = [
    (
        [sin(1.7 * k + 0.9 * s) for k in 1:nrow],
        a1 == :time ? 0.5 + 0.25 * s : [cos(0.6 * k + 1.1 * s) for k in 1:ncol],
    ) for s in 1:2
]
_ht(name) = name == "forcing_bracket" ? 0.37 : 0.7

# ∇ Σᵢ μᵢᵀ (∂f/∂a1) wᵢ by ForwardDiff over ForwardDiff through the tree-walk
# RHS of the promoted document, read off at `hv.col2names`.
function _oracle_hvp(name, hv, u, ps, t)
    path = joinpath(FIXTURES, "$name.esm")
    file = EarthSciAST.load_path(path)
    model = first(keys(file.models))
    a1, a2 = hv.wrt
    promote = a2 == :states ? String[] : unique(first.(something(hv.columns, [])))
    src = isempty(promote) ? file : _promoted_any(path, model, promote)
    f!, x0, pp, _, vm = build_evaluator(src; _hkw(name)...)
    x = copy(x0)
    rows = [vm[r] for r in hv.rownames]
    x[rows] .= u
    rhs(y, τ) = (du = similar(y, promote_type(eltype(y), typeof(τ))); f!(du, y, pp, τ); du)
    function φ(y)
        acc = zero(eltype(y))
        for (μ, w) in ps
            d = if a1 == :time
                ForwardDiff.derivative(τ -> rhs(y, τ), t) .* w
            else
                dir = zeros(length(y)); dir[rows] .= w
                ForwardDiff.derivative(s -> rhs(y .+ s .* dir, t), 0.0)
            end
            acc += sum(μ .* d[rows])
        end
        return acc
    end
    g = ForwardDiff.gradient(φ, x)
    return [g[vm[c]] for c in hv.col2names]
end

@testset "second order: goldens" begin
    golden_dir = joinpath(@__DIR__, "..", "..", "..", "tests", "goldens")
    for (name, wrt) in SECOND_ORDER
        file = fixture("$name.esm")
        model = first(keys(file.models))
        gold(kind) = joinpath(golden_dir, "$name.$(_pairtag(wrt)).$kind.json")
        entries = jacobian_bands2(file, model; wrt = wrt)
        @test stable_json(Any[serialize_band(en) for en in entries]) == read(gold("bands"), String)
        block = jacobian_document(file, model; wrt = Pair(wrt...))["jacobians"][model]
        @test block["wrt"] == Any[String(wrt[1]), String(wrt[2])]
        @test !haskey(block, "factorization")
        @test haskey(block, "columns") == (wrt[2] in (:parameters, :forcing))
        @test stable_json(block) == read(gold("block"), String)
        w, back = parse_jacobian_block(block)
        @test w == wrt
        @test stable_json(Any[serialize_band(en) for en in back]) ==
            stable_json(Any[serialize_band(en) for en in entries])
    end
end

@testset "second order: what the entries are" begin
    # bd_chem's bimolecular terms: k3·B·C gives both orientations of (B, C)
    # in A's row, and B² a single (B, B) entry; the linear k1 term none
    e2 = jacobian_bands2(fixture("bd_chem.esm"), "Chem")
    trip = Set((en.u, en.v, en.w) for en in e2)
    @test trip == Set([("A", "B", "C"), ("A", "C", "B"), ("B", "B", "B"), ("C", "B", "B")])
    # every orientation comes out on its own: (r, c, s) ⇒ (r, s, c)
    for (name, model) in (("bd_chem", "Chem"), ("arrhenius_chem", "Chem"))
        t = Set((en.u, en.v, en.w) for en in jacobian_bands2(fixture("$name.esm"), model))
        @test all(((r, c, s),) -> (r, s, c) in t, t)
    end
    # the Arrhenius rate is an observed of parameters only: opaque on the
    # state pair, inlined on the parameter pair so T's sites are seen
    ss = jacobian_bands2(fixture("arrhenius_chem.esm"), "Chem")
    @test any(en -> "k1" in EarthSciAST.free_variables(en.band.coef), ss)
    sp = jacobian_bands2(fixture("arrhenius_chem.esm"), "Chem"; wrt = (:states, :parameters))
    @test any(en -> en.w == "T" && stable_json(EarthSciASTDiff._ser_expr(only(col2_idx(en)))) == "\"i\"", sp)
    @test all(en -> !("k1" in EarthSciAST.free_variables(en.band.coef)), sp)
    # minmod's limiter is piecewise linear: its first-order coefficients are
    # branch-wise constant, so the second-order pattern is empty (§3.2)
    @test isempty(jacobian_bands2(fixture("minmod_adv.esm"), "AdvectionMinmodTVD"))
    # the bracket's records stay distinct second columns
    fb = jacobian_bands2(fixture("forcing_bracket.esm"), "Bracket"; wrt = (:states, :forcing))
    @test any(en -> en.w == "G" && !isempty(en.band.contracted), fb)
    @test Set(stable_json(EarthSciASTDiff._ser_expr(col2_idx(en)[1])) for en in fb if en.u == "c" && en.w == "F") ==
        Set(["1", "2"])
end

@testset "second order: structure" begin
    blk(name, model, wrt) = jacobian_document(fixture("$name.esm"), model; wrt = Pair(wrt...))["jacobians"][model]
    @test blk("bd_chem", "Chem", (:states, :states))["structure"] == "block_diagonal"
    @test blk("arrhenius_chem", "Chem", (:states, :states))["structure"] == "block_diagonal"
    # a scalar parameter column is shared by every lane: not lane-local
    @test !haskey(blk("arrhenius_chem", "Chem", (:states, :parameters)), "structure")
    @test !haskey(blk("forcing_bracket", "Bracket", (:time, :states)), "structure")
end

# What the oracle needs to know of a pair: its axes, the second axis's column
# space, and the row and second-column names of `doc_hvp`.
function _hpair(file, wrt; build_kwargs = NamedTuple())
    sv = sysview(file, first(keys(file.models)))
    rows = _doc_source(file, nothing, build_kwargs).rownames
    columns = wrt[2] in (:parameters, :forcing) ? jacobian_columns(sv; wrt = wrt[2]) : nothing
    return (
        wrt = wrt, columns = columns, rownames = rows,
        colnames = wrt[1] == :time ? ["t"] : rows,
        col2names = jacobian_pattern(file; wrt = wrt[2], build_kwargs = build_kwargs).colnames,
    )
end

@testset "second order: the contraction document against ForwardDiff over ForwardDiff" begin
    for (name, wrt) in SECOND_ORDER
        file = fixture("$name.esm")
        hv = _hpair(file, wrt; build_kwargs = _hkw(name))
        u = _hpoint(length(hv.rownames))
        ps = _hpairs(length(hv.rownames), length(hv.colnames), wrt[1])
        t = _ht(name)
        ref = _oracle_hvp(name, hv, u, ps, t)
        g, cols = doc_hvp(file, ps; wrt = wrt, u = u, t = t, build_kwargs = _hkw(name))
        @test cols == hv.col2names
        @test isapprox(g, ref; rtol = 1.0e-10, atol = 1.0e-12)
        if wrt == (:states, :states)
            gs, _ = doc_hvp(
                file, ps; wrt = wrt, form = :symmetric, u = u, t = t, build_kwargs = _hkw(name)
            )
            @test isapprox(gs, ref; rtol = 1.0e-10, atol = 1.0e-12)
        end
    end
end

@testset "second order: document forms" begin
    file = fixture("forcing_bracket.esm")
    jb = jacobian_document(file, "Bracket"; wrt = [:states, (:states, :forcing), [:time, :states]])["jacobians"]["Bracket"]
    @test [b["wrt"] for b in jb] == Any["states", Any["states", "forcing"], Any["time", "states"]]
    parsed = parse_jacobian_blocks(jb)
    @test first.(parsed) == Any[:states, (:states, :forcing), (:time, :states)]
    @test parsed[2][2] isa Vector{JacEntry2}
    @test_throws ArgumentError jacobian_document(file, "Bracket"; wrt = [(:states, :states), (:states, :states)])
    @test_throws ArgumentError jacobian_document(file, "Bracket"; wrt = [(:states, :states), :states => :states])
    # a bare Tuple reads both as a pair and as a list of axes: refused
    @test_throws ArgumentError jacobian_document(file, "Bracket"; wrt = (:states, :forcing))
    @test_throws ArgumentError jacobian_document(file, "Bracket"; wrt = ("states", "forcing"))
    # a Pair anywhere a pair goes
    @test jacobian_document(file, "Bracket"; wrt = :states => :forcing)["jacobians"]["Bracket"] ==
        jb[2]
    @test jacobian_document(file, "Bracket"; wrt = [:states, :states => :forcing, :time => :states])["jacobians"]["Bracket"] ==
        jb
    @test serialize_band.(jacobian_bands2(sysview(file, "Bracket"); wrt = :time => :states)) ==
        serialize_band.(jacobian_bands2(sysview(file, "Bracket"); wrt = (:time, :states)))
    # one emission shares its first-order passes: the blocks equal those
    # emitted alone
    for name in ("arrhenius_chem", "bd_chem")
        f = fixture("$name.esm")
        m = first(keys(f.models))
        wrts = [:states, :states => :states, :states => :parameters, :time]
        both = jacobian_document(f, m; wrt = wrts)["jacobians"][m]
        @test both == Any[jacobian_document(f, m; wrt = w)["jacobians"][m] for w in wrts]
    end
end

@testset "second order: refusals" begin
    sv = sysview(fixture("forcing_bracket.esm"), "Bracket")
    @test_throws ArgumentError jacobian_bands2(sv; wrt = (:forcing, :states))
    @test_throws ArgumentError jacobian_bands2(sv; wrt = (:states, :time))
    @test_throws ArgumentError jacobian_bands2(sv; wrt = (:states,))
    @test_throws ArgumentError hvp_document(fixture("bd_chem.esm"); stages = 1, form = :nope)
    @test_throws ArgumentError hvp_document(
        fixture("forcing_bracket.esm"); stages = 1, wrt = (:time, :states), form = :symmetric
    )
    # a second column read through a parameter offset is one-hot over the
    # buffer's extent (spec §3.5), every point a placeable column
    O(kv...) = Dict{String, Any}(kv...)
    ix(v, i...) = O("op" => "index", "args" => Any[v, i...])
    body = O("op" => "*", "args" => Any[ix("c", "i"), ix("c", "i"), ix("F", 1, O("op" => "+", "args" => Any["i", "s"]))])
    doc = EarthSciAST.load_document(
        O(
            "esm" => "1.1.0", "metadata" => O("name" => "B"),
            "index_sets" => O("x" => O("kind" => "interval", "size" => 4), "xf" => O("kind" => "interval", "size" => 5), "rec" => O("kind" => "interval", "size" => 2)),
            "models" => O(
                "B" => O(
                    "variables" => O(
                        "c" => O("type" => "unknown", "shape" => Any["x"], "default" => 0.5),
                        "F" => O("type" => "parameter", "shape" => Any["rec", "xf"], "default" => 1.0),
                        "s" => O("type" => "parameter", "default" => 1),
                    ),
                    "equations" => Any[
                        O(
                            "lhs" => O("op" => "D", "args" => Any["c"], "wrt" => "t"),
                            "rhs" => O("op" => "faq", "args" => Any[], "output_idx" => Any["i"], "ranges" => O("i" => O("from" => "x")), "expr" => body)
                        ),
                    ]
                )
            )
        )
    )
    e2 = jacobian_bands2(sysview(doc, "B"); wrt = (:states, :forcing), forcing = ["F"])
    @test !isempty(e2) && all(en -> any(c -> startswith(c[1], "_oh"), en.band.contracted), e2)
end
