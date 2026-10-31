# Filtered faq nodes, non-`+` reductions and value-indexed reads
# (esm-jacobian-spec.md §3.5): every case is checked against ForwardDiff
# through the evaluator, and the guarded cases for an EXACT pattern — the
# window's points, no padding.

using JSON3
using EarthSciASTDiff: BandError, DerivativeRuleError

module _U9
    O(kv...) = Dict{String, Any}(kv...)
    idx(v, i...) = O("op" => "index", "args" => Any[v, i...])
    op(o, a...) = O("op" => o, "args" => Any[a...])
    faq(oidx, ranges, body; kw...) = O(
        "op" => "faq", "args" => Any[], "output_idx" => Any[oidx...],
        "ranges" => O(ranges...), "expr" => body, (String(k) => v for (k, v) in kw)...
    )
    # One model over `x` (size n) and, optionally, `z`; `u` is the state.
    function doc(rhs; n = 5, nz = 0, ushape = ["x"], vars = O(), eqs = Any[])
        sets = O("x" => O("kind" => "interval", "size" => n))
        nz > 0 && (sets["z"] = O("kind" => "interval", "size" => nz))
        return O(
            "esm" => "1.1.0",
            "metadata" => O("name" => "U9"),
            "index_sets" => sets,
            "models" => O(
                "U9" => O(
                    "variables" => O(
                        "u" => O("type" => "unknown", "shape" => Any[ushape...], "default" => 0.5),
                        "w" => O("type" => "parameter", "default" => 0.7),
                        vars...
                    ),
                    "equations" => Any[
                        eqs...,
                        O("lhs" => O("op" => "D", "args" => Any["u"], "wrt" => "t"), "rhs" => rhs),
                    ]
                )
            )
        )
    end
end

@testset "filters, semiring reductions, value-indexed reads" begin
    U = _U9
    O, idx, op, faq = U.O, U.idx, U.op, U.faq
    oi(body; n = 5, kw...) = faq(["i"], ["i" => Any[1, n], "j" => Any[1, n]], body; kw...)
    le(a, b) = op("<=", a, b); lt(a, b) = op("<", a, b)
    ge(a, b) = op(">=", a, b); gt(a, b) = op(">", a, b)
    neg(x) = op("-", x)

    function check(d; u, label, npat = nothing)
        mktempdir() do dir
            path = joinpath(dir, "u9.esm")
            open(io -> JSON3.write(io, d), path, "w")
            file = EarthSciAST.load_path(path)
            f!, _, p, _, _ = build_evaluator(file; compiler = :interpreter)
            Jad = ad_jacobian(f!, u, p, 0.0)
            res = assemble_jacobian(file; u = u, p = p, build_kwargs = (compiler = :interpreter,))
            @testset "$label" begin
                @test maximum(abs.(Matrix(res.J) .- Matrix(Jad)); init = 0.0) ≤ 1.0e-12
                @test all(res.pattern .| (Jad .== 0))           # superset of the local pattern
                npat === nothing || @test nnz(res.pattern) == npat
            end
            return res, file
        end
    end
    u5 = [0.3, 0.9, 0.2, 0.6, 0.45]

    # ── filtered `+` at array level ──
    scan(f) = neg(oi(op("*", "w", idx("u", "j")); filter = f))
    check(U.doc(scan(le("j", "i"))); u = u5, label = "inclusive forward scan", npat = 15)
    check(U.doc(scan(lt("j", "i"))); u = u5, label = "exclusive forward scan", npat = 10)
    check(U.doc(scan(ge("j", "i"))); u = u5, label = "inclusive reverse scan", npat = 15)
    check(U.doc(scan(gt("i", "j"))); u = u5, label = "mirrored spelling", npat = 10)
    check(U.doc(scan(le("j", 2))); u = u5, label = "constant window (box only)", npat = 10)
    check(
        U.doc(scan(op("and", le("j", "i"), ge("j", 2)))); u = u5,
        label = "window clipped by a constant", npat = 10
    )
    # the window read squared, through the chain rule, and a value gate
    check(
        U.doc(neg(oi(op("^", idx("u", "j"), 2); filter = op("and", le("j", "i"), gt(idx("u", "j"), 0.4))))),
        u = u5, label = "value-dependent filter conjunct", npat = 15
    )
    # a column that reads no contracted index: the filtered scalar sum
    check(
        U.doc(neg(oi(op("*", idx("u", "i"), idx("u", "j")); filter = lt("j", "i")))); u = u5,
        label = "column with and without the contracted index"
    )
    # the Mz shape: (gi, gke) rows over a level contraction gk < gke
    mz = faq(
        ["gi", "gke"], ["gi" => Any[1, 3], "gke" => Any[1, 4], "gk" => Any[1, 4]],
        op("*", "w", idx("u", "gi", "gk")); filter = lt("gk", "gke")
    )
    u34 = collect(range(0.1, 1.2; length = 12))
    res, _ = check(
        U.doc(neg(mz); n = 3, nz = 4, ushape = ["x", "z"]); u = u34,
        label = "Mz: exclusive level scan", npat = 3 * 6
    )
    @test any(en -> en.band.guard !== nothing, res.entries)
    # a row-only filter on a non-contracting node: rows clipped, no guard
    res, _ = check(
        U.doc(faq(["i"], ["i" => Any[1, 5]], op("*", "w", idx("u", "i")); filter = ge("i", 3))); u = u5,
        label = "row-only filter", npat = 3
    )
    @test all(en -> en.band.guard === nothing, res.entries)

    # ── a filtered sum read through an observed (nested scalar sum) ──
    svar = O("s" => O("type" => "unknown", "shape" => Any["x"]))
    seq = Any[O("lhs" => "s", "rhs" => oi(op("^", idx("u", "j"), 2); filter = le("j", "i")))]
    res, _ = check(
        U.doc(faq(["i"], ["i" => Any[1, 5]], neg(op("*", idx("s", "i"), idx("u", "i")))); vars = svar, eqs = seq),
        u = u5, label = "nested filtered sum", npat = 15
    )
    @test any(en -> en.band.guard !== nothing, res.entries)

    # ── max / min ──
    mx(f; red = "max") = oi(op("*", idx("u", "j"), idx("u", "j")); filter = f, reduce = red)
    check(U.doc(neg(mx(le("j", "i")))); u = u5, label = "running max", npat = 15)
    check(U.doc(neg(mx(ge("j", "i"); red = "min"))); u = u5, label = "suffix min", npat = 15)
    check(
        U.doc(neg(oi(idx("u", "j"); semiring = "max_product"))); u = u5,
        label = "max_product semiring", npat = 25
    )
    # nested: D(u)[i] = -u[i]·max_j u[j]
    mxs = faq([], ["j" => Any[1, 5]], idx("u", "j"); reduce = "max")
    check(
        U.doc(faq(["i"], ["i" => Any[1, 5]], neg(op("*", idx("u", "i"), mxs)))); u = u5,
        label = "nested max"
    )
    # Exact ties: the first extremum in ascending order takes the derivative.
    utie = [0.3, 0.9, 0.2, 0.9, 0.45]
    mktempdir() do dir
        path = joinpath(dir, "tie.esm")
        open(io -> JSON3.write(io, U.doc(neg(mx(le("j", "i"))))), path, "w")
        file = EarthSciAST.load_path(path)
        _, _, p, _, _ = build_evaluator(file; compiler = :interpreter)
        J = Matrix(assemble_jacobian(file; u = utie, p = p, build_kwargs = (compiler = :interpreter,)).J)
        @test J[4, 2] ≈ -2 * 0.9 && J[4, 4] == 0 && J[5, 2] ≈ -2 * 0.9 && J[5, 4] == 0
    end
    # a max over two contracted indices has no ascending order: refused
    mx2 = faq(
        ["i"], ["i" => Any[1, 3], "j" => Any[1, 3], "k" => Any[1, 4]],
        idx("u", "j", "k"); reduce = "max"
    )
    mktempdir() do dir
        path = joinpath(dir, "mx2.esm")
        open(io -> JSON3.write(io, U.doc(mx2; n = 3, nz = 4, ushape = ["x", "z"])), path, "w")
        @test_throws BandError jacobian_bands(EarthSciAST.load_path(path), "U9")
    end

    # ── products ──
    pr = oi(op("+", 1, idx("u", "j")); filter = le("j", "i"), reduce = "*")
    check(U.doc(neg(pr)); u = u5, label = "prefix product", npat = 15)
    check(U.doc(neg(pr)); u = [0.3, -1.0, 0.2, 0.6, 0.45], label = "prefix product, zero factor", npat = 15)

    # ── a level search used as an index (one-hot) ──
    # κ = min{k : u[k] > 0.5}; D(u)[i] = -u[κ]·u[i]
    kvar = O("kap" => O("type" => "unknown"))
    keq = Any[
        O(
            "lhs" => "kap", "rhs" => faq(
                [], ["k" => Any[1, 5]], "k"; reduce = "min", filter = gt(idx("u", "k"), 0.5)
            )
        ),
    ]
    res, _ = check(
        U.doc(faq(["i"], ["i" => Any[1, 5]], neg(op("*", idx("u", "kap"), idx("u", "i")))); vars = kvar, eqs = keq),
        u = u5, label = "level search as an index", npat = 25
    )
    @test any(en -> any(c -> startswith(c[1], "_oh"), en.band.contracted), res.entries)
end

@testset "column_block structure" begin
    U = _U9
    O, idx, op, faq = U.O, U.idx, U.op, U.faq
    load(d) = mktempdir() do dir
        path = joinpath(dir, "cb.esm")
        open(io -> JSON3.write(io, d), path, "w")
        EarthSciAST.load_path(path)
    end
    R = ["i" => Any[1, 3], "k" => Any[1, 4]]
    # an exclusive level scan along z plus a local term: dense lower lines
    scan = faq(["i", "k"], [R..., "m" => Any[1, 4]], op("*", "w", idx("u", "i", "m")); filter = op("<", "m", "k"))
    loc = faq(["i", "k"], R, op("-", idx("u", "i", "k")))
    file = load(U.doc(op("+", scan, loc); n = 3, nz = 4, ushape = ["x", "z"]))
    sv = sysview(file, "U9")
    entries = jacobian_bands(sv)
    @test detect_structure(entries, sv) == :general
    @test detect_structure(entries, sv; column_axis = "z") == :column_block
    @test detect_structure(entries, sv; column_axis = "x") == :general
    cl = column_lines(entries, sv, "z")
    @test (cl.pattern, cl.kl, cl.ku, cl.line_length, cl.block_vars) == ("lower", 3, 0, 4, ["u"])
    block = jacobian_document(file, "U9"; column_axis = "z")["jacobians"]["U9"]
    @test block["structure"] == "column_block"
    @test block["factorization"] == Dict(
        "type" => "column_block", "axis" => "z", "line_length" => 4,
        "block_vars" => ["u"], "pattern" => "lower", "bandwidth" => [3, 0]
    )
    # without the axis the block is unchanged
    @test jacobian_document(file, "U9")["jacobians"]["U9"]["structure"] == "general"
    # a one-step stencil along z: banded lines
    sten = faq(["i", "k"], ["i" => Any[1, 3], "k" => Any[2, 4]], op("*", "w", idx("u", "i", op("-", "k", 1))))
    file2 = load(U.doc(op("+", sten, loc); n = 3, nz = 4, ushape = ["x", "z"]))
    sv2 = sysview(file2, "U9")
    cl2 = column_lines(jacobian_bands(sv2), sv2, "z")
    @test (cl2.pattern, cl2.kl, cl2.ku) == ("banded", 1, 0)
    @test detect_structure(jacobian_bands(sv2), sv2; column_axis = "z") == :column_block
    # a chemistry-shaped (per-cell) Jacobian stays block-diagonal
    file3 = load(U.doc(loc; n = 3, nz = 4, ushape = ["x", "z"]))
    sv3 = sysview(file3, "U9")
    @test detect_structure(jacobian_bands(sv3), sv3; column_axis = "z") == :block_diagonal
end

@testset "silent paths: faq-LHS observed, unresolvable columns" begin
    U = _U9
    O, idx, op, faq = U.O, U.idx, U.op, U.faq
    # `s[i] ~ Σ_j u[j]·u[i]` written with a faq LHS is an observed, inlined,
    # so the bands go through it to `u` — never a column of `s`.
    svar = O("s" => O("type" => "unknown", "shape" => Any["x"]))
    seq = Any[
        O(
            "lhs" => faq(["i"], ["i" => Any[1, 5]], idx("s", "i")),
            "rhs" => faq([], ["j" => Any[1, 5]], op("*", idx("u", "j"), idx("u", "i"))),
        ),
    ]
    rhs = faq(["i"], ["i" => Any[1, 5]], op("-", idx("s", "i")))
    mktempdir() do dir
        path = joinpath(dir, "lhs.esm")
        open(io -> JSON3.write(io, U.doc(rhs; vars = svar, eqs = seq)), path, "w")
        file = EarthSciAST.load_path(path)
        entries = jacobian_bands(file, "U9")
        @test !isempty(entries) && all(en -> en.v == "u", entries)
        f!, _, p, _, _ = build_evaluator(file; compiler = :interpreter)
        u = [0.3, 0.9, 0.2, 0.6, 0.45]
        res = assemble_jacobian(file; u = u, p = p, build_kwargs = (compiler = :interpreter,))
        @test Matrix(res.J) ≈ Matrix(ad_jacobian(f!, u, p, 0.0)) atol = 1.0e-13
    end
    # A column whose variable the layout does not hold at all is refused at
    # assembly; a cell past a state's extent is a ghost read.
    slot, _, _ = EarthSciASTDiff._colmap(Dict("u[1]" => 1, "u[2]" => 2), nothing, :states)
    @test slot("u[3]") == 0
    @test_throws BandError EarthSciASTDiff._colslot(slot, "V[1]", "u[1]", :states)
end
