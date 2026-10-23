# A CONTRACTING observed (issue #6): `s[i] = Σ_k f(u[i,k])`, a faq whose
# ranges bind `k` beyond its output index. Inlining it as the bare body drops
# the sum, and a reading faq that binds `k` too CAPTURES the leftover index —
# the column sum collapses to the local cell, with no error. The read must
# keep the sum (inline.jl `_beta_faq`), and the band calculus must follow the
# contraction through it (`_dsum`, `_lift_nested`).
#
#   s[i] = Σ_k u[i,k]²,  D(u)[i,k] = g(s[i], u[i,k])
#   ⇒ ∂ D(u)[i,k] / ∂ u[i,k'] ≠ 0 for every k'

using JSON3
using EarthSciASTDiff: BandError, DerivativeRuleError

@testset "contracting observed keeps its sum" begin
    O(kv...) = Dict{String,Any}(kv...)
    idx(v, i...) = O("op" => "index", "args" => Any[v, i...])
    sq(x) = O("op" => "^", "args" => Any[x, 2])
    sdef(; kw...) = O("op" => "faq", "args" => Any[], "output_idx" => Any["i"],
        "ranges" => O("i" => Any[1, 2], "k" => Any[1, 3]),
        "reduce" => "+", "expr" => sq(idx("u", "i", "k")), (String(k) => v for (k, v) in kw)...)
    doc(rhs_body; oidx = ["i", "k"], s = sdef(), vars = O(), eqs = Any[]) = O("esm" => "1.1.0",
        "metadata" => O("name" => "ContractObs"),
        "index_sets" => O("x" => O("kind" => "interval", "size" => 2),
                          "z" => O("kind" => "interval", "size" => 3)),
        "models" => O("ContractObs" => O(
            "variables" => O(
                "u" => O("type" => "unknown", "shape" => Any["x", "z"], "default" => 0.5),
                "s" => O("type" => "unknown", "shape" => Any["x"]), vars...),
            "equations" => Any[
                O("lhs" => "s", "rhs" => s), eqs...,
                O("lhs" => O("op" => "D", "args" => Any["u"], "wrt" => "t"),
                  "rhs" => O("op" => "faq", "args" => Any[], "output_idx" => Any[oidx...],
                             "ranges" => O(oidx[1] => Any[1, 2], oidx[2] => Any[1, 3]),
                             "expr" => rhs_body(oidx)))])))
    neg(x) = O("op" => "-", "args" => Any[x])
    si(o) = idx("s", o[1])
    bodies = [
        "-s[i]"         => o -> neg(si(o)),
        "-s[i]^2"       => o -> neg(sq(si(o))),                 # chain rule through Σ
        "-s[i]*u[i,k]"  => o -> neg(O("op" => "*", "args" => Any[si(o), idx("u", o...)])),
    ]

    mul(a...) = O("op" => "*", "args" => Any[a...])
    ksum(body) = O("op" => "faq", "args" => Any[], "output_idx" => Any[],
                   "ranges" => O("k" => Any[1, 3]), "expr" => body)
    cvar = O("c" => O("type" => "unknown"))
    ceq(rhs) = Any[O("lhs" => "c", "rhs" => rhs)]
    # (label, rhs body, extra variables, extra equations)
    more = [
        # a scalar sum observed (a bare read, no subscripts)
        ("-c*u[i,k], c = Σ_k u[1,k]", o -> neg(mul("c", idx("u", o...))), cvar,
         ceq(ksum(idx("u", 1, "k")))),
        # an observed whose sum is NOT its root: spliced as is, then renamed
        # apart from the reading faq's k
        ("-c*u[i,k], c = 2·Σ_k u[2,k]²", o -> neg(mul("c", idx("u", o...))), cvar,
         ceq(mul(2, ksum(sq(idx("u", 2, "k")))))),
        # a hand-written sum shadowing the reading faq's k, with the same
        # site u[i,k] both outside and inside it
        ("-u[i,k]·Σ_k u[i,k]²", o -> neg(mul(idx("u", o...), ksum(sq(idx("u", "i", "k"))))),
         O(), Any[]),
    ]

    mktempdir() do d
        load(dd) = (path = joinpath(d, "contract_obs.esm");
                    open(io -> JSON3.write(io, dd), path, "w");
                    EarthSciAST.load_path(path))
        u = collect(0.1:0.1:0.6)
        for (label, body, vars, eqs) in more
            @testset "$label" begin
                file = load(doc(body; vars = vars, eqs = eqs))
                f!, _, p, _, _ = build_evaluator(file)
                Jad = ad_jacobian(f!, u, p, 0.0)
                res = assemble_jacobian(file; u = u, p = p,
                                        build_kwargs = (compiler = :interpreter,))
                @test maximum(abs.(Matrix(res.J) .- Matrix(Jad))) ≤ 1e-13
                @test res.pattern == (Jad .!= 0)
            end
        end

        # Two reads of one contracting observed inline to the SAME tree, so
        # interning / template CSE / hoisting still share the sum.
        file = load(doc(o -> O("op" => "-", "args" => Any[neg(si(o)), sq(si(o))])))
        sv = EarthSciASTDiff.sysview(file, "ContractObs")
        obs = EarthSciASTDiff._observed_defs(sv)
        eq = only(e for e in sv.equations if EarthSciASTDiff.lhs_state(e) !== nothing)
        sums = OpExpr[]
        walk(x) = (x isa OpExpr && (x.op == "faq" && isempty(something(x.output_idx, Any[])) &&
                                    push!(sums, x);
                                    EarthSciAST.map_children(y -> (walk(y); y), x)); x)
        walk(EarthSciASTDiff.inline_observed(eq.rhs, obs))
        @test length(sums) == 2
        ix = EarthSciASTDiff.StructIndex()
        @test EarthSciASTDiff.sid(ix, sums[1]) == EarthSciASTDiff.sid(ix, sums[2])
        @test !haskey(sums[1].ranges, "k")          # renamed apart from the reader's k

        # An index named like an observed: an index the enclosing faq binds
        # is that index, not a read of the observed. (Upstream's evaluator
        # reads it as the observed, so there is no oracle; check the inliner.)
        obsk = copy(obs)
        obsk["k"] = NumExpr(7.0)
        @test EarthSciAST.to_ascii(EarthSciASTDiff.inline_observed(eq.rhs, obsk)) ==
              EarthSciAST.to_ascii(EarthSciASTDiff.inline_observed(eq.rhs, obs))

        # A scalar sum observed built with `output_idx` absent (the schema
        # requires it on the wire; the typed AST allows `nothing`) inlines
        # like one with an empty list.
        file = load(doc(o -> neg(mul("c", idx("u", o...))); vars = cvar,
                        eqs = ceq(ksum(idx("u", 1, "k")))))
        sv = EarthSciASTDiff.sysview(file, "ContractObs")
        obs = EarthSciASTDiff._observed_defs(sv)
        eq = only(e for e in sv.equations if EarthSciASTDiff.lhs_state(e) !== nothing)
        obs2 = copy(obs)
        obs2["c"] = EarthSciAST.reconstruct(obs["c"]; output_idx = nothing)
        a = EarthSciASTDiff.inline_observed(eq.rhs, obs)
        b = EarthSciASTDiff.inline_observed(eq.rhs, obs2)
        @test EarthSciAST.to_ascii(a) == EarthSciAST.to_ascii(b)

        # oidx [i, k]: the reading faq binds the contracted name (the capture);
        # [i, j]: it does not (the leftover index was free).
        for oidx in (["i", "k"], ["i", "j"]), (label, body) in bodies
            @testset "$label over $(join(oidx, ","))" begin
                file = load(doc(body; oidx = oidx))
                f!, _, p, _, _ = build_evaluator(file)
                Jad = ad_jacobian(f!, u, p, 0.0)
                # The derived model reads the sum per cell, which the native
                # compiler declines (`compiler_refused_rule`); the interpreter
                # evaluates every cell.
                res = assemble_jacobian(file; u = u, p = p,
                                        build_kwargs = (compiler = :interpreter,))
                @test maximum(abs.(Matrix(res.J) .- Matrix(Jad))) ≤ 1e-13
                @test res.pattern == (Jad .!= 0)
            end
        end

        # A filter on the contracting observed is refused, as it is inline
        # (bands.jl `_bands_faq!`) — never silently dropped.
        filt = O("op" => "<", "args" => Any["k", 3])
        file = load(doc(o -> neg(si(o)); s = sdef(filter = filt)))
        @test_throws Union{BandError,DerivativeRuleError} jacobian_bands(file, "ContractObs")

        # So is a max: `semiring` supersedes `reduce`, and a `max_product`
        # node with no `reduce` is not a sum.
        mx = sdef(); delete!(mx, "reduce"); mx["semiring"] = "max_product"
        file = load(doc(o -> neg(si(o)); s = mx))
        @test_throws Union{BandError,DerivativeRuleError} jacobian_bands(file, "ContractObs")
    end

    # A range over an index set the context does not declare is a BandError,
    # not a raw KeyError from inside the band calculus.
    ctx = EarthSciASTDiff.Ctx(Dict("x" => 2), Dict{String,Vector{Int}}(), Set{String}())
    @test_throws BandError EarthSciASTDiff._range_of(Dict("from" => "nope"), ctx)
end

@testset "contracting observed: shared fixture matches the AD oracle" begin
    # tests/valid/contracting_observed.esm — the cross-binding golden for the
    # nested-sum rule (spec §3.5).
    file = fixture("contracting_observed.esm")
    f!, u0, p, _, _ = build_evaluator(file)
    u = [0.2 + 0.6 * abs(sin(2.3 * i)) for i in 1:length(u0)]
    Jad = ad_jacobian(f!, u, p, 0.7)
    res = assemble_jacobian(file; u = u, t = 0.7,
                            build_kwargs = (compiler = :interpreter,))
    @test maximum(abs.(Matrix(res.J) .- Matrix(Jad))) ≤ 1e-13
    @test res.pattern == (Jad .!= 0)
end
