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
    doc(rhs_body; oidx = ["i", "k"], s = sdef()) = O("esm" => "1.1.0",
        "metadata" => O("name" => "ContractObs"),
        "index_sets" => O("x" => O("kind" => "interval", "size" => 2),
                          "z" => O("kind" => "interval", "size" => 3)),
        "models" => O("ContractObs" => O(
            "variables" => O(
                "u" => O("type" => "unknown", "shape" => Any["x", "z"], "default" => 0.5),
                "s" => O("type" => "unknown", "shape" => Any["x"])),
            "equations" => Any[
                O("lhs" => "s", "rhs" => s),
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

    mktempdir() do d
        load(dd) = (path = joinpath(d, "contract_obs.esm");
                    open(io -> JSON3.write(io, dd), path, "w");
                    EarthSciAST.load_path(path))
        u = collect(0.1:0.1:0.6)
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
    end
end
