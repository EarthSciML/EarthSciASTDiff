# Arrayed observed (system.jl `_observed_defs`): an observed written the
# indexed way — `y[i] ~ f(i)`, spelled `faq{expr: index(y, i)} ~ f(i)` — is an
# observed, not a solver slot. Missed, it is a SILENT failure twice over:
# `view_states` counts `y` as a spurious column target, and `y` is never
# inlined, so the chain rule through it is lost.
#
#   y[i] = k·u[i]²,  D(u)[i] = -y[i]   ⇒   ∂/∂u[i] = -2k·u[i]
#
# The bindings carry the same cases (test_observed.py, tests/observed.rs).

using JSON3
using EarthSciASTDiff: _observed_defs, view_states, sysview, BandError

@testset "arrayed observed: indexed definition is inlined" begin
    K = 3.0
    O(kv...) = Dict{String,Any}(kv...)
    _idx(v, i) = O("op" => "index", "args" => Any[v, i])
    _faq(body) = O("op" => "faq", "args" => Any[], "output_idx" => Any["i"],
                   "ranges" => O("i" => Any[1, 4]), "expr" => body)
    doc(ylhs) = O("esm" => "1.1.0",
        "metadata" => O("name" => "ArrObs"),
        "index_sets" => O("x" => O("kind" => "interval", "size" => 4)),
        "models" => O("ArrObs" => O(
            "variables" => O(
                "u" => O("type" => "unknown", "shape" => Any["x"], "default" => 0.5),
                "y" => O("type" => "unknown", "shape" => Any["x"]),
                "k" => O("type" => "parameter", "default" => K)),
            "equations" => Any[
                O("lhs" => ylhs,
                  "rhs" => O("op" => "*", "args" => Any["k",
                      O("op" => "^", "args" => Any[_idx("u", "i"), 2])])),
                O("lhs" => O("op" => "D", "args" => Any["u"], "wrt" => "t"),
                  "rhs" => _faq(O("op" => "-", "args" => Any[_idx("y", "i")])))])))

    mktempdir() do d
        load(dd) = (path = joinpath(d, "arr_obs.esm");
                    open(io -> JSON3.write(io, dd), path, "w");
                    EarthSciAST.load_path(path))

        file = load(doc(_faq(_idx("y", "i"))))
        sv = sysview(file, "ArrObs")
        obs = _observed_defs(sv)
        @test haskey(obs, "y")
        @test view_states(sv, obs) == ["u"]        # y is not a solver slot
        entries = jacobian_bands(file, "ArrObs")
        @test !isempty(entries) && all(e -> e.v == "u", entries)

        f!, _, p, _, _ = build_evaluator(file)
        u = [0.3, 0.7, 1.1, 1.9]
        res = assemble_jacobian(file; u = u, p = p)
        @test maximum(abs.(Matrix(res.J) .- Matrix(ad_jacobian(f!, u, p, 0.0)))) ≤ 1e-14
        @test Matrix(res.J) ≈ Diagonal(-2K .* u) atol = 1e-14

        # Pinned (`y[1] ~ …`): an observed, but with no whole-array form —
        # loud, not a silent state slot.
        pinned = load(doc(_idx("y", 1)))
        @test_throws BandError _observed_defs(sysview(pinned, "ArrObs"))
    end
end
