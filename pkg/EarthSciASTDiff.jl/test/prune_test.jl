# Inline pruning (system.jl `inlinable_observed`): an observed that cannot move
# with the differentiation target is NOT inlined — it stays an opaque read that
# the derived evaluation model resolves from its own definition.
#
# The hazard this file exists to guard is a SILENT ZERO: pruning is exactly the
# kind of optimisation that produces a plausible, fast, wrong Jacobian. So every
# assertion here is numeric (ForwardDiff through the same tree-walk RHS), and
# the `:parameters` axis — where the very same observed is target-DEPENDENT and
# must survive — is checked alongside the `:states` axis on one fixture.
#
# `src = 2k` depends on the parameter `k` and on no state, and it appears
#   * as a multiplicative COEFFICIENT of a state (`-(src·u[i])`), so the state
#     band's coefficient still names the pruned observed and the derived model
#     has to evaluate it — a wrong value here is a wrong Jacobian, not an error;
#   * as an additive term, so `∂/∂k` is non-zero even where `∂/∂u` is not.

using JSON3
using EarthSciASTDiff: inlinable_observed, _observed_defs, sysview

@testset "inline pruning: target-independent observed" begin
    K = 3.0
    O(kv...) = Dict{String,Any}(kv...)
    _idx(v, i) = O("op" => "index", "args" => Any[v, i])
    # D(u)[i] = -(src * u[i]) + src, with `src = 2k` an observed that depends on
    # the PARAMETER k and on no state.
    body = O("op" => "+", "args" => Any[
        O("op" => "-", "args" => Any[O("op" => "*", "args" => Any["src", _idx("u", "i")])]),
        "src"])
    rhs = O("op" => "aggregate", "args" => Any[], "output_idx" => Any["i"],
            "ranges" => O("i" => Any[1, 4]), "expr" => body)
    doc = O("esm" => "0.8.0",
            "metadata" => O("name" => "PruneProbe"),
            "index_sets" => O("x" => O("kind" => "interval", "size" => 4)),
            "models" => O("PruneProbe" => O(
                "variables" => O(
                    "u" => O("type" => "state", "shape" => Any["x"], "default" => 0.5),
                    "k" => O("type" => "parameter", "default" => K),
                    "src" => O("type" => "observed",
                               "expression" => O("op" => "*", "args" => Any["k", 2]))),
                "equations" => Any[
                    O("lhs" => O("op" => "D", "args" => Any["u"], "wrt" => "t"),
                      "rhs" => rhs)])))

    mktempdir() do d
        path = joinpath(d, "prune_probe.esm")
        open(io -> JSON3.write(io, doc), path, "w")
        file = EarthSciAST.load(path)
        sv = sysview(file, "PruneProbe")
        obs = _observed_defs(sv)
        @test haskey(obs, "src")

        # ---- the pruning decision itself, per axis --------------------------
        keep_u = inlinable_observed(obs, sv.equations, Set{String}(["u"]))
        keep_k = inlinable_observed(obs, sv.equations, Set{String}(["k"]))
        @test !haskey(keep_u, "src")        # ∂src/∂u ≡ 0 → never inlined
        @test haskey(keep_k, "src")         # ∂src/∂k ≠ 0 → MUST be inlined

        # ---- and the numbers, against the AD oracle ------------------------
        f!, u0, p, _, _ = build_evaluator(file)
        u = [0.3, 0.7, 1.1, 1.9]

        res = assemble_jacobian(file; u = u, p = p)
        Jad = ad_jacobian(f!, u, p, 0.0)
        @test maximum(abs.(Matrix(res.J) .- Matrix(Jad))) ≤ 1e-14
        # The pruned observed really is in the coefficient: ∂/∂u[i] = -2k, so a
        # coefficient that silently dropped it would read -1, and one that
        # zeroed it would read 0.
        @test all(diag(Matrix(res.J)) .≈ -2K)

        resp = assemble_jacobian(file; wrt = :parameters, u = u, p = p)
        pv = collect(Float64, p)
        Jp_ad = ForwardDiff.jacobian(pv) do q
            pp = NamedTuple{keys(p)}(Tuple(q))
            du = similar(q, length(u)); f!(du, u, pp, 0.0); du
        end
        @test maximum(abs.(Matrix(resp.J) .- Jp_ad)) ≤ 1e-12
        # THE anti-silent-zero assertion: ∂f/∂k = 2 − 2u[i] flows THROUGH the
        # observed the state axis prunes. A parameter-axis prune would make the
        # whole column zero and every other check here would still pass.
        kcol = findfirst(==("k"), resp.colnames)
        @test kcol !== nothing
        @test Matrix(resp.J)[:, kcol] ≈ 2 .- 2 .* u
        @test !all(iszero, Matrix(resp.J)[:, kcol])
    end
end
