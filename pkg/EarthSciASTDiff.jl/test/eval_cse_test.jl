# Evaluation-model CSE (eval_cse.jl): repeated coefficient subtrees hoist
# into observed array variables of the derived evaluation document, so the
# tree-walk's factored array-observed buffers evaluate them once per cell
# per RHS call instead of once per occurrence.

using EarthSciASTDiff: Band, hoist_observed, _evaluation_document, skey

# A ≥ 12-node subtree in the row index `n`: sin(u[n]) + cos(u[n+1]) * exp(u[n])
_bigfrag(n) = begin
    rd(o) = OpExpr("index", ASTExpr[VarExpr("u"),
        o == 0 ? VarExpr(n) : OpExpr("+", ASTExpr[VarExpr(n), IntExpr(o)])])
    OpExpr("+", ASTExpr[OpExpr("sin", ASTExpr[rd(0)]),
        OpExpr("*", ASTExpr[OpExpr("cos", ASTExpr[rd(1)]),
                            OpExpr("exp", ASTExpr[rd(0)])])])
end

_sv() = EarthSciASTDiff.SysView(Dict{String,Any}(), [], Dict{String,Any}())

@testset "hoist_observed" begin
    # the fragment repeats across two bands → one observed over the union box
    frag = _bigfrag("i")
    mk(coef, lo, hi) = JacEntry("u", "u",
        Band([(lo, hi)], Union{String,Int}["i"], ASTExpr[VarExpr("i")], coef))
    e1 = mk(OpExpr("*", ASTExpr[frag, NumExpr(2.0)]), 2, 8)
    e2 = mk(OpExpr("+", ASTExpr[frag, NumExpr(1.5)]), 1, 7)
    h = hoist_observed([e1, e2], _sv(); min_nodes = 8)
    @test h !== nothing && length(h.obs) == 1
    nm = first(keys(h.obs))
    @test startswith(nm, "Jcse")
    vd = h.obs[nm]
    @test vd["type"] == "unknown" && length(vd["shape"]) == 1
    @test collect(values(h.axes)) == [8]              # extent = max hi over boxes
    # esm 1.0.0: the hoisted definition comes back in `obs_eqs`, not in the
    # variable -- `variable.expression` no longer exists.
    @test length(h.obs_eqs) == 1 && first(h.obs_eqs).first == nm
    regs = first(h.obs_eqs).second["regions"]
    @test regs[1] == Any[Any[1, 8]]                   # zero default over the axis
    @test Any[Any[2, 8]] in regs && Any[Any[1, 7]] in regs
    # both rewritten coefficients read the observed at the row index
    for c in h.coefs
        @test occursin("\"$nm\"", skey(c))
        @test !occursin("\"sin\"", skey(c))
    end
    # nothing to hoist → nothing (fragments below threshold)
    @test hoist_observed([mk(VarExpr("a"), 1, 8), mk(VarExpr("a"), 1, 8)],
                         _sv(); min_nodes = 8) === nothing

    # a scalar fragment (no free row-index names) hoists to a scalar observed
    sfrag = OpExpr("+", ASTExpr[OpExpr("sin", ASTExpr[VarExpr("c")]),
        OpExpr("*", ASTExpr[OpExpr("cos", ASTExpr[VarExpr("c")]),
                            OpExpr("exp", ASTExpr[VarExpr("d")])])])
    s1 = mk(OpExpr("*", ASTExpr[sfrag, VarExpr("i")]), 1, 8)
    s2 = mk(OpExpr("+", ASTExpr[sfrag, VarExpr("i")]), 1, 8)
    hs = hoist_observed([s1, s2], _sv(); min_nodes = 6)
    @test hs !== nothing && length(hs.obs) == 1
    svd = first(values(hs.obs))
    @test !haskey(svd, "shape")

    # an axis name colliding with a using band's missing binder bails out:
    # the fragment is free in "i", but the second band binds only "j"
    t1 = mk(OpExpr("*", ASTExpr[frag, NumExpr(2.0)]), 1, 8)
    t2 = JacEntry("w", "u", Band([(1, 8)], Union{String,Int}["j"],
        ASTExpr[VarExpr("j")], OpExpr("+", ASTExpr[frag, NumExpr(1.5)])))
    @test hoist_observed([t1, t2], _sv(); min_nodes = 8) === nothing
end

@testset "evaluation document carries the hoisted observeds" begin
    file = fixture("minmod_adv.esm")
    mname = first(keys(file.models))
    sv = sysview(file, mname)
    entries = jacobian_bands(file, mname)
    doc, _ = _evaluation_document(sv, entries)
    vars = doc["models"]["JacobianEval"]["variables"]
    hoisted = [k for k in keys(vars) if startswith(String(k), "Jcse")]
    @test !isempty(hoisted)
    doc0, _ = _evaluation_document(sv, entries; cse = false)
    @test !any(startswith(String(k), "Jcse")
               for k in keys(doc0["models"]["JacobianEval"]["variables"]))
    # the CSE'd document is materially smaller
    @test length(EarthSciASTDiff.stable_json(doc)) <
          length(EarthSciASTDiff.stable_json(doc0))
end

@testset "minmod: eval-CSE Jacobian matches the plain path and AD" begin
    file = fixture("minmod_adv.esm")
    mname = first(keys(file.models))
    f!, u0, p0, _, _ = build_evaluator(file)
    u = 0.2 .+ 0.6 .* abs.(sin.((1:length(u0)) .* 2.3))
    Jad = ad_jacobian(f!, u, p0, 0.0)
    jc = prepare_jacobian(file; model_name = mname)                    # default: CSE on
    jp = prepare_jacobian(file; model_name = mname, eval_cse = false)
    Jc = copy(jc.prototype); jc(Jc, u, p0, 0.0)
    Jp = copy(jp.prototype); jp(Jp, u, p0, 0.0)
    @test Jc.colptr == Jp.colptr && Jc.rowval == Jp.rowval
    @test Matrix(Jc) ≈ Matrix(Jp) rtol = 1e-14
    @test Matrix(Jc) ≈ Matrix(Jad) rtol = 1e-10
end
