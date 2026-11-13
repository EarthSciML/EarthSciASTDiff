# Per-op derivative rules checked against a central finite difference of
# `EarthSciAST.evaluate_expr` — the same evaluator the tree-walk uses, so the
# rules and the runtime can never drift apart. Test points avoid kinks; the
# kink policy itself (which branch at a tie) is pinned separately.

ex(d) = EarthSciAST.expression_from_json(d)

function numderiv(expr, x; h = 1.0e-6)
    f(v) = evaluate_expr(expr, Dict("x" => v))
    return (f(x + h) - f(x - h)) / 2h
end

function symderiv(expr, x)
    dexpr = dscalar(expr, Site(VarExpr("x")))
    return evaluate_expr(dexpr, Dict("x" => x))
end

@testset "scalar rules vs finite differences" begin
    cases = [
        (Dict("op" => "+", "args" => ["x", "x", 2.0]), 0.7),
        (Dict("op" => "-", "args" => [3.0, "x"]), 0.7),
        (Dict("op" => "*", "args" => ["x", "x", "x"]), 0.7),
        (Dict("op" => "/", "args" => [Dict("op" => "+", "args" => ["x", 1.0]), "x"]), 0.7),
        (Dict("op" => "^", "args" => ["x", 3]), 0.7),
        (Dict("op" => "^", "args" => [2.0, "x"]), 0.7),
        (Dict("op" => "exp", "args" => ["x"]), 0.7),
        (Dict("op" => "log", "args" => ["x"]), 0.7),
        (Dict("op" => "log10", "args" => ["x"]), 0.7),
        (Dict("op" => "sqrt", "args" => ["x"]), 0.7),
        (Dict("op" => "sin", "args" => ["x"]), 0.7),
        (Dict("op" => "cos", "args" => ["x"]), 0.7),
        (Dict("op" => "tan", "args" => ["x"]), 0.7),
        (Dict("op" => "tanh", "args" => ["x"]), 0.7),
        (Dict("op" => "sinh", "args" => ["x"]), 0.7),
        (Dict("op" => "cosh", "args" => ["x"]), 0.7),
        (Dict("op" => "asin", "args" => ["x"]), 0.4),
        (Dict("op" => "acos", "args" => ["x"]), 0.4),
        (Dict("op" => "atan", "args" => ["x"]), 0.7),
        (Dict("op" => "atan", "args" => ["x", 2.0]), 0.7),
        (Dict("op" => "abs", "args" => ["x"]), -0.7),
        (
            Dict(
                "op" => "ifelse",
                "args" => [
                    Dict("op" => ">", "args" => ["x", 0.5]),
                    Dict("op" => "*", "args" => ["x", "x"]),
                    Dict("op" => "*", "args" => [3.0, "x"]),
                ]
            ), 0.7,
        ),
        (
            Dict(
                "op" => "ifelse",
                "args" => [
                    Dict("op" => ">", "args" => ["x", 0.5]),
                    Dict("op" => "*", "args" => ["x", "x"]),
                    Dict("op" => "*", "args" => [3.0, "x"]),
                ]
            ), 0.2,
        ),
        (
            Dict(
                "op" => "max", "args" => [
                    "x", 0.3,
                    Dict("op" => "*", "args" => [0.5, "x"]),
                ]
            ), 0.7,
        ),
        (Dict("op" => "min", "args" => [Dict("op" => "*", "args" => ["x", "x"]), "x"]), 0.7),
        # `Pre` is the identity in a right-hand side (spec §3.2)
        (Dict("op" => "*", "args" => [Dict("op" => "Pre", "args" => ["x"]), "x"]), 0.7),
        (Dict("op" => "exp", "args" => [Dict("op" => "Pre", "args" => ["x"])]), 0.7),
    ]
    for (d, x) in cases
        expr = ex(d)
        @test symderiv(expr, x) ≈ numderiv(expr, x) rtol = 1.0e-5 atol = 1.0e-8
    end
end

@testset "structural zeros and policies" begin
    s = Site(VarExpr("x"))
    # A subtree without the site differentiates to a literal zero.
    @test EarthSciASTDiff.iszero_lit(dscalar(ex(Dict("op" => "*", "args" => ["y", "z"])), s))
    # Comparisons appearing multiplicatively carry no derivative of their own.
    e = ex(Dict("op" => "*", "args" => [Dict("op" => ">", "args" => ["x", 0.0]), 2.0]))
    @test evaluate_expr(dscalar(e, s), Dict("x" => 1.0)) == 0.0
    # max tie resolves to the FIRST argument's derivative (documented policy).
    e = ex(Dict("op" => "max", "args" => ["x", "x"]))
    @test evaluate_expr(dscalar(e, s), Dict("x" => 0.7)) == 1.0
    # `index` of a literal array is piecewise-constant in its integer
    # subscripts: derivative 0 (spec §3.1).
    tbl = ex(Dict("op" => "const", "args" => Any[], "value" => [1.0, 3.0, 2.0]))
    fx = OpExpr("floor", ASTExpr[VarExpr("x")])
    @test EarthSciASTDiff.iszero_lit(dscalar(OpExpr("index", ASTExpr[tbl, fx]), s))
    # ... but a site-dependent read of a variable has no rule (another
    # variable may depend on the site: a recurrence array is never inlined),
    # nor a read of an array expression that reads variables
    @test_throws EarthSciASTDiff.DerivativeRuleError dscalar(
        OpExpr("index", ASTExpr[VarExpr("q"), fx]), s
    )
    ui = Site(OpExpr("index", ASTExpr[VarExpr("u"), IntExpr(1)]))
    @test_throws EarthSciASTDiff.DerivativeRuleError dscalar(
        OpExpr("index", ASTExpr[VarExpr("u"), OpExpr("floor", ASTExpr[ui.expr])]), ui
    )
    @test_throws EarthSciASTDiff.DerivativeRuleError dscalar(
        OpExpr("index", ASTExpr[OpExpr("+", ASTExpr[tbl, VarExpr("y")]), fx]), s
    )
    # `Pre(x)` differentiates as `x` (spec §3.2): it is the identity in a
    # right-hand side, as EarthSciAST evaluates it, not a constant.
    @test dscalar(ex(Dict("op" => "Pre", "args" => ["x"])), s) == IntExpr(1)
    # Rewrite-target ops must be lowered first.
    @test_throws EarthSciASTDiff.DerivativeRuleError dscalar(
        OpExpr("D", ASTExpr[VarExpr("x")]; wrt = "lon"), s
    )
end

# The esm:std templates (esm 2.0.0) that replaced the `fn` closed functions
# differentiate through their expansion by the core rules plus the `index`
# rule above: a 2.0.0 document of one model `M` whose state `w` (and `v`)
# reads `rhs`, checked against the AD oracle.
function _std_doc(rhs, imports; v = false)
    vars = Dict{String, Any}("w" => Dict{String, Any}("type" => "unknown", "default" => 0.4))
    eqs = Any[
        Dict{String, Any}(
            "lhs" => Dict{String, Any}("op" => "D", "args" => Any["w"], "wrt" => "t"),
            "rhs" => rhs
        ),
    ]
    if v
        vars["v"] = Dict{String, Any}("type" => "unknown", "default" => 7.0)
        push!(
            eqs, Dict{String, Any}(
                "lhs" => Dict{String, Any}("op" => "D", "args" => Any["v"], "wrt" => "t"),
                "rhs" => 0.0
            )
        )
    end
    return EarthSciAST.load_document(
        Dict{String, Any}(
            "esm" => "2.0.0",
            "metadata" => Dict{String, Any}("name" => "StdTemplates"),
            "models" => Dict{String, Any}(
                "M" => Dict{String, Any}(
                    "expression_template_imports" => Any[
                        Dict{String, Any}("ref" => "esm:std/$f.esm", "only" => Any[n])
                            for (f, n) in imports
                    ],
                    "variables" => vars, "equations" => eqs
                )
            )
        )
    )
end
_cnst(v) = Dict{String, Any}("op" => "const", "args" => Any[], "value" => v)
_apply(name, b) = Dict{String, Any}(
    "op" => "apply_expression_template", "args" => Any[], "name" => name, "bindings" => b
)

@testset "esm:std bilinear template: both partials" begin
    file = _std_doc(
        _apply(
            "bilinear", Dict{String, Any}(
                "table" => _cnst([[1.0, 2.0, 4.0], [3.0, 5.0, 9.0], [4.0, 7.0, 13.0]]),
                "axis_x" => _cnst([0.0, 1.0, 2.0]), "axis_y" => _cnst([0.0, 10.0, 30.0]),
                "x" => "w", "y" => "v", "nx" => 3, "ny" => 3
            )
        ), (("interp", "bilinear"),); v = true
    )
    f!, u0, p, _, _ = build_evaluator(file)
    # every cell, each clamp region and the corners: both states feed the
    # same lookup, so the Jacobian row carries BOTH partials
    for u in (
            [0.4, 7.0], [1.5, 7.0], [0.4, 21.0], [1.5, 21.0], [-1.0, 12.0],
            [3.0, 12.0], [0.7, -5.0], [0.7, 45.0], [-1.0, -5.0], [3.0, 45.0],
        )
        res = doc_jacobian(file; u = u, p = p)
        @test Matrix(res.J) ≈ Matrix(ad_jacobian(f!, u, p, 0.0)) atol = 1.0e-12
    end
end

@testset "esm:std linear template: segment slope" begin
    file = _std_doc(
        _apply(
            "linear", Dict{String, Any}(
                "table" => _cnst([0.0, 10.0, 12.0]), "axis" => _cnst([0.0, 1.0, 2.0]),
                "x" => "w", "n" => 3
            )
        ), (("interp", "linear"),)
    )
    f!, u0, p, _, _ = build_evaluator(file)
    # inside each segment, and flat outside the axis
    for (x, slope) in ((0.4, 10.0), (1.6, 2.0), (2.5, 0.0), (-0.5, 0.0))
        res = doc_jacobian(file; u = [x], p = p)
        @test Matrix(res.J) ≈ [slope;;] atol = 1.0e-12
        @test Matrix(res.J) ≈ Matrix(ad_jacobian(f!, [x], p, 0.0)) atol = 1.0e-12
    end
end

@testset "esm:std calendar templates: hour is flat, julian_day is not" begin
    # hour(w) is piecewise-constant: no entry; julian_day(3w) has slope
    # 3/86400 (it carries the fractional time of day)
    hour = _std_doc(
        Dict{String, Any}(
            "op" => "*", "args" => Any[_apply("hour", Dict{String, Any}("t" => "w")), 2.0]
        ), (("calendar", "hour"),)
    )
    @test isempty(jacobian_bands(hour, "M"))
    jd = _std_doc(
        _apply(
            "julian_day", Dict{String, Any}(
                "t" => Dict{String, Any}("op" => "*", "args" => Any[3, "w"])
            )
        ), (("calendar", "julian_day"),)
    )
    _, _, p, _, _ = build_evaluator(jd)
    for x in (0.4, 1.0e9 + 0.3)
        @test Matrix(doc_jacobian(jd; u = [x], p = p).J) ≈ [3 / 86400;;] rtol = 1.0e-12
    end
end
