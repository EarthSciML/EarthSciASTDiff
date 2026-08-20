# Per-op derivative rules checked against a central finite difference of
# `EarthSciAST.evaluate_expr` — the same evaluator the tree-walk uses, so the
# rules and the runtime can never drift apart. Test points avoid kinks; the
# kink policy itself (which branch at a tie) is pinned separately.

ex(d) = EarthSciAST.parse_expression(d)

function numderiv(expr, x; h = 1e-6)
    f(v) = evaluate_expr(expr, Dict("x" => v))
    (f(x + h) - f(x - h)) / 2h
end

function symderiv(expr, x)
    dexpr = dscalar(expr, Site(VarExpr("x")))
    evaluate_expr(dexpr, Dict("x" => x))
end

@testset "scalar rules vs finite differences" begin
    cases = [
        (Dict("op"=>"+", "args"=>["x", "x", 2.0]),                    0.7),
        (Dict("op"=>"-", "args"=>[3.0, "x"]),                         0.7),
        (Dict("op"=>"*", "args"=>["x", "x", "x"]),                    0.7),
        (Dict("op"=>"/", "args"=>[Dict("op"=>"+","args"=>["x",1.0]), "x"]), 0.7),
        (Dict("op"=>"^", "args"=>["x", 3]),                           0.7),
        (Dict("op"=>"^", "args"=>[2.0, "x"]),                         0.7),
        (Dict("op"=>"exp", "args"=>["x"]),                            0.7),
        (Dict("op"=>"log", "args"=>["x"]),                            0.7),
        (Dict("op"=>"log10", "args"=>["x"]),                          0.7),
        (Dict("op"=>"sqrt", "args"=>["x"]),                           0.7),
        (Dict("op"=>"sin", "args"=>["x"]),                            0.7),
        (Dict("op"=>"cos", "args"=>["x"]),                            0.7),
        (Dict("op"=>"tan", "args"=>["x"]),                            0.7),
        (Dict("op"=>"tanh", "args"=>["x"]),                           0.7),
        (Dict("op"=>"sinh", "args"=>["x"]),                           0.7),
        (Dict("op"=>"cosh", "args"=>["x"]),                           0.7),
        (Dict("op"=>"asin", "args"=>["x"]),                           0.4),
        (Dict("op"=>"acos", "args"=>["x"]),                           0.4),
        (Dict("op"=>"atan", "args"=>["x"]),                           0.7),
        (Dict("op"=>"atan", "args"=>["x", 2.0]),                      0.7),
        (Dict("op"=>"abs", "args"=>["x"]),                           -0.7),
        (Dict("op"=>"ifelse",
              "args"=>[Dict("op"=>">","args"=>["x",0.5]),
                       Dict("op"=>"*","args"=>["x","x"]),
                       Dict("op"=>"*","args"=>[3.0,"x"])]),           0.7),
        (Dict("op"=>"ifelse",
              "args"=>[Dict("op"=>">","args"=>["x",0.5]),
                       Dict("op"=>"*","args"=>["x","x"]),
                       Dict("op"=>"*","args"=>[3.0,"x"])]),           0.2),
        (Dict("op"=>"max", "args"=>["x", 0.3,
              Dict("op"=>"*","args"=>[0.5,"x"])]),                    0.7),
        (Dict("op"=>"min", "args"=>[Dict("op"=>"*","args"=>["x","x"]), "x"]), 0.7),
    ]
    for (d, x) in cases
        expr = ex(d)
        @test symderiv(expr, x) ≈ numderiv(expr, x) rtol = 1e-5 atol = 1e-8
    end
end

@testset "structural zeros and policies" begin
    s = Site(VarExpr("x"))
    # A subtree without the site differentiates to a literal zero.
    @test EarthSciASTDiff.iszero_lit(dscalar(ex(Dict("op"=>"*","args"=>["y","z"])), s))
    # Comparisons appearing multiplicatively carry no derivative of their own.
    e = ex(Dict("op"=>"*","args"=>[Dict("op"=>">","args"=>["x",0.0]), 2.0]))
    @test evaluate_expr(dscalar(e, s), Dict("x"=>1.0)) == 0.0
    # max tie resolves to the FIRST argument's derivative (documented policy).
    e = ex(Dict("op"=>"max","args"=>["x", "x"]))
    @test evaluate_expr(dscalar(e, s), Dict("x"=>0.7)) == 1.0
    # datetime.* / searchsorted are piecewise-constant: derivative 0.
    e = OpExpr("fn", ASTExpr[VarExpr("x")]; name = "datetime.year")
    @test EarthSciASTDiff.iszero_lit(dscalar(e, s))
    # Rewrite-target ops must be lowered first.
    @test_throws EarthSciASTDiff.DerivativeRuleError dscalar(
        OpExpr("D", ASTExpr[VarExpr("x")]; wrt = "lon"), s)
end

@testset "interp.bilinear two-slope rule" begin
    # interp.bilinear(table, axis_x, axis_y, x, y) — row-major literal table,
    # per-axis clamp (spec §9.2). Both partials checked against a central
    # finite difference of the evaluator's own blend, inside every cell and
    # in every clamp region.
    tbl = ex(Dict("op"=>"const", "args"=>Any[],
                  "value"=>[[1.0, 2.0, 4.0], [3.0, 5.0, 9.0], [4.0, 7.0, 13.0]]))
    axx = ex(Dict("op"=>"const", "args"=>Any[], "value"=>[0.0, 1.0, 2.0]))
    axy = ex(Dict("op"=>"const", "args"=>Any[], "value"=>[0.0, 10.0, 30.0]))
    e = OpExpr("fn", ASTExpr[tbl, axx, axy, VarExpr("x"), VarExpr("y")];
               name = "interp.bilinear")
    dx = dscalar(e, Site(VarExpr("x")))
    dy = dscalar(e, Site(VarExpr("y")))
    f(x, y) = evaluate_expr(e, Dict("x" => x, "y" => y))
    h = 1e-5
    pts = [(0.4, 7.0), (1.5, 7.0), (0.4, 21.0), (1.5, 21.0),   # all four cells
           (-1.0, 12.0), (3.0, 12.0),                          # x clamped, y interior
           (0.7, -5.0), (0.7, 45.0),                           # y clamped, x interior
           (-1.0, -5.0), (3.0, 45.0)]                          # corners
    for (x, y) in pts
        @test evaluate_expr(dx, Dict("x" => x, "y" => y)) ≈
              (f(x + h, y) - f(x - h, y)) / 2h atol = 1e-8
        @test evaluate_expr(dy, Dict("x" => x, "y" => y)) ≈
              (f(x, y + h) - f(x, y - h)) / 2h atol = 1e-8
    end
    # knot convention (spec §9.2 step 2): a query on an interior knot selects
    # the cell whose weight is 0 — the RIGHT cell's slope on the knot's axis
    @test evaluate_expr(dx, Dict("x" => 1.0, "y" => 7.0)) ≈
          evaluate_expr(dx, Dict("x" => 1.3, "y" => 7.0))
    @test evaluate_expr(dy, Dict("x" => 0.4, "y" => 10.0)) ≈
          evaluate_expr(dy, Dict("x" => 0.4, "y" => 12.0))
    # y-query with no x-dependence contributes only the ∂/∂y term
    @test evaluate_expr(dy, Dict("x" => -9.0, "y" => 12.0)) ≈
          (f(0.0, 12.0 + h) - f(0.0, 12.0 - h)) / 2h atol = 1e-8
    # non-literal tables fail loudly (spec violation, never a silent zero)
    ev = OpExpr("fn", ASTExpr[VarExpr("tbl"), axx, axy, VarExpr("x"), VarExpr("y")];
                name = "interp.bilinear")
    @test_throws EarthSciASTDiff.DerivativeRuleError dscalar(ev, Site(VarExpr("x")))
    # End-to-end through a model, against the AD oracle: both states feed
    # the same lookup, so the Jacobian row carries BOTH partials.
    doc = Dict{String,Any}("esm" => "0.8.0",
        "metadata" => Dict{String,Any}("name" => "BilinearSlope"),
        "models" => Dict{String,Any}("M" => Dict{String,Any}(
            "variables" => Dict{String,Any}(
                "w" => Dict{String,Any}("type" => "state", "default" => 0.4),
                "v" => Dict{String,Any}("type" => "state", "default" => 7.0)),
            "equations" => Any[Dict{String,Any}(
                "lhs" => Dict{String,Any}("op" => "D", "args" => Any["w"], "wrt" => "t"),
                "rhs" => Dict{String,Any}("op" => "fn", "name" => "interp.bilinear",
                    "args" => Any[
                        Dict{String,Any}("op"=>"const","args"=>Any[],
                            "value"=>[[1.0,2.0,4.0],[3.0,5.0,9.0],[4.0,7.0,13.0]]),
                        Dict{String,Any}("op"=>"const","args"=>Any[],"value"=>[0.0,1.0,2.0]),
                        Dict{String,Any}("op"=>"const","args"=>Any[],"value"=>[0.0,10.0,30.0]),
                        "w", "v"]),
                ), Dict{String,Any}(
                "lhs" => Dict{String,Any}("op" => "D", "args" => Any["v"], "wrt" => "t"),
                "rhs" => 0.0)])))
    file = EarthSciAST.load(doc)
    f!, u0, p, _, _ = build_evaluator(file)
    for u in ([0.4, 7.0], [-1.0, 45.0])   # interior cell + double clamp
        res = assemble_jacobian(file; u = u, p = p)
        Jad = ad_jacobian(f!, u, p, 0.0)
        @test Matrix(res.J) ≈ Matrix(Jad) atol = 1e-12
    end
end

@testset "interp.linear slope rule" begin
    # interp.linear(table, axis, x) — TABLE first, AXIS second (spec §9.2),
    # extrapolate-flat outside the axis. Slopes precomputed into a segment
    # ifelse, so the derivative is directly evaluable.
    table = ex(Dict("op"=>"const", "args"=>Any[], "value"=>[0.0, 10.0, 12.0]))
    axis  = ex(Dict("op"=>"const", "args"=>Any[], "value"=>[0.0, 1.0, 2.0]))
    e = OpExpr("fn", ASTExpr[table, axis, VarExpr("x")]; name = "interp.linear")
    d = dscalar(e, Site(VarExpr("x")))
    @test evaluate_expr(d, Dict("x" => 0.4)) ≈ 10.0    # first segment
    @test evaluate_expr(d, Dict("x" => 1.6)) ≈ 2.0     # second segment
    @test evaluate_expr(d, Dict("x" => -0.5)) == 0.0   # flat extrapolation
    @test evaluate_expr(d, Dict("x" => 2.5)) == 0.0
    # Non-literal tables are a spec violation: fail loudly, never guess.
    ev = OpExpr("fn", ASTExpr[VarExpr("tbl"), VarExpr("ax"), VarExpr("x")];
                name = "interp.linear")
    @test_throws EarthSciASTDiff.DerivativeRuleError dscalar(ev, Site(VarExpr("x")))
    # End-to-end through a model, against the AD oracle.
    doc = Dict{String,Any}("esm" => "0.8.0",
        "metadata" => Dict{String,Any}("name" => "InterpSlope"),
        "models" => Dict{String,Any}("M" => Dict{String,Any}(
            "variables" => Dict{String,Any}(
                "w" => Dict{String,Any}("type" => "state", "default" => 0.4)),
            "equations" => Any[Dict{String,Any}(
                "lhs" => Dict{String,Any}("op" => "D", "args" => Any["w"], "wrt" => "t"),
                "rhs" => Dict{String,Any}("op" => "fn", "name" => "interp.linear",
                    "args" => Any[
                        Dict{String,Any}("op"=>"const","args"=>Any[],"value"=>[0.0,10.0,12.0]),
                        Dict{String,Any}("op"=>"const","args"=>Any[],"value"=>[0.0,1.0,2.0]),
                        "w"]))])))
    file = EarthSciAST.load(doc)
    f!, u0, p, _, _ = build_evaluator(file)
    for (x, slope) in ((0.4, 10.0), (1.6, 2.0), (2.5, 0.0))
        res = assemble_jacobian(file; u = [x], p = p)
        @test Matrix(res.J) ≈ [slope;;] atol = 1e-12
        Jad = ad_jacobian(f!, [x], p, 0.0)
        @test Matrix(res.J) ≈ Matrix(Jad) atol = 1e-12
    end
end
