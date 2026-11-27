# Operator references (esm-spec §9.6) are differentiated without being
# expanded (operator_refs.jl): scalar definitions through derived definitions
# and the chain rule, array (stencil) definitions through their bands composed
# at each use. The reference answers here come from EXPANDING the same system
# in the test (EarthSciAST's explicit expansion utilities) and from AD through
# EarthSciAST's own evaluator; the package itself never expands.

using EarthSciASTDiff: _evaluation_document

const _REFS_DOC = """
{"esm": "2.0.0", "metadata": {"name": "Refs"},
 "models": {"M": {
   "expression_template_imports": [{"ref": "esm:std/interp.esm", "only": ["linear"]}],
   "expression_templates": {
     "rate": {"params": ["A", "E"],
              "body": {"op": "*", "args": ["A",
                {"op": "exp", "args": [{"op": "/", "args": [{"op": "-", "args": ["E"]}, "T"]}]}, "x"]}},
     "sq": {"params": ["a"],
            "body": {"op": "+", "args": [{"op": "*", "args": ["a", "a"]},
              {"op": "apply_expression_template", "args": [], "name": "rate",
               "bindings": {"A": "a", "E": 100}}]}}
   },
   "variables": {"x": {"type": "unknown", "default": 0.5},
                 "y": {"type": "unknown", "default": 1.2},
                 "T": {"type": "parameter", "default": 290.0},
                 "k": {"type": "parameter", "default": 0.3}},
   "equations": [
     {"lhs": {"op": "D", "args": ["x"], "wrt": "t"},
      "rhs": {"op": "*", "args": [{"op": "-", "args": ["k"]},
        {"op": "apply_expression_template", "args": [], "name": "sq", "bindings": {"a": "y"}}, "x"]}},
     {"lhs": {"op": "D", "args": ["y"], "wrt": "t"},
      "rhs": {"op": "-", "args": [
        {"op": "apply_expression_template", "args": [], "name": "rate", "bindings": {"A": "k", "E": 50}},
        {"op": "apply_expression_template", "args": [], "name": "linear",
         "bindings": {"table": {"op": "const", "args": [], "value": [1.0, 3.0, 2.0]},
                      "axis": {"op": "const", "args": [], "value": [0.0, 1.0, 2.0]},
                      "x": {"op": "*", "args": ["x", "y"]}, "n": 3}}]}}]}}}
"""

_nrefs(x) = x isa AbstractDict ?
    (get(x, "op", nothing) in ("apply_operator", "apply_expression_template")) +
    sum(_nrefs, values(x); init = 0) :
    x isa AbstractVector ? sum(_nrefs, x; init = 0) : 0
_has_op(x, o) = x isa AbstractDict ? (get(x, "op", nothing) == o || any(v -> _has_op(v, o), values(x))) :
    x isa AbstractVector ? any(v -> _has_op(v, o), x) : false

@testset "operator references: scalar definitions, chain rule through derived definitions" begin
    file = EarthSciAST.load_string(_REFS_DOC)
    flat = EarthSciAST.flatten(file)
    expanded = EarthSciAST.expand_flattened_refs(flat)     # the reference answer only
    f!, u0, p, _, _ = tw_build(flat)
    u = [0.7, 1.9]
    for sv in (sysview(file, "M"), sysview(flat))
        entries = jacobian_bands(sv)
        # each definition differentiated once, into derived definitions
        names = collect(keys(sv.refs.derived))
        @test any(endswith(n, "rate.d_x") for n in names)
        @test any(endswith(n, "sq.d_a") for n in names)
        @test any(endswith(n, "linear.d_x") for n in names)
        @test all(e -> e["reference_only"] === true, values(sv.refs.derived))
        # coefficients write references, never an expanded body
        coefs = Any[EarthSciAST.serialize_expression(en.band.coef) for en in entries]
        @test _nrefs(coefs) > 0
        @test !_has_op(coefs, "exp")
    end
    # values: through the documents, against the expanded system's and AD
    J = doc_jacobian(flat; u = u, p = p)
    Je = doc_jacobian(expanded; u = u, p = p)
    vm = Dict(r => i for (i, r) in enumerate(J.rownames))
    Jad = ad_jacobian(f!, u, p, 0.0)
    @test J.rownames == Je.rownames
    @test Matrix(J.J) ≈ Matrix(Je.J) rtol = 1.0e-12
    @test Matrix(J.J) ≈ Matrix(Jad) rtol = 1.0e-10
    @test J.pattern == Je.pattern
    # structure-only mode
    @test jacobian_pattern(flat; coefficients = false).pattern ==
        jacobian_pattern(expanded; coefficients = false).pattern
end

@testset "operator references: the evaluation document carries definitions" begin
    flat = EarthSciAST.flatten(EarthSciAST.load_string(_REFS_DOC))
    sv1 = sysview(flat)
    entries = jacobian_bands(sv1)
    # emitted from ANOTHER view of the system: the derived definitions are
    # regenerated from their names and the uses' bindings
    doc, fields = evaluation_document(flat, entries)
    blocks = [m["operators"] for (_, m) in doc["models"] if haskey(m, "operators")]
    @test !isempty(blocks)
    jm = doc["models"]["JacobianEval"]
    @test haskey(jm, "operators")
    @test any(k -> occursin(".d_", k), keys(jm["operators"]))
    for (k, e) in jm["operators"]
        occursin(".d_", k) && @test e["reference_only"] === true
    end
    @test _nrefs(jm["equations"]) > 0
    @test !_has_op(jm["equations"], "exp")
    back = EarthSciAST.flatten(EarthSciAST.load_string(stable_json(doc)))
    @test all(f -> haskey(back.state_variables, "JacobianEval.$f"), fields)
    @test !isempty(back.template_registry)
end

@testset "operator references: array (stencil) definitions, bands composed at the use" begin
    file = fixture("coupled_chem_advection.esm")
    flat = EarthSciAST.flatten(file)
    expanded = EarthSciAST.expand_flattened_refs(flat)
    sv = sysview(flat)
    for co in (true, false)
        a = jacobian_bands(sv; coefficients = co)
        b = jacobian_bands(sysview(expanded); coefficients = co)
        function pts(es)
            out = Set{Any}()
            for en in es
                EarthSciASTDiff._foreach_point(en.band) do cell, _, ci
                    push!(out, (en.u, en.v, copy(cell), ci))
                end
            end
            return sort!(collect(out))
        end
        @test pts(a) == pts(b)
    end
    # band definitions: one per band of each stencil, generic in the stencil's params
    names = collect(keys(sv.refs.derived))
    @test any(n -> occursin("central_grad_lon_zero_grad_bc.d_f.b", n), names)
    @test any(n -> occursin("central_grad_lat_zero_grad_bc.d_f.b", n), names)
    entries = jacobian_bands(sv)
    coefs = Any[EarthSciAST.serialize_expression(en.band.coef) for en in entries]
    @test !_has_op(coefs, "makearray")
    @test _nrefs(coefs) > 0
    # values against the expanded system and AD through EarthSciAST
    f!, u0, p, _, _ = tw_build(flat)
    u = u0 .* (0.5 .+ rand(length(u0)))
    J = doc_jacobian(flat; u = u, p = p)
    Je = doc_jacobian(expanded; u = u, p = p)
    @test J.pattern == Je.pattern
    @test Matrix(J.J) ≈ Matrix(Je.J) rtol = 1.0e-12
    @test Matrix(J.J) ≈ Matrix(ad_jacobian(f!, u, p, 0.0)) rtol = 1.0e-10 atol = 1.0e-14
    @test detect_structure(entries, sv) == detect_structure(jacobian_bands(sysview(expanded)), sysview(expanded))
end

# A stencil `sq` (nonlinear in its array argument, a structural range bound
# `n` reaching its coefficient) used at two binding shapes, once with an
# argument that is itself an expression.
const _STENCIL_DOC = """
{"esm": "2.4.0", "metadata": {"name": "Stencils"},
 "index_sets": {"x": {"kind": "interval", "size": 6}, "y": {"kind": "interval", "size": 2}},
 "models": {"M": {
   "operators": {
     "sq": {"params": ["f", "n"],
            "body": {"op": "faq", "output_idx": ["i"], "args": ["f"], "ranges": {"i": [1, "n"]},
              "expr": {"op": "/", "args": [{"op": "*", "args": [
                {"op": "index", "args": ["f", "i"]}, {"op": "index", "args": ["f", "i"]}]}, "n"]}}}
   },
   "variables": {"a": {"type": "unknown", "shape": ["x"], "default": 0.5},
                 "b": {"type": "unknown", "shape": ["y"], "default": 1.5},
                 "k": {"type": "parameter", "default": 0.3}},
   "equations": [
     {"lhs": {"op": "D", "args": ["a"], "wrt": "t"},
      "rhs": {"op": "faq", "output_idx": ["i"], "args": [], "ranges": {"i": {"from": "x"}},
              "expr": {"op": "*", "args": [{"op": "-", "args": ["k"]},
                {"op": "index", "args": [
                  {"op": "apply_operator", "args": [], "name": "sq", "bindings": {
                    "f": {"op": "faq", "output_idx": ["j"], "args": ["a"], "ranges": {"j": {"from": "x"}},
                          "expr": {"op": "*", "args": [{"op": "index", "args": ["a", "j"]},
                                                       {"op": "index", "args": ["a", "j"]}]}},
                    "n": 6}}, "i"]}]}}},
     {"lhs": {"op": "D", "args": ["b"], "wrt": "t"},
      "rhs": {"op": "faq", "output_idx": ["i"], "args": [], "ranges": {"i": {"from": "y"}},
              "expr": {"op": "*", "args": [{"op": "-", "args": ["k"]},
                {"op": "index", "args": [
                  {"op": "apply_operator", "args": [], "name": "sq", "bindings": {"f": "b", "n": 2}},
                  "i"]}]}}}]}}}
"""

@testset "operator references: one stencil at two binding shapes, expression arguments" begin
    flat = EarthSciAST.flatten(EarthSciAST.load_string(_STENCIL_DOC))
    expanded = EarthSciAST.expand_flattened_refs(flat)
    sv = sysview(flat)
    entries = jacobian_bands(sv)
    # one band definition per binding shape, the shape in its name
    names = sort!(collect(keys(sv.refs.derived)))
    @test any(n -> endswith(n, ".d_f.b1@6,6"), names)
    @test any(n -> endswith(n, ".d_f.b1@2,2"), names)
    # no coefficient reads an index name its band does not bind
    for en in entries
        bound = Set{String}(r for r in en.band.ridx if r isa String)
        foreach(c -> push!(bound, c[1]), en.band.contracted)
        free = EarthSciAST.free_variables(en.band.coef)
        @test issubset(Set(n for n in free if !haskey(sv.variables, n)), bound)
    end
    f!, u0, p, _, _ = tw_build(flat)
    u = u0 .* (0.5 .+ rand(length(u0)))
    J = doc_jacobian(flat; u = u, p = p)
    Je = doc_jacobian(expanded; u = u, p = p)
    @test J.pattern == Je.pattern
    @test Matrix(J.J) ≈ Matrix(Je.J) rtol = 1.0e-12
    @test Matrix(J.J) ≈ Matrix(ad_jacobian(f!, u, p, 0.0)) rtol = 1.0e-10
    # the derived band definitions are regenerated from their names (the
    # document is written from another view of the system)
    doc, _ = evaluation_document(flat, entries)
    ops = doc["models"]["JacobianEval"]["operators"]
    @test any(n -> endswith(n, "@6,6"), keys(ops)) && any(n -> endswith(n, "@2,2"), keys(ops))
end

# An array reference inside a first-order coefficient (`∂/∂c` of
# `c[i] * half(a)[i]` reads the stencil's array), differentiated again.
const _COEF_REF_DOC = """
{"esm": "2.4.0", "metadata": {"name": "CoefRefs"},
 "index_sets": {"x": {"kind": "interval", "size": 5}, "z": {"kind": "interval", "size": 4}},
 "models": {"M": {
   "operators": {
     "shift": {"params": ["f"],
               "body": {"op": "faq", "output_idx": ["i"], "args": ["f"], "ranges": {"i": [1, 4]},
                 "expr": {"op": "-", "args": [{"op": "index", "args": ["f", {"op": "+", "args": ["i", 1]}]},
                                              {"op": "*", "args": [0.5, {"op": "index", "args": ["f", "i"]}]}]}}},
     "rate": {"params": ["T"],
              "body": {"op": "exp", "args": [{"op": "/", "args": [-300, "T"]}]}}
   },
   "variables": {"a": {"type": "unknown", "shape": ["x"], "default": 0.5},
                 "c": {"type": "unknown", "shape": ["z"], "default": 1.5},
                 "T": {"type": "unknown", "default": 290.0}},
   "equations": [
     {"lhs": {"op": "D", "args": ["c"], "wrt": "t"},
      "rhs": {"op": "faq", "output_idx": ["i"], "args": [], "ranges": {"i": {"from": "z"}},
              "expr": {"op": "*", "args": [{"op": "-", "args": [{"op": "index", "args": ["c", "i"]}]},
                {"op": "apply_operator", "args": [], "name": "rate", "bindings": {"T": "T"}},
                {"op": "index", "args": [
                  {"op": "apply_operator", "args": [], "name": "shift", "bindings": {"f": "a"}}, "i"]}]}}},
     {"lhs": {"op": "D", "args": ["T"], "wrt": "t"}, "rhs": 0},
     {"lhs": {"op": "D", "args": ["a"], "wrt": "t"},
      "rhs": {"op": "faq", "output_idx": ["i"], "args": [], "ranges": {"i": {"from": "x"}},
              "expr": {"op": "*", "args": [-0.1, {"op": "index", "args": ["a", "i"]}]}}}]}}}
"""

@testset "operator references: second order through scalar and array references" begin
    flat = EarthSciAST.flatten(EarthSciAST.load_string(_COEF_REF_DOC))
    expanded = EarthSciAST.expand_flattened_refs(flat)
    e2 = jacobian_bands2(sysview(flat))
    e2x = jacobian_bands2(sysview(expanded))
    triples(es) = sort!(unique([(en.u, en.v, en.w) for en in es]))
    @test triples(e2) == triples(e2x)
    @test ("M.c", "M.c", "M.a") in triples(e2) && ("M.c", "M.c", "M.T") in triples(e2)
    _, u0, p, _, _ = tw_build(flat)
    n = length(u0)
    u = u0 .* (0.5 .+ rand(n))
    pairs = [(rand(n), rand(n)) for _ in 1:2]
    g, cols = doc_hvp(flat, pairs; u = u, p = p)
    gx, colsx = doc_hvp(expanded, pairs; u = u, p = p)
    @test cols == colsx
    @test g ≈ gx rtol = 1.0e-12
end

@testset "operator references: a flattened registry is closed in the view" begin
    file = EarthSciAST.load_string(_REFS_DOC)
    function pts(es)
        out = Set{Any}()
        for en in es
            EarthSciASTDiff._foreach_point(en.band) do cell, _, ci
                push!(out, (en.u, en.v, copy(cell), ci))
            end
        end
        return sort!(collect(out))
    end
    closed = EarthSciAST.flatten(file)
    ref = pts(jacobian_bands(sysview(closed)))
    _, u0, p, _, _ = tw_build(closed)
    u = u0 .* (0.5 .+ rand(length(u0)))
    Jref = doc_jacobian(closed; u = u, p = p)
    # Open the registry again, as a flatten that only scopes bodies leaves
    # it: every definition back to its declared parameters, its body reading
    # the closing names as the system's (scoped) variables, and no reference
    # binding them. The view must close it as `flatten` does, not
    # differentiate each body with respect to its declared parameters alone.
    flat = EarthSciAST.flatten(file)
    declared = Dict(
        String(k) => String[String(x) for x in v["params"]]
            for (k, v) in EarthSciAST.JSON3.read(_REFS_DOC, Dict{String, Any})["models"]["M"]["expression_templates"]
    )
    declared["linear"] = ["table", "axis", "x", "n"]
    unbind(x) = x
    function unbind(e::EarthSciAST.OpExpr)
        e2 = EarthSciAST.map_children(unbind, e)
        EarthSciASTDiff._is_ref(e2) || return e2
        keep = declared[String(e2.name)]
        b = Dict{String, EarthSciAST.ASTExpr}(k => v for (k, v) in e2.bindings if k in keep)
        return EarthSciAST.reconstruct(e2; bindings = b)
    end
    reg = flat.template_registry
    opened = 0
    for (k, d) in collect(reg)
        ps = String[String(x) for x in d["params"]]
        extra = [q for q in ps if !(q in declared[String(k)])]
        isempty(extra) && continue
        body = EarthSciAST.expression_from_json(d["body"])
        body = EarthSciASTDiff._subst_free(
            body, Dict{String, EarthSciAST.ASTExpr}(q => VarExpr("M." * q) for q in extra)
        )
        e = Dict{String, Any}(String(a) => b for (a, b) in pairs(d))
        e["params"] = Any[declared[String(k)]...]
        e["body"] = EarthSciAST.serialize_expression(unbind(body))
        reg[k] = e
        opened += 1
    end
    @test opened == 2
    for (i, eq) in enumerate(flat.equations)
        flat.equations[i] = EarthSciAST.Equation(unbind(eq.lhs), unbind(eq.rhs); _comment = eq._comment)
    end
    @test pts(jacobian_bands(sysview(flat))) == ref
    # values too: a dropped free read can hide behind another entry's points
    J = doc_jacobian(flat; u = u, p = p)
    @test J.rownames == Jref.rownames
    @test Matrix(J.J) ≈ Matrix(Jref.J) rtol = 1.0e-12
end

# A scalar definition read through its array parameters: at sites directly,
# under a sum in its body, and through a nested definition it passes one to
# whole; used inside each row's own faq with bindings that read the row index
# (an intermediate over that index), once with an argument that is itself an
# array expression (spec §3.6).
@testset "operator references: a scalar definition through its array parameters" begin
    file = fixture("ref_array_param.esm")
    flat = EarthSciAST.flatten(file)
    expanded = EarthSciAST.expand_flattened_refs(flat)
    sv = sysview(flat)
    function pts(es)
        out = Set{Any}()
        for en in es
            EarthSciASTDiff._foreach_point(en.band) do cell, _, ci
                push!(out, (en.u, en.v, copy(cell), ci))
            end
        end
        return sort!(collect(out))
    end
    for co in (true, false)
        @test pts(jacobian_bands(sv; coefficients = co)) ==
            pts(jacobian_bands(sysview(expanded); coefficients = co))
    end
    entries = jacobian_bands(sv)
    golden = joinpath(@__DIR__, "..", "..", "..", "tests", "goldens", "ref_array_param.bands.json")
    @test stable_json(Any[EarthSciASTDiff.serialize_band(en) for en in entries]) == read(golden, String)
    names = collect(keys(sv.refs.derived))
    @test any(startswith("pair.d_w.s1@"), names) && any(startswith("pair.d_q.s1@"), names)
    @test any(startswith("inner.d_w.s1@"), names)
    f!, u0, p, _, _ = tw_build(flat)
    u = u0 .* (0.5 .+ rand(length(u0)))
    J = doc_jacobian(flat; u = u, p = p)
    Je = doc_jacobian(expanded; u = u, p = p)
    @test J.pattern == Je.pattern
    @test Matrix(J.J) ≈ Matrix(Je.J) rtol = 1.0e-12
    @test Matrix(J.J) ≈ Matrix(ad_jacobian(f!, u, p, 0.0)) rtol = 1.0e-10 atol = 1.0e-14
end
