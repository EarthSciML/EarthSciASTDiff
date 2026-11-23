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
