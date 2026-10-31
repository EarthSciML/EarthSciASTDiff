# Regenerate tests/goldens/*.bands.json from tests/valid fixtures.
# Run after an INTENDED derivative-rule or block-format change; the diff is
# the review surface.
using EarthSciAST, EarthSciASTDiff
root = normpath(joinpath(@__DIR__, ".."))
for name in (
        "bd_chem", "adv_interior", "contracted_ops", "flux_form_adv", "minmod_adv",
        "contracting_observed", "tgrad_photolysis", "forcing_bracket",
        "filtered_scan", "semiring_reduce",
    )
    file = EarthSciAST.load_path(joinpath(root, "tests", "valid", "$name.esm"))
    mname = first(keys(file.models))
    entries = jacobian_bands(file, mname)
    text = stable_json(Any[EarthSciASTDiff.serialize_band(en) for en in entries])
    path = joinpath(root, "tests", "goldens", "$name.bands.json")
    write(path, text)
    println("wrote ", path)
    # the emitted block (template-CSE'd entries + structure + templates [+
    # factorization where a plan exists]) — bindings without the
    # factorization port compare with that key stripped
    doc = jacobian_document(file, mname)
    block = doc["jacobians"][mname]
    bpath = joinpath(root, "tests", "goldens", "$name.block.json")
    write(bpath, stable_json(block))
    println("wrote ", bpath)
    # the derived evaluation document prepare_jacobian compiles: source
    # derivatives zeroed, hoisted observed, one band field per entry
    sv = sysview(file, mname)
    evdoc, _ = EarthSciASTDiff._evaluation_document(sv, jacobian_bands(sv))
    epath = joinpath(root, "tests", "goldens", "$name.evaldoc.json")
    write(epath, stable_json(evdoc))
    println("wrote ", epath)
    # the same without hoisting (prepare_jacobian's `eval_cse = false`), kept
    # only where it differs: a fixture without one has nothing to hoist
    evdoc0, _ = EarthSciASTDiff._evaluation_document(sv, jacobian_bands(sv); cse = false)
    epath0 = joinpath(root, "tests", "goldens", "$name.evaldoc-nocse.json")
    if stable_json(evdoc0) == stable_json(evdoc)
        rm(epath0; force = true)
    else
        write(epath0, stable_json(evdoc0))
        println("wrote ", epath0)
    end
end
# the other axes: `<name>.<axis>.*` goldens. Time-derivative bands (spec §4.2)
# for the fixtures whose right-hand side reads `t` explicitly; forcing and
# parameter bands (spec §4.3, array parameters as real columns) for the
# forcing fixture, its forcing set inferred from the `data` updates. No
# scatter plan: plans cover the state Jacobian only (spec §8).
for (name, wrt) in (
        ("tgrad_photolysis", :time),
        ("forcing_bracket", :forcing),
        ("forcing_bracket", :parameters),
    )
    file = EarthSciAST.load_path(joinpath(root, "tests", "valid", "$name.esm"))
    mname = first(keys(file.models))
    sv = sysview(file, mname)
    entries = jacobian_bands(sv; wrt = wrt)
    gold(kind) = joinpath(root, "tests", "goldens", "$name.$wrt.$kind.json")
    write(gold("bands"), stable_json(Any[EarthSciASTDiff.serialize_band(en) for en in entries]))
    doc = jacobian_document(file, mname; wrt = wrt)
    write(gold("block"), stable_json(doc["jacobians"][mname]))
    evdoc, _ = EarthSciASTDiff._evaluation_document(sv, entries)
    write(gold("evaldoc"), stable_json(evdoc))
    evdoc0, _ = EarthSciASTDiff._evaluation_document(sv, entries; cse = false)
    if stable_json(evdoc0) == stable_json(evdoc)
        rm(gold("evaldoc-nocse"); force = true)
    else
        write(gold("evaldoc-nocse"), stable_json(evdoc0))
    end
    println("wrote ", name, ".$wrt.{bands,block,evaldoc}.json")
end
# second-order bands (spec §4.4): `<name>.<a1>-<a2>.{bands,block}.json` for
# the axis pairs a Rosenbrock adjoint contracts. No evaluation document: the
# derived model is the first-order one's, over these coefficients.
for (name, wrt) in (
        ("bd_chem", (:states, :states)),
        ("arrhenius_chem", (:states, :states)),
        ("arrhenius_chem", (:states, :parameters)),
        ("forcing_bracket", (:states, :forcing)),
        ("forcing_bracket", (:time, :states)),
        ("minmod_adv", (:states, :states)),
        ("filtered_scan", (:states, :states)),
        ("semiring_reduce", (:states, :states)),
    )
    file = EarthSciAST.load_path(joinpath(root, "tests", "valid", "$name.esm"))
    mname = first(keys(file.models))
    tag = "$(wrt[1])-$(wrt[2])"
    gold(kind) = joinpath(root, "tests", "goldens", "$name.$tag.$kind.json")
    entries = jacobian_bands2(file, mname; wrt = wrt)
    write(gold("bands"), stable_json(Any[EarthSciASTDiff.serialize_band(en) for en in entries]))
    doc = jacobian_document(file, mname; wrt = Pair(wrt...))
    write(gold("block"), stable_json(doc["jacobians"][mname]))
    println("wrote ", name, ".$tag.{bands,block}.json")
end
# product documents (spec §9.6): `Jv` and `Jᵀλ` for every axis with a value
# golden, and the §4.4 contraction (two stages, as in the `*.hvals.json`) for
# every pair, plus its symmetric form on a (states, states) pair.
for (name, wrt) in (
        ("bd_chem", :states), ("adv_interior", :states), ("contracted_ops", :states),
        ("flux_form_adv", :states), ("minmod_adv", :states),
        ("contracting_observed", :states), ("tgrad_photolysis", :states),
        ("forcing_bracket", :states), ("filtered_scan", :states),
        ("semiring_reduce", :states), ("tgrad_photolysis", :time),
        ("forcing_bracket", :forcing), ("forcing_bracket", :parameters),
    )
    file = EarthSciAST.load_path(joinpath(root, "tests", "valid", "$name.esm"))
    sv = sysview(file, first(keys(file.models)))
    entries = jacobian_bands(sv; wrt = wrt)
    tag = wrt == :states ? "" : ".$wrt"
    for (kind, build) in (
            ("jvp", EarthSciASTDiff._jvp_document), ("vjp", EarthSciASTDiff._vjp_document),
        )
        doc, _ = build(sv, entries, wrt, nothing)
        path = joinpath(root, "tests", "goldens", "$name$tag.$kind.evaldoc.json")
        write(path, stable_json(doc))
        println("wrote ", path)
    end
end
for (name, wrt) in (
        ("bd_chem", (:states, :states)),
        ("arrhenius_chem", (:states, :states)),
        ("arrhenius_chem", (:states, :parameters)),
        ("forcing_bracket", (:states, :forcing)),
        ("forcing_bracket", (:time, :states)),
        ("minmod_adv", (:states, :states)),
        ("filtered_scan", (:states, :states)),
        ("semiring_reduce", (:states, :states)),
    )
    file = EarthSciAST.load_path(joinpath(root, "tests", "valid", "$name.esm"))
    sv = sysview(file, first(keys(file.models)))
    entries = jacobian_bands2(sv; wrt = wrt)
    forms = wrt == (:states, :states) ? (:contract, :symmetric) : (:contract,)
    for form in forms
        doc, _ = EarthSciASTDiff._hvp_document(sv, entries, wrt, nothing, 2; form = form)
        kind = form == :contract ? "hvp" : "hvp-sym"
        path = joinpath(root, "tests", "goldens", "$name.$(wrt[1])-$(wrt[2]).$kind.evaldoc.json")
        write(path, stable_json(doc))
        println("wrote ", path)
    end
end
# scatter plans (spec §8): one golden per (fixture, applicable layout); the
# fixtures not listed for a layout refuse it (test/scatter_test.jl)
for (name, tag, layout) in (
        ("bd_chem", "lane_blocks", LaneBlocks()),
        ("bd_chem", "lane_sparse_blocks", LaneSparseBlocks()),
        ("bd_chem", "column_banded", ColumnBanded("z")),
        ("bd_chem", "csr", Csr()),
        ("adv_interior", "column_banded", ColumnBanded("x")),
        ("adv_interior", "csr", Csr()),
        ("flux_form_adv", "csr", Csr()),
        ("minmod_adv", "csr", Csr()),
        ("contracted_ops", "csr", Csr()),
        ("contracting_observed", "csr", Csr()),
        ("filtered_scan", "csr", Csr()),
        ("semiring_reduce", "csr", Csr()),
    )
    file = EarthSciAST.load_path(joinpath(root, "tests", "valid", "$name.esm"))
    ppath = joinpath(root, "tests", "goldens", "$name.$tag.plan.json")
    write(ppath, stable_json(serialize_scatter_plan(scatter_plan(file, layout))))
    println("wrote ", ppath)
end
