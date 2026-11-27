# Regenerate tests/goldens/*.bands.json from tests/valid fixtures.
# Run after an INTENDED derivative-rule or block-format change; the diff is
# the review surface.
using EarthSciAST, EarthSciASTDiff
root = normpath(joinpath(@__DIR__, ".."))
for name in (
        "bd_chem", "adv_interior", "contracted_ops", "flux_form_adv", "minmod_adv",
        "contracting_observed", "tgrad_photolysis", "forcing_bracket",
        "filtered_scan", "semiring_reduce", "faq_operand_list",
        "pre_operator", "recurrence_sweep", "pbl_rank1",
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
    # the derived evaluation document (`evaluation_document`): source
    # derivatives zeroed, hoisted observed, one band field per entry
    sv = sysview(file, mname)
    evdoc, _ = EarthSciASTDiff._evaluation_document(sv, jacobian_bands(sv))
    epath = joinpath(root, "tests", "goldens", "$name.evaldoc.json")
    write(epath, stable_json(evdoc))
    println("wrote ", epath)
    # the same without hoisting (`evaluation_document(…; cse = false)`), kept
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
# the evaluation document of a flattened system (spec §8.1): a coupled
# document's scoped names laid out as its component tree. The bands are the
# flattened system's, through `sysview(flat)`. A flattened system carries no
# `data_sources`, so the view takes the document's: `data_fed_flat`'s
# data-fed parameters keep their `data` rules and the sources they name.
for name in ("coupled_chem_advection", "lift_same_size_index_sets", "data_fed_flat")
    file = EarthSciAST.load_path(joinpath(root, "tests", "valid", "$name.esm"))
    flat = EarthSciAST.flatten(file)
    sv = sysview(flat; data_sources = file.data_sources)
    entries = jacobian_bands(sv)
    doc = stable_json(first(EarthSciASTDiff._evaluation_document(sv, entries)))
    path = joinpath(root, "tests", "goldens", "$name.evaldoc.json")
    write(path, doc)
    println("wrote ", path)
    # without hoisting, kept only where it differs (as for the models above)
    doc0 = stable_json(first(EarthSciASTDiff._evaluation_document(sv, entries; cse = false)))
    path0 = joinpath(root, "tests", "goldens", "$name.evaldoc-nocse.json")
    if doc0 == doc
        rm(path0; force = true)
    else
        write(path0, doc0)
        println("wrote ", path0)
    end
end
# the state bands of the flattened data-fed document: each data-fed parameter
# a constant of the state Jacobian, as without its rule
let name = "data_fed_flat"
    flat = EarthSciAST.flatten(EarthSciAST.load_path(joinpath(root, "tests", "valid", "$name.esm")))
    path = joinpath(root, "tests", "goldens", "$name.bands.json")
    write(path, stable_json(Any[EarthSciASTDiff.serialize_band(en) for en in jacobian_bands(sysview(flat))]))
    println("wrote ", path)
end
# structure-only entries (spec §2.1): `<name>.structure.bands.json`, the state
# entries without coefficients, for every model fixture above and the
# flattened coupled document (the list the tests check, shared with them)
include(joinpath(root, "pkg", "EarthSciASTDiff.jl", "test", "structure_fixtures.jl"))
for name in (STRUCTURE_FIXTURES..., "recurrence_sweep", "pbl_rank1", "diamond_chain")
    src = EarthSciAST.load_path(joinpath(root, "tests", "valid", "$name.esm"))
    sv = name == "coupled_chem_advection" ? sysview(EarthSciAST.flatten(src)) :
        sysview(src, first(keys(src.models)))
    entries = jacobian_bands(sv; coefficients = false)
    path = joinpath(root, "tests", "goldens", "$name.structure.bands.json")
    write(path, stable_json(Any[EarthSciASTDiff.serialize_band(en) for en in entries]))
    println("wrote ", path)
end
# the band goldens of the summand-splitting fixtures (spec §2.2)
for name in ("split_sum_2d", "split_diff_2d")
    file = EarthSciAST.load_path(joinpath(root, "tests", "valid", "$name.esm"))
    entries = jacobian_bands(file, "S")
    path = joinpath(root, "tests", "goldens", "$name.bands.json")
    write(path, stable_json(Any[EarthSciASTDiff.serialize_band(en) for en in entries]))
    println("wrote ", path)
end
# the bands of the flattened coupled document on the axes it has bands on:
# `<name>.bands.json` (states) and `<name>.parameters.bands.json`
let name = "coupled_chem_advection"
    flat = EarthSciAST.flatten(EarthSciAST.load_path(joinpath(root, "tests", "valid", "$name.esm")))
    sv = sysview(flat)
    for wrt in (:states, :parameters)
        entries = jacobian_bands(sv; wrt = wrt)
        tag = wrt == :states ? "" : ".$wrt"
        path = joinpath(root, "tests", "goldens", "$name$tag.bands.json")
        write(path, stable_json(Any[EarthSciASTDiff.serialize_band(en) for en in entries]))
        println("wrote ", path)
    end
end
# one stencil definition used at two binding shapes (spec §3.6): derived band
# definitions named with their shape, `sq.d_f.b<k>@<shape>`, one per shape
let name = "two_shape_stencil"
    flat = EarthSciAST.flatten(EarthSciAST.load_path(joinpath(root, "tests", "valid", "$name.esm")))
    sv = sysview(flat)
    entries = jacobian_bands(sv)
    path = joinpath(root, "tests", "goldens", "$name.bands.json")
    write(path, stable_json(Any[EarthSciASTDiff.serialize_band(en) for en in entries]))
    evpath = joinpath(root, "tests", "goldens", "$name.evaldoc.json")
    write(evpath, stable_json(first(EarthSciASTDiff._evaluation_document(sv, entries))))
    println("wrote ", path, ", ", evpath)
end
# Time bands with data held fixed (spec §4.2): the band list only, which
# every binding must reproduce byte for byte (tests/README.md).
let name = "tgrad_data_fixed"
    file = EarthSciAST.load_path(joinpath(root, "tests", "valid", "$name.esm"))
    entries = jacobian_bands(sysview(file, "Box"); wrt = :time)
    path = joinpath(root, "tests", "goldens", "$name.time.bands.json")
    write(path, stable_json(Any[EarthSciASTDiff.serialize_band(en) for en in entries]))
    println("wrote ", path)
end
# An elementwise observed read at a cell, its operands aligned by index-set
# name (esm-spec §4.3.4): the band list only (tests/README.md).
let name = "elementwise_aligned"
    file = EarthSciAST.load_path(joinpath(root, "tests", "valid", "$name.esm"))
    entries = jacobian_bands(sysview(file, "Box"))
    path = joinpath(root, "tests", "goldens", "$name.bands.json")
    write(path, stable_json(Any[EarthSciASTDiff.serialize_band(en) for en in entries]))
    println("wrote ", path)
end
# A makearray whose region values are cell reads (scalars), read at a cell
# through an observed: the band list only (tests/README.md).
let name = "makearray_of_reads"
    file = EarthSciAST.load_path(joinpath(root, "tests", "valid", "$name.esm"))
    entries = jacobian_bands(sysview(file, "M"))
    path = joinpath(root, "tests", "goldens", "$name.bands.json")
    write(path, stable_json(Any[EarthSciASTDiff.serialize_band(en) for en in entries]))
    println("wrote ", path)
end
# A regrid through a spatial overlap join (spec §3.5): the bands of the
# states and parameters axes, their evaluation documents, and the Jv / Jᵀλ
# documents of both, all carrying the join (tests/README.md).
let name = "join_regrid"
    file = EarthSciAST.load_path(joinpath(root, "tests", "valid", "$name.esm"))
    sv = sysview(file, "Regrid")
    for wrt in (:states, :parameters)
        tag = wrt == :states ? "" : ".$wrt"
        gold(kind) = joinpath(root, "tests", "goldens", "$name$tag.$kind.json")
        entries = jacobian_bands(sv; wrt = wrt)
        write(gold("bands"), stable_json(Any[EarthSciASTDiff.serialize_band(en) for en in entries]))
        write(gold("block"), stable_json(jacobian_document(file, "Regrid"; wrt = wrt)["jacobians"]["Regrid"]))
        evdoc, _ = EarthSciASTDiff._evaluation_document(sv, entries)
        write(gold("evaldoc"), stable_json(evdoc))
        for (kind, build) in (("jvp", jvp_document), ("vjp", vjp_document))
            doc, _ = build(file; wrt = wrt)
            write(gold("$kind.evaldoc"), stable_json(doc))
        end
        println("wrote ", name, tag, ".{bands,block,evaldoc,jvp.evaldoc,vjp.evaldoc}.json")
    end
end
# A chosen parameter set (spec §4.3): the forcing fixture's parameters axis
# over `E` and `k` only.
let name = "forcing_bracket"
    file = EarthSciAST.load_path(joinpath(root, "tests", "valid", "$name.esm"))
    sv = sysview(file, "Bracket")
    gold(kind) = joinpath(root, "tests", "goldens", "$name.parameters-subset.$kind.json")
    entries = jacobian_bands(sv; wrt = :parameters, parameters = ["E", "k"])
    write(gold("bands"), stable_json(Any[EarthSciASTDiff.serialize_band(en) for en in entries]))
    doc = jacobian_document(file, "Bracket"; wrt = :parameters, parameters = ["E", "k"])
    write(gold("block"), stable_json(doc["jacobians"]["Bracket"]))
    println("wrote ", name, ".parameters-subset.{bands,block}.json")
end
# the other axes: `<name>.<axis>.*` goldens. Time-derivative bands (spec §4.2)
# for the fixtures whose right-hand side reads `t` explicitly; forcing and
# parameter bands (spec §4.3, array parameters as real columns) for the
# forcing fixture, its forcing set inferred from the `data` updates. No
# scatter plan: plans cover the state Jacobian only (spec §8).
for (name, wrt) in (
        ("tgrad_photolysis", :time),
        ("tgrad_named_steps", :time),
        ("tgrad_surface", :time),
        ("forcing_bracket", :forcing),
        ("forcing_bracket", :parameters),
        ("recurrence_sweep", :parameters),
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
        ("forcing_bracket", (:time, :parameters)),
        ("forcing_bracket", (:time, :forcing)),
        ("tgrad_named_steps", (:time, :states)),
        ("tgrad_named_steps", (:time, :parameters)),
        ("tgrad_named_steps", (:time, :forcing)),
        ("tgrad_photolysis", (:time, :states)),
        ("minmod_adv", (:states, :states)),
        ("filtered_scan", (:states, :states)),
        ("semiring_reduce", (:states, :states)),
        ("pre_operator", (:states, :states)),
        ("recurrence_sweep", (:states, :states)),
        ("recurrence_sweep", (:states, :parameters)),
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
# every pair, plus its symmetric form on a (states, states) pair. Through the
# public builders: through a recurrence they are the §9.7 sweeps over the
# extended system, never the materialized bands.
for (name, wrt) in (
        ("bd_chem", :states), ("adv_interior", :states), ("contracted_ops", :states),
        ("flux_form_adv", :states), ("minmod_adv", :states),
        ("contracting_observed", :states), ("tgrad_photolysis", :states),
        ("forcing_bracket", :states), ("filtered_scan", :states),
        ("semiring_reduce", :states), ("tgrad_photolysis", :time),
        ("tgrad_named_steps", :time),
        ("tgrad_surface", :time),
        ("pre_operator", :states),
        ("forcing_bracket", :forcing), ("forcing_bracket", :parameters),
        ("recurrence_sweep", :states), ("recurrence_sweep", :parameters),
        ("pbl_rank1", :states),
    )
    file = EarthSciAST.load_path(joinpath(root, "tests", "valid", "$name.esm"))
    tag = wrt == :states ? "" : ".$wrt"
    for (kind, build) in (("jvp", jvp_document), ("vjp", vjp_document))
        doc, _ = build(file; wrt = wrt)
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
        ("forcing_bracket", (:time, :parameters)),
        ("forcing_bracket", (:time, :forcing)),
        ("tgrad_named_steps", (:time, :states)),
        ("tgrad_named_steps", (:time, :parameters)),
        ("tgrad_named_steps", (:time, :forcing)),
        ("tgrad_photolysis", (:time, :states)),
        ("minmod_adv", (:states, :states)),
        ("filtered_scan", (:states, :states)),
        ("semiring_reduce", (:states, :states)),
        ("pre_operator", (:states, :states)),
        ("recurrence_sweep", (:states, :states)),
        ("recurrence_sweep", (:states, :parameters)),
    )
    file = EarthSciAST.load_path(joinpath(root, "tests", "valid", "$name.esm"))
    forms = wrt == (:states, :states) ? (:contract, :symmetric) : (:contract,)
    for form in forms
        doc, _ = hvp_document(file; stages = 2, wrt = wrt, form = form)
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
        ("pre_operator", "csr", Csr()),
        ("recurrence_sweep", "csr", Csr()),
        ("pbl_rank1", "csr", Csr()),
    )
    file = EarthSciAST.load_path(joinpath(root, "tests", "valid", "$name.esm"))
    ppath = joinpath(root, "tests", "goldens", "$name.$tag.plan.json")
    write(ppath, stable_json(serialize_scatter_plan(scatter_plan(file, layout))))
    println("wrote ", ppath)
end
# the blocks with a column axis (spec §6): `column_block` and its plan — the
# `diag_plus_rank1` pattern of `pbl_rank1`
for name in ("recurrence_sweep", "pbl_rank1")
    file = EarthSciAST.load_path(joinpath(root, "tests", "valid", "$name.esm"))
    mname = first(keys(file.models))
    doc = jacobian_document(file, mname; column_axis = :lev)
    path = joinpath(root, "tests", "goldens", "$name.column-lev.block.json")
    write(path, stable_json(doc["jacobians"][mname]))
    println("wrote ", path)
end
# the float goldens: `[bits, text]` pairs, the bits the corpus (ties, powers
# of two, RFC 8785 Appendix B, subnormals, layout switches; tests/README.md)
# and the text this writer's. Add a case by adding its bits, then run this.
let path = joinpath(root, "tests", "goldens", "canonical_floats.json")
    bits = [String(p[1]) for p in EarthSciASTDiff.JSON3.read(read(path, String))]
    rows = map(bits) do b
        x = reinterpret(Float64, parse(UInt64, b; base = 16))
        text = stable_json(Any[x])[2:(end - 1)]
        "  [\"$b\", \"$text\"]"
    end
    write(path, "[\n" * join(rows, ",\n") * "\n]\n")
    println("wrote ", path)
end
