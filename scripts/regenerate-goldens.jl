# Regenerate tests/goldens/*.bands.json from tests/valid fixtures.
# Run after an INTENDED derivative-rule or block-format change; the diff is
# the review surface.
using EarthSciAST, EarthSciASTDiff
root = normpath(joinpath(@__DIR__, ".."))
for name in ("bd_chem", "adv_interior", "contracted_ops", "flux_form_adv", "minmod_adv",
             "contracting_observed")
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
