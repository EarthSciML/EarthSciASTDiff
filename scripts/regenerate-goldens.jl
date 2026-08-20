# Regenerate tests/goldens/*.bands.json from tests/valid fixtures.
# Run after an INTENDED derivative-rule or block-format change; the diff is
# the review surface.
using EarthSciAST, EarthSciASTDiff
root = normpath(joinpath(@__DIR__, ".."))
for name in ("bd_chem", "adv_interior", "contracted_ops", "flux_form_adv", "minmod_adv")
    file = EarthSciAST.load(joinpath(root, "tests", "valid", "$name.esm"))
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
end
