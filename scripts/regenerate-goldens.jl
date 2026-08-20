# Regenerate tests/goldens/*.bands.json from tests/valid fixtures.
# Run after an INTENDED derivative-rule or block-format change; the diff is
# the review surface.
using EarthSciAST, EarthSciASTDiff
root = normpath(joinpath(@__DIR__, ".."))
for name in ("bd_chem", "adv_interior", "contracted_ops", "flux_form_adv", "minmod_adv")
    file = EarthSciAST.load(joinpath(root, "tests", "valid", "$name.esm"))
    entries = jacobian_bands(file, first(keys(file.models)))
    text = stable_json(Any[EarthSciASTDiff.serialize_band(en) for en in entries])
    path = joinpath(root, "tests", "goldens", "$name.bands.json")
    write(path, text)
    println("wrote ", path)
end
