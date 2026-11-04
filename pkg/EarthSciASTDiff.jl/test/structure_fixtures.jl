# The fixtures whose state structure-only goldens
# (`<name>.structure.bands.json`, esm-jacobian-spec.md §2.1) the Julia
# reference writes: every model fixture with a state band golden, and the
# flattened coupled document, and `onehot_clipped` (structure-only goldens
# alone: a one-hot column split by clipping, a state of two equations). One
# list for the writer
# (scripts/regenerate-goldens.jl) and the checker (structure_only_test.jl).
const STRUCTURE_FIXTURES = (
    "bd_chem", "adv_interior", "contracted_ops", "flux_form_adv", "minmod_adv",
    "contracting_observed", "tgrad_photolysis", "forcing_bracket",
    "filtered_scan", "semiring_reduce", "faq_operand_list", "pre_operator",
    "coupled_chem_advection", "onehot_clipped",
)
