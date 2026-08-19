# Shared conformance fixtures

Cross-language test surface for EarthSciASTDiff (mirrors EarthSciAST's
`tests/`). Every language binding must produce byte-identical goldens.

- `valid/` — fixture `.esm` models exercising the differentiation surface:
  - `bd_chem.esm` — pure per-cell chemistry → block-diagonal structure +
    a `block_diagonal_lu` factorization plan
  - `adv_interior.esm` — interior stencil, constant faces → banded structure
  - `adv_react_2d.esm` — stencil + chemistry + scalar↔array coupling,
    zero-gradient faces, `max` limiter kink → general structure
  - `coupled_chem_advection.esm` — reaction system ⊕ advection through
    `operator_compose` + pointwise lift (differentiates the *flattened*
    system)
- `goldens/` — `<fixture>.bands.json`: the `entries` list of the `"jacobians"`
  block in deterministic JSON (sorted object keys; entries in equation order,
  then sorted target variable, then sorted canonical site key). Regenerate
  with `scripts/regenerate-goldens.jl` after an INTENDED rule or format
  change; the diff is the review surface.

Value-level conformance (Jacobian entries at sample points) is asserted in
each binding's own test suite against a machine-precision oracle — for Julia,
ForwardDiff through the eltype-generic tree-walk RHS.
