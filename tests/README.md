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
  - `minmod_adv.esm` — minmod-TVD periodic advection (N = 16), a
    self-contained serialization of the EarthSciDiscretizations
    `advection_1d_periodic_minmod_tvd` problem — the limiter-family
    regression target for coefficient-size control (spec §4.1 templates)
  - `flux_form_adv.esm` — flux-form upwind advection with an inflow face:
    an observed edge-flux field defined by a makearray, read per cell
    (`F[i]`, `F[i+1]`) — the `index(makearray…)` lowering path; the static
    region clipping regression target (exact pattern, guard-free
    coefficients, no ghost entry on the inflow row)
  - `contracted_ops.esm` — contracted-index (`reduce: "+"`) aggregates: a
    dense gathered matvec column (`u[j]`), a contraction-independent column
    (symbolic-sum coefficient), a row+contracted affine stencil (`u[i+k]`)
    under a makearray region, and a `conn[]`-table gather (`u[conn[i,k]]`)
    with a duplicated neighbor so contracted points accumulate — the
    `contracted` entry-field (§4) regression target
- `goldens/` — `<fixture>.block.json`: the emitted `jacobians` block
  (template-CSE'd entries + structure + `expression_templates`, plus the
  `factorization` plan where one exists); bindings without the factorization
  port compare byte-exactly with that key stripped.
- `goldens/` — `<fixture>.jvals.json`: assembled Jacobian VALUES at a pinned
  state (Julia-produced via `scripts/regenerate-value-goldens.jl`); bindings'
  numeric assemblies must reproduce the exact (row, col) key set and every
  value to 1e-10 relative — bit-equality across runtimes is not required.
- `goldens/` — `<fixture>.bands.json` (all five fixtures): the `entries` list of the `"jacobians"`
  block in deterministic JSON (sorted object keys; entries in equation order,
  then sorted target variable, then sorted canonical site key). Regenerate
  with `scripts/regenerate-goldens.jl` after an INTENDED rule or format
  change; the diff is the review surface.

Value-level conformance (Jacobian entries at sample points) is asserted in
each binding's own test suite against a machine-precision oracle — for Julia,
ForwardDiff through the eltype-generic tree-walk RHS.
