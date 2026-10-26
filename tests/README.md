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
  - `contracted_ops.esm` — contracted-index (`reduce: "+"`) faq nodes: a
    dense gathered matvec column (`u[j]`), a contraction-independent column
    (symbolic-sum coefficient), a row+contracted affine stencil (`u[i+k]`)
    under a makearray region, and a `conn[]`-table gather (`u[conn[i,k]]`)
    with a duplicated neighbor so contracted points accumulate — the
    `contracted` entry-field (§4) regression target
  - `contracting_observed.esm` — reads of contracting observed (§3.5): an
    array observed `s[i] = Σ_k u[i,k]²` read under a faq over `(i, k)` that
    binds the same `k` (the capture case), squared (chain rule through the
    nested sum), and a scalar observed whose sum is not its root — nested
    sums renamed apart, column-reading sites as `contracted` dimensions
  - `tgrad_photolysis.esm` — the time-derivative (`wrt: "time"`, spec §4.2)
    target: a photolysis rate `max(0, cos(ω t + φᵢ))` through an array
    observed, `interp.linear` of `t` through a scalar observed, a
    `datetime.hour` factor (piecewise-constant: no time band), a
    `datetime.julian_day` source (continuous: `d[t]/86400`), and one
    autonomous row
- `goldens/` — `<fixture>.block.json`: the emitted `jacobians` block
  (template-CSE'd entries + structure + `expression_templates`, plus the
  `factorization` plan where one exists); bindings compare byte-exactly.
- `goldens/` — `<fixture>.evaldoc.json`: the derived evaluation document
  `prepare_jacobian` compiles — the source's observed definitions that the
  coefficients read, every state derivative zeroed (`D(u) = 0`), the
  `hoist_observed` buffers, and one band-field state `J_k` per entry.
  `<fixture>.evaldoc-nocse.json` is the same with `eval_cse = false`, present
  only where it differs (a fixture with something to hoist). Julia and Rust
  compare byte-exactly; it is the input the Rust production path (#4) compiles.
- `goldens/` — `<fixture>.<layout>.plan.json`: scatter plans (spec §8) —
  band field → storage position for `lane_blocks`, `lane_sparse_blocks`,
  `column_banded` and `csr`, one per layout the fixture's structure admits
  (the list lives in `scripts/regenerate-goldens.jl`). Julia and Rust
  compare byte-exactly; Python does not port scatter plans yet.
- `goldens/` — `<fixture>.jvals.json`: assembled Jacobian VALUES at a pinned
  state (Julia-produced via `scripts/regenerate-value-goldens.jl`); bindings'
  numeric assemblies must reproduce the exact (row, col) key set and every
  value to 1e-10 relative — bit-equality across runtimes is not required.
- `goldens/` — `<fixture>.bands.json` (every fixture): the `entries` list of the `"jacobians"`
  block in deterministic JSON (sorted object keys; entries in equation order,
  then sorted target variable, then sorted canonical site key). Regenerate
  with `scripts/regenerate-goldens.jl` after an INTENDED rule or format
  change; the diff is the review surface.
- `goldens/` — `<fixture>.time.{bands,block,evaldoc,evaldoc-nocse,jvals}.json`:
  the same goldens for `wrt: "time"` (spec §4.2), present only for a fixture
  whose right-hand side reads `t` (currently `tgrad_photolysis`). A fixture
  without them is autonomous: its time band list MUST be empty.
- Floats in every golden use the reference layout of spec §5 item 2 (an
  integral float keeps its `.0`; scientific notation for magnitudes outside `[1e-4, 1e6)`).

Value-level conformance (Jacobian entries at sample points) is asserted in
each binding's own test suite against a machine-precision oracle — for Julia,
ForwardDiff through the eltype-generic tree-walk RHS.
