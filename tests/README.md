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
  - `data_fed_flat.esm` — data-fed parameters through a flattened system
    (spec §8.1): two components read three data-fed array parameters
    (`Chem.T`, `Chem.E`, `Box.P`, each with a scalar default) from two of
    the document's three data sources. Its evaluation document
    (`data_fed_flat.evaldoc.json`) keeps each parameter's `data` rule and
    carries the two sources they name, not the unread third; every binding
    checks that EarthSciAST builds the three as inputs (forcing slots of
    `rhs_program` in Julia and Rust, `loader_fields` in Python) and that the
    state bands (`data_fed_flat.bands.json`) are those of the same document
    without the rules
  - `lift_same_size_index_sets.esm` — EarthSciAST's pointwise-lift
    conformance case of that name: a scalar state lifted onto a 3 × 3 grid
    whose index sets `y` and `x` have the same size, named `[x, y]` by the
    operand `Wind.u` (esm-libraries-spec §4.7.5), here with a different `u`
    in every cell so the values pin the cell layout. Pins the shape
    `flatten` records on a lifted state through the view and the evaluation
    document (`lift_same_size_index_sets.evaldoc.json`, `.jvals.json`);
    skipped by a binding whose `flatten` records no lifted shape (before
    EarthSciAST 0.8)
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
  - `filtered_scan.esm` — filtered faq nodes (§3.5): an inclusive forward
    and an exclusive reverse scan, a filtered contracting observed read
    through a nested sum, a window with an index conjunct and a value
    conjunct, reseact's `Mz` level scan (`gk < gke` over `(gi, gke)` rows)
    and a constant-clipped window — the `where` entry-field (§4) target
  - `semiring_reduce.esm` — non-`+` reductions (§3.5): a running `max`
    (first argmax), a nested `min` observed, a suffix product (exclusion
    products), a `max_product` semiring, and a level search used as a
    subscript (one-hot over the axis)
  - `faq_operand_list.esm` — faq nodes carrying their operand list `args`
    (esm-spec §4.3.1), as reseact's boundary-layer mixing observed are
    written: a column sum `colsum[i] = Σ_k dp[k]·q[i,k]` with `args: ["q"]`,
    its broadcast mean, a bare-read scalar observed `c = Σ_k q[1,k]` with
    `args: ["q"]`, and a tendency `(mean − q)/tau + 0.01·c`. The operand list
    names whole arrays but reads nothing (spec §3.5): the bands are those of
    the same document with every `args` emptied (a column band contracted
    over `lev`, the diagonal, and the row read by `c`)
  - `pre_operator.esm` — `Pre` in a right-hand side, which is the identity
    there, so its derivative is its operand's (spec §3.2), and which is
    lowered to its operand before differentiation, so no emitted coefficient
    or product document carries a `Pre`: `Pre` of an indexed array state in
    a faq body, of a scalar state read by every cell, of one cell read by a
    scalar equation, a state read both through `Pre` and directly, nonlinear
    reads (`Pre(b)·a[i]`, `exp(Pre(b))·Pre(a[2])`), a read that equals a
    plain one only once `Pre` is lowered (`Pre(b)·b − b·b`), and whole-array
    `Pre` at array level (`D(c) = −k·Pre(c) + 0.3·Pre(a)`). Its bands equal
    those of the same document with every `Pre` removed
  - `onehot_clipped.esm` — structure-only merging (spec §2.1), its only
    golden `onehot_clipped.structure.bands.json`: a level-search subscript
    under a row-index `ifelse` (`c[kap]` on rows 1..2, `c[6 - kap]` on rows
    3..5), so clipping splits one one-hot column into two adjacent regions
    whose selections differ — two full bands, ONE structure-only band — and
    a state `e` defined by two equations whose same-structure bands on `c`
    meet at a row boundary and stay apart in both modes
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
  (`evaluation_document`, model `JacobianEval`) — the source's observed
  definitions that the coefficients read, every state derivative zeroed (`D(u) = 0`), the
  `hoist_observed` buffers, and one band-field state `J_k` per entry; a
  data-fed parameter keeps its `data` rules, with the `data_sources` they
  name (`forcing_bracket`, `data_fed_flat`).
  `<fixture>.evaldoc-nocse.json` is the same with `eval_cse = false`, present
  only where it differs (a fixture with something to hoist). Julia, Rust and
  Python compare byte-exactly. `coupled_chem_advection.evaldoc[-nocse].json`
  is the evaluation document of a FLATTENED system: its scoped names
  (`SimpleOzone.O3`) declared in the component tree (models `SimpleOzone`
  and `Advection`, band fields in `JacobianEval`; spec §8.1), with
  `coupled_chem_advection.jvals.json` the values its right-hand side gives,
  checked in every binding by loading and building the document with
  EarthSciAST, and the flattened system's `JvpEval` / `VjpEval` documents
  (component tree too, spec §9.1) checked against it by value. Its bands,
  `coupled_chem_advection[.parameters].bands.json`, are the flattened
  system's (`sysview(flat)`) on the states and parameters axes, compared
  byte-exactly by every binding.
- `goldens/` — `<fixture>[.<axis>].{jvp,vjp}.evaldoc.json` and
  `<fixture>.<a1>-<a2>.{hvp,hvp-sym}.evaldoc.json`: the product documents
  (spec §9), models `JvpEval`, `VjpEval`, `HvpEval` (two stages) and, on a
  `states-states` pair, `HvpEvalSymmetric`. One per axis with a `*.jvals.json`
  and per second-order pair. Julia, Rust and Python compare byte-exactly, and
  each binding checks them by value by evaluating each with its EarthSciAST
  against the product formed from the `*.jvals.json` / `*.hvals.json` golden.
- `goldens/` — `<fixture>.<layout>.plan.json`: scatter plans (spec §8) —
  band field → storage position for `lane_blocks`, `lane_sparse_blocks`,
  `column_banded` and `csr`, one per layout the fixture's structure admits
  (the list lives in `scripts/regenerate-goldens.jl`). Julia, Rust and
  Python compare byte-exactly.
- `goldens/` — `<fixture>.jvals.json`: assembled Jacobian VALUES at a pinned
  state (Julia-produced via `scripts/regenerate-value-goldens.jl`); bindings'
  value tests must reproduce the exact (row, col) key set and every
  value to 1e-10 relative — bit-equality across runtimes is not required.
  - `recurrence_sweep.esm` — causal recurrences (spec §4.5, §9.7): a Thomas
    solve per column (forward elimination, forward substitution, back
    substitution on the REVERSED axis: dense lines), a lag-2 recurrence with
    a base guard, a non-uniform implicit-trapezoid sweep with a parameter, a
    Newton iteration on a non-state axis (pointwise), and a window filtered
    by a `const` mask (a `where` gather)
  - `pbl_rank1.esm` — mixing toward a column mean plus per-cell titration:
    the `column_block` plan's `diag_plus_rank1` pattern (spec §6)
- `goldens/` — `<fixture>.column-<axis>.block.json`: the state block emitted
  with a column axis (`column_block` and its plan), for `recurrence_sweep`
  and `pbl_rank1`.
- `goldens/` — **who writes them.** Every golden is written by the Julia
  reference (`scripts/regenerate-goldens.jl`,
  `scripts/regenerate-value-goldens.jl`) except those of `recurrence_sweep`
  and `pbl_rank1`, which the Rust binding writes
  (`cargo test --test write_goldens -- --ignored`) until EarthSciAST.jl runs
  recurrences; Rust and Python compare them byte-exactly, and their values
  are checked against central differences of the source right-hand side.
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
each binding's own test suite, every value obtained by running the emitted
documents through that binding's EarthSciAST — never by evaluating band
coefficients in the test itself.
