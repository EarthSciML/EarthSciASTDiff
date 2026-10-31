# EarthSciASTDiff

Analytical sparse Jacobians for [EarthSciAST](https://github.com/EarthSciML/EarthSciAST)
(`.esm`) models, by symbolic differentiation of the expression AST.

Given an arbitrary `.esm` model (or a coupled document's flattened system),
EarthSciASTDiff computes `∂f/∂u` (states), `∂f/∂p` (parameters, array-valued
ones cell by cell), `∂f/∂B` (live forcing buffers) and `∂f/∂t` of the ODE
right-hand side as a list of structured **bands** — rectangular index
regions with symbolic column-index expressions and ESM coefficient
expressions. From the bands it derives, without any numeric probing:

- the exact **structural sparsity pattern** (global: `ifelse`/`min`/`max`
  branches unioned, so it is valid at every point),
- a **structure classification** (`:block_diagonal` / `:banded` /
  `:general`, and `:column_block` along a column axis the caller names) for
  choosing a matrix type,
- optional **symbolic factorization info** (general interface; per-cell
  block LU with Markowitz ordering and fill-in — the chemistry shape — is
  the first implementation),
- a serialized **`"jacobians"` document block**, so the Jacobian is itself
  an ESM artifact any language binding can evaluate,
- **product documents**: `J`'s band fields, `J v`, `Jᵀ λ` and the
  second-order contraction, each as a derived ESM document an unmodified
  EarthSciAST evaluator runs,
- **second-order bands** for Rosenbrock discrete adjoints, contracted to
  `∇(μᵀ J w)` per step without ever forming a Hessian.

The Julia package manipulates ASTs only: every number comes from
EarthSciAST evaluating one of its documents, and
[EarthSciIntegrator](https://github.com/EarthSciML/EarthSciIntegrator)
coordinates those evaluations in a solver. (The Python and Rust bindings
still carry their own evaluators, until they emit the same documents.)

Correctness contract: the derivative is taken on the *same* expanded,
lowered tree `EarthSciAST.build_evaluator` compiles, and the test suite pins
the assembled Jacobian against ForwardDiff through the tree-walk RHS to
machine precision.

## Quick start (Julia)

```julia
using EarthSciAST, EarthSciASTDiff

file = EarthSciAST.load_path("model.esm")

entries = jacobian_bands(file, "MyModel")    # the Jacobian as structured bands
pat = jacobian_pattern(file)                 # structural pattern + row/column names
detect_structure(entries, sysview(file, "MyModel"))  # :block_diagonal | :banded | :general
doc = jacobian_document(file, "MyModel")     # document + "jacobians" block

# Values are EarthSciAST's to compute. Each product is a document:
ev, fields = evaluation_document(file)       # J's band fields  (model "JacobianEval")
jv, io = jvp_document(file)                  # J v              (model "JvpEval")
vj, io = vjp_document(file)                  # Jᵀ λ             (model "VjpEval")
hv, io = hvp_document(file; stages = 2)      # ∇_u Σᵢ μᵢᵀ J(u) wᵢ (model "HvpEval")

# Build any of them with EarthSciAST, write u into the source states' slots
# and the inputs (io.v, io.lam, io.mu / io.w) into theirs, call the RHS once,
# and read the outputs (io.jvp, io.vjp, io.hvp) from du.
f!, x0, p, _, vm = build_evaluator(EarthSciAST.load_document(jv); model_name = io.model)

# Coupled documents go through flatten first:
flat = flatten(file)
jv, io = jvp_document(flat)
```

`∂f/∂p`: pass `wrt = :parameters`. `∂f/∂B`, with respect to the cells of
the forcing buffers (by default every array parameter with a `data` update;
pass `forcing = [names…]` to choose), is `wrt = :forcing`:
`vjp_document(file; wrt = :forcing)` is the forcing cotangent
`ḡ_B = (∂f/∂B)ᵀ λ`, its buffers bound live through EarthSciAST's
`param_arrays`. `∂f/∂t` is `wrt = :time`.

Second order, for a Rosenbrock adjoint: an axis pair `(a1, a2)`,
`a1 ∈ (:states, :time)`, `a2 ∈ (:states, :parameters, :forcing)`:

```julia
hv, io = hvp_document(file; stages = 2, wrt = (:states, :parameters))
doc = jacobian_document(file, "MyModel"; wrt = [:states, (:states, :states)])
```

In `jacobian_document` a single pair block is `wrt = :states => :states`; a
bare tuple there is refused, since it also reads as a list of axes.

A `Jᵀ λ` or second-order document is an equality-filtered `faq` scatter
(spec §9.3): EarthSciAST.jl's interpreter evaluates it today
(`compiler = :interpreter`), and its native compiler gains it with the
equality-filter scatter optimization (EarthSciAST#613).

Placing `J`'s band values in a matrix of your own layout is a scatter plan:
`scatter_plan(file, Csr())` (or `LaneBlocks()`, `LaneSparseBlocks()`,
`ColumnBanded(axis)`), checked against the source by `bind_plan`, whose
`pairs` say which `JacobianEval` field cell goes to which storage position.

## Documentation

The documentation site — getting-started guide, cross-language API reference and the
specification — is at <https://earthsci.dev/astdiff/>. It is built from `docs/` by
the shared tooling in [EarthSciML/earthsci.dev](https://github.com/EarthSciML/earthsci.dev)
(`.github/workflows/docs.yml` publishes it on every push to `main`); preview locally
with `python3 <earthsci.dev>/tools/api_docs/render.py --repo-root .` (see that repo's
`tools/api_docs/README.md`).

## Specification

The `"jacobians"` block format, the per-operator derivative semantics
(including the non-smooth policies: branch-union sparsity, `max`/`min` tie
rules, closed-function table), and the factorization-info interface are
specified in [`esm-jacobian-spec.md`](esm-jacobian-spec.md) — a draft to be
merged into `esm-spec.md` once stable. Cross-language conformance fixtures
and goldens live in [`tests/`](tests/).

## Repository layout

Mirrors EarthSciAST to make room for future language bindings:

```
esm-jacobian-spec.md         draft spec (block format + derivative semantics)
pkg/
  EarthSciASTDiff.jl/        Julia implementation (reference)
  earthsci-astdiff-py/       Python binding (band calculus + §4 serialization)
  earthsci-astdiff-rs/       Rust binding (same scope, golden-gated)
tests/
  valid/                     shared fixture models
  goldens/                   band-list goldens (deterministic JSON)
scripts/
  test-conformance.sh        cross-language conformance runner
```

Planned bindings follow EarthSciAST's naming: `earthsci-astdiff-py`,
`earthsci-astdiff-rs`, `earthsci-astdiff-go`, `earthsci-astdiff-ts` — each
implements the §3 rule table and band calculus against its language's
EarthSciAST implementation and is gated by the same goldens.

## Design notes

- **Bands, not a rewritten array AST.** A band is the symbolic analogue of
  the tree-walk's `_AccDesc` state-read descriptors: `col == row` is a
  per-cell (chemistry) coupling, `col == row ± k` an affine stencil offset.
  Structure is read off the column expressions.
- **Sparsity is structural occurrence** — never a symbolic-zero or numeric
  probe (the Symbolics/KPP/AMICI/OpenModelica consensus). Stored zeros are
  kept so solver preparations can be reused across the integration.
- **Evaluation is EarthSciAST's.** Band coefficient fields and every
  product (spec §9) are emitted as derived ESM documents evaluated by the
  unmodified EarthSciAST evaluator; this package computes no values, so a
  faster evaluator is an EarthSciAST change, never a kernel here.
- **Factorization info is a general interface** (`FactorizationPlan`,
  dispatched on the detected structure class); `BlockDiagonalPlan`
  (per-cell block LU: Markowitz ordering + symbolic fill-in, the KPP
  precedent) is the first implementation. Banded and general-CSC plans slot
  in without changing the interface.

## Roadmap

- [x] Coefficient-size control, stage 1: branch-aware simplification
      (`simplify_branches` — equal-branch collapse, same-condition pruning)
      in the band calculus, plus hash-consed template CSE at emission
      (`cse_templates` → zero-param `expression_templates` in the block,
      spec §4.1).
- [x] Coefficient-size control, stage 2 (PPM-scale): CSE inside the
      evaluation model (`hoist_observed` — repeated coefficient subtrees
      become factored observed-array buffers of the derived document,
      evaluated once per cell per RHS call). PPM: derived document 4×
      smaller, `jac!` 8× faster per call (beating ForwardDiff through the
      RHS by an order of magnitude); values bit-identical to the plain
      path. Symbolic reverse mode turned out unnecessary at this scale —
      the remaining `prepare_jacobian` cost is tree-walk kernel JIT, i.e.
      the dedicated compiled `jac!` kernel already noted in Design notes.
- [x] Trimmed evaluation document: the source's state derivatives are
      zeroed in the derived model, so a `jac!` call evaluates the band
      fields and the observed they read but no longer recomputes `f`
      (6–36% less time per `jac!` call on the fixtures, copy-in and scatter
      included; the saving grows with the cost of `f`). Pinned by
      `*.evaldoc.json`.
- [x] Scatter plans (spec §8): a checked, serializable map from the band
      fields to a consumer's storage — lane-SoA dense per-cell blocks
      (Markowitz `block_vars` order), KPP-style sparse lane blocks, per-column
      bands and CSR. Lane layouts are per-cell templates (O(nnz)); the lane
      capacity is a bind-time choice. `bind_plan` refuses a plan from another
      band list or state set, and `jac_into!` fills the layout from a prepared
      evaluator. Pinned by `*.<layout>.plan.json`, which Rust reproduces.
- [x] Static region clipping for `index(makearray…)` reads (`clip.jl`):
      decidable membership guards are decided exactly by splitting band row
      ranges at their affine breakpoints (equal-coefficient sub-bands
      re-merge), so region structure never reaches the coefficients or
      pads the pattern; state-dependent (limiter) branches remain unioned.
- [x] Contracted-index bands: `reduce: "+"` faq nodes over gathered
      columns (`u[j]`, `u[i+k]`, `u[conn[i,k]]` const tables) emit entries
      with free `contracted` column dimensions (accumulating at assembly);
      contraction-independent columns emit ordinary entries with a
      symbolic-sum coefficient.
- [x] `interp.bilinear` derivative (two clamped-weight partials, per-cell
      slope constants precomputed into nested segment `ifelse`, spec §3.4)
      and `tgrad` (`wrt = :time` through the same band calculus;
      `ode_components`/`odefunction` attach the analytical `tgrad!`, with a
      zero fill for autonomous systems).
- [x] Time-derivative bands specified and golden-gated
      ([#11](https://github.com/EarthSciML/EarthSciASTDiff/issues/11)):
      `wrt: "time"` is spec §4.2 (one scalar column `t`, every read observed
      inlined, the partial with parameters held fixed, data-borne time
      dependence excluded, an empty list meaning `f_t ≡ 0`), and
      `tgrad_photolysis.time.{bands,block,evaldoc,jvals}.json` pin Julia,
      Rust and Python against each other. `datetime.julian_day` now
      differentiates to `1/86400` (it is continuous; the other `datetime.*`
      fields stay 0), and the Rust and Python golden writers emit floats in
      the reference's layout (`2.4405875e6`, not `2440587.5`).
- [x] Forcing-buffer and array-parameter columns
      ([#1](https://github.com/EarthSciML/EarthSciASTDiff/issues/1), spec
      §4.3): `wrt: "forcing"` over the cells of the live data buffers (the
      forcing set inferred from `data` updates or passed in), the record
      bracket an ordinary leading length-2 axis so the time blend's chain rule
      falls out of `*`/`+`; array parameters are real columns of
      `wrt: "parameters"` instead of being dropped at assembly. Blocks carry
      their `columns` (the canonical column ordinal) and a model's
      `"jacobians"` entry may be an array of blocks. `vjp_forcing!` applies
      the transposed bands as a scatter-add into buffer-shaped accumulators
      (Python and Rust: `prepare_jacobian` once, then `vjp_forcing` per stage).
      Pinned by `forcing_bracket.{forcing,parameters}.*` goldens.
- [x] Second-order bands for Rosenbrock discrete adjoints
      ([#2](https://github.com/EarthSciML/EarthSciASTDiff/issues/2), spec
      §4.4): the band calculus applied to first-order coefficients, emitted
      as `wrt: [a1, a2]` blocks (`col2`, `col2_idx`), and contracted per
      adjoint step by `hvp!` (`g[col2] += coef2 · Σᵢ μᵢ[row] wᵢ[col]`, or
      the symmetric form on the state pair) — checked against ForwardDiff
      over ForwardDiff. Pinned by `<fixture>.<a1>-<a2>.*` goldens on
      `bd_chem`, `arrhenius_chem`, `forcing_bracket` and `minmod_adv`.
      Python and Rust reproduce the band and block goldens byte-for-byte
      and the contraction values (`prepare_hvp` once, then `hvp` per
      step) on their numeric oracles. Rust also contracts them on its
      production path (below): `HvpProgram` compiles the second-order band
      fields through the same `RhsProgram` route, and an `HvpPlan`
      contracts them against the integrator's own layout (whole-domain
      only for now).
- [x] Filtered faq nodes, `max`/`min`/`*` reductions and value-indexed reads
      ([#3](https://github.com/EarthSciML/EarthSciASTDiff/issues/3), spec
      §3.5): a filter is a gate whose Dirac term is dropped; its index-only
      conjuncts split the band box exactly and leave a triangular window as
      the entry's `where` guard (a level scan's `gk < gke`), its value
      conjuncts stay in the coefficient. `max`/`min` weight each term by the
      first-extremum indicator (the left-fold tie rule), products by the
      exclusion product (no division). A subscript that reads a value (a
      level search `x[κ(u)]`) is one-hot over the axis. `detect_structure(…;
      column_axis)` reports `:column_block` with a line-by-line
      factorization plan. Pinned by `filtered_scan` and `semiring_reduce`
      goldens. Causal recurrences (tangent/adjoint sweeps) are the
      remaining part of #3.
- [x] Public expansion seam in EarthSciAST (`EarthSciAST.expanded_model`,
      ≥ 0.9.1) and `ODEFunction(f!; jac, jac_prototype)` / `ODEProblem`
      wiring (`odefunction` / `odeproblem` via the SciMLBase package
      extension; `ode_components` is the solver-free half).
- [x] Python binding, band-calculus half (`pkg/earthsci-astdiff-py`):
      the §3 rule table, observed inlining, static clipping, contracted
      bands, and the §4 serialization — byte-identical to the Julia
      reference on every fixture × axis (goldens now cover all five
      fixtures). Includes a faithful port of the reference `simplify`
      (the two packages' own simplifiers order commutative arguments
      differently). Numeric assembly is in too (`assemble_jacobian`,
      per-cell evaluation through the numpy interpreter, gated by
      Julia-produced VALUE goldens `tests/goldens/*.jvals.json` — exact
      pattern key set, 1e-10 relative values), as is the emission side —
      `jacobian_block` with §4.1 template CSE and the block-LU
      factorization plan, byte-identical to the reference's blocks
      (`*.block.json` goldens). Still to come there: production-rate
      assembly.
- [x] Rust binding (`pkg/earthsci-astdiff-rs`), same scope as Python:
      band calculus, block emission (template CSE + structure + block-LU
      plan), and numeric assembly — all three golden gates pass
      byte-for-byte / at tolerance on every fixture. Caught up with esm
      1.0.0/1.1.0 (the `faq` tag, the unknown/parameter variable model,
      inline pruning) together with Python. Emits the derived evaluation
      document (with `hoist_observed`) byte-identically to Julia
      (`*.evaldoc.json`) and the scatter plans (`*.plan.json`), with
      `band_fields` + `ScatterPlan::scatter` checked against the oracle on
      every fixture.
- [x] Production-rate Rust assembly
      ([#4](https://github.com/EarthSciML/EarthSciASTDiff/issues/4) W2, W4,
      W5): `JacobianProgram` compiles the evaluation document once through
      earthsci-ast-rs's own `RhsProgram` (`native`, selected to the band
      fields, so no source `D` equation runs), and a `JacPlan` binds a
      scatter plan to an executor — the whole-domain program, or a lane
      program over run-time cells or columns (RFC U3) — and to the
      integrator's program for the same operator (states, parameters and
      forcing slots resolved by name; a source from another build is
      refused). `eval_into` fills any of the four layouts (`f64` or `f32`
      storage) and leaves the band slab for matrix-free `apply` (J·v) and
      `apply_transpose` (Jᵀ·λ, `f64`). `Send + Sync` program and plan, one
      `JacWorkspace` per thread; bitwise invariant to thread count, lane
      order and padding; equal to the oracle to 1e-13 on every plan
      fixture. On `bd_chem` at 6,552 cells it is 0.65–0.95× Julia's `jac!`
      per call (lane layouts fastest, CSR slowest) and builds in
      milliseconds where `prepare_jacobian` takes ~55 s
      (`examples/bench_production.rs`, `scripts/bench-production.jl`).
      U8's second-order contraction rides the same route: `HvpProgram`
      compiles the second-order band fields (an empty list, as on
      `minmod_adv`, is a no-op program), and an `HvpPlan` resolves each
      point to the source's `u ++ aux`, `p` or forcing slots by name for
      `contract` (both forms), equal to the `hvp` oracle and the `hvals`
      goldens to 1e-10. Still to come: tile (stencil) programs, a lane
      form of the second-order contraction (refused today; the domain
      plan covers it), the reseact / CBM-Z benchmarks (B2, B3), and lane programs for
      column bands that stop short of the column ends (refused by
      earthsci-ast's lane rewrite today; the domain plan covers them).
- [ ] diffsol solver integration — not needed by EarthSciIntegrator, which
      owns its linear algebra; it could be built on the `Csr` layout.

## License

AGPL-3.0 — see [LICENSE](LICENSE).
