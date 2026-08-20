# EarthSciASTDiff

Analytical sparse Jacobians for [EarthSciAST](https://github.com/EarthSciML/EarthSciAST)
(`.esm`) models, by symbolic differentiation of the expression AST.

Given an arbitrary `.esm` model (or a coupled document's flattened system),
EarthSciASTDiff computes `∂f/∂u` (states) and `∂f/∂p` (parameters) of the ODE
right-hand side as a list of structured **bands** — rectangular index
regions with symbolic column-index expressions and ESM coefficient
expressions. From the bands it derives, without any numeric probing:

- the exact **structural sparsity pattern** (global: `ifelse`/`min`/`max`
  branches unioned, so it is valid at every point),
- a **structure classification** (`:block_diagonal` / `:banded` /
  `:general`) for choosing a matrix type,
- optional **symbolic factorization info** (general interface; per-cell
  block LU with Markowitz ordering and fill-in — the chemistry shape — is
  the first implementation),
- a serialized **`"jacobians"` document block**, so the Jacobian is itself
  an ESM artifact any language binding can evaluate,
- a prepared evaluator `jac!(J, u, p, t)` over a fixed sparse prototype.

Correctness contract: the derivative is taken on the *same* expanded,
lowered tree `EarthSciAST.build_evaluator` compiles, and the test suite pins
the assembled Jacobian against ForwardDiff through the tree-walk RHS to
machine precision.

## Quick start (Julia)

```julia
using EarthSciAST, EarthSciASTDiff

file = EarthSciAST.load("model.esm")

jac = prepare_jacobian(file)             # differentiate + compile, once
J = jac_prototype(jac)                   # SparseMatrixCSC on the fixed pattern
jac(J, u, p, t)                          # fill J in place, any (u, p, t)

jac.structure                            # :block_diagonal | :banded | :general
doc = jacobian_document(file, "MyModel") # document + "jacobians" block

# Coupled documents go through flatten first:
flat = flatten(file)
res = assemble_jacobian(flat; u = u, p = p)   # one-shot: J, pattern, entries…
```

`∂f/∂p`: pass `wrt = :parameters` to any of the above.

Solving with the analytical Jacobian (any stiff OrdinaryDiffEq solver;
loading one activates the SciMLBase package extension):

```julia
using OrdinaryDiffEqRosenbrock

prob = odeproblem(file, (0.0, 3600.0))   # ODEFunction(f!; jac, jac_prototype)
sol = solve(prob, Rodas5P())
```

(`ode_components(file)` returns the same pieces — `f!`, `jac!`,
`jac_prototype`, `u0`, `p`, `var_map` — with no solver dependency.)

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
- **v1 evaluation reuses the tree-walk.** Band coefficient fields are
  emitted as a derived ESM model evaluated by the unmodified
  `EarthSciAST.build_evaluator`; a dedicated compiled `jac!` kernel is a
  planned optimization, not a correctness need.
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
- [x] Static region clipping for `index(makearray…)` reads (`clip.jl`):
      decidable membership guards are decided exactly by splitting band row
      ranges at their affine breakpoints (equal-coefficient sub-bands
      re-merge), so region structure never reaches the coefficients or
      pads the pattern; state-dependent (limiter) branches remain unioned.
- [x] Contracted-index bands: `reduce: "+"` aggregates over gathered
      columns (`u[j]`, `u[i+k]`, `u[conn[i,k]]` const tables) emit entries
      with free `contracted` column dimensions (accumulating at assembly);
      contraction-independent columns emit ordinary entries with a
      symbolic-sum coefficient. Filtered aggregates and non-`"+"` semirings
      still fail loudly.
- [x] `interp.bilinear` derivative (two clamped-weight partials, per-cell
      slope constants precomputed into nested segment `ifelse`, spec §3.4)
      and `tgrad` (`wrt = :time` through the same band calculus;
      `ode_components`/`odefunction` attach the analytical `tgrad!`, with a
      zero fill for autonomous systems).
- [ ] Forcing-buffer columns (∂f/∂(streamed forcing values), the EarthSciIO
      coupling) — needs the streaming-forcing representation to settle.
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
      `jacobian_block` with §4.1 template CSE, byte-identical to the
      reference's blocks (`*.block.json` goldens, `factorization` key
      excepted). Still to come there: production-rate assembly and the
      factorization plans.
- [ ] Rust binding (diffsol wants an analytic sparse J).

## License

AGPL-3.0 — see [LICENSE](LICENSE).
