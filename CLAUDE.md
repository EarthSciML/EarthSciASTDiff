# Project Instructions for AI Agents

## Build & Test

```bash
# Julia (reference implementation)
julia --project=pkg/EarthSciASTDiff.jl -e 'using Pkg; Pkg.test()'

# Cross-language conformance (currently Julia-only)
./scripts/test-conformance.sh

# Regenerate band goldens after an INTENDED rule/format change
julia --project=pkg/EarthSciASTDiff.jl scripts/regenerate-goldens.jl
```

NOTE for agents on the shared UIUC cluster: `Pkg.test()` resolving a fresh env
can contend on the shared depot. Prefer a persistent scratch env that
`Pkg.develop`s both this package and a local EarthSciAST checkout, then
`include("pkg/EarthSciASTDiff.jl/test/runtests.jl")`. Never commit a
Manifest.toml or a `[deps]` path override.

## Architecture

EarthSciASTDiff differentiates `.esm` expression ASTs symbolically and emits
the Jacobian as structured bands (see `esm-jacobian-spec.md`, the draft spec
that will merge into EarthSciAST's `esm-spec.md`). Layout mirrors EarthSciAST:
`pkg/` holds language implementations (Julia is the reference), `tests/`
holds shared fixtures + goldens that all future bindings must match.

Julia source map (`pkg/EarthSciASTDiff.jl/src/`):
- `scalar_rules.jl` — per-op derivative table; MUST match spec §3 row-for-row
- `simplify_branches.jl` — `ifelse`-aware simplification (equal-branch
  collapse, same-condition path pruning); `_simp` = the coefficient simplifier
- `cse.jl`          — hash-consed repeated-subtree extraction into zero-param
  `expression_templates` (`cse_templates` / `expand_templates`, spec §4.1)
- `bands.jl`        — array-level band calculus (aggregate/makearray)
- `inline.jl`       — observed inlining + index-of-makearray lowering
- `emit.jl`         — "jacobians" block, goldens writer, derived eval model
- `assemble.jl`     — pattern, scatter map, prepared `jac!`
- `structure.jl`    — structure classification from column-index expressions
- `factorization.jl`— FactorizationPlan interface + block-diagonal LU plan
- `ode.jl`          — solver-free ODE pieces (`ode_components`) + dispatchers
  into `ext/EarthSciASTDiffSciMLBaseExt.jl` (`odefunction` / `odeproblem`)

## Conventions

- Conventional commits: `type(scope): description` (e.g. `feat(julia): …`,
  `spec: …`).
- Sparsity is STRUCTURAL (occurrence; branches unioned). Never derive a
  pattern from numeric evaluation, and never silently return 0 for an op
  without a rule — throw.
- Any change to derivative rules or the block format is a spec change:
  update `esm-jacobian-spec.md` and the goldens in the same commit.
