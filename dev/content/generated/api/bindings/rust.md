---
title: "Rust API index"
description: "Every stable name Rust exports, and its extension seams."
type: "api"
---


**Package:** `earthsci-astdiff` 0.1.0

Documentation extracted with rustdoc JSON (rustdoc 1.101.0-nightly (21b707e3f 2026-09-30)), format_version 61.

Extension-tier names are on the [Rust extension page](/generated/api/extension/rust).

## Stable names

| Rust | Canonical | Kind | Summary |
|---|---|---|---|
| `assemble_jacobian` | [`assemble_jacobian`](/generated/api/ref/assemble_jacobian) | function | Evaluate the band entries at a point: `{(rowname, colname): value}`, contributions summed (§4 additivity, contracted points included). |
| `Band` | [`band`](/generated/api/ref/band) | type | One rectangular region of a Jacobian block: the rows it covers (`rows`, 1-based inclusive ranges, named by `ridx`), the column-index expressions (`cidx`) and the coefficient… |
| `BandError` | [`band_error`](/generated/api/ref/band_error) | error | An expression has a form the band calculus cannot differentiate (for example a filtered `faq` node or a non-`"+"` semiring). |
| `block_diagonal_lu` | [`plan_factorization`](/generated/api/ref/plan_factorization) | function | Symbolic factorization info for a block-diagonal band list: the per-cell block LU with Markowitz ordering and fill-in (`BlockLu`). |
| `BlockLu` | [`block_diagonal_plan`](/generated/api/ref/block_diagonal_plan) | type | The per-cell block LU of a block-diagonal band list (factorization.jl's `BlockDiagonalPlan`): block variables in Markowitz order, the 1-based ordering, and the permuted pattern… |
| `cse_templates` | [`cse_templates`](/generated/api/ref/cse_templates) | function | Greedy largest-first extraction; deterministic (first-encounter order, first-max tie-break) to match the reference byte-for-byte. |
| `DerivativeRuleError` | [`derivative_rule_error`](/generated/api/ref/derivative_rule_error) | error | A scalar operator has no derivative rule (spec §3). |
| `detect_structure` | [`detect_structure`](/generated/api/ref/detect_structure) | function | Classify the band list as `"block_diagonal"`, `"banded"` or `"general"` (or `"empty"` / `"other"`), read off the column-index expressions without evaluating anything. |
| `dscalar` | [`dscalar`](/generated/api/ref/dscalar) | function | The derivative of the scalar expression `e` with respect to the site `s` (a variable or an indexed read), by the §3 rule table. |
| `forcing_names` | [`forcing_names`](/generated/api/ref/forcing_names) | function | The forcing set of a system, sorted by name: the parameters whose cells are the columns of the `Wrt::Forcing` axis (spec §4.3). |
| `hvp` | [`hvp`](/generated/api/ref/hvp) | function | The second-order contraction at a point (spec §4.4), with no tensor formed: |
| `HvpEvaluator` | [`hvp_evaluator`](/generated/api/ref/hvp_evaluator) | type | A prepared second-order contraction for one axis pair, built once by [`prepare_hvp`]: the second-order bands, their observed-inlined coefficients, the second axis's column space… |
| `JacEntry` | [`jac_entry`](/generated/api/ref/jac_entry) | type | One block of the Jacobian: row variable `u`, column variable `v`, and the `Band` giving the rows it covers, the column-index expressions and the coefficient. |
| `JacEntry2` | [`jac_entry2`](/generated/api/ref/jac_entry2) | type | One second-order band: `∂²(equation for u)/∂v ∂w`. |
| `jacobian_bands` | [`jacobian_bands`](/generated/api/ref/jacobian_bands) | function | Differentiate a model's right-hand side into Jacobian bands with respect to `wrt` (states, parameters, forcing or time). |
| `jacobian_bands2` | [`jacobian_bands2`](/generated/api/ref/jacobian_bands2) | function | Second-order bands for the axis pair `wrt = (a1, a2)`, `a1` states or time and `a2` states, parameters or forcing: every first-order entry of the `a1` axis… |
| `jacobian_block` | [`jacobian_block`](/generated/api/ref/jacobian_block) | function | The `jacobians` block for one model (spec §4): entries (template-CSE'd when `cse` is set, hoisting subtrees of at least `cse_min_nodes` nodes) and — on the state and time axes —… |
| `jacobian_columns` | [`jacobian_columns`](/generated/api/ref/jacobian_columns) | function | The column space of a parameters or forcing block: each column variable with its integer shape (empty for a scalar), sorted by name — the `columns` field of the emitted block… |
| `JacobianEvaluator` | [`jacobian_evaluator`](/generated/api/ref/jacobian_evaluator) | type | A prepared band evaluation for one model and axis, built once by [`prepare_jacobian`]: the band calculus, the observed-inlined coefficients, the column space and the parameter… |
| `lu_fill` | [`lu_fill`](/generated/api/ref/lu_fill) | function | Symbolic (no-pivoting) LU: the pattern plus every fill-in entry. |
| `markowitz_ordering` | [`markowitz_ordering`](/generated/api/ref/markowitz_ordering) | function | A Markowitz pivot ordering of a block pattern (1-based), minimizing fill-in greedily. |
| `parse_jacobian_blocks` | [`parse_jacobian_blocks`](/generated/api/ref/parse_jacobian_blocks) | function | `parse_jacobian_block` / `parse_jacobian_block2` over a model's `jacobians` entry, which is either one block or an array of blocks with distinct `wrt` (spec §4); one… |
| `prepare_hvp` | [`prepare_hvp`](/generated/api/ref/prepare_hvp) | function | Build an `HvpEvaluator` once for the axis pair `wrt` (first axis states or time, second states, parameters or forcing). |
| `prepare_jacobian` | [`prepare_jacobian`](/generated/api/ref/prepare_jacobian) | function | Build a `JacobianEvaluator` once: differentiation, observed inlining, the column space (parameters and forcing axes) and the parameter defaults. |
| `ScatterPlan` | [`scatter_plan`](/generated/api/ref/scatter_plan-type) | type | Band fields → storage positions for one (band list, layout). |
| `serialize_band` | [`serialize_band`](/generated/api/ref/serialize_band) | function | One entry of the `jacobians` block, as JSON (spec §4). |
| `simplify_branches` | [`simplify_branches`](/generated/api/ref/simplify_branches) | function | `ifelse`-aware simplification, the swell control for coefficients of limiter-heavy schemes: equal branches collapse and same-condition tests are pruned. |
| `Site` | [`site`](/generated/api/ref/site) | type | A bare variable reference or one symbolic `index(v, i…)` cell. |
| `stable_json` | [`stable_json`](/generated/api/ref/stable_json) | function | Deterministic JSON: sorted object keys at every level, floats in the reference's layout (`canonical_float`), every other scalar by serde_json. |
| `SysView` | [`sys_view`](/generated/api/ref/sys_view) | type | A model as the band calculus sees it: its variables, equations and index sets. |
| `sysview` | [`sysview`](/generated/api/ref/sysview) | function | View a model of `file` for differentiation. |
| `vjp` | [`vjp`](/generated/api/ref/vjp) | function | The transposed product `ḡ += Jᵀ λ` at a point, as a scatter-add over the band points: `ḡ[col] += coef · λ[row]` for every point, with no matrix (and no transpose) formed. |
| `vjp_forcing` | [`vjp_forcing`](/generated/api/ref/vjp_forcing) | function | `vjp` on the forcing axis, into buffer-shaped accumulators: the forcing cotangent of one stage evaluation, `ḡ_B += (∂f/∂B)ᵀ λ` (spec §4.3). |
