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
| `JacEntry` | [`jac_entry`](/generated/api/ref/jac_entry) | type | One block of the Jacobian: row variable `u`, column variable `v`, and the `Band` giving the rows it covers, the column-index expressions and the coefficient. |
| `jacobian_bands` | [`jacobian_bands`](/generated/api/ref/jacobian_bands) | function | Differentiate a model's right-hand side into Jacobian bands with respect to `wrt` (states, parameters or time). |
| `jacobian_block` | [`jacobian_block`](/generated/api/ref/jacobian_block) | function | The `jacobians` block for one model (spec §4): entries (template-CSE'd when `cse` is set, hoisting subtrees of at least `cse_min_nodes` nodes), the structure classification, and… |
| `lu_fill` | [`lu_fill`](/generated/api/ref/lu_fill) | function | Symbolic (no-pivoting) LU: the pattern plus every fill-in entry. |
| `markowitz_ordering` | [`markowitz_ordering`](/generated/api/ref/markowitz_ordering) | function | A Markowitz pivot ordering of a block pattern (1-based), minimizing fill-in greedily. |
| `ScatterPlan` | [`scatter_plan`](/generated/api/ref/scatter_plan-type) | type | Band fields → storage positions for one (band list, layout). |
| `serialize_band` | [`serialize_band`](/generated/api/ref/serialize_band) | function | One entry of the `jacobians` block, as JSON (spec §4). |
| `simplify_branches` | [`simplify_branches`](/generated/api/ref/simplify_branches) | function | `ifelse`-aware simplification, the swell control for coefficients of limiter-heavy schemes: equal branches collapse and same-condition tests are pruned. |
| `Site` | [`site`](/generated/api/ref/site) | type | A bare variable reference or one symbolic `index(v, i…)` cell. |
| `stable_json` | [`stable_json`](/generated/api/ref/stable_json) | function | Deterministic JSON: sorted object keys at every level, floats in the reference's layout (`canonical_float`), every other scalar by serde_json. |
| `SysView` | [`sys_view`](/generated/api/ref/sys_view) | type | A model as the band calculus sees it: its variables, equations and index sets. |
| `sysview` | [`sysview`](/generated/api/ref/sysview) | function | View a model of `file` for differentiation. |
