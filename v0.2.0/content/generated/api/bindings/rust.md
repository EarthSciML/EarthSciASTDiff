---
title: "Rust API index"
description: "Every stable name Rust exports, and its extension seams."
type: "api"
---


**Package:** `earthsci-astdiff` 0.2.0

Documentation extracted with rustdoc JSON (rustdoc 1.101.0-nightly (21b707e3f 2026-09-30)), format_version 61.

Extension-tier names are on the [Rust extension page](/generated/api/extension/rust).

## Stable names

| Rust | Canonical | Kind | Summary |
|---|---|---|---|
| `Band` | [`band`](/generated/api/ref/band) | type | One rectangular region of a Jacobian block: the rows it covers (`rows`, 1-based inclusive ranges, named by `ridx`), the column-index expressions (`cidx`) and the coefficient… |
| `BandError` | [`band_error`](/generated/api/ref/band_error) | error | An expression has a form the band calculus cannot differentiate (for example a joined `faq` node, the `bool_and_or` semiring, or a `max` over two contracted indices). |
| `BlockLu` | [`block_diagonal_plan`](/generated/api/ref/block_diagonal_plan) | type | The per-cell block LU of a block-diagonal band list (factorization.jl's `BlockDiagonalPlan`): block variables in Markowitz order, the 1-based ordering, and the permuted pattern… |
| `column_lines` | [`column_lines`](/generated/api/ref/column_lines) | function | The per-line structure of a state Jacobian along index set `axis`, or `None` when a coupling leaves its line: every band couples two array states of one common shape that holds… |
| `ColumnLines` | [`column_block_plan`](/generated/api/ref/column_block_plan) | type | The per-line structure of a state Jacobian whose couplings all stay within one line along an index set (structure.jl `column_lines`). |
| `cse_templates` | [`cse_templates`](/generated/api/ref/cse_templates) | function | Greedy largest-first extraction; deterministic (first-encounter order, first-max tie-break) to match the reference byte-for-byte. |
| `DerivativeRuleError` | [`derivative_rule_error`](/generated/api/ref/derivative_rule_error) | error | A scalar operator has no derivative rule (spec §3). |
| `detect_structure` | [`detect_structure`](/generated/api/ref/detect_structure) | function | Classify the band list as `"block_diagonal"`, `"banded"` or `"general"` (or `"empty"` / `"other"`), read off the column-index expressions without evaluating anything. |
| `dscalar` | [`dscalar`](/generated/api/ref/dscalar) | function | The derivative of the scalar expression `e` with respect to the site `s` (a variable or an indexed read), by the §3 rule table. |
| `evaluation_document` | [`evaluation_document`](/generated/api/ref/evaluation_document) | function | The derived evaluation document and its band-state names (`J_1`, …). |
| `extended_bands` | [`extended_bands`](/generated/api/ref/extended_bands) | function | The extended first-order system of axis `wrt` (spec §4.5): every state equation's bands with respect to the axis's targets and the active recurrences, and each active recurrence… |
| `forcing_names` | [`forcing_names`](/generated/api/ref/forcing_names) | function | The forcing set of a system, sorted by name: the parameters whose cells are the columns of the `Wrt::Forcing` axis (spec §4.3). |
| `hvp_document` | [`hvp_document`](/generated/api/ref/hvp_document) | function | The second-order product document of spec §9.4 for the axis pair `wrt`, the second-order `entries` (as `crate::jacobian_bands2` gives them) and `stages` stage pairs: its… |
| `JacEntry` | [`jac_entry`](/generated/api/ref/jac_entry) | type | One block of the Jacobian: row variable `u`, column variable `v`, and the `Band` giving the rows it covers, the column-index expressions and the coefficient. |
| `JacEntry2` | [`jac_entry2`](/generated/api/ref/jac_entry2) | type | One second-order band: `∂²(equation for u)/∂v ∂w`. |
| `jacobian_bands` | [`jacobian_bands`](/generated/api/ref/jacobian_bands) | function | Differentiate a model's right-hand side into Jacobian bands with respect to `wrt` (states, parameters, forcing or time). |
| `jacobian_bands2` | [`jacobian_bands2`](/generated/api/ref/jacobian_bands2) | function | Second-order bands for the axis pair `wrt = (a1, a2)`, `a1` states or time and `a2` states, parameters or forcing: every first-order entry of the `a1` axis… |
| `jacobian_block` | [`jacobian_block`](/generated/api/ref/jacobian_block) | function | The `jacobians` block for one model (spec §4): entries (template-CSE'd when `cse` is set, hoisting subtrees of at least `cse_min_nodes` nodes) and — on the state and time axes —… |
| `jacobian_columns` | [`jacobian_columns`](/generated/api/ref/jacobian_columns) | function | The column space of a parameters or forcing block: each column variable with its integer shape (empty for a scalar), sorted by name — the `columns` field of the emitted block… |
| `jvp_document` | [`jvp_document`](/generated/api/ref/jvp_document) | function | The `J v` product document of spec §9.2 for the first-order `entries` on axis `wrt` (as `crate::jacobian_bands_view` gives them): a derived ESM document (model `"JvpEval"`) whose… |
| `lu_fill` | [`lu_fill`](/generated/api/ref/lu_fill) | function | Symbolic (no-pivoting) LU: the pattern plus every fill-in entry. |
| `markowitz_ordering` | [`markowitz_ordering`](/generated/api/ref/markowitz_ordering) | function | A Markowitz pivot ordering of a block pattern (1-based), minimizing fill-in greedily. |
| `parse_jacobian_blocks` | [`parse_jacobian_blocks`](/generated/api/ref/parse_jacobian_blocks) | function | `parse_jacobian_block` / `parse_jacobian_block2` over a model's `jacobians` entry, which is either one block or an array of blocks with distinct `wrt` (spec §4); one… |
| `plan_factorization` | [`plan_factorization`](/generated/api/ref/plan_factorization) | function | Symbolic factorization info for the band list's structure class: the reference's `plan_factorization(entries, sv; structure)`. |
| `recurrences` | [`recurrences`](/generated/api/ref/recurrences) | function | Every recurrence of the view, in equation order: each observed whose definition reads itself, with its frame and axis read off the definition. |
| `ScatterPlan` | [`scatter_plan`](/generated/api/ref/scatter_plan-type) | type | Band fields → storage positions for one (band list, layout). |
| `serialize_band` | [`serialize_band`](/generated/api/ref/serialize_band) | function | One entry of the `jacobians` block, as JSON (spec §4). |
| `serialize_plan` | [`serialize_plan`](/generated/api/ref/serialize_plan) | function | A factorization plan as the `factorization` field of a `jacobians` block (spec §4): the reference's `serialize_plan`. |
| `simplify_branches` | [`simplify_branches`](/generated/api/ref/simplify_branches) | function | `ifelse`-aware simplification, the swell control for coefficients of limiter-heavy schemes: equal branches collapse and same-condition tests are pruned. |
| `Site` | [`site`](/generated/api/ref/site) | type | A bare variable reference or one symbolic `index(v, i…)` cell. |
| `stable_json` | [`stable_json`](/generated/api/ref/stable_json) | function | Deterministic JSON: sorted object keys at every level, floats in the reference's layout (`canonical_float`), every other scalar by serde_json. |
| `SysView` | [`sys_view`](/generated/api/ref/sys_view) | type | A model as the band calculus sees it: its variables, equations and index sets. |
| `sysview` | [`sysview`](/generated/api/ref/sysview) | function | View a model of `file` for differentiation. |
| `transfer_definition` | [`transfer_definition`](/generated/api/ref/transfer_definition) | function | The transfer recurrence `T` of `rec` (spec §4.5): over the frame plus one output index `j` on the axis, |
| `vjp_document` | [`vjp_document`](/generated/api/ref/vjp_document) | function | The `Jᵀ λ` product document of spec §9.3 (model `"VjpEval"`): as `jvp_document`, with the cotangent inputs `VjpIo::lam` and the outputs `VjpIo::vjp`. |
