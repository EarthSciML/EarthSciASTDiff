---
title: "Python API index"
description: "Every stable name Python exports, and its extension seams."
type: "api"
---


**Package:** `earthsci_astdiff` 0.1.0

Documentation extracted with griffe 2.3.2.

Extension-tier names are on the [Python extension page](/generated/api/extension/python).

## Stable names

| Python | Canonical | Kind | Summary |
|---|---|---|---|
| `assemble_jacobian` | [`assemble_jacobian`](/generated/api/ref/assemble_jacobian) | function | Evaluate the band entries at a point: `{(rowname, colname): value}` with contributions to one (row, col) pair SUMMED (§4 additivity, incl. |
| `Band` | [`band`](/generated/api/ref/band) | type | One structured block of ∂(rhs of the equation for u)/∂v — see the Julia reference docstring and esm-jacobian-spec.md §4. |
| `BandError` | [`band_error`](/generated/api/ref/band_error) | error | Raised when an expression has a form the band calculus cannot differentiate (for example a filtered `faq` node or a non-`"+"` semiring). |
| `BlockDiagonalPlan` | [`block_diagonal_plan`](/generated/api/ref/block_diagonal_plan) | type | Per-cell block LU of a block-diagonal Jacobian: the block variables in Markowitz order, the block pattern, the 1-based ordering and the pattern of the LU factors including fill-in. |
| `cse_templates` | [`cse_templates`](/generated/api/ref/cse_templates) | function | Greedy largest-first extraction of repeated subtrees. |
| `DerivativeRuleError` | [`derivative_rule_error`](/generated/api/ref/derivative_rule_error) | error | Raised when a scalar operator has no derivative rule (spec §3). |
| `detect_structure` | [`detect_structure`](/generated/api/ref/detect_structure) | function | Classify the band list: `"block_diagonal"`, `"banded"`, `"general"` (or `"empty"` / `"other"`), read off the column-index expressions without evaluating anything. |
| `dscalar` | [`dscalar`](/generated/api/ref/dscalar) | function | Derivative of the scalar expression `e` w.r.t. |
| `JacEntry` | [`jac_entry`](/generated/api/ref/jac_entry) | type | One block of the Jacobian: row variable `u`, column variable `v`, and the `Band` giving the rows it covers, the column-index expressions and the coefficient. |
| `jacobian_bands` | [`jacobian_bands`](/generated/api/ref/jacobian_bands) | function | Differentiate a model's right-hand side into a list of Jacobian bands. |
| `jacobian_block` | [`jacobian_block`](/generated/api/ref/jacobian_block) | function | The §4 `jacobians` block for one model: entries (template-CSE'd by default), the structure classification, and the block-local `expression_templates`, plus the `factorization`… |
| `jacobian_pattern` | [`jacobian_pattern`](/generated/api/ref/jacobian_pattern) | function | The structural sparsity pattern as sorted (rowname, colname) pairs plus the entries — dependency-light (no scipy). |
| `lu_fill` | [`lu_fill`](/generated/api/ref/lu_fill) | function | Symbolic (no-pivoting) LU: the pattern plus every fill-in entry. |
| `markowitz_ordering` | [`markowitz_ordering`](/generated/api/ref/markowitz_ordering) | function | Greedy Markowitz ordering (1-based), first minimal cost wins. |
| `plan_factorization` | [`plan_factorization`](/generated/api/ref/plan_factorization) | function | Symbolic factorization info for the detected structure. |
| `serialize_band` | [`serialize_band`](/generated/api/ref/serialize_band) | function | One entry of the `jacobians` block, as a JSON-ready dict (spec §4). |
| `simplify_branches` | [`simplify_branches`](/generated/api/ref/simplify_branches) | function | Literal-condition fold, equal-branch collapse, same-condition pruning. |
| `Site` | [`site`](/generated/api/ref/site) | type | A bare variable reference or one symbolic `index(v, i…)` cell. |
| `stable_json` | [`stable_json`](/generated/api/ref/stable_json) | function | Deterministic JSON: sorted object keys at every level, floats in the reference's layout (`canonical_float`) — emit.jl's `stable_json`. |
| `SysView` | [`sys_view`](/generated/api/ref/sys_view) | type | A model or flattened system as the band calculus sees it: its variables, equations and index sets, with subsystems already expanded. |
| `sysview` | [`sysview`](/generated/api/ref/sysview) | function | View a model of an `EsmFile`, or a `FlattenedSystem`, for differentiation. |
