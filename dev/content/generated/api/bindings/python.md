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
| `forcing_names` | [`forcing_names`](/generated/api/ref/forcing_names) | function | The forcing set of a system, sorted by name: the parameters whose cells are the columns of the `wrt="forcing"` axis (esm-jacobian-spec.md §4.3). |
| `hvp` | [`hvp`](/generated/api/ref/hvp) | function | The second-order contraction at a point, as a scatter-add over the second-order band points with no tensor formed: |
| `HVPEvaluator` | [`hvp_evaluator`](/generated/api/ref/hvp_evaluator) | type | A prepared second-order contraction for one axis pair, built once by `prepare_hvp`: the second-order bands (`jacobian_bands2`) and every band point's (row, column, second column)… |
| `JacEntry` | [`jac_entry`](/generated/api/ref/jac_entry) | type | One block of the Jacobian: row variable `u`, column variable `v`, and the `Band` giving the rows it covers, the column-index expressions and the coefficient. |
| `JacEntry2` | [`jac_entry2`](/generated/api/ref/jac_entry2) | type | One second-order band: ∂²(equation for `u`)/∂`v` ∂`w`. |
| `jacobian_bands` | [`jacobian_bands`](/generated/api/ref/jacobian_bands) | function | Differentiate a model's right-hand side into a list of Jacobian bands. |
| `jacobian_bands2` | [`jacobian_bands2`](/generated/api/ref/jacobian_bands2) | function | Second-order bands for the axis pair `wrt = (a1, a2)`, `a1` in `("states", "time")` and `a2` in `("states", "parameters", "forcing")`: every first-order entry of the `a1` axis… |
| `jacobian_block` | [`jacobian_block`](/generated/api/ref/jacobian_block) | function | The §4 `jacobians` block for one model: entries (template-CSE'd by default) and the block-local `expression_templates`; on the state and time axes the structure classification… |
| `jacobian_columns` | [`jacobian_columns`](/generated/api/ref/jacobian_columns) | function | The column space of a `"parameters"` or `"forcing"` block: each column variable with its integer shape (`[]` for a scalar), sorted by name — the `columns` field of the emitted… |
| `jacobian_pattern` | [`jacobian_pattern`](/generated/api/ref/jacobian_pattern) | function | The structural sparsity pattern as sorted (rowname, colname) pairs plus the entries — dependency-light (no scipy). |
| `JacobianEvaluator` | [`jacobian_evaluator`](/generated/api/ref/jacobian_evaluator) | type | A prepared band evaluation for one model and axis, built once by `prepare_jacobian`: the band calculus, the observed definitions, the column space and every band point's (row,… |
| `lu_fill` | [`lu_fill`](/generated/api/ref/lu_fill) | function | Symbolic (no-pivoting) LU: the pattern plus every fill-in entry. |
| `markowitz_ordering` | [`markowitz_ordering`](/generated/api/ref/markowitz_ordering) | function | Greedy Markowitz ordering (1-based), first minimal cost wins. |
| `parse_jacobian_blocks` | [`parse_jacobian_blocks`](/generated/api/ref/parse_jacobian_blocks) | function | `parse_jacobian_block` over a model's `jacobians` entry, which is either one block or a list of blocks with distinct `wrt` (spec §4); returns one `(wrt, entries)` per block, in… |
| `plan_factorization` | [`plan_factorization`](/generated/api/ref/plan_factorization) | function | Symbolic factorization info for the detected structure. |
| `prepare_hvp` | [`prepare_hvp`](/generated/api/ref/prepare_hvp) | function | Build an `HVPEvaluator` once — second-order differentiation, column spaces and point placement — for repeated `hvp` calls (one per Rosenbrock step of an adjoint sweep). |
| `prepare_jacobian` | [`prepare_jacobian`](/generated/api/ref/prepare_jacobian) | function | Build a `JacobianEvaluator` once — differentiation, column space and point placement — for repeated evaluation at different points (an adjoint loop calling `vjp_forcing` once per… |
| `serialize_band` | [`serialize_band`](/generated/api/ref/serialize_band) | function | One entry of the `jacobians` block, as a JSON-ready dict (spec §4; a `JacEntry2` adds `col2` / `col2_idx`, §4.4). |
| `simplify_branches` | [`simplify_branches`](/generated/api/ref/simplify_branches) | function | Literal-condition fold, equal-branch collapse, same-condition pruning. |
| `Site` | [`site`](/generated/api/ref/site) | type | A bare variable reference or one symbolic `index(v, i…)` cell. |
| `stable_json` | [`stable_json`](/generated/api/ref/stable_json) | function | Deterministic JSON: sorted object keys at every level, floats in the reference's layout (`canonical_float`) — emit.jl's `stable_json`. |
| `SysView` | [`sys_view`](/generated/api/ref/sys_view) | type | A model or flattened system as the band calculus sees it: its variables, equations and index sets, with subsystems already expanded. |
| `sysview` | [`sysview`](/generated/api/ref/sysview) | function | View a model of an `EsmFile`, or a `FlattenedSystem`, for differentiation. |
| `vjp` | [`vjp`](/generated/api/ref/vjp) | function | The transposed product `g += Jᵀ λ` at a point, as a scatter-add over the band points: `g[col] += coef · λ[row]`, with no matrix formed. |
| `vjp_forcing` | [`vjp_forcing`](/generated/api/ref/vjp_forcing) | function | `vjp` on the forcing axis: the forcing cotangent of one stage evaluation, `g_B += (∂f/∂B)ᵀ λ` (spec §4.3). |
