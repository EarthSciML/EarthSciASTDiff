---
title: "Python API index"
description: "Every stable name Python exports, and its extension seams."
type: "api"
---


**Package:** `earthsci_astdiff` 0.2.0

Documentation extracted with griffe 2.3.2.

Extension-tier names are on the [Python extension page](/generated/api/extension/python).

## Stable names

| Python | Canonical | Kind | Summary |
|---|---|---|---|
| `Band` | [`band`](/generated/api/ref/band) | type | One structured block of ∂(rhs of the equation for u)/∂v — see the Julia reference docstring and esm-jacobian-spec.md §4. |
| `BandError` | [`band_error`](/generated/api/ref/band_error) | error | Raised when an expression has a form the band calculus cannot differentiate (for example a joined `faq` node or a boolean semiring). |
| `bind_plan` | [`bind_plan`](/generated/api/ref/bind_plan) | function | Check `plan` against the state Jacobian of `obj` (an `EsmFile`, a flattened system or a `SysView`) and precompute its gather. |
| `BlockDiagonalPlan` | [`block_diagonal_plan`](/generated/api/ref/block_diagonal_plan) | type | Per-cell block LU of a block-diagonal Jacobian: the block variables in Markowitz order, the block pattern, the 1-based ordering and the pattern of the LU factors including fill-in. |
| `column_lines` | [`column_lines`](/generated/api/ref/column_lines) | function | The per-line structure of a state Jacobian whose couplings all stay within one line along index set `axis`, or `None` when one does not (structure.jl `column_lines`): every band… |
| `ColumnBlockPlan` | [`column_block_plan`](/generated/api/ref/column_block_plan) | type | Line-by-line factorization info for a `"column_block"` Jacobian: every coupling stays within one line along `axis`, so the matrix is one `(len(block_vars)·line_length)`-square… |
| `cse_templates` | [`cse_templates`](/generated/api/ref/cse_templates) | function | Greedy largest-first extraction of repeated subtrees. |
| `DerivativeRuleError` | [`derivative_rule_error`](/generated/api/ref/derivative_rule_error) | error | Raised when a scalar operator has no derivative rule (spec §3). |
| `detect_structure` | [`detect_structure`](/generated/api/ref/detect_structure) | function | Classify the band list: `"block_diagonal"`, `"banded"`, `"general"` (or `"empty"` / `"other"`), read off the column-index expressions without evaluating anything. |
| `dscalar` | [`dscalar`](/generated/api/ref/dscalar) | function | Derivative of the scalar expression `e` w.r.t. |
| `evaluation_document` | [`evaluation_document`](/generated/api/ref/evaluation_document) | function | The derived evaluation document of the first-order bands on axis `wrt` (spec §8, model `"JacobianEval"`): the source's observed the coefficients read, every source derivative… |
| `extended_bands` | [`extended_bands`](/generated/api/ref/extended_bands) | function | The extended first-order system of axis `wrt` (spec §4.5, `ExtendedBands`): every state equation's bands with respect to the axis's targets and the active recurrences, and each… |
| `forcing_names` | [`forcing_names`](/generated/api/ref/forcing_names) | function | The forcing set of a system, sorted by name: the parameters whose cells are the columns of the `wrt="forcing"` axis (esm-jacobian-spec.md §4.3). |
| `hvp_document` | [`hvp_document`](/generated/api/ref/hvp_document) | function | The second-order product document of spec §9.4 for the axis pair `wrt` and `stages` stage pairs: its right-hand side leaves `g = ∇_{a2} Σᵢ μᵢᵀ J(u) wᵢ` (`∇_{a2} (Σᵢ wᵢ μᵢ)ᵀ f_t`… |
| `JacEntry` | [`jac_entry`](/generated/api/ref/jac_entry) | type | One block of the Jacobian: row variable `u`, column variable `v`, and the `Band` giving the rows it covers, the column-index expressions and the coefficient. |
| `JacEntry2` | [`jac_entry2`](/generated/api/ref/jac_entry2) | type | One second-order band: ∂²(equation for `u`)/∂`v` ∂`w`. |
| `jacobian_bands` | [`jacobian_bands`](/generated/api/ref/jacobian_bands) | function | Differentiate a model's right-hand side into a list of Jacobian bands. |
| `jacobian_bands2` | [`jacobian_bands2`](/generated/api/ref/jacobian_bands2) | function | Second-order bands for the axis pair `wrt = (a1, a2)`, `a1` in `("states", "time")` and `a2` in `("states", "parameters", "forcing")`: every first-order entry of the `a1` axis… |
| `jacobian_block` | [`jacobian_block`](/generated/api/ref/jacobian_block) | function | The §4 `jacobians` block for one model: entries (template-CSE'd by default) and the block-local `expression_templates`; on the state and time axes the structure classification… |
| `jacobian_columns` | [`jacobian_columns`](/generated/api/ref/jacobian_columns) | function | The column space of a `"parameters"` or `"forcing"` block: each column variable with its integer shape (`[]` for a scalar), sorted by name — the `columns` field of the emitted… |
| `jacobian_pattern` | [`jacobian_pattern`](/generated/api/ref/jacobian_pattern) | function | The structural sparsity pattern as sorted (rowname, colname) pairs plus the entries — dependency-light (no scipy). |
| `jvp_document` | [`jvp_document`](/generated/api/ref/jvp_document) | function | The `J v` product document of spec §9.2 for one first-order axis. |
| `lu_fill` | [`lu_fill`](/generated/api/ref/lu_fill) | function | Symbolic (no-pivoting) LU: the pattern plus every fill-in entry. |
| `markowitz_ordering` | [`markowitz_ordering`](/generated/api/ref/markowitz_ordering) | function | Greedy Markowitz ordering (1-based), first minimal cost wins. |
| `parse_jacobian_blocks` | [`parse_jacobian_blocks`](/generated/api/ref/parse_jacobian_blocks) | function | `parse_jacobian_block` over a model's `jacobians` entry, which is either one block or a list of blocks with distinct `wrt` (spec §4); returns one `(wrt, entries)` per block, in… |
| `plan_factorization` | [`plan_factorization`](/generated/api/ref/plan_factorization) | function | Symbolic factorization info for the detected structure. |
| `recurrences` | [`recurrences`](/generated/api/ref/recurrences) | function | Every recurrence of the view, in equation order: each observed whose definition reads itself, with its frame and axis read off the definition. |
| `scatter_plan` | [`scatter_plan`](/generated/api/ref/scatter_plan-function) | function | Build the scatter plan of a model's state Jacobian for `layout`. |
| `ScatterPlan` | [`scatter_plan`](/generated/api/ref/scatter_plan-type) | type |  |
| `serialize_band` | [`serialize_band`](/generated/api/ref/serialize_band) | function | One entry of the `jacobians` block, as a JSON-ready dict (spec §4; a `JacEntry2` adds `col2` / `col2_idx`, §4.4). |
| `serialize_plan` | [`serialize_plan`](/generated/api/ref/serialize_plan) | function |  |
| `simplify_branches` | [`simplify_branches`](/generated/api/ref/simplify_branches) | function | Literal-condition fold, equal-branch collapse, same-condition pruning. |
| `Site` | [`site`](/generated/api/ref/site) | type | A bare variable reference or one symbolic `index(v, i…)` cell. |
| `stable_json` | [`stable_json`](/generated/api/ref/stable_json) | function | Deterministic JSON: sorted object keys at every level, floats in the reference's layout (`canonical_float`) — emit.jl's `stable_json`. |
| `SysView` | [`sys_view`](/generated/api/ref/sys_view) | type | A model or flattened system as the band calculus sees it: its variables, equations and index sets, with subsystems already expanded. |
| `sysview` | [`sysview`](/generated/api/ref/sysview) | function | View a model of an `EsmFile`, or a `FlattenedSystem`, for differentiation. |
| `transfer_definition` | [`transfer_definition`](/generated/api/ref/transfer_definition) | function | The transfer recurrence `T` of `rec` (spec §4.5): over the frame plus one output index `j` on the axis, `T[o, k, j] = [k == j] + Σ_{self bands} coef · T[o, k − lag, j]`, so `T[o,… |
| `vjp_document` | [`vjp_document`](/generated/api/ref/vjp_document) | function | The `Jᵀ λ` product document of spec §9.3 for one first-order axis (model `"VjpEval"`), as `jvp_document`: `io["lam"]` maps each state to its adjoint input and `io["vjp"]` each… |
