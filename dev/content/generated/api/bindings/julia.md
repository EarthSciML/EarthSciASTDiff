---
title: "Julia API index"
description: "Every stable name Julia exports, and its extension seams."
type: "api"
---


**Package:** `EarthSciASTDiff` 0.1.0

Documentation extracted with Base.Docs (Julia 1.13.3).

Extension-tier names are on the [Julia extension page](/generated/api/extension/julia).

## Stable names

| Julia | Canonical | Kind | Summary |
|---|---|---|---|
| `assemble_jacobian` | [`assemble_jacobian`](/generated/api/ref/assemble_jacobian) | function | One-shot convenience over `prepare_jacobian`: evaluate the analytical Jacobian at a point (defaults: the document's initial state and parameters). |
| `Band` | [`band`](/generated/api/ref/band) | type | One structured block of `∂(rhs of the equation for u)/∂v`: |
| `BandError` | [`band_error`](/generated/api/ref/band_error) | error | An expression has a form the band calculus cannot differentiate (for example a filtered `faq` node or a non-`"+"` semiring). |
| `BlockDiagonalPlan` | [`block_diagonal_plan`](/generated/api/ref/block_diagonal_plan) | type | Per-cell block factorization info for a `:block_diagonal` Jacobian (the chemistry shape: every coupling is within-cell, so the global matrix is one identical `ns × ns`-patterned… |
| `cse_templates` | [`cse_templates`](/generated/api/ref/cse_templates) | function | Greedy largest-first extraction of repeated subtrees over `exprs`. |
| `DerivativeRuleError` | [`derivative_rule_error`](/generated/api/ref/derivative_rule_error) | error | A scalar operator has no derivative rule (spec §3). |
| `detect_structure` | [`detect_structure`](/generated/api/ref/detect_structure) | function | Classify the Jacobian's structure from the band list: |
| `dscalar` | [`dscalar`](/generated/api/ref/dscalar) | function | Derivative of the scalar expression `e` with respect to site `s`, as a new ESM expression. |
| `forcing_names` | [`forcing_names`](/generated/api/ref/forcing_names) | function | The forcing set of a system, sorted by name: the parameters whose cells are the columns of the `wrt = :forcing` axis (esm-jacobian-spec.md §4.3). |
| `hvp!` | [`hvp`](/generated/api/ref/hvp) | function | The second-order contraction at `(u, p, t)`: |
| `HVPEvaluator` | [`hvp_evaluator`](/generated/api/ref/hvp_evaluator) | type | Prepared second-order contraction for one axis pair: call `hvp!``(g, hv, pairs, u, p, t)`. |
| `jac_prototype` | [`jac_prototype`](/generated/api/ref/jac_prototype) | function | A freshly zeroed sparse prototype on the structural pattern — the object to pass as `ODEFunction(f!; jac_prototype = …)`. |
| `JacEntry` | [`jac_entry`](/generated/api/ref/jac_entry) | type | One band of the model Jacobian: `∂(equation for state `u`)/∂`v`` with its `Band`. |
| `JacEntry2` | [`jac_entry2`](/generated/api/ref/jac_entry2) | type | One second-order band: `∂²(equation for u)/∂v ∂w`. |
| `jacobian_bands` | [`jacobian_bands`](/generated/api/ref/jacobian_bands) | function | Differentiate every differential equation of the system with respect to every state (`wrt = :states`), every parameter outside the forcing set (`wrt = :parameters`, scalar and… |
| `jacobian_bands2` | [`jacobian_bands2`](/generated/api/ref/jacobian_bands2) | function | Second-order bands for the axis pair `wrt = (a1, a2)`, `a1 ∈ (:states, :time)` and `a2 ∈ (:states, :parameters, :forcing)`: every first-order entry of the `a1` axis… |
| `jacobian_columns` | [`jacobian_columns`](/generated/api/ref/jacobian_columns) | function | The column space of a `:parameters` or `:forcing` block: each column variable with its integer shape (`Int[]` for a scalar), sorted by name — the `columns` field of the emitted… |
| `jacobian_document` | [`jacobian_document`](/generated/api/ref/jacobian_document) | function | The serialized input document plus the draft `"jacobians"` top-level block (esm-jacobian-spec.md §4): per model, the differentiation axis, the band entries, and — on the state… |
| `jacobian_pattern` | [`jacobian_pattern`](/generated/api/ref/jacobian_pattern) | function | The structural (global, point-independent) sparsity pattern of the Jacobian as a `SparseMatrixCSC{Bool}`. |
| `JacobianEvaluator` | [`jacobian_evaluator`](/generated/api/ref/jacobian_evaluator) | type | Prepared analytical Jacobian: call as `jac!(J, u, p, t)` where `J` is (a copy of) `jac.prototype`. |
| `lu_fill` | [`lu_fill`](/generated/api/ref/lu_fill) | function | Symbolic (no-pivoting) LU of a structural pattern: the input pattern plus every fill-in entry Gaussian elimination creates, in natural order. |
| `ode_components` | [`ode_components`](/generated/api/ref/ode_components) | function | Everything a stiff ODE solver needs, with no solver dependency: |
| `parse_jacobian_blocks` | [`parse_jacobian_blocks`](/generated/api/ref/parse_jacobian_blocks) | function | `parse_jacobian_block` over a model's `"jacobians"` entry, which is either one block or an array of blocks with distinct `wrt` (esm-jacobian-spec.md §4); returns one `(wrt,… |
| `plan_factorization` | [`plan_factorization`](/generated/api/ref/plan_factorization) | function | Symbolic factorization info for the detected structure class, or `nothing` when no plan is implemented for that class (the Jacobian itself is still fully usable — a plan is an… |
| `prepare_hvp` | [`prepare_hvp`](/generated/api/ref/prepare_hvp) | function | Build the second-order contraction once (`jacobian_bands2`): each second-order coefficient field becomes a state of the derived evaluation model (the same mechanism as… |
| `prepare_jacobian` | [`prepare_jacobian`](/generated/api/ref/prepare_jacobian) | function | Build the analytical Jacobian evaluator once: differentiates the system, compiles the derived band-field model through the ordinary `EarthSciAST.build_evaluator`, and precomputes… |
| `scatter_plan` | [`scatter_plan`](/generated/api/ref/scatter_plan-function) | function | Build the scatter plan of a model's state Jacobian for `layout`. |
| `ScatterPlan` | [`scatter_plan`](/generated/api/ref/scatter_plan-type) | type | Band fields → storage positions for one (band list, layout). |
| `simplify_branches` | [`simplify_branches`](/generated/api/ref/simplify_branches) | function | Conditional-specific simplification, complementing `EarthSciAST.simplify`: |
| `Site` | [`site`](/generated/api/ref/site) | type | The thing a scalar derivative is taken with respect to: a bare `VarExpr` (scalar variable) or an `index(v, i…)` node (one symbolic cell of an array variable). |
| `stable_json` | [`stable_json`](/generated/api/ref/stable_json) | function | Deterministic JSON: object keys emitted in sorted order at every level, scalars formatted by JSON3. |
| `SysView` | [`sys_view`](/generated/api/ref/sys_view) | type | Uniform system view: `variables` (name → `ModelVariable`), `equations`, and the document's `index_sets`. |
| `sysview` | [`sysview`](/generated/api/ref/sysview) | function |  |
| `vjp!` | [`vjp`](/generated/api/ref/vjp) | function | The transposed product `ḡ += Jᵀ λ` at `(u, p, t)`, as a scatter-add over the band points: `ḡ[col] += coef · λ[row]` for every point, with no matrix (and no transpose) formed. |
| `vjp_forcing!` | [`vjp_forcing`](/generated/api/ref/vjp_forcing) | function | `vjp!` on a `wrt = :forcing` evaluator: the forcing cotangent of one stage evaluation, `ḡ_B += (∂f/∂B)ᵀ λ` (esm-jacobian-spec.md §4.3). |
