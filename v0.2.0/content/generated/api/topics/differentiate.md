---
title: "Differentiate a model"
description: "Take a model or flattened system, differentiate its right-hand side, and get back structured Jacobian bands."
type: "api"
---


| Operation | Kind | Bindings | Summary |
|---|---|---|---|
| [`dscalar`](/generated/api/ref/dscalar) | function | Julia, Python, Rust | Derivative of the scalar expression `e` with respect to site `s`, as a new ESM expression. |
| [`extended_bands`](/generated/api/ref/extended_bands) | function | Python, Rust | The extended first-order system of axis `wrt` (spec §4.5, `ExtendedBands`): every state equation's bands with respect to the axis's targets and the active recurrences, and each… |
| [`forcing_names`](/generated/api/ref/forcing_names) | function | Julia, Python, Rust | The forcing set of a system, sorted by name: the parameters whose cells are the columns of the `wrt = :forcing` axis (esm-jacobian-spec.md §4.3). |
| [`jacobian_bands`](/generated/api/ref/jacobian_bands) | function | Julia, Python, Rust | Differentiate every differential equation of the system with respect to every state (`wrt = :states`), every parameter outside the forcing set (`wrt = :parameters`, scalar and… |
| [`jacobian_bands2`](/generated/api/ref/jacobian_bands2) | function | Julia, Python, Rust | Second-order bands for the axis pair `wrt = (a1, a2)`, `a1 ∈ (:states, :time)` and `a2 ∈ (:states, :parameters, :forcing)`: every first-order entry of the `a1` axis… |
| [`jacobian_columns`](/generated/api/ref/jacobian_columns) | function | Julia, Python, Rust | The column space of a `:parameters` or `:forcing` block: each column variable with its integer shape (`Int[]` for a scalar), sorted by name — the `columns` field of the emitted… |
| [`recurrences`](/generated/api/ref/recurrences) | function | Python, Rust | Every recurrence of the view, in equation order: each observed whose definition reads itself, with its frame and axis read off the definition. |
| [`simplify_branches`](/generated/api/ref/simplify_branches) | function | Julia, Python, Rust | Conditional-specific simplification, complementing `EarthSciAST.simplify`: |
| [`sysview`](/generated/api/ref/sysview) | function | Julia, Python, Rust | View a model of an `EsmFile`, or a `FlattenedSystem`, for differentiation. |
| [`transfer_definition`](/generated/api/ref/transfer_definition) | function | Python, Rust | The transfer recurrence `T` of `rec` (spec §4.5): over the frame plus one output index `j` on the axis, `T[o, k, j] = [k == j] + Σ_{self bands} coef · T[o, k − lag, j]`, so `T[o,… |
| [`band`](/generated/api/ref/band) | type | Julia, Python, Rust | One structured block of `∂(rhs of the equation for u)/∂v`: |
| [`jac_entry`](/generated/api/ref/jac_entry) | type | Julia, Python, Rust | One band of the model Jacobian: `∂(equation for state `u`)/∂`v`` with its `Band`. |
| [`jac_entry2`](/generated/api/ref/jac_entry2) | type | Julia, Python, Rust | One second-order band: `∂²(equation for u)/∂v ∂w`. |
| [`site`](/generated/api/ref/site) | type | Julia, Python, Rust | The thing a scalar derivative is taken with respect to: a bare `VarExpr` (scalar variable) or an `index(v, i…)` node (one symbolic cell of an array variable). |
| [`sys_view`](/generated/api/ref/sys_view) | type | Julia, Python, Rust | Uniform system view: `variables` (name → `ModelVariable`), `equations`, and the document's `index_sets`. |
| [`band_error`](/generated/api/ref/band_error) | error | Julia, Python, Rust | An expression has a form the band calculus cannot differentiate (for example a filtered `faq` node or a non-`"+"` semiring). |
| [`derivative_rule_error`](/generated/api/ref/derivative_rule_error) | error | Julia, Python, Rust | A scalar operator has no derivative rule (spec §3). |

[All topics](/generated/api)
