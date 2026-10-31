---
title: "Evaluate the Jacobian"
description: "Numeric values at a point, and prepared evaluators and ODE pieces that fill a fixed sparse prototype."
type: "api"
---


| Operation | Kind | Bindings | Summary |
|---|---|---|---|
| [`assemble_jacobian`](/generated/api/ref/assemble_jacobian) | function | Julia, Python, Rust | One-shot convenience over `prepare_jacobian`: evaluate the analytical Jacobian at a point (defaults: the document's initial state and parameters). |
| [`hvp`](/generated/api/ref/hvp) | function | Julia, Python, Rust | The second-order contraction at `(u, p, t)`: |
| [`jac_prototype`](/generated/api/ref/jac_prototype) | function | Julia | A freshly zeroed sparse prototype on the structural pattern — the object to pass as `ODEFunction(f!; jac_prototype = …)`. |
| [`ode_components`](/generated/api/ref/ode_components) | function | Julia | Everything a stiff ODE solver needs, with no solver dependency: |
| [`prepare_hvp`](/generated/api/ref/prepare_hvp) | function | Julia, Python, Rust | Build the second-order contraction once (`jacobian_bands2`): each second-order coefficient field becomes a state of the derived evaluation model (the same mechanism as… |
| [`prepare_jacobian`](/generated/api/ref/prepare_jacobian) | function | Julia, Python, Rust | Build the analytical Jacobian evaluator once: differentiates the system, compiles the derived band-field model through the ordinary `EarthSciAST.build_evaluator`, and precomputes… |
| [`scatter_plan`](/generated/api/ref/scatter_plan-function) | function | Julia | Build the scatter plan of a model's state Jacobian for `layout`. |
| [`vjp`](/generated/api/ref/vjp) | function | Julia, Python, Rust | The transposed product `ḡ += Jᵀ λ` at `(u, p, t)`, as a scatter-add over the band points: `ḡ[col] += coef · λ[row]` for every point, with no matrix (and no transpose) formed. |
| [`vjp_forcing`](/generated/api/ref/vjp_forcing) | function | Julia, Python, Rust | `vjp!` on a `wrt = :forcing` evaluator: the forcing cotangent of one stage evaluation, `ḡ_B += (∂f/∂B)ᵀ λ` (esm-jacobian-spec.md §4.3). |
| [`hvp_evaluator`](/generated/api/ref/hvp_evaluator) | type | Julia, Python, Rust | Prepared second-order contraction for one axis pair: call `hvp!``(g, hv, pairs, u, p, t)`. |
| [`jacobian_evaluator`](/generated/api/ref/jacobian_evaluator) | type | Julia, Python, Rust | Prepared analytical Jacobian: call as `jac!(J, u, p, t)` where `J` is (a copy of) `jac.prototype`. |
| [`scatter_plan`](/generated/api/ref/scatter_plan-type) | type | Julia, Rust | Band fields → storage positions for one (band list, layout). |

[All topics](/generated/api)
