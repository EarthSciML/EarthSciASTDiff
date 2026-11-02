---
title: "Evaluate the Jacobian"
description: "Numeric values at a point, and prepared evaluators and ODE pieces that fill a fixed sparse prototype."
type: "api"
---


| Operation | Kind | Bindings | Summary |
|---|---|---|---|
| [`assemble_jacobian`](/generated/api/ref/assemble_jacobian) | function | Julia, Python, Rust | One-shot convenience over `prepare_jacobian`: evaluate the analytical Jacobian at a point (defaults: the document's initial state and parameters). |
| [`jac_prototype`](/generated/api/ref/jac_prototype) | function | Julia | A freshly zeroed sparse prototype on the structural pattern — the object to pass as `ODEFunction(f!; jac_prototype = …)`. |
| [`ode_components`](/generated/api/ref/ode_components) | function | Julia | Everything a stiff ODE solver needs, with no solver dependency: |
| [`prepare_jacobian`](/generated/api/ref/prepare_jacobian) | function | Julia | Build the analytical Jacobian evaluator once: differentiates the system, compiles the derived band-field model through the ordinary `EarthSciAST.build_evaluator`, and precomputes… |
| [`scatter_plan`](/generated/api/ref/scatter_plan-function) | function | Julia | Build the scatter plan of a model's state Jacobian for `layout`. |
| [`jacobian_evaluator`](/generated/api/ref/jacobian_evaluator) | type | Julia | Prepared analytical Jacobian: call as `jac!(J, u, p, t)` where `J` is (a copy of) `jac.prototype`. |
| [`scatter_plan`](/generated/api/ref/scatter_plan-type) | type | Julia, Rust | Band fields → storage positions for one (band list, layout). |

[All topics](/generated/api)
