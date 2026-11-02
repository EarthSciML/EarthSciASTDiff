---
title: "Evaluate the Jacobian"
description: "The Jacobian and its products as documents an EarthSciAST evaluator runs, the storage map that places the band values."
type: "api"
---


| Operation | Kind | Bindings | Summary |
|---|---|---|---|
| [`bind_plan`](/generated/api/ref/bind_plan) | function | Julia, Python | Check `plan` against the state Jacobian of `input` (an `EsmFile` or a `FlattenedSystem`) and precompute its gather. |
| [`evaluation_document`](/generated/api/ref/evaluation_document) | function | Julia, Python, Rust | The derived evaluation document of the first-order bands on axis `wrt` (esm-jacobian-spec.md §8, model `"JacobianEval"`): the source's observed the coefficients read, every… |
| [`hvp_document`](/generated/api/ref/hvp_document) | function | Julia, Python, Rust | The second-order product document of esm-jacobian-spec.md §9.4 for the axis pair `wrt` and `stages` stage pairs: its right-hand side leaves `g = ∇_{a2} Σᵢ μᵢᵀ J(u) wᵢ` (`∇_{a2}… |
| [`jac_prototype`](/generated/api/ref/jac_prototype) | function | Julia | A freshly zeroed sparse prototype on the structural pattern — the object to pass as `ODEFunction(f!; jac_prototype = …)`. |
| [`jvp_document`](/generated/api/ref/jvp_document) | function | Julia, Python, Rust | The product documents of esm-jacobian-spec.md §9.2 / §9.3 for one first-order axis: a derived ESM document (model `"JvpEval"` / `"VjpEval"`) whose right-hand side, evaluated by… |
| [`scatter_plan`](/generated/api/ref/scatter_plan-function) | function | Julia, Python | Build the scatter plan of a model's state Jacobian for `layout`. |
| [`vjp_document`](/generated/api/ref/vjp_document) | function | Julia, Python, Rust | The product documents of esm-jacobian-spec.md §9.2 / §9.3 for one first-order axis: a derived ESM document (model `"JvpEval"` / `"VjpEval"`) whose right-hand side, evaluated by… |
| [`scatter_plan`](/generated/api/ref/scatter_plan-type) | type | Julia, Python, Rust | Band fields → storage positions for one (band list, layout). |

[All topics](/generated/api)
