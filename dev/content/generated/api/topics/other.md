---
title: "Other"
description: "Stable operations not yet placed in a topic."
type: "api"
---


| Operation | Kind | Bindings | Summary |
|---|---|---|---|
| [`jacobian_structure`](/generated/api/ref/jacobian_structure) | function | Julia, Python, Rust | Differentiate every differential equation of the system with respect to every state (`wrt = :states`), every parameter outside the forcing set (`wrt = :parameters`, scalar and… |
| [`serialize_structure_band`](/generated/api/ref/serialize_structure_band) | function | Rust | A structure-only entry (spec §2.1) as JSON: the fields of `serialize_band` without `coef`. |
| [`structure_entry`](/generated/api/ref/structure_entry) | type | Rust | One block of the structure-only Jacobian (spec §2.1): a `JacEntry` without its coefficient. |

[All topics](/generated/api)
