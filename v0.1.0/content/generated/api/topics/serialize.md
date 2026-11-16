---
title: "Serialize as a document block"
description: "The `jacobians` document block, so the Jacobian is itself an `.esm` artifact any binding can evaluate."
type: "api"
---


| Operation | Kind | Bindings | Summary |
|---|---|---|---|
| [`cse_templates`](/generated/api/ref/cse_templates) | function | Julia, Python, Rust | Greedy largest-first extraction of repeated subtrees over `exprs`. |
| [`jacobian_block`](/generated/api/ref/jacobian_block) | function | Python, Rust | The §4 `jacobians` block for one model: entries (template-CSE'd by default), the structure classification, and the block-local `expression_templates`, plus the `factorization`… |
| [`jacobian_document`](/generated/api/ref/jacobian_document) | function | Julia | The serialized input document plus the draft `"jacobians"` top-level block (esm-jacobian-spec.md §4): per model, the differentiation axis, the band entries, the detected… |
| [`serialize_band`](/generated/api/ref/serialize_band) | function | Python, Rust | One entry of the `jacobians` block, as a JSON-ready dict (spec §4). |
| [`stable_json`](/generated/api/ref/stable_json) | function | Julia, Python, Rust | Deterministic JSON: object keys emitted in sorted order at every level, scalars formatted by JSON3. |

[All topics](/generated/api)
