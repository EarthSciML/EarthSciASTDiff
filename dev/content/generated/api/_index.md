---
title: "API reference"
description: "Every stable operation of the three EarthSciASTDiff bindings, side by side."
type: "api"
api_tabs: true
---


EarthSciASTDiff has three bindings — Julia, Python and Rust — that differentiate the same `.esm` models symbolically into structured Jacobian bands. This reference has one page per **stable** operation, with a tab for each binding showing its spelling, signatures and documentation, taken from the binding's own documentation tooling, along with runnable examples. The examples run in CI, so what is shown here is what the code does.

The surface has two tiers. **Stable** names are harmonized across bindings and break only at a major version. **Extension** names are a binding's own documented seams; they may differ between bindings and may change in a minor release, and each binding lists them on one page.

## Guides

- [Getting started](/generated/api/guides/getting-started) — Differentiate a model, read its sparsity pattern and structure, evaluate the Jacobian, and serialize it as a document block.

## Topics

- [Differentiate a model](/generated/api/topics/differentiate) (14) — Take a model or flattened system, differentiate its right-hand side, and get back structured Jacobian bands.
- [Sparsity and structure](/generated/api/topics/structure) (6) — The exact structural sparsity pattern, the structure classification, and the symbolic factorization plan.
- [Evaluate the Jacobian](/generated/api/topics/evaluate) (12) — Numeric values at a point, and prepared evaluators and ODE pieces that fill a fixed sparse prototype.
- [Serialize as a document block](/generated/api/topics/serialize) (6) — The `jacobians` document block, so the Jacobian is itself an `.esm` artifact any binding can evaluate.

Or browse [all stable operations A–Z](/generated/api/ref).

## Bindings

| Binding | Package | Version | Docs from | Stable | Extension |
|---|---|---|---|---|---|
| [Julia](/generated/api/bindings/julia) | `EarthSciASTDiff` | 0.1.0 | Base.Docs (Julia 1.13.3) | 35 | [0](/generated/api/extension/julia) |
| [Python](/generated/api/bindings/python) | `earthsci_astdiff` | 0.1.0 | griffe 2.3.2 | 33 | [0](/generated/api/extension/python) |
| [Rust](/generated/api/bindings/rust) | `earthsci-astdiff` | 0.1.0 | rustdoc JSON (rustdoc 1.101.0-nightly (21b707e3f 2026-09-30)), format_version 61 | 33 | [0](/generated/api/extension/rust) |

[Documentation coverage](/generated/api/coverage) lists the stable names each binding has not documented yet.
