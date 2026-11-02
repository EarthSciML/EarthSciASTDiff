---
title: "Sparsity and structure"
description: "The exact structural sparsity pattern, the structure classification, and the symbolic factorization plan."
type: "api"
---


| Operation | Kind | Bindings | Summary |
|---|---|---|---|
| [`detect_structure`](/generated/api/ref/detect_structure) | function | Julia, Python, Rust | Classify the Jacobian's structure from the band list: |
| [`jacobian_pattern`](/generated/api/ref/jacobian_pattern) | function | Julia, Python | The structural (global, point-independent) sparsity pattern of the Jacobian as a `SparseMatrixCSC{Bool}`. |
| [`lu_fill`](/generated/api/ref/lu_fill) | function | Julia, Python, Rust | Symbolic (no-pivoting) LU of a structural pattern: the input pattern plus every fill-in entry Gaussian elimination creates, in natural order. |
| [`markowitz_ordering`](/generated/api/ref/markowitz_ordering) | function | Python, Rust | Greedy Markowitz ordering (1-based), first minimal cost wins. |
| [`plan_factorization`](/generated/api/ref/plan_factorization) | function | Julia, Python, Rust | Symbolic factorization info for the detected structure class, or `nothing` when no plan is implemented for that class (the Jacobian itself is still fully usable — a plan is an… |
| [`block_diagonal_plan`](/generated/api/ref/block_diagonal_plan) | type | Julia, Python, Rust | Per-cell block factorization info for a `:block_diagonal` Jacobian (the chemistry shape: every coupling is within-cell, so the global matrix is one identical `ns × ns`-patterned… |

[All topics](/generated/api)
