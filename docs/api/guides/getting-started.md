---
title: "Getting started"
description: "Differentiate a model, read its sparsity pattern and structure, emit the Jacobian as documents EarthSciAST evaluates, and serialize it as a document block."
weight: 10
---

EarthSciASTDiff turns an EarthSciAST (`.esm`) model into its analytical sparse
Jacobian by symbolic differentiation. This guide takes one small model through
each step. Each step shows the same task in every binding; pick your language
on any tab and the rest of the site follows. Every snippet is cut from a
program that runs, and checks its own result, in CI.

The examples use `bd_chem.esm` from
[`tests/valid/`](https://github.com/EarthSciML/EarthSciASTDiff/tree/main/tests/valid):
per-cell chemistry among three species `A`, `B` and `C` on a four-cell column,
so the Jacobian is block-diagonal: four blocks of three species.

## Differentiate

The Jacobian is a list of **bands**: each is a rectangular region of the
equation's index space, a symbolic column-index expression, and a coefficient
expression. One entry is produced for each (row variable, column variable)
pair that occurs in the model, so the derivative is exact and does not depend
on the point where you later evaluate it.

<!-- example: bands -->

See [`jacobian_bands`](/generated/api/ref/jacobian_bands) and
[`sysview`](/generated/api/ref/sysview).

## Sparsity pattern

The pattern is read off structural occurrence, never off numeric probing, so it
is valid at every point. Branches of `ifelse`, `min` and `max` are unioned, and
entries that happen to be zero at one point stay in the pattern.

<!-- example: pattern -->

See [`jacobian_pattern`](/generated/api/ref/jacobian_pattern).

## Structure and factorization

The bands are classified as block-diagonal, banded or general, which tells a
solver what matrix type to use. For block-diagonal systems a symbolic
per-cell block LU plan (Markowitz ordering and fill-in) is available too.

<!-- example: structure -->

See [`detect_structure`](/generated/api/ref/detect_structure) and
[`plan_factorization`](/generated/api/ref/plan_factorization).

## Evaluate

EarthSciASTDiff evaluates nothing itself. It emits the Jacobian as derived
ESM documents, and EarthSciAST evaluates them like any other model: the band
fields `J_k` (the evaluation document), and the matrix-free products `J·v`,
`Jᵀ·λ` and the second-order contraction (spec §9).

<!-- example: evaluate -->

See [`evaluation_document`](/generated/api/ref/evaluation_document) and
[`jvp_document`](/generated/api/ref/jvp_document).

## Prototype and storage

To use the Jacobian inside a solver, take the fixed sparse prototype once,
and place the band values in your own storage with a scatter plan that is
checked against the source before use.

<!-- example: prepare -->

See [`jac_prototype`](/generated/api/ref/jac_prototype),
[`scatter_plan`](/generated/api/ref/scatter_plan) and
[`bind_plan`](/generated/api/ref/bind_plan).

## Serialize

The bands can be written into the document itself as a `jacobians` block, so
any binding that can evaluate `.esm` expressions can evaluate the Jacobian.
The block carries the entries, the structure classification and the
factorization plan.

<!-- example: block -->

See [`jacobian_block`](/generated/api/ref/jacobian_block) and
[`jacobian_document`](/generated/api/ref/jacobian_document).
