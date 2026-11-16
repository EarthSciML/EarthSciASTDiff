---
title: "EarthSciASTDiff"
description: "Analytical sparse Jacobians for EarthSciAST (.esm) models by symbolic differentiation of the expression AST."
---

EarthSciASTDiff computes the analytical Jacobian of an
[EarthSciAST](https://earthsci.dev/ast/) (`.esm`) model's ODE right-hand side: `∂f/∂u` with respect
to the states, `∂f/∂p` with respect to the parameters, and `∂f/∂t`. The result
is a list of structured **bands** — rectangular index regions with symbolic
column-index expressions and ESM coefficient expressions — from which it derives,
without any numeric probing:

- the exact **structural sparsity pattern**, valid at every point;
- a **structure classification** (block-diagonal, banded or general) for
  choosing a matrix type;
- optional **symbolic factorization info**, such as per-cell block LU with
  Markowitz ordering and fill-in;
- a serialized **`jacobians` document block**, so the Jacobian is itself an ESM
  artifact any language binding can evaluate;
- a prepared evaluator that fills a fixed sparse prototype in place.

The derivative is taken on the same expanded, lowered tree that EarthSciAST
compiles, and the test suite pins the assembled Jacobian against ForwardDiff
through the tree-walk right-hand side to machine precision.

## Where to go

- [Getting started](/generated/api/guides/getting-started/): differentiate a
  model, read its sparsity and structure, and evaluate the Jacobian, in Julia,
  Python and Rust.
- [API reference](/generated/api/): every operation, side by side across the
  three bindings.
- [Specification](/generated/spec/): the `jacobians` block format and the
  derivative semantics of every operator (a draft, to be merged into the
  EarthSciAST format specification).

## Quick start (Julia)

```julia
using EarthSciAST, EarthSciASTDiff

file = EarthSciAST.load_path("model.esm")

jac = prepare_jacobian(file)             # differentiate + compile, once
J = jac_prototype(jac)                   # SparseMatrixCSC on the fixed pattern
jac(J, u, p, t)                          # fill J in place, any (u, p, t)

jac.structure                            # :block_diagonal | :banded | :general
```

Source and issue tracker:
[github.com/EarthSciML/EarthSciASTDiff](https://github.com/EarthSciML/EarthSciASTDiff).
