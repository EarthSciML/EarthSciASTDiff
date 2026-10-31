---
title: "ode_components"
description: "Stable function · Evaluate the Jacobian"
type: "api"
api_tabs: true
---


**Tier:** stable · **Kind:** function · **Topic:** [Evaluate the Jacobian](/generated/api/topics/evaluate)

| Julia | Python | Rust |
|---|---|---|
| `ode_components` | not exported | not exported |


<div class="api-tabs">

<div class="api-tab" data-lang="julia" data-label="Julia">
<p class="api-tab-label">Julia</p>

```julia
ode_components(input; model_name, build_kwargs)
```

```julia
ode_components(input; model_name = nothing, build_kwargs = NamedTuple())
    -> (f!, jac!, jac_prototype, tgrad!, u0, p, var_map, jacobian)
```

Everything a stiff ODE solver needs, with no solver dependency:

  - `f!` — the tree-walk RHS, `f!(du, u, p, t)`.
  - `jac!` — the prepared analytical Jacobian, `jac!(J, u, p, t)`, filling
    the nonzeros of a matrix shaped like `jac_prototype` in place.
  - `jac_prototype::SparseMatrixCSC{Float64}` — the fixed structural
    (point-independent) pattern with stored zeros, ready to hand to a solver.
  - `tgrad!` — the analytical time gradient, `tgrad!(dT, u, p, t)` filling
    `∂f/∂t` in place (`wrt = :time` through the same band calculus). For a
    system whose RHS never reads `t` this is a zero fill with no second
    evaluator built.
  - `u0`, `p` — the document's initial state and parameter values.
  - `var_map` — state-element name → index in `u`.
  - `jacobian::JacobianEvaluator` — the underlying prepared evaluator
    (pattern, band entries, detected `structure`, …).

`input` is an `EsmFile` (with `model_name` when it holds several models) or
a `FlattenedSystem`. Always `wrt = :states` — that is the Jacobian an ODE
solver consumes. Wrap the pieces yourself for any solver interface, or load
SciMLBase and call `odefunction` / `odeproblem`.

[Source](https://github.com/EarthSciML/EarthSciASTDiff/blob/a9ca79cb254eab4630ba5d3dc17b785c72fcfc79/pkg/EarthSciASTDiff.jl/src/ode.jl#L9)

</div>

<div class="api-tab" data-lang="python" data-label="Python">
<p class="api-tab-label">Python</p>

<p class="api-absent">Python does not export this function.</p>

</div>

<div class="api-tab" data-lang="rust" data-label="Rust">
<p class="api-tab-label">Rust</p>

<p class="api-absent">Rust does not export this function.</p>

</div>

</div>
