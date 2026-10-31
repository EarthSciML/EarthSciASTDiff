---
title: "prepare_hvp"
description: "Stable function · Evaluate the Jacobian"
type: "api"
api_tabs: true
---


**Tier:** stable · **Kind:** function · **Topic:** [Evaluate the Jacobian](/generated/api/topics/evaluate)

| Julia | Python | Rust |
|---|---|---|
| `prepare_hvp` | `prepare_hvp` | `prepare_hvp` |


<div class="api-tabs">

<div class="api-tab" data-lang="julia" data-label="Julia">
<p class="api-tab-label">Julia</p>

```julia
prepare_hvp(input; wrt, model_name, forcing, build_kwargs, source_layout, eval_cse, eval_cse_min_nodes)
```

```julia
prepare_hvp(input; wrt = (:states, :states), model_name = nothing, forcing = nothing,
            build_kwargs = NamedTuple(), source_layout = nothing) -> HVPEvaluator
```

Build the second-order contraction once ([`jacobian_bands2`](/generated/api/ref/jacobian_bands2)): each
second-order coefficient field becomes a state of the derived evaluation model
(the same mechanism as [`prepare_jacobian`](/generated/api/ref/prepare_jacobian), whose keywords and
`source_layout` contract this shares), and a precomputed
`(field slot, row, col, col2)` list drives the contraction. A second column
outside its variable's extent is a ghost read on the state and parameter
axes and an error on the forcing axis, as for the first-order columns.

[Source](https://github.com/EarthSciML/EarthSciASTDiff/blob/a9ca79cb254eab4630ba5d3dc17b785c72fcfc79/pkg/EarthSciASTDiff.jl/src/second_order.jl#L216)

</div>

<div class="api-tab" data-lang="python" data-label="Python">
<p class="api-tab-label">Python</p>

```python
prepare_hvp(obj, model_name: str | None = None, wrt=('states', 'states'), forcing=None) -> HVPEvaluator
```

Build an [`HVPEvaluator`](/generated/api/ref/hvp_evaluator) once — second-order differentiation,
column spaces and point placement — for repeated [`hvp`](/generated/api/ref/hvp) calls (one
per Rosenbrock step of an adjoint sweep).

[Source](https://github.com/EarthSciML/EarthSciASTDiff/blob/a9ca79cb254eab4630ba5d3dc17b785c72fcfc79/pkg/earthsci-astdiff-py/src/earthsci_astdiff/numeric.py#L433)

</div>

<div class="api-tab" data-lang="rust" data-label="Rust">
<p class="api-tab-label">Rust</p>

```rust
pub fn prepare_hvp(file: &EsmFile, model_name: Option<&str>, wrt: (Wrt, Wrt), forcing: Option<&[String]>) -> Result<HvpEvaluator, BandError>
```

Build an [`HvpEvaluator`](/generated/api/ref/hvp_evaluator) once for the axis pair `wrt` (first axis states
or time, second states, parameters or forcing).

[Source](https://github.com/EarthSciML/EarthSciASTDiff/blob/a9ca79cb254eab4630ba5d3dc17b785c72fcfc79/pkg/earthsci-astdiff-rs/src/numeric.rs#L710) · [Rust docs](https://docs.rs/earthsci-astdiff/latest/earthsci_astdiff/numeric/fn.prepare_hvp.html)

</div>

</div>
