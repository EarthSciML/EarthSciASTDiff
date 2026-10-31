---
title: "vjp_forcing"
description: "Stable function · Evaluate the Jacobian"
type: "api"
api_tabs: true
---


**Tier:** stable · **Kind:** function · **Topic:** [Evaluate the Jacobian](/generated/api/topics/evaluate)

| Julia | Python | Rust |
|---|---|---|
| `vjp_forcing!` | `vjp_forcing` | `vjp_forcing` |


<div class="api-tabs">

<div class="api-tab" data-lang="julia" data-label="Julia">
<p class="api-tab-label">Julia</p>

```julia
vjp_forcing!(ḡ::AbstractVector, jac::JacobianEvaluator, λ, u, p, t::Real)
vjp_forcing!(ḡ::AbstractDict, jac::JacobianEvaluator, λ, u, p, t::Real)
```

```julia
vjp_forcing!(ḡ, jac::JacobianEvaluator, λ, u, p, t) -> ḡ
```

[`vjp!`](/generated/api/ref/vjp) on a `wrt = :forcing` evaluator: the forcing cotangent of one
stage evaluation, `ḡ_B += (∂f/∂B)ᵀ λ` (esm-jacobian-spec.md §4.3). `ḡ` is
either a flat vector over the canonical column ordinal (`jac.colnames`) or a
`Dict` of buffer-shaped accumulators keyed by buffer name (one per
`jac.columns` entry, each the buffer's shape). It accumulates; which dataset
record a buffer slot holds, and when to flush, is the caller's bookkeeping.

[Source](https://github.com/EarthSciML/EarthSciASTDiff/blob/a9ca79cb254eab4630ba5d3dc17b785c72fcfc79/pkg/EarthSciASTDiff.jl/src/assemble.jl#L443)

</div>

<div class="api-tab" data-lang="python" data-label="Python">
<p class="api-tab-label">Python</p>

```python
vjp_forcing(g, obj, lam: dict, model_name: str | None = None, states: dict | None = None, params: dict | None = None, t: float = 0.0, forcing=None)
```

[`vjp`](/generated/api/ref/vjp) on the forcing axis: the forcing cotangent of one stage
evaluation, `g_B += (∂f/∂B)ᵀ λ` (spec §4.3). `obj` is a
`wrt="forcing"` [`JacobianEvaluator`](/generated/api/ref/jacobian_evaluator) — build it once with
[`prepare_jacobian`](/generated/api/ref/prepare_jacobian) and reuse it across stages — or anything
[`assemble_jacobian`](/generated/api/ref/assemble_jacobian) takes. `g` is either a flat sequence over the
canonical column ordinal or a dict of buffer-shaped numpy accumulators
keyed by buffer name (one per forcing buffer, each of the buffer's shape,
indexed by cell). It accumulates; which dataset record a buffer slot
holds is the caller's bookkeeping.

[Source](https://github.com/EarthSciML/EarthSciASTDiff/blob/a9ca79cb254eab4630ba5d3dc17b785c72fcfc79/pkg/earthsci-astdiff-py/src/earthsci_astdiff/numeric.py#L310)

</div>

<div class="api-tab" data-lang="rust" data-label="Rust">
<p class="api-tab-label">Rust</p>

```rust
pub fn vjp_forcing(file: &EsmFile, model_name: Option<&str>, forcing: Option<&[String]>, env: &Env, lambda: &HashMap<String, f64>, gbar: &mut HashMap<String, Vec<f64>>) -> Result<(), BandError>
```

[`vjp`](/generated/api/ref/vjp) on the forcing axis, into buffer-shaped accumulators: the forcing
cotangent of one stage evaluation, `ḡ_B += (∂f/∂B)ᵀ λ` (spec §4.3). `gbar`
holds one accumulator per forcing buffer, keyed by buffer name, each the
buffer's cells COLUMN-MAJOR (first index fastest). It accumulates; which
dataset record a buffer slot holds, and when to flush, is the caller's
bookkeeping. This prepares the evaluator for one call; an adjoint loop
calls `JacobianEvaluator::vjp_forcing` on one [`prepare_jacobian`](/generated/api/ref/prepare_jacobian)
instead.

[Source](https://github.com/EarthSciML/EarthSciASTDiff/blob/a9ca79cb254eab4630ba5d3dc17b785c72fcfc79/pkg/earthsci-astdiff-rs/src/numeric.rs#L625) · [Rust docs](https://docs.rs/earthsci-astdiff/latest/earthsci_astdiff/numeric/fn.vjp_forcing.html)

</div>

</div>
