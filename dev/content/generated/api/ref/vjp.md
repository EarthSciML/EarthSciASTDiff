---
title: "vjp"
description: "Stable function · Evaluate the Jacobian"
type: "api"
api_tabs: true
---


**Tier:** stable · **Kind:** function · **Topic:** [Evaluate the Jacobian](/generated/api/topics/evaluate)

| Julia | Python | Rust |
|---|---|---|
| `vjp!` | `vjp` | `vjp` |


<div class="api-tabs">

<div class="api-tab" data-lang="julia" data-label="Julia">
<p class="api-tab-label">Julia</p>

```julia
vjp!(ḡ::AbstractVector, jac::JacobianEvaluator, λ::AbstractVector, u::AbstractVector, p, t::Real)
```

```julia
vjp!(ḡ, jac::JacobianEvaluator, λ, u, p, t) -> ḡ
```

The transposed product `ḡ += Jᵀ λ` at `(u, p, t)`, as a scatter-add over the
band points: `ḡ[col] += coef · λ[row]` for every point, with no matrix (and
no transpose) formed. `λ` is indexed like the rows (`jac.rownames`, the
state vector) and `ḡ` like the columns (`jac.colnames`); `ḡ` ACCUMULATES —
zero it first for a single product.

[Source](https://github.com/EarthSciML/EarthSciASTDiff/blob/a9ca79cb254eab4630ba5d3dc17b785c72fcfc79/pkg/EarthSciASTDiff.jl/src/assemble.jl#L420)

</div>

<div class="api-tab" data-lang="python" data-label="Python">
<p class="api-tab-label">Python</p>

```python
vjp(g, obj, lam: dict, model_name: str | None = None, wrt: str | None = None, states: dict | None = None, params: dict | None = None, t: float = 0.0, forcing=None)
```

The transposed product `g += Jᵀ λ` at a point, as a scatter-add over
the band points: `g[col] += coef · λ[row]`, with no matrix formed.

`obj` is a [`JacobianEvaluator`](/generated/api/ref/jacobian_evaluator) (built once with
[`prepare_jacobian`](/generated/api/ref/prepare_jacobian); its axis and forcing set apply) or anything
[`assemble_jacobian`](/generated/api/ref/assemble_jacobian) takes, prepared for this one call with `wrt`
(default `"forcing"`). `lam` maps row (state cell) names to values; a
row a band point needs that `lam` does not bind raises `ValueError`.
`g` is a flat mutable sequence (list or numpy vector) over the
canonical column ordinal of the `"parameters"` / `"forcing"` axis
([`jacobian_columns`](/generated/api/ref/jacobian_columns), cells column-major) or of `"time"` (`[t]`).
`g` ACCUMULATES — zero it first for a single product. Returns `g`.

[Source](https://github.com/EarthSciML/EarthSciASTDiff/blob/a9ca79cb254eab4630ba5d3dc17b785c72fcfc79/pkg/earthsci-astdiff-py/src/earthsci_astdiff/numeric.py#L273)

</div>

<div class="api-tab" data-lang="rust" data-label="Rust">
<p class="api-tab-label">Rust</p>

```rust
pub fn vjp(file: &EsmFile, model_name: Option<&str>, wrt: Wrt, forcing: Option<&[String]>, env: &Env, lambda: &HashMap<String, f64>, gbar: &mut [f64]) -> Result<(), BandError>
```

The transposed product `ḡ += Jᵀ λ` at a point, as a scatter-add over the
band points: `ḡ[col] += coef · λ[row]` for every point, with no matrix
(and no transpose) formed. `lambda` is keyed by row cell name (`"c[2]"`,
`"S"`); a row it does not bind is an error. `gbar` is indexed by the
canonical column ordinal of a parameters or forcing axis
(`crate::jacobian::column_names` of [`crate::jacobian_columns`](/generated/api/ref/jacobian_columns)) and
ACCUMULATES — zero it first for a single product. This prepares the
evaluator for one call; a loop calls `JacobianEvaluator::vjp` on one
[`prepare_jacobian`](/generated/api/ref/prepare_jacobian) instead.

[Source](https://github.com/EarthSciML/EarthSciASTDiff/blob/a9ca79cb254eab4630ba5d3dc17b785c72fcfc79/pkg/earthsci-astdiff-rs/src/numeric.rs#L605) · [Rust docs](https://docs.rs/earthsci-astdiff/latest/earthsci_astdiff/numeric/fn.vjp.html)

</div>

</div>
