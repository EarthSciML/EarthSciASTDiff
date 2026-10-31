---
title: "hvp"
description: "Stable function · Evaluate the Jacobian"
type: "api"
api_tabs: true
---


**Tier:** stable · **Kind:** function · **Topic:** [Evaluate the Jacobian](/generated/api/topics/evaluate)

| Julia | Python | Rust |
|---|---|---|
| `hvp!` | `hvp` | `hvp` |


<div class="api-tabs">

<div class="api-tab" data-lang="julia" data-label="Julia">
<p class="api-tab-label">Julia</p>

```julia
hvp!(g::AbstractVector, hv::HVPEvaluator, pairs, u::AbstractVector, p, t::Real; form)
```

```julia
hvp!(g, hv::HVPEvaluator, pairs, u, p, t; form = :contract) -> g
```

The second-order contraction at `(u, p, t)`:

    g += ∇_{a2} Σᵢ μᵢᵀ J(u) wᵢ          (pair (:states, a2))
    g += ∇_{a2} (Σᵢ wᵢ μᵢ)ᵀ f_t         (pair (:time, a2); each wᵢ a scalar)

with `J = ∂f/∂(a1)`. `pairs` is one `(μ, w)` tuple or a collection of them
(one per Rosenbrock stage); `μ` is indexed like `hv.rownames`, `w` like
`hv.colnames` and `g` like `hv.col2names`. `g` ACCUMULATES — zero it first
for a single product.

`form = :contract` (default) sums the stages into `M(row, col)` once and runs
`g[col2] += coef2 · M(row, col)` over the points. On a `(:states, :states)`
pair `form = :symmetric` uses the symmetry of `∇²(μᵀf)` instead,
`g[col] += μ[row] · coef2 · w[col2]` per stage, which needs no `M`.

[Source](https://github.com/EarthSciML/EarthSciASTDiff/blob/a9ca79cb254eab4630ba5d3dc17b785c72fcfc79/pkg/EarthSciASTDiff.jl/src/second_order.jl#L283)

</div>

<div class="api-tab" data-lang="python" data-label="Python">
<p class="api-tab-label">Python</p>

```python
hvp(g, obj, pairs, model_name: str | None = None, wrt=None, states: dict | None = None, params: dict | None = None, t: float = 0.0, forcing=None, form: str = 'contract')
```

The second-order contraction at a point, as a scatter-add over the
second-order band points with no tensor formed:

```python
g += ∇_{a2} Σᵢ μᵢᵀ J wᵢ          (pair (states, a2))
g += ∇_{a2} (Σᵢ wᵢ μᵢ)ᵀ f_t      (pair (time, a2); each wᵢ a scalar)
```

with `J = ∂f/∂(a1)`. `obj` is an [`HVPEvaluator`](/generated/api/ref/hvp_evaluator) (built once
with [`prepare_hvp`](/generated/api/ref/prepare_hvp); its pair and forcing set apply) or anything
[`prepare_hvp`](/generated/api/ref/prepare_hvp) takes, prepared for this one call with `wrt`
(default `("states", "states")`). `pairs` is one `(μ, w)` tuple or
a collection of them (one per Rosenbrock stage): `μ` maps row (state
cell) names to values, `w` maps column names to values — or is a
number on a time pair. A name a band point needs that they do not bind
raises `ValueError`.

`g` is a dict keyed by second-column name, or (on a `"parameters"` /
`"forcing"` second axis) a flat sequence over `col2names`. It
ACCUMULATES — zero it first for a single product; a dict gains a key for
every second column a band point reaches, zero-valued or not.

`form="contract"` (default) runs `g[col2] += coef2 · Σᵢ μᵢ[row] wᵢ[col]` over the points. On a `("states", "states")` pair
`form="symmetric"` uses the symmetry of ∇²(μᵀf) instead, `g[col] += μ[row] · coef2 · w[col2]` per stage (`g` then keyed by the first
column). Returns `g`.

[Source](https://github.com/EarthSciML/EarthSciASTDiff/blob/a9ca79cb254eab4630ba5d3dc17b785c72fcfc79/pkg/earthsci-astdiff-py/src/earthsci_astdiff/numeric.py#L458)

</div>

<div class="api-tab" data-lang="rust" data-label="Rust">
<p class="api-tab-label">Rust</p>

```rust
pub fn hvp(file: &EsmFile, model_name: Option<&str>, wrt: (Wrt, Wrt), forcing: Option<&[String]>, env: &Env, pairs: &[HvpPair], form: HvpForm, g: &mut BTreeMap<String, f64>) -> Result<(), BandError>
```

The second-order contraction at a point (spec §4.4), with no tensor
formed:

```text
g += ∇_{a2} Σᵢ μᵢᵀ J(u) wᵢ          (pair (states, a2))
g += ∇_{a2} (Σᵢ wᵢ μᵢ)ᵀ f_t         (pair (time, a2); wᵢ binds "t")
```

with `J = ∂f/∂(a1)`. `pairs` holds one `(μ, w)` per Rosenbrock stage,
`μ` keyed by row cell name and `w` by first-column cell name; a name a
point reads that they do not bind is an error. `g` is keyed by
second-column cell name (`"T[2]"`, `"k3"`, `"F[1,4]"`; on
`HvpForm::Symmetric` by first-column name), gains a key for every
column a band point reaches, and ACCUMULATES — clear it first for a
single product. Parameters `env` leaves unbound read the document
defaults. This prepares the evaluator for one call; a loop calls
`HvpEvaluator::hvp` on one [`prepare_hvp`](/generated/api/ref/prepare_hvp) instead.

[Source](https://github.com/EarthSciML/EarthSciASTDiff/blob/a9ca79cb254eab4630ba5d3dc17b785c72fcfc79/pkg/earthsci-astdiff-rs/src/numeric.rs#L858) · [Rust docs](https://docs.rs/earthsci-astdiff/latest/earthsci_astdiff/numeric/fn.hvp.html)

</div>

</div>
