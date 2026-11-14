---
title: "scatter_plan (type)"
description: "Stable type · Evaluate the Jacobian"
type: "api"
api_tabs: true
---


**Tier:** stable · **Kind:** type · **Topic:** [Evaluate the Jacobian](/generated/api/topics/evaluate)

| Julia | Python | Rust |
|---|---|---|
| `ScatterPlan` | not exported | `ScatterPlan` |


<div class="api-tabs">

<div class="api-tab" data-lang="julia" data-label="Julia">
<p class="api-tab-label">Julia</p>

```julia
abstract type ScatterPlan
```

```julia
ScatterPlan
```

Band fields → storage positions for one (band list, layout). Built by
[`scatter_plan`](/generated/api/ref/scatter_plan-function), serialized by `serialize_scatter_plan`,
bound to an evaluator by `bind_plan`. Every plan carries the
canonical `states` (name, shape) and the `bands` (row var, column var, field
shape) it was built from, which is what binding checks.

[Source](https://github.com/EarthSciML/EarthSciASTDiff/blob/6f151823f210d8951df2266333997c204f54e68e/pkg/EarthSciASTDiff.jl/src/scatter.jl#L75)

</div>

<div class="api-tab" data-lang="python" data-label="Python">
<p class="api-tab-label">Python</p>

<p class="api-absent">Python does not export this type.</p>

</div>

<div class="api-tab" data-lang="rust" data-label="Rust">
<p class="api-tab-label">Rust</p>

```rust
pub struct ScatterPlan {
    pub states: Vec<(String, Vec<i64>)>,
    pub bands: Vec<(String, String, Vec<i64>)>,
    pub kind: PlanKind,
}

impl ScatterPlan { pub fn from_file(file: &EsmFile, model_name: Option<&str>, layout: &Layout) -> Result<ScatterPlan, BandError> }

impl ScatterPlan { pub fn build(sv: &SysView<'_>, entries: &[JacEntry], layout: &Layout) -> Result<ScatterPlan, BandError> }

impl ScatterPlan { pub fn to_json(&self) -> Value }

impl ScatterPlan { pub fn lanes(&self) -> usize }

impl ScatterPlan { pub fn out_len(&self, l: usize) -> usize }

impl ScatterPlan { pub fn bind(&self, file: &EsmFile, model_name: Option<&str>, l: usize) -> Result<BoundScatter, BandError> }

impl ScatterPlan { pub fn bind_entries(&self, sv: &SysView<'_>, entries: &[JacEntry], l: usize) -> Result<BoundScatter, BandError> }

impl ScatterPlan { pub fn targets(&self, l: usize) -> Vec<Option<(String, String)>> }
```

Band fields → storage positions for one (band list, layout). Plain data:
`Send + Sync`, built once, shared by every leaf using the operator.

[Source](https://github.com/EarthSciML/EarthSciASTDiff/blob/6f151823f210d8951df2266333997c204f54e68e/pkg/earthsci-astdiff-rs/src/scatter.rs#L90) · [Rust docs](https://docs.rs/earthsci-astdiff/latest/earthsci_astdiff/scatter/struct.ScatterPlan.html)

</div>

</div>
