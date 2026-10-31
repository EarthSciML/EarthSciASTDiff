---
title: "scatter_plan (function)"
description: "Stable function · Evaluate the Jacobian"
type: "api"
api_tabs: true
---


**Tier:** stable · **Kind:** function · **Topic:** [Evaluate the Jacobian](/generated/api/topics/evaluate)

| Julia | Python | Rust |
|---|---|---|
| `scatter_plan` | not exported | not exported |


<div class="api-tabs">

<div class="api-tab" data-lang="julia" data-label="Julia">
<p class="api-tab-label">Julia</p>

```julia
scatter_plan(input, layout::StorageLayout; model_name)
scatter_plan(entries::Vector{JacEntry}, sv::SysView, layout::StorageLayout)
```

```julia
scatter_plan(input, layout::StorageLayout; model_name = nothing) -> ScatterPlan
scatter_plan(entries, sv::SysView, layout) -> ScatterPlan
```

Build the scatter plan of a model's state Jacobian for `layout`. Throws an
`ArgumentError` naming the reason when the layout does not fit the Jacobian
(a stencil under `LaneBlocks`, a periodic wrap under `ColumnBanded`, …), and
always for an empty band list on a model with array states — that is the
silent all-zero Jacobian, never a valid plan.

[Source](https://github.com/EarthSciML/EarthSciASTDiff/blob/a9ca79cb254eab4630ba5d3dc17b785c72fcfc79/pkg/EarthSciASTDiff.jl/src/scatter.jl#L140)

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
