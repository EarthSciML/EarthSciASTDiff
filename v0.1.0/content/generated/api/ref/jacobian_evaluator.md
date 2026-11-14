---
title: "jacobian_evaluator"
description: "Stable type · Evaluate the Jacobian"
type: "api"
api_tabs: true
---


**Tier:** stable · **Kind:** type · **Topic:** [Evaluate the Jacobian](/generated/api/topics/evaluate)

| Julia | Python | Rust |
|---|---|---|
| `JacobianEvaluator` | not exported | not exported |


<div class="api-tabs">

<div class="api-tab" data-lang="julia" data-label="Julia">
<p class="api-tab-label">Julia</p>

```julia
struct JacobianEvaluator
JacobianEvaluator(fJ!::F, oop::Bool, uj::Vector{Float64}, duj::Vector{Float64}, umap::Vector{Int64}, p_default::P, scatter::Vector{Tuple{Int64, Int64}}, field_shapes::Vector{Vector{Int64}}, slots::Dict{String, Int64}, prototype::SparseArrays.SparseMatrixCSC{Float64, Int64}, pattern::SparseArrays.SparseMatrixCSC{Bool, Int64}, entries::Vector{JacEntry}, rownames::Vector{String}, colnames::Vector{String}, structure::Symbol, wrt::Symbol) where {F, P}
```

```julia
JacobianEvaluator
```

Prepared analytical Jacobian: call as `jac!(J, u, p, t)` where `J` is (a copy
of) `jac.prototype`. Fields of note: `prototype::SparseMatrixCSC{Float64}`,
`pattern::SparseMatrixCSC{Bool}`, `entries`, `colnames`, `structure`.

[Source](https://github.com/EarthSciML/EarthSciASTDiff/blob/6f151823f210d8951df2266333997c204f54e68e/pkg/EarthSciASTDiff.jl/src/assemble.jl#L192)

</div>

<div class="api-tab" data-lang="python" data-label="Python">
<p class="api-tab-label">Python</p>

<p class="api-absent">Python does not export this type.</p>

</div>

<div class="api-tab" data-lang="rust" data-label="Rust">
<p class="api-tab-label">Rust</p>

<p class="api-absent">Rust does not export this type.</p>

</div>

</div>
