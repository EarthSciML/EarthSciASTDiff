---
title: "jac_prototype"
description: "Stable function · Evaluate the Jacobian"
type: "api"
api_tabs: true
---


**Tier:** stable · **Kind:** function · **Topic:** [Evaluate the Jacobian](/generated/api/topics/evaluate)

| Julia | Python | Rust |
|---|---|---|
| `jac_prototype` | not exported | not exported |


<div class="api-tabs">

<div class="api-tab" data-lang="julia" data-label="Julia">
<p class="api-tab-label">Julia</p>

```julia
jac_prototype(jac::JacobianEvaluator)
jac_prototype(input; wrt, model_name)
```

```julia
jac_prototype(jac::JacobianEvaluator) -> SparseMatrixCSC{Float64,Int}
jac_prototype(input; wrt = :states, model_name = nothing)
```

A freshly zeroed sparse prototype on the structural pattern — the object to
pass as `ODEFunction(f!; jac_prototype = …)`. Stored (structural) zeros are
kept so the pattern is point-independent.

[Source](https://github.com/EarthSciML/EarthSciASTDiff/blob/6f151823f210d8951df2266333997c204f54e68e/pkg/EarthSciASTDiff.jl/src/structure.jl#L69)

**Example** (`prepare`)

```julia
jac = prepare_jacobian(file)                 # differentiate and compile once
J = jac_prototype(jac)                       # sparse matrix on the fixed pattern
u = fill(2.0, 12)
p = jac.p_default                            # the document's parameter values
jac(J, u, p, 0.0)                            # fill J in place, at any (u, p, t)
@assert size(J) == (12, 12)
@assert nnz(J) == 24
@assert J[1, 1] == -0.04
```

<p class="api-src">From <a href="https://github.com/EarthSciML/EarthSciASTDiff/blob/6f151823f210d8951df2266333997c204f54e68e/docs/api/examples/julia/jacobians.jl#L41-L48">docs/api/examples/julia/jacobians.jl</a></p>

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
