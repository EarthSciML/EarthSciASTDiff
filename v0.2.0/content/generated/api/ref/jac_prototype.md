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
jac_prototype(input; wrt, model_name)
```

```julia
jac_prototype(input; wrt = :states, model_name = nothing)
```

A freshly zeroed sparse prototype on the structural pattern — the object to
pass as `ODEFunction(f!; jac_prototype = …)`. Stored (structural) zeros are
kept so the pattern is point-independent.

[Source](https://github.com/EarthSciML/EarthSciASTDiff/blob/2a7603b0bd3d2f1c582d4b8dc2f8f9c2960584df/pkg/EarthSciASTDiff.jl/src/structure.jl#L135)

**Example** (`evaluate`)

```julia
# EarthSciASTDiff evaluates nothing: it emits ESM documents that EarthSciAST
# evaluates. The band fields J_k, as a derived model whose right-hand side,
# run once, leaves every band value in `du`. `fields[k]` names entry k's field.
doc, fields = evaluation_document(file)
@assert length(fields) == length(entries)
prob = EarthSciAST.esm_problem(
    JSON3.read(stable_json(doc)), (0.0, 1.0)
)
du = zero(prob.u0)
prob.f!(du, prob.u0, prob.p, 0.0)            # at the document's initial state
k = findfirst(en -> (en.u, en.v) == ("A", "A"), entries)
@assert du[prob.var_map["JacobianEval.$(fields[k])[1]"]] == -0.04   # -k1
# J·v as a derived model: inputs `_v_<C>`, outputs `_jvp_<R>`.
jdoc, (model, v, jvp) = jvp_document(file)
@assert model == "JvpEval"
J = jac_prototype(file)                      # sparse matrix on the fixed pattern
@assert size(J) == (12, 12) && nnz(J) == 24
# A scatter plan places the band fields in your own storage (here CSR),
# checked against the source before it is used.
bound = bind_plan(file, scatter_plan(file, Csr()))
@assert bound.len == 24
@assert all(1 <= k <= length(fields) for (k, _, _) in bound.pairs)
```

<p class="api-src">From <a href="https://github.com/EarthSciML/EarthSciASTDiff/blob/2a7603b0bd3d2f1c582d4b8dc2f8f9c2960584df/docs/api/examples/julia/jacobians.jl#L36-L57">docs/api/examples/julia/jacobians.jl</a></p>

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
