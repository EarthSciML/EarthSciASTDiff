---
title: "bind_plan"
description: "Stable function · Evaluate the Jacobian"
type: "api"
api_tabs: true
---


**Tier:** stable · **Kind:** function · **Topic:** [Evaluate the Jacobian](/generated/api/topics/evaluate)

| Julia | Python | Rust |
|---|---|---|
| `bind_plan` | `bind_plan` | not exported |


**Note:** The bound plan's `pairs` follow each language's own indexing: Julia gives `(band, field cell, position)` triples, all 1-based; Rust (`ScatterPlan::bind`) the same triples, all 0-based; Python `(band-field cell name, position)` pairs with a 0-based position. The plan file itself (`*.plan.json`) is 0-based in every binding (spec §8). A loop ported between bindings must shift positions accordingly.


<div class="api-tabs">

<div class="api-tab" data-lang="julia" data-label="Julia">
<p class="api-tab-label">Julia</p>

```julia
bind_plan(input, plan::ScatterPlan; lanes, model_name)
```

```julia
bind_plan(input, plan::ScatterPlan; lanes = nothing, model_name = nothing) -> BoundScatter
```

Check `plan` against the state Jacobian of `input` (an `EsmFile` or a
`FlattenedSystem`) and precompute its gather. Binding is strict: the
plan's band list (row var, column var, field shape per band) and canonical
states must equal the source's, and every contribution is decoded
against the source's own bands — each must land on the (row, column) cell
its band point differentiates, and every band point with a column must be
covered exactly once. A plan from another build, another document or another
stencil is refused, never silently misapplied. `lanes` is the lane capacity
`L` of a lane or column layout (default: exactly the lane count; padding
lanes stay zero).

[Source](https://github.com/EarthSciML/EarthSciASTDiff/blob/2a7603b0bd3d2f1c582d4b8dc2f8f9c2960584df/pkg/EarthSciASTDiff.jl/src/scatter.jl#L458)

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

```python
bind_plan(obj, plan: ScatterPlan, lanes: int | None = None, model_name: str | None = None) -> BoundScatter
```

Check `plan` against the state Jacobian of `obj` (an `EsmFile`, a
flattened system or a [`SysView`](/generated/api/ref/sys_view)) and precompute its gather.

Binding is strict: the plan's band list (row var, column var, field shape
per band) and canonical states must equal the source's, and every
contribution is decoded against the source's own bands — each must land
on the (row, column) cell its band point differentiates, and every band
point with a column must be covered exactly once. A plan from another
build, another document or another stencil is refused (`ValueError`),
never silently misapplied. `lanes` is the lane capacity `L` of a lane
or column layout (default: exactly the lane count; padding lanes stay
zero).

[Source](https://github.com/EarthSciML/EarthSciASTDiff/blob/2a7603b0bd3d2f1c582d4b8dc2f8f9c2960584df/pkg/earthsci-astdiff-py/src/earthsci_astdiff/scatter.py#L485)

</div>

<div class="api-tab" data-lang="rust" data-label="Rust">
<p class="api-tab-label">Rust</p>

<p class="api-absent">Rust does not export this function.</p>

</div>

</div>
