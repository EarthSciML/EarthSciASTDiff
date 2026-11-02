---
title: "scatter_plan (function)"
description: "Stable function · Evaluate the Jacobian"
type: "api"
api_tabs: true
---


**Tier:** stable · **Kind:** function · **Topic:** [Evaluate the Jacobian](/generated/api/topics/evaluate)

| Julia | Python | Rust |
|---|---|---|
| `scatter_plan` | `scatter_plan` | not exported |


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

[Source](https://github.com/EarthSciML/EarthSciASTDiff/blob/2a9b4a92b2b429cd94fc2ddac236c438425d94f9/pkg/EarthSciASTDiff.jl/src/scatter.jl#L141)

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

<p class="api-src">From <a href="https://github.com/EarthSciML/EarthSciASTDiff/blob/2a9b4a92b2b429cd94fc2ddac236c438425d94f9/docs/api/examples/julia/jacobians.jl#L36-L57">docs/api/examples/julia/jacobians.jl</a></p>

</div>

<div class="api-tab" data-lang="python" data-label="Python">
<p class="api-tab-label">Python</p>

```python
scatter_plan(obj, layout: StorageLayout, model_name: str | None = None, entries: list[JacEntry] | None = None) -> ScatterPlan
```

Build the scatter plan of a model's state Jacobian for `layout`.

`obj` is an `EsmFile`, a flattened system or a [`SysView`](/generated/api/ref/sys_view);
`entries` (the reference's `scatter_plan(entries, sv, layout)`) is a
state band list to plan instead of the model's own. Raises
`ValueError` naming the reason when the layout does not fit the
Jacobian (a stencil under `LaneBlocks`, a periodic wrap under
`ColumnBanded`, …), and always for an empty band list on a model
with array states — that is the silent all-zero Jacobian, never a valid
plan.

[Source](https://github.com/EarthSciML/EarthSciASTDiff/blob/2a9b4a92b2b429cd94fc2ddac236c438425d94f9/pkg/earthsci-astdiff-py/src/earthsci_astdiff/scatter.py#L155)

</div>

<div class="api-tab" data-lang="rust" data-label="Rust">
<p class="api-tab-label">Rust</p>

<p class="api-absent">Rust does not export this function.</p>

</div>

</div>
