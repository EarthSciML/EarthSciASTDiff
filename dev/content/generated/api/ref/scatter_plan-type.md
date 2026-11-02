---
title: "scatter_plan (type)"
description: "Stable type · Evaluate the Jacobian"
type: "api"
api_tabs: true
---


**Tier:** stable · **Kind:** type · **Topic:** [Evaluate the Jacobian](/generated/api/topics/evaluate)

| Julia | Python | Rust |
|---|---|---|
| `ScatterPlan` | `ScatterPlan` | `ScatterPlan` |


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
checked against a source by [`bind_plan`](/generated/api/ref/bind_plan). Every plan carries the
canonical `states` (name, shape) and the `bands` (row var, column var, field
shape) it was built from, which is what binding checks.

[Source](https://github.com/EarthSciML/EarthSciASTDiff/blob/2a9b4a92b2b429cd94fc2ddac236c438425d94f9/pkg/EarthSciASTDiff.jl/src/scatter.jl#L76)

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
ScatterPlan = LaneBlocksPlan | LaneSparseBlocksPlan | ColumnBandedPlan | CsrPlan
```

<p class="api-absent">No docstring.</p>

[Source](https://github.com/EarthSciML/EarthSciASTDiff/blob/2a9b4a92b2b429cd94fc2ddac236c438425d94f9/pkg/earthsci-astdiff-py/src/earthsci_astdiff/scatter.py#L126)

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

[Source](https://github.com/EarthSciML/EarthSciASTDiff/blob/2a9b4a92b2b429cd94fc2ddac236c438425d94f9/pkg/earthsci-astdiff-rs/src/scatter.rs#L91) · [Rust docs](https://docs.rs/earthsci-astdiff/latest/earthsci_astdiff/scatter/struct.ScatterPlan.html)

</div>

</div>
