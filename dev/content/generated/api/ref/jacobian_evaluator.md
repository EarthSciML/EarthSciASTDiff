---
title: "jacobian_evaluator"
description: "Stable type · Evaluate the Jacobian"
type: "api"
api_tabs: true
---


**Tier:** stable · **Kind:** type · **Topic:** [Evaluate the Jacobian](/generated/api/topics/evaluate)

| Julia | Python | Rust |
|---|---|---|
| `JacobianEvaluator` | `JacobianEvaluator` | `JacobianEvaluator` |


<div class="api-tabs">

<div class="api-tab" data-lang="julia" data-label="Julia">
<p class="api-tab-label">Julia</p>

```julia
struct JacobianEvaluator
JacobianEvaluator(fJ!::F, oop::Bool, uj::Vector{Float64}, duj::Vector{Float64}, umap::Vector{Int64}, p_default::P, scatter::Vector{Tuple{Int64, Int64}}, items::Vector{Tuple{Int64, Int64, Int64}}, field_shapes::Vector{Vector{Int64}}, slots::Dict{String, Int64}, prototype::SparseArrays.SparseMatrixCSC{Float64, Int64}, pattern::SparseArrays.SparseMatrixCSC{Bool, Int64}, entries::Vector{JacEntry}, rownames::Vector{String}, colnames::Vector{String}, columns::Union{Nothing, Vector{Tuple{String, Vector{Int64}}}}, structure::Union{Nothing, Symbol}, wrt::Symbol) where {F, P}
```

```julia
JacobianEvaluator
```

Prepared analytical Jacobian: call as `jac!(J, u, p, t)` where `J` is (a copy
of) `jac.prototype`. Fields of note: `prototype::SparseMatrixCSC{Float64}`,
`pattern::SparseMatrixCSC{Bool}`, `entries`, `colnames`, `structure`.

[Source](https://github.com/EarthSciML/EarthSciASTDiff/blob/a9ca79cb254eab4630ba5d3dc17b785c72fcfc79/pkg/EarthSciASTDiff.jl/src/assemble.jl#L253)

</div>

<div class="api-tab" data-lang="python" data-label="Python">
<p class="api-tab-label">Python</p>

```python
class JacobianEvaluator
JacobianEvaluator(sv: SysView, wrt: str, forcing=None)
JacobianEvaluator.values(states: dict | None = None, params: dict | None = None, t: float = 0.0) -> dict[tuple[str, str], float]
```

A prepared band evaluation for one model and axis, built once by
[`prepare_jacobian`](/generated/api/ref/prepare_jacobian): the band calculus, the observed definitions,
the column space and every band point's (row, column) placement. Each
later evaluation (`values`, [`vjp`](/generated/api/ref/vjp), [`vjp_forcing`](/generated/api/ref/vjp_forcing)) only
resolves the coefficients at the point — the Julia reference's
`JacobianEvaluator` in this binding's correctness-surface form.

Fields of note: `wrt`, `entries`, `columns` (the
[`jacobian_columns`](/generated/api/ref/jacobian_columns) space on the `"parameters"` / `"forcing"`
axes, else `None`) and `colnames` (that space's cell names, `["t"]`
on `"time"`, `None` on `"states"`).

**Members**

- `values` — The band entries at a point, as [`assemble_jacobian`](/generated/api/ref/assemble_jacobian) returns them.

[Source](https://github.com/EarthSciML/EarthSciASTDiff/blob/a9ca79cb254eab4630ba5d3dc17b785c72fcfc79/pkg/earthsci-astdiff-py/src/earthsci_astdiff/numeric.py#L145)

</div>

<div class="api-tab" data-lang="rust" data-label="Rust">
<p class="api-tab-label">Rust</p>

```rust
pub struct JacobianEvaluator {
    pub wrt: Wrt,
    pub entries: Vec<JacEntry>,
    pub columns: Option<Vec<(String, Vec<i64>)>>,
    pub colnames: Vec<String>,
    /* private fields */
}

impl JacobianEvaluator { pub fn values(&self, env: &Env) -> Result<BTreeMap<(String, String), f64>, BandError> }

impl JacobianEvaluator { pub fn vjp(&self, env: &Env, lambda: &HashMap<String, f64>, gbar: &mut [f64]) -> Result<(), BandError> }

impl JacobianEvaluator { pub fn vjp_forcing(&self, env: &Env, lambda: &HashMap<String, f64>, gbar: &mut HashMap<String, Vec<f64>>) -> Result<(), BandError> }
```

A prepared band evaluation for one model and axis, built once by
[`prepare_jacobian`]: the band calculus, the observed-inlined
coefficients, the column space and the parameter defaults. Each later
evaluation (`JacobianEvaluator::values`, `JacobianEvaluator::vjp`,
`JacobianEvaluator::vjp_forcing`) only walks the band points at the
given `Env` — so an adjoint loop that takes one forcing cotangent per
stage does not redo the build-time work per stage.

[Source](https://github.com/EarthSciML/EarthSciASTDiff/blob/a9ca79cb254eab4630ba5d3dc17b785c72fcfc79/pkg/earthsci-astdiff-rs/src/numeric.rs#L426) · [Rust docs](https://docs.rs/earthsci-astdiff/latest/earthsci_astdiff/numeric/struct.JacobianEvaluator.html)

</div>

</div>
