---
title: "jacobian_columns"
description: "Stable function · Differentiate a model"
type: "api"
api_tabs: true
---


**Tier:** stable · **Kind:** function · **Topic:** [Differentiate a model](/generated/api/topics/differentiate)

| Julia | Python | Rust |
|---|---|---|
| `jacobian_columns` | `jacobian_columns` | `jacobian_columns` |


<div class="api-tabs">

<div class="api-tab" data-lang="julia" data-label="Julia">
<p class="api-tab-label">Julia</p>

```julia
jacobian_columns(sv::SysView; wrt, forcing)
```

```julia
jacobian_columns(sv::SysView; wrt, forcing = nothing) -> Vector{Tuple{String,Vector{Int}}}
```

The column space of a `:parameters` or `:forcing` block: each column variable
with its integer shape (`Int[]` for a scalar), sorted by name — the `columns`
field of the emitted block (esm-jacobian-spec.md §4). The canonical column
ordinal enumerates these variables in order, each one's cells column-major
(first index fastest).

[Source](https://github.com/EarthSciML/EarthSciASTDiff/blob/4093ace7f71a8249dc4b70b8a275bedb6d07ba7b/pkg/EarthSciASTDiff.jl/src/jacobian.jl#L70)

</div>

<div class="api-tab" data-lang="python" data-label="Python">
<p class="api-tab-label">Python</p>

```python
jacobian_columns(obj, model_name: str | None = None, wrt: str = 'parameters', forcing=None)
```

The column space of a `"parameters"` or `"forcing"` block: each
column variable with its integer shape (`[]` for a scalar), sorted by
name — the `columns` field of the emitted block (spec §4). The canonical
column ordinal enumerates these variables in order, each one's cells
column-major (first index fastest).

[Source](https://github.com/EarthSciML/EarthSciASTDiff/blob/4093ace7f71a8249dc4b70b8a275bedb6d07ba7b/pkg/earthsci-astdiff-py/src/earthsci_astdiff/jacobian.py#L126)

</div>

<div class="api-tab" data-lang="rust" data-label="Rust">
<p class="api-tab-label">Rust</p>

```rust
pub fn jacobian_columns(sv: &SysView<'_>, wrt: Wrt, forcing: Option<&[String]>) -> Result<Vec<(String, Vec<i64>)>, BandError>
```

The column space of a parameters or forcing block: each column variable
with its integer shape (empty for a scalar), sorted by name — the
`columns` field of the emitted block (spec §4). The canonical column
ordinal enumerates these variables in order, each one's cells
column-major (first index fastest).

[Source](https://github.com/EarthSciML/EarthSciASTDiff/blob/4093ace7f71a8249dc4b70b8a275bedb6d07ba7b/pkg/earthsci-astdiff-rs/src/jacobian.rs#L140) · [Rust docs](https://docs.rs/earthsci-astdiff/latest/earthsci_astdiff/jacobian/fn.jacobian_columns.html)

</div>

</div>
