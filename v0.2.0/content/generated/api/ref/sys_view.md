---
title: "sys_view"
description: "Stable type · Differentiate a model"
type: "api"
api_tabs: true
---


**Tier:** stable · **Kind:** type · **Topic:** [Differentiate a model](/generated/api/topics/differentiate)

| Julia | Python | Rust |
|---|---|---|
| `SysView` | `SysView` | `SysView` |


<div class="api-tabs">

<div class="api-tab" data-lang="julia" data-label="Julia">
<p class="api-tab-label">Julia</p>

```julia
struct SysView
SysView(variables::Dict{String, Any}, equations::Vector, index_sets::Dict{String, Any})
SysView(variables, equations, index_sets)
```

```julia
SysView
```

Uniform system view: `variables` (name → `ModelVariable`), `equations`, and
the document's `index_sets`. Construct with [`sysview`](/generated/api/ref/sysview).

[Source](https://github.com/EarthSciML/EarthSciASTDiff/blob/2a7603b0bd3d2f1c582d4b8dc2f8f9c2960584df/pkg/EarthSciASTDiff.jl/src/system.jl#L6)

</div>

<div class="api-tab" data-lang="python" data-label="Python">
<p class="api-tab-label">Python</p>

```python
class SysView
SysView(variables: dict[str, object], equations: list, index_sets: dict[str, object])
```

A model or flattened system as the band calculus sees it: its
variables, equations and index sets, with subsystems already expanded.
Build one with [`sysview`](/generated/api/ref/sysview).

[Source](https://github.com/EarthSciML/EarthSciASTDiff/blob/2a7603b0bd3d2f1c582d4b8dc2f8f9c2960584df/pkg/earthsci-astdiff-py/src/earthsci_astdiff/system.py#L17)

</div>

<div class="api-tab" data-lang="rust" data-label="Rust">
<p class="api-tab-label">Rust</p>

```rust
pub struct SysView<'a> {
    pub variables: IndexMap<String, Cow<'a, ModelVariable>>,
    pub equations: &'a Vec<Equation>,
    pub index_sets: IndexMap<String, IndexSet>,
}
```

A model as the band calculus sees it: its variables, equations and index
sets. Build one with [`sysview`](/generated/api/ref/sysview) (one model of a document) or
`sysview_flat` (a flattened system).

Each variable is borrowed from its model or flattened system, except a
pointwise-lifted state of a flattened system, which the view owns with
its grid shape (see `sysview_flat`).

[Source](https://github.com/EarthSciML/EarthSciASTDiff/blob/2a7603b0bd3d2f1c582d4b8dc2f8f9c2960584df/pkg/earthsci-astdiff-rs/src/system.rs#L25) · [Rust docs](https://docs.rs/earthsci-astdiff/latest/earthsci_astdiff/system/struct.SysView.html)

</div>

</div>
