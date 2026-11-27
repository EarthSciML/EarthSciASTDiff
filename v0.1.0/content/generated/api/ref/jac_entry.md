---
title: "jac_entry"
description: "Stable type · Differentiate a model"
type: "api"
api_tabs: true
---


**Tier:** stable · **Kind:** type · **Topic:** [Differentiate a model](/generated/api/topics/differentiate)

| Julia | Python | Rust |
|---|---|---|
| `JacEntry` | `JacEntry` | `JacEntry` |


<div class="api-tabs">

<div class="api-tab" data-lang="julia" data-label="Julia">
<p class="api-tab-label">Julia</p>

```julia
struct JacEntry
JacEntry(u::String, v::String, band::Band)
JacEntry(u, v, band)
```

```julia
JacEntry
```

One band of the model Jacobian: `∂(equation for state `u`)/∂`v`` with its
[`Band`](/generated/api/ref/band).

[Source](https://github.com/EarthSciML/EarthSciASTDiff/blob/6f151823f210d8951df2266333997c204f54e68e/pkg/EarthSciASTDiff.jl/src/jacobian.jl#L4)

</div>

<div class="api-tab" data-lang="python" data-label="Python">
<p class="api-tab-label">Python</p>

```python
class JacEntry
JacEntry(u: str, v: str, band: Band)
```

One block of the Jacobian: row variable `u`, column variable `v`,
and the [`Band`](/generated/api/ref/band) giving the rows it covers, the column-index
expressions and the coefficient.

[Source](https://github.com/EarthSciML/EarthSciASTDiff/blob/6f151823f210d8951df2266333997c204f54e68e/pkg/earthsci-astdiff-py/src/earthsci_astdiff/jacobian.py#L16)

</div>

<div class="api-tab" data-lang="rust" data-label="Rust">
<p class="api-tab-label">Rust</p>

```rust
pub struct JacEntry {
    pub u: String,
    pub v: String,
    pub band: Band,
}
```

One block of the Jacobian: row variable `u`, column variable `v`, and the
[`Band`](/generated/api/ref/band) giving the rows it covers, the column-index expressions and the
coefficient.

[Source](https://github.com/EarthSciML/EarthSciASTDiff/blob/6f151823f210d8951df2266333997c204f54e68e/pkg/earthsci-astdiff-rs/src/jacobian.rs#L34) · [Rust docs](https://docs.rs/earthsci-astdiff/latest/earthsci_astdiff/jacobian/struct.JacEntry.html)

</div>

</div>
