---
title: "structure_entry"
description: "Stable type · Other"
type: "api"
api_tabs: true
---


**Tier:** stable · **Kind:** type · **Topic:** [Other](/generated/api/topics/other)

| Julia | Python | Rust |
|---|---|---|
| not exported | not exported | `StructureEntry` |


**Note:** Julia and Python return a `JacEntry` whose band has no coefficient.


<div class="api-tabs">

<div class="api-tab" data-lang="julia" data-label="Julia">
<p class="api-tab-label">Julia</p>

<p class="api-absent">Julia does not export this type.</p>

</div>

<div class="api-tab" data-lang="python" data-label="Python">
<p class="api-tab-label">Python</p>

<p class="api-absent">Python does not export this type.</p>

</div>

<div class="api-tab" data-lang="rust" data-label="Rust">
<p class="api-tab-label">Rust</p>

```rust
pub struct StructureEntry {
    pub u: String,
    pub v: String,
    pub band: StructureBand,
}
```

One block of the structure-only Jacobian (spec §2.1): a [`JacEntry`](/generated/api/ref/jac_entry)
without its coefficient.

[Source](https://github.com/EarthSciML/EarthSciASTDiff/blob/4093ace7f71a8249dc4b70b8a275bedb6d07ba7b/pkg/earthsci-astdiff-rs/src/jacobian.rs#L239) · [Rust docs](https://docs.rs/earthsci-astdiff/latest/earthsci_astdiff/jacobian/struct.StructureEntry.html)

</div>

</div>
