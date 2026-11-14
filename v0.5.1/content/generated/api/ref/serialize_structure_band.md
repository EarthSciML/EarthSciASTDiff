---
title: "serialize_structure_band"
description: "Stable function · Other"
type: "api"
api_tabs: true
---


**Tier:** stable · **Kind:** function · **Topic:** [Other](/generated/api/topics/other)

| Julia | Python | Rust |
|---|---|---|
| not exported | not exported | `serialize_structure_band` |


**Note:** Python's `serialize_band` omits `coef` for a structure-only entry.


<div class="api-tabs">

<div class="api-tab" data-lang="julia" data-label="Julia">
<p class="api-tab-label">Julia</p>

<p class="api-absent">Julia does not export this function.</p>

</div>

<div class="api-tab" data-lang="python" data-label="Python">
<p class="api-tab-label">Python</p>

<p class="api-absent">Python does not export this function.</p>

</div>

<div class="api-tab" data-lang="rust" data-label="Rust">
<p class="api-tab-label">Rust</p>

```rust
pub fn serialize_structure_band(en: &StructureEntry) -> Value
```

A structure-only entry (spec §2.1) as JSON: the fields of
[`serialize_band`](/generated/api/ref/serialize_band) without `coef`.

[Source](https://github.com/EarthSciML/EarthSciASTDiff/blob/2c83701848fda145f8d65f939fb8cf382ee2681e/pkg/earthsci-astdiff-rs/src/emit.rs#L40) · [Rust docs](https://docs.rs/earthsci-astdiff/latest/earthsci_astdiff/emit/fn.serialize_structure_band.html)

</div>

</div>
