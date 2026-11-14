---
title: "serialize_band"
description: "Stable function · Serialize as a document block"
type: "api"
api_tabs: true
---


**Tier:** stable · **Kind:** function · **Topic:** [Serialize as a document block](/generated/api/topics/serialize)

| Julia | Python | Rust |
|---|---|---|
| not exported | `serialize_band` | `serialize_band` |


<div class="api-tabs">

<div class="api-tab" data-lang="julia" data-label="Julia">
<p class="api-tab-label">Julia</p>

<p class="api-absent">Julia does not export this function.</p>

</div>

<div class="api-tab" data-lang="python" data-label="Python">
<p class="api-tab-label">Python</p>

```python
serialize_band(en: JacEntry | JacEntry2) -> dict
```

One entry of the `jacobians` block, as a JSON-ready dict (spec §4;
a [`JacEntry2`](/generated/api/ref/jac_entry2) adds `col2` / `col2_idx`, §4.4).

[Source](https://github.com/EarthSciML/EarthSciASTDiff/blob/9891c3005a31b74a0547498d0d2981ebb69cdc14/pkg/earthsci-astdiff-py/src/earthsci_astdiff/emit.py#L24)

</div>

<div class="api-tab" data-lang="rust" data-label="Rust">
<p class="api-tab-label">Rust</p>

```rust
pub fn serialize_band(en: &JacEntry) -> Value
```

One entry of the `jacobians` block, as JSON (spec §4).

[Source](https://github.com/EarthSciML/EarthSciASTDiff/blob/9891c3005a31b74a0547498d0d2981ebb69cdc14/pkg/earthsci-astdiff-rs/src/emit.rs#L23) · [Rust docs](https://docs.rs/earthsci-astdiff/latest/earthsci_astdiff/emit/fn.serialize_band.html)

</div>

</div>
