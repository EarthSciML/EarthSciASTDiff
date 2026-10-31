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
serialize_band(en: JacEntry) -> dict
```

One entry of the `jacobians` block, as a JSON-ready dict (spec §4).

[Source](https://github.com/EarthSciML/EarthSciASTDiff/blob/6f151823f210d8951df2266333997c204f54e68e/pkg/earthsci-astdiff-py/src/earthsci_astdiff/emit.py#L16)

</div>

<div class="api-tab" data-lang="rust" data-label="Rust">
<p class="api-tab-label">Rust</p>

```rust
pub fn serialize_band(en: &JacEntry) -> Value
```

One entry of the `jacobians` block, as JSON (spec §4).

[Source](https://github.com/EarthSciML/EarthSciASTDiff/blob/6f151823f210d8951df2266333997c204f54e68e/pkg/earthsci-astdiff-rs/src/emit.rs#L19) · [Rust docs](https://docs.rs/earthsci-astdiff/latest/earthsci_astdiff/emit/fn.serialize_band.html)

</div>

</div>
