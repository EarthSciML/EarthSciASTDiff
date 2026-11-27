---
title: "stable_json"
description: "Stable function · Serialize as a document block"
type: "api"
api_tabs: true
---


**Tier:** stable · **Kind:** function · **Topic:** [Serialize as a document block](/generated/api/topics/serialize)

| Julia | Python | Rust |
|---|---|---|
| `stable_json` | `stable_json` | `stable_json` |


<div class="api-tabs">

<div class="api-tab" data-lang="julia" data-label="Julia">
<p class="api-tab-label">Julia</p>

```julia
stable_json(x)
```

```julia
stable_json(x) -> String
```

Deterministic JSON: object keys emitted in sorted order at every level,
scalars formatted by JSON3. Used for conformance goldens and for structural
expression keys (`skey`) — equal trees always produce equal text.

[Source](https://github.com/EarthSciML/EarthSciASTDiff/blob/6f151823f210d8951df2266333997c204f54e68e/pkg/EarthSciASTDiff.jl/src/emit.jl#L5)

</div>

<div class="api-tab" data-lang="python" data-label="Python">
<p class="api-tab-label">Python</p>

```python
stable_json(x: Any) -> str
```

Deterministic JSON: sorted object keys at every level, floats in the
reference's layout (`canonical_float`) — emit.jl's `stable_json`.

[Source](https://github.com/EarthSciML/EarthSciASTDiff/blob/6f151823f210d8951df2266333997c204f54e68e/pkg/earthsci-astdiff-py/src/earthsci_astdiff/expr_helpers.py#L122)

</div>

<div class="api-tab" data-lang="rust" data-label="Rust">
<p class="api-tab-label">Rust</p>

```rust
pub fn stable_json(v: &Value) -> String
```

Deterministic JSON: sorted object keys at every level, floats in the
reference's layout (`canonical_float`), every other scalar by
serde_json.

[Source](https://github.com/EarthSciML/EarthSciASTDiff/blob/6f151823f210d8951df2266333997c204f54e68e/pkg/earthsci-astdiff-rs/src/helpers.rs#L92) · [Rust docs](https://docs.rs/earthsci-astdiff/latest/earthsci_astdiff/helpers/fn.stable_json.html)

</div>

</div>
