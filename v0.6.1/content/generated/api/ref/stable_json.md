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

A `const` node's `value` is written in canonical-number form (EarthSciAST
CONFORMANCE_SPEC §5.5.3.1 rule 1: an integral, `Int64`-representable number
as an integer literal, at any depth), as EarthSciAST ≥ 0.10 serializes it, so
the text does not depend on the EarthSciAST release or on whether a table was
authored `[1.0, 2.0]` or built in code.

[Source](https://github.com/EarthSciML/EarthSciASTDiff/blob/9891c3005a31b74a0547498d0d2981ebb69cdc14/pkg/EarthSciASTDiff.jl/src/emit.jl#L5)

</div>

<div class="api-tab" data-lang="python" data-label="Python">
<p class="api-tab-label">Python</p>

```python
stable_json(x: Any) -> str
```

Deterministic JSON: sorted object keys at every level, floats in the
reference's layout (`canonical_float`), a `const` node's `value` in
canonical-number form (`_canon_numbers`, as EarthSciAST >= 0.10 writes
it) — emit.jl's `stable_json`.

[Source](https://github.com/EarthSciML/EarthSciASTDiff/blob/9891c3005a31b74a0547498d0d2981ebb69cdc14/pkg/earthsci-astdiff-py/src/earthsci_astdiff/expr_helpers.py#L138)

</div>

<div class="api-tab" data-lang="rust" data-label="Rust">
<p class="api-tab-label">Rust</p>

```rust
pub fn stable_json(v: &Value) -> String
```

Deterministic JSON: sorted object keys at every level, floats in the
reference's layout (`canonical_float`), every other scalar by
serde_json. A `const` node's `value` is written in canonical-number form
(CONFORMANCE_SPEC §5.5.3.1 rule 1: an integral, i64-representable number
as an integer, at any depth), as earthsci-ast ≥ 0.10 serializes it, so the
text does not depend on the earthsci-ast release.

[Source](https://github.com/EarthSciML/EarthSciASTDiff/blob/9891c3005a31b74a0547498d0d2981ebb69cdc14/pkg/earthsci-astdiff-rs/src/helpers.rs#L121) · [Rust docs](https://docs.rs/earthsci-astdiff/latest/earthsci_astdiff/helpers/fn.stable_json.html)

</div>

</div>
