---
title: "transfer_definition"
description: "Stable function · Differentiate a model"
type: "api"
api_tabs: true
---


**Tier:** stable · **Kind:** function · **Topic:** [Differentiate a model](/generated/api/topics/differentiate)

| Julia | Python | Rust |
|---|---|---|
| not exported | `transfer_definition` | `transfer_definition` |


**Note:** The transfer recurrence T = (I − G)⁻¹ a block's `auxiliary` field carries (spec §4.5).


<div class="api-tabs">

<div class="api-tab" data-lang="julia" data-label="Julia">
<p class="api-tab-label">Julia</p>

<p class="api-absent">Julia does not export this function.</p>

</div>

<div class="api-tab" data-lang="python" data-label="Python">
<p class="api-tab-label">Python</p>

```python
transfer_definition(rec: Recurrence, obs: dict, recs: list[Recurrence], ctx, tname: str)
```

The transfer recurrence `T` of `rec` (spec §4.5): over the frame
plus one output index `j` on the axis, `T[o, k, j] = [k == j] + Σ_{self bands} coef · T[o, k − lag, j]`, so `T[o, ·, ·] = (I − G)⁻¹`
on each line. Lower triangular on each line.

[Source](https://github.com/EarthSciML/EarthSciASTDiff/blob/2a7603b0bd3d2f1c582d4b8dc2f8f9c2960584df/pkg/earthsci-astdiff-py/src/earthsci_astdiff/recurrence.py#L452)

</div>

<div class="api-tab" data-lang="rust" data-label="Rust">
<p class="api-tab-label">Rust</p>

```rust
pub fn transfer_definition(rec: &Recurrence, obs: &HashMap<String, Expr>, recs: &[Recurrence], ctx: &Ctx, tname: &str) -> Result<Expr, BandError>
```

The transfer recurrence `T` of `rec` (spec §4.5): over the frame plus one
output index `j` on the axis,

```text
T[o, k, j] = [k == j] + Σ_{self bands} coef · T[o, k − lag, j]
```

so `T[o, ·, ·] = (I − G)⁻¹` on each line: `∂V[o, k] / ∂(source at [o, j])`.
Lower triangular on each line (zero for `j > k`).

[Source](https://github.com/EarthSciML/EarthSciASTDiff/blob/2a7603b0bd3d2f1c582d4b8dc2f8f9c2960584df/pkg/earthsci-astdiff-rs/src/recurrence.rs#L626) · [Rust docs](https://docs.rs/earthsci-astdiff/latest/earthsci_astdiff/recurrence/fn.transfer_definition.html)

</div>

</div>
