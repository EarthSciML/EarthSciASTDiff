---
title: "extended_bands"
description: "Stable function · Differentiate a model"
type: "api"
api_tabs: true
---


**Tier:** stable · **Kind:** function · **Topic:** [Differentiate a model](/generated/api/topics/differentiate)

| Julia | Python | Rust |
|---|---|---|
| not exported | `extended_bands` | `extended_bands` |


**Note:** The extended first-order system through causal recurrences (spec §4.5): state bands with recurrence columns and each recurrence definition's own bands. Julia follows once EarthSciAST.jl runs recurrences.


<div class="api-tabs">

<div class="api-tab" data-lang="julia" data-label="Julia">
<p class="api-tab-label">Julia</p>

<p class="api-absent">Julia does not export this function.</p>

</div>

<div class="api-tab" data-lang="python" data-label="Python">
<p class="api-tab-label">Python</p>

```python
extended_bands(obj, model_name: str | None = None, wrt: str = 'states', forcing=None, second=None)
```

The extended first-order system of axis `wrt` (spec §4.5,
`ExtendedBands`): every state
equation's bands with respect to the axis's targets and the active
recurrences, and each active recurrence definition's own bands. With
`second` (an axis) the pass is a pair's first pass: seeded with the
second axis's targets, so the recurrences that move with either axis are
active and the observed that move with either are inlined (spec §4.4).
Without a recurrence `rows` is [`jacobian_bands`](/generated/api/ref/jacobian_bands)'s list and the
rest is empty.

[Source](https://github.com/EarthSciML/EarthSciASTDiff/blob/2a7603b0bd3d2f1c582d4b8dc2f8f9c2960584df/pkg/earthsci-astdiff-py/src/earthsci_astdiff/jacobian.py#L161)

</div>

<div class="api-tab" data-lang="rust" data-label="Rust">
<p class="api-tab-label">Rust</p>

```rust
pub fn extended_bands(sv: &SysView<'_>, wrt: Wrt, forcing: Option<&[String]>, second: Option<Wrt>) -> Result<ExtendedBands, BandError>
```

The extended first-order system of axis `wrt` (spec §4.5): every state
equation's bands with respect to the axis's targets and the active
recurrences, and each active recurrence definition's own bands. With
`second` the pass is a pair's first pass: seeded with the second axis's
targets, so the recurrences that move with either axis are active and
the observed that move with either are inlined (spec §4.4). Without a
recurrence `rows` is `jacobian_bands_view`'s list and the rest is
empty.

[Source](https://github.com/EarthSciML/EarthSciASTDiff/blob/2a7603b0bd3d2f1c582d4b8dc2f8f9c2960584df/pkg/earthsci-astdiff-rs/src/jacobian.rs#L278) · [Rust docs](https://docs.rs/earthsci-astdiff/latest/earthsci_astdiff/jacobian/fn.extended_bands.html)

</div>

</div>
