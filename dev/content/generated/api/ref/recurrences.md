---
title: "recurrences"
description: "Stable function · Differentiate a model"
type: "api"
api_tabs: true
---


**Tier:** stable · **Kind:** function · **Topic:** [Differentiate a model](/generated/api/topics/differentiate)

| Julia | Python | Rust |
|---|---|---|
| not exported | `recurrences` | `recurrences` |


**Note:** The causal self-reference recurrences of a view (spec §4.5), with frame and axis.


<div class="api-tabs">

<div class="api-tab" data-lang="julia" data-label="Julia">
<p class="api-tab-label">Julia</p>

<p class="api-absent">Julia does not export this function.</p>

</div>

<div class="api-tab" data-lang="python" data-label="Python">
<p class="api-tab-label">Python</p>

```python
recurrences(sv, obs: dict, ctx) -> list[Recurrence]
```

Every recurrence of the view, in equation order: each observed whose
definition reads itself, with its frame and axis read off the
definition. A self-read the producer cannot place on one axis is refused
(EarthSciAST rejects such a document anyway; the producer never
guesses).

[Source](https://github.com/EarthSciML/EarthSciASTDiff/blob/2a9b4a92b2b429cd94fc2ddac236c438425d94f9/pkg/earthsci-astdiff-py/src/earthsci_astdiff/recurrence.py#L144)

</div>

<div class="api-tab" data-lang="rust" data-label="Rust">
<p class="api-tab-label">Rust</p>

```rust
pub fn recurrences(sv: &SysView<'_>, obs: &HashMap<String, Expr>, ctx: &Ctx) -> Result<Vec<Recurrence>, BandError>
```

Every recurrence of the view, in equation order: each observed whose
definition reads itself, with its frame and axis read off the
definition. A self-read the producer cannot place on one axis is
refused (EarthSciAST rejects such a document anyway; the producer never
guesses).

[Source](https://github.com/EarthSciML/EarthSciASTDiff/blob/2a9b4a92b2b429cd94fc2ddac236c438425d94f9/pkg/earthsci-astdiff-rs/src/recurrence.rs#L143) · [Rust docs](https://docs.rs/earthsci-astdiff/latest/earthsci_astdiff/recurrence/fn.recurrences.html)

</div>

</div>
