---
title: "band_error"
description: "Stable error · Differentiate a model"
type: "api"
api_tabs: true
---


**Tier:** stable · **Kind:** error · **Topic:** [Differentiate a model](/generated/api/topics/differentiate)

| Julia | Python | Rust |
|---|---|---|
| `BandError` | `BandError` | `BandError` |


<div class="api-tabs">

<div class="api-tab" data-lang="julia" data-label="Julia">
<p class="api-tab-label">Julia</p>

```julia
struct BandError <: Exception
BandError(msg::String)
BandError(msg)
```

```julia
BandError(msg)
```

An expression has a form the band calculus cannot differentiate (for example a
filtered `faq` node or a non-`"+"` semiring).

[Source](https://github.com/EarthSciML/EarthSciASTDiff/blob/6f151823f210d8951df2266333997c204f54e68e/pkg/EarthSciASTDiff.jl/src/bands.jl#L48)

</div>

<div class="api-tab" data-lang="python" data-label="Python">
<p class="api-tab-label">Python</p>

```python
class BandError(Exception)
```

Raised when an expression has a form the band calculus cannot differentiate
(for example a filtered `faq` node or a non-`"+"` semiring).

[Source](https://github.com/EarthSciML/EarthSciASTDiff/blob/6f151823f210d8951df2266333997c204f54e68e/pkg/earthsci-astdiff-py/src/earthsci_astdiff/bands.py#L40)

</div>

<div class="api-tab" data-lang="rust" data-label="Rust">
<p class="api-tab-label">Rust</p>

```rust
pub struct BandError(pub String);
```

An expression has a form the band calculus cannot differentiate (for
example a filtered `faq` node or a non-`"+"` semiring).

[Source](https://github.com/EarthSciML/EarthSciASTDiff/blob/6f151823f210d8951df2266333997c204f54e68e/pkg/earthsci-astdiff-rs/src/bands.rs#L40) · [Rust docs](https://docs.rs/earthsci-astdiff/latest/earthsci_astdiff/bands/struct.BandError.html)

</div>

</div>
