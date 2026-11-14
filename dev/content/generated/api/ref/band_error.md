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

[Source](https://github.com/EarthSciML/EarthSciASTDiff/blob/9891c3005a31b74a0547498d0d2981ebb69cdc14/pkg/EarthSciASTDiff.jl/src/bands.jl#L64)

</div>

<div class="api-tab" data-lang="python" data-label="Python">
<p class="api-tab-label">Python</p>

```python
class BandError(Exception)
```

Raised when an expression has a form the band calculus cannot differentiate
(for example a joined `faq` node or a boolean semiring).

[Source](https://github.com/EarthSciML/EarthSciASTDiff/blob/9891c3005a31b74a0547498d0d2981ebb69cdc14/pkg/earthsci-astdiff-py/src/earthsci_astdiff/bands.py#L51)

</div>

<div class="api-tab" data-lang="rust" data-label="Rust">
<p class="api-tab-label">Rust</p>

```rust
pub struct BandError(pub String);
```

An expression has a form the band calculus cannot differentiate (for
example a joined `faq` node, the `bool_and_or` semiring, or a `max` over
two contracted indices).

[Source](https://github.com/EarthSciML/EarthSciASTDiff/blob/9891c3005a31b74a0547498d0d2981ebb69cdc14/pkg/earthsci-astdiff-rs/src/bands.rs#L54) · [Rust docs](https://docs.rs/earthsci-astdiff/latest/earthsci_astdiff/bands/struct.BandError.html)

</div>

</div>
