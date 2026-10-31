---
title: "markowitz_ordering"
description: "Stable function · Sparsity and structure"
type: "api"
api_tabs: true
---


**Tier:** stable · **Kind:** function · **Topic:** [Sparsity and structure](/generated/api/topics/structure)

| Julia | Python | Rust |
|---|---|---|
| not exported | `markowitz_ordering` | `markowitz_ordering` |


<div class="api-tabs">

<div class="api-tab" data-lang="julia" data-label="Julia">
<p class="api-tab-label">Julia</p>

<p class="api-absent">Julia does not export this function.</p>

</div>

<div class="api-tab" data-lang="python" data-label="Python">
<p class="api-tab-label">Python</p>

```python
markowitz_ordering(pattern: list[list[bool]]) -> list[int]
```

Greedy Markowitz ordering (1-based), first minimal cost wins.

[Source](https://github.com/EarthSciML/EarthSciASTDiff/blob/a9ca79cb254eab4630ba5d3dc17b785c72fcfc79/pkg/earthsci-astdiff-py/src/earthsci_astdiff/factorization.py#L39)

</div>

<div class="api-tab" data-lang="rust" data-label="Rust">
<p class="api-tab-label">Rust</p>

```rust
pub fn markowitz_ordering(pattern: &[Vec<bool>]) -> Vec<usize>
```

A Markowitz pivot ordering of a block pattern (1-based), minimizing fill-in
greedily.

[Source](https://github.com/EarthSciML/EarthSciASTDiff/blob/a9ca79cb254eab4630ba5d3dc17b785c72fcfc79/pkg/earthsci-astdiff-rs/src/emit.rs#L299) · [Rust docs](https://docs.rs/earthsci-astdiff/latest/earthsci_astdiff/emit/fn.markowitz_ordering.html)

</div>

</div>
