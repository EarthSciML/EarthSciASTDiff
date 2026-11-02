---
title: "derivative_rule_error"
description: "Stable error · Differentiate a model"
type: "api"
api_tabs: true
---


**Tier:** stable · **Kind:** error · **Topic:** [Differentiate a model](/generated/api/topics/differentiate)

| Julia | Python | Rust |
|---|---|---|
| `DerivativeRuleError` | `DerivativeRuleError` | `DerivativeRuleError` |


<div class="api-tabs">

<div class="api-tab" data-lang="julia" data-label="Julia">
<p class="api-tab-label">Julia</p>

```julia
struct DerivativeRuleError <: Exception
DerivativeRuleError(msg::String)
DerivativeRuleError(msg)
```

```julia
DerivativeRuleError(msg)
```

A scalar operator has no derivative rule (spec §3).

[Source](https://github.com/EarthSciML/EarthSciASTDiff/blob/6f151823f210d8951df2266333997c204f54e68e/pkg/EarthSciASTDiff.jl/src/scalar_rules.jl#L47)

</div>

<div class="api-tab" data-lang="python" data-label="Python">
<p class="api-tab-label">Python</p>

```python
class DerivativeRuleError(Exception)
```

Raised when a scalar operator has no derivative rule (spec §3).

[Source](https://github.com/EarthSciML/EarthSciASTDiff/blob/6f151823f210d8951df2266333997c204f54e68e/pkg/earthsci-astdiff-py/src/earthsci_astdiff/scalar_rules.py#L31)

</div>

<div class="api-tab" data-lang="rust" data-label="Rust">
<p class="api-tab-label">Rust</p>

```rust
pub struct DerivativeRuleError(pub String);
```

A scalar operator has no derivative rule (spec §3).

[Source](https://github.com/EarthSciML/EarthSciASTDiff/blob/6f151823f210d8951df2266333997c204f54e68e/pkg/earthsci-astdiff-rs/src/rules.rs#L14) · [Rust docs](https://docs.rs/earthsci-astdiff/latest/earthsci_astdiff/rules/struct.DerivativeRuleError.html)

</div>

</div>
