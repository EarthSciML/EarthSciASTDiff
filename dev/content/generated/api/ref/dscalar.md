---
title: "dscalar"
description: "Stable function · Differentiate a model"
type: "api"
api_tabs: true
---


**Tier:** stable · **Kind:** function · **Topic:** [Differentiate a model](/generated/api/topics/differentiate)

| Julia | Python | Rust |
|---|---|---|
| `dscalar` | `dscalar` | `dscalar` |


<div class="api-tabs">

<div class="api-tab" data-lang="julia" data-label="Julia">
<p class="api-tab-label">Julia</p>

```julia
dscalar(e::EarthSciAST.ASTExpr, s::Site)
dscalar(e::EarthSciAST.ASTExpr, s::Site, occ::IdDict{EarthSciAST.OpExpr, Bool})
```

```julia
dscalar(e::ASTExpr, s::Site) -> ASTExpr
```

Derivative of the scalar expression `e` with respect to site `s`, as a new
ESM expression. Structural rules only; a subtree in which `s` does not occur
differentiates to a literal zero without descent. Throws
[`DerivativeRuleError`](/generated/api/ref/derivative_rule_error) on an op with no rule (rewrite targets,
joined or value-invention `faq` nodes in scalar position, unknown closed
functions).

[Source](https://github.com/EarthSciML/EarthSciASTDiff/blob/2a9b4a92b2b429cd94fc2ddac236c438425d94f9/pkg/EarthSciASTDiff.jl/src/scalar_rules.jl#L61)

</div>

<div class="api-tab" data-lang="python" data-label="Python">
<p class="api-tab-label">Python</p>

```python
dscalar(e: Expr, s: Site) -> Expr
```

Derivative of the scalar expression `e` w.r.t. site `s`.

[Source](https://github.com/EarthSciML/EarthSciASTDiff/blob/2a9b4a92b2b429cd94fc2ddac236c438425d94f9/pkg/earthsci-astdiff-py/src/earthsci_astdiff/scalar_rules.py#L59)

</div>

<div class="api-tab" data-lang="rust" data-label="Rust">
<p class="api-tab-label">Rust</p>

```rust
pub fn dscalar(e: &Expr, s: &Site) -> Result<Expr, DerivativeRuleError>
```

The derivative of the scalar expression `e` with respect to the site `s`
(a variable or an indexed read), by the §3 rule table.

[Source](https://github.com/EarthSciML/EarthSciASTDiff/blob/2a9b4a92b2b429cd94fc2ddac236c438425d94f9/pkg/earthsci-astdiff-rs/src/rules.rs#L45) · [Rust docs](https://docs.rs/earthsci-astdiff/latest/earthsci_astdiff/rules/fn.dscalar.html)

</div>

</div>
