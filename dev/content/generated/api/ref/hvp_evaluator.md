---
title: "hvp_evaluator"
description: "Stable type · Evaluate the Jacobian"
type: "api"
api_tabs: true
---


**Tier:** stable · **Kind:** type · **Topic:** [Evaluate the Jacobian](/generated/api/topics/evaluate)

| Julia | Python | Rust |
|---|---|---|
| `HVPEvaluator` | `HVPEvaluator` | `HvpEvaluator` |


<div class="api-tabs">

<div class="api-tab" data-lang="julia" data-label="Julia">
<p class="api-tab-label">Julia</p>

```julia
struct HVPEvaluator
HVPEvaluator(fJ!::F, oop::Bool, uj::Vector{Float64}, duj::Vector{Float64}, umap::Vector{Int64}, p_default::P, items::Vector{NTuple{4, Int64}}, rc::Vector{Tuple{Int64, Int64}}, rcidx::Vector{Int64}, M::Vector{Float64}, entries::Vector{JacEntry2}, rownames::Vector{String}, colnames::Vector{String}, col2names::Vector{String}, columns::Union{Nothing, Vector{Tuple{String, Vector{Int64}}}}, structure::Union{Nothing, Symbol}, wrt::Tuple{Symbol, Symbol}) where {F, P}
```

```julia
HVPEvaluator
```

Prepared second-order contraction for one axis pair: call
[`hvp!`](/generated/api/ref/hvp)`(g, hv, pairs, u, p, t)`. Fields of note: `entries`,
`rownames` (μ's index space, the states), `colnames` (w's: the states, or
`["t"]` on a time pair), `col2names` (g's: the second axis, in its canonical
order), `columns` (the second axis's column space on `:parameters` /
`:forcing`), `wrt`.

[Source](https://github.com/EarthSciML/EarthSciASTDiff/blob/a9ca79cb254eab4630ba5d3dc17b785c72fcfc79/pkg/EarthSciASTDiff.jl/src/second_order.jl#L186)

</div>

<div class="api-tab" data-lang="python" data-label="Python">
<p class="api-tab-label">Python</p>

```python
class HVPEvaluator
HVPEvaluator(sv: SysView, wrt, forcing=None)
```

A prepared second-order contraction for one axis pair, built once by
[`prepare_hvp`](/generated/api/ref/prepare_hvp): the second-order bands ([`jacobian_bands2`](/generated/api/ref/jacobian_bands2))
and every band point's (row, column, second column) placement. Each
[`hvp`](/generated/api/ref/hvp) call only resolves the coefficients at the point — the
Julia reference's `HVPEvaluator` in this binding's correctness-surface
form.

Fields of note: `wrt` (the pair), `entries`, `structure`,
`columns` (the second axis's [`jacobian_columns`](/generated/api/ref/jacobian_columns) space on
`"parameters"` / `"forcing"`, else `None`), `col2names` (that
space's cell names, else `None`) and `colnames` (`["t"]` on a time
pair, else `None`).

[Source](https://github.com/EarthSciML/EarthSciASTDiff/blob/a9ca79cb254eab4630ba5d3dc17b785c72fcfc79/pkg/earthsci-astdiff-py/src/earthsci_astdiff/numeric.py#L356)

</div>

<div class="api-tab" data-lang="rust" data-label="Rust">
<p class="api-tab-label">Rust</p>

```rust
pub struct HvpEvaluator {
    pub wrt: (Wrt, Wrt),
    pub entries: Vec<JacEntry2>,
    pub columns: Option<Vec<(String, Vec<i64>)>>,
    pub colnames: Vec<String>,
    pub structure: Option<&'static str>,
    /* private fields */
}

impl HvpEvaluator { pub fn hvp(&self, env: &Env, pairs: &[HvpPair], form: HvpForm, g: &mut BTreeMap<String, f64>) -> Result<(), BandError> }
```

A prepared second-order contraction for one axis pair, built once by
[`prepare_hvp`]: the second-order bands, their observed-inlined
coefficients, the second axis's column space and the parameter defaults.
Each `HvpEvaluator::hvp` only walks the band points at the given
`Env`.

[Source](https://github.com/EarthSciML/EarthSciASTDiff/blob/a9ca79cb254eab4630ba5d3dc17b785c72fcfc79/pkg/earthsci-astdiff-rs/src/numeric.rs#L692) · [Rust docs](https://docs.rs/earthsci-astdiff/latest/earthsci_astdiff/numeric/struct.HvpEvaluator.html)

</div>

</div>
