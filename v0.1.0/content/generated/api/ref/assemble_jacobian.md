---
title: "assemble_jacobian"
description: "Stable function · Evaluate the Jacobian"
type: "api"
api_tabs: true
---


**Tier:** stable · **Kind:** function · **Topic:** [Evaluate the Jacobian](/generated/api/topics/evaluate)

| Julia | Python | Rust |
|---|---|---|
| `assemble_jacobian` | `assemble_jacobian` | `assemble_jacobian` |


<div class="api-tabs">

<div class="api-tab" data-lang="julia" data-label="Julia">
<p class="api-tab-label">Julia</p>

```julia
assemble_jacobian(input; wrt, u, p, t, model_name, build_kwargs, source_layout)
```

```julia
assemble_jacobian(input; wrt = :states, u = nothing, p = nothing, t = 0.0,
                  model_name = nothing)
    -> (J, pattern, rownames, colnames, entries, structure)
```

One-shot convenience over [`prepare_jacobian`](/generated/api/ref/prepare_jacobian): evaluate the analytical
Jacobian at a point (defaults: the document's initial state and parameters).

[Source](https://github.com/EarthSciML/EarthSciASTDiff/blob/6f151823f210d8951df2266333997c204f54e68e/pkg/EarthSciASTDiff.jl/src/assemble.jl#L331)

**Example** (`assemble`)

```julia
res = assemble_jacobian(file)                # at the document's initial state
@assert res.structure == :block_diagonal
@assert res.J[1, 1] == -0.04                 # -k1
@assert res.J[5, 5] == -3.0                  # -2 k2 B, with B = 0.5 by default
```

<p class="api-src">From <a href="https://github.com/EarthSciML/EarthSciASTDiff/blob/6f151823f210d8951df2266333997c204f54e68e/docs/api/examples/julia/jacobians.jl#L34-L37">docs/api/examples/julia/jacobians.jl</a></p>

</div>

<div class="api-tab" data-lang="python" data-label="Python">
<p class="api-tab-label">Python</p>

```python
assemble_jacobian(obj, model_name: str | None = None, wrt: str = 'states', states: dict | None = None, params: dict | None = None, t: float = 0.0) -> dict[tuple[str, str], float]
```

Evaluate the band entries at a point: `{(rowname, colname): value}`
with contributions to one (row, col) pair SUMMED (§4 additivity, incl.
contracted-point accumulation). `states`/`params` map variable name
→ nested list (1-based cells) or scalar; defaults come from the document.

[Source](https://github.com/EarthSciML/EarthSciASTDiff/blob/6f151823f210d8951df2266333997c204f54e68e/pkg/earthsci-astdiff-py/src/earthsci_astdiff/numeric.py#L132)

**Example** (`assemble`)

```python
values = earthsci_astdiff.assemble_jacobian(file, states={"A": [2.0] * 4})
assert values[("A[1]", "A[1]")] == -0.04  # -k1
assert values[("B[1]", "B[1]")] == -3.0   # -2 k2 B, with B = 0.5 by default
```

<p class="api-src">From <a href="https://github.com/EarthSciML/EarthSciASTDiff/blob/6f151823f210d8951df2266333997c204f54e68e/docs/api/examples/python/jacobians.py#L40-L42">docs/api/examples/python/jacobians.py</a></p>

</div>

<div class="api-tab" data-lang="rust" data-label="Rust">
<p class="api-tab-label">Rust</p>

```rust
pub fn assemble_jacobian(file: &EsmFile, model_name: Option<&str>, wrt: Wrt, env: &Env) -> Result<BTreeMap<(String, String), f64>, BandError>
```

Evaluate the band entries at a point: `{(rowname, colname): value}`,
contributions summed (§4 additivity, contracted points included).

[Source](https://github.com/EarthSciML/EarthSciASTDiff/blob/6f151823f210d8951df2266333997c204f54e68e/pkg/earthsci-astdiff-rs/src/numeric.rs#L298) · [Rust docs](https://docs.rs/earthsci-astdiff/latest/earthsci_astdiff/numeric/fn.assemble_jacobian.html)

**Example** (`assemble`)

```rust
let env = Env {
    states: HashMap::from([
        ("A".to_string(), json!([2.0, 2.0, 2.0, 2.0])),
        ("B".to_string(), json!([0.5, 0.5, 0.5, 0.5])),
        ("C".to_string(), json!([0.1, 0.1, 0.1, 0.1])),
    ]),
    params: HashMap::from([
        ("k1".to_string(), json!(0.04)),
        ("k2".to_string(), json!(3.0)),
        ("k3".to_string(), json!(1.0)),
    ]),
    t: 0.0,
};
let values = assemble_jacobian(&file, None, Wrt::States, &env)?;
assert_eq!(values[&("A[1]".to_string(), "A[1]".to_string())], -0.04); // -k1
assert_eq!(values[&("B[1]".to_string(), "B[1]".to_string())], -3.0); // -2 k2 B
```

<p class="api-src">From <a href="https://github.com/EarthSciML/EarthSciASTDiff/blob/6f151823f210d8951df2266333997c204f54e68e/docs/api/examples/rust/jacobians.rs#L40-L55">docs/api/examples/rust/jacobians.rs</a></p>

</div>

</div>
