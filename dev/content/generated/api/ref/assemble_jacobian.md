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
assemble_jacobian(input; wrt, u, p, t, model_name, forcing, build_kwargs, source_layout)
```

```julia
assemble_jacobian(input; wrt = :states, u = nothing, p = nothing, t = 0.0,
                  model_name = nothing, forcing = nothing)
    -> (J, pattern, rownames, colnames, entries, structure)
```

One-shot convenience over [`prepare_jacobian`](/generated/api/ref/prepare_jacobian): evaluate the analytical
Jacobian at a point (defaults: the document's initial state and parameters).

[Source](https://github.com/EarthSciML/EarthSciASTDiff/blob/a9ca79cb254eab4630ba5d3dc17b785c72fcfc79/pkg/EarthSciASTDiff.jl/src/assemble.jl#L477)

**Example** (`assemble`)

```julia
res = assemble_jacobian(file)                # at the document's initial state
@assert res.structure == :block_diagonal
@assert res.J[1, 1] == -0.04                 # -k1
@assert res.J[5, 5] == -3.0                  # -2 k2 B, with B = 0.5 by default
```

<p class="api-src">From <a href="https://github.com/EarthSciML/EarthSciASTDiff/blob/a9ca79cb254eab4630ba5d3dc17b785c72fcfc79/docs/api/examples/julia/jacobians.jl#L34-L37">docs/api/examples/julia/jacobians.jl</a></p>

</div>

<div class="api-tab" data-lang="python" data-label="Python">
<p class="api-tab-label">Python</p>

```python
assemble_jacobian(obj, model_name: str | None = None, wrt: str | None = None, states: dict | None = None, params: dict | None = None, t: float = 0.0, forcing=None) -> dict[tuple[str, str], float]
```

Evaluate the band entries at a point: `{(rowname, colname): value}`
with contributions to one (row, col) pair SUMMED (§4 additivity, incl.
contracted-point accumulation). `states`/`params` map variable name
→ nested list (1-based cells, row-major) or scalar; defaults come from
the document (array parameters and forcing buffers included, from their
inline array `default`).

On the `"parameters"` and `"forcing"` axes the column names are the
cells of the canonical column space ([`jacobian_columns`](/generated/api/ref/jacobian_columns)). A
forcing column outside its buffer's extent raises [`BandError`](/generated/api/ref/band_error) (no
ghost skip); a parameter or state column read past its extent is a ghost
read and is skipped. `wrt` defaults to `"states"`; `obj` may also
be a [`JacobianEvaluator`](/generated/api/ref/jacobian_evaluator), whose own axis applies.

[Source](https://github.com/EarthSciML/EarthSciASTDiff/blob/a9ca79cb254eab4630ba5d3dc17b785c72fcfc79/pkg/earthsci-astdiff-py/src/earthsci_astdiff/numeric.py#L248)

**Example** (`assemble`)

```python
values = earthsci_astdiff.assemble_jacobian(file, states={"A": [2.0] * 4})
assert values[("A[1]", "A[1]")] == -0.04  # -k1
assert values[("B[1]", "B[1]")] == -3.0  # -2 k2 B, with B = 0.5 by default
```

<p class="api-src">From <a href="https://github.com/EarthSciML/EarthSciASTDiff/blob/a9ca79cb254eab4630ba5d3dc17b785c72fcfc79/docs/api/examples/python/jacobians.py#L45-L47">docs/api/examples/python/jacobians.py</a></p>

</div>

<div class="api-tab" data-lang="rust" data-label="Rust">
<p class="api-tab-label">Rust</p>

```rust
pub fn assemble_jacobian(file: &EsmFile, model_name: Option<&str>, wrt: Wrt, env: &Env) -> Result<BTreeMap<(String, String), f64>, BandError>
```

Evaluate the band entries at a point: `{(rowname, colname): value}`,
contributions summed (§4 additivity, contracted points included). The
forcing set is inferred; see `assemble_jacobian_with_forcing`.

[Source](https://github.com/EarthSciML/EarthSciASTDiff/blob/a9ca79cb254eab4630ba5d3dc17b785c72fcfc79/pkg/earthsci-astdiff-rs/src/numeric.rs#L569) · [Rust docs](https://docs.rs/earthsci-astdiff/latest/earthsci_astdiff/numeric/fn.assemble_jacobian.html)

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

<p class="api-src">From <a href="https://github.com/EarthSciML/EarthSciASTDiff/blob/a9ca79cb254eab4630ba5d3dc17b785c72fcfc79/docs/api/examples/rust/jacobians.rs#L40-L55">docs/api/examples/rust/jacobians.rs</a></p>

</div>

</div>
