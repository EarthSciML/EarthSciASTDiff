---
title: "jacobian_bands"
description: "Stable function · Differentiate a model"
type: "api"
api_tabs: true
---


**Tier:** stable · **Kind:** function · **Topic:** [Differentiate a model](/generated/api/topics/differentiate)

| Julia | Python | Rust |
|---|---|---|
| `jacobian_bands` | `jacobian_bands` | `jacobian_bands` |


<div class="api-tabs">

<div class="api-tab" data-lang="julia" data-label="Julia">
<p class="api-tab-label">Julia</p>

```julia
jacobian_bands(file::EarthSciAST.EsmFile, model_name; kw...)
jacobian_bands(flat::EarthSciAST.FlattenedSystem; kw...)
jacobian_bands(sv::SysView; wrt)
```

```julia
jacobian_bands(file::EsmFile, model_name; wrt = :states) -> Vector{JacEntry}
jacobian_bands(flat::FlattenedSystem;    wrt = :states) -> Vector{JacEntry}
jacobian_bands(sv::SysView;              wrt = :states) -> Vector{JacEntry}
```

Differentiate every differential equation of the system with respect to every
state (`wrt = :states`) or every parameter (`wrt = :parameters`) that
structurally occurs in its (observed-inlined) RHS — or with respect to time
itself (`wrt = :time`, the single scalar site `t`; the integer-valued
`datetime.*` calendar fields are piecewise-constant and differentiate to 0,
the continuous `datetime.julian_day` to `d[t]/86400`, and
`interp.linear`/`interp.bilinear` of `t` to the active-segment slope).

The pattern implied by the result is GLOBAL: `ifelse`/`min`/`max` branches
are unioned, so it is valid at every point and is a superset of any local
(point-wise) pattern.

[Source](https://github.com/EarthSciML/EarthSciASTDiff/blob/6f151823f210d8951df2266333997c204f54e68e/pkg/EarthSciASTDiff.jl/src/jacobian.jl#L16)

**Example** (`bands`)

```julia
file = EarthSciAST.load_path(joinpath(FIXTURES, "bd_chem.esm"))
view = sysview(file, "Chem")
entries = jacobian_bands(view)
# One entry per (row variable, column variable) pair that occurs in the model.
@assert sort(unique([(en.u, en.v) for en in entries])) ==
        [("A", "A"), ("A", "B"), ("A", "C"), ("B", "A"), ("B", "B"), ("C", "B")]
```

<p class="api-src">From <a href="https://github.com/EarthSciML/EarthSciASTDiff/blob/6f151823f210d8951df2266333997c204f54e68e/docs/api/examples/julia/jacobians.jl#L10-L15">docs/api/examples/julia/jacobians.jl</a></p>

</div>

<div class="api-tab" data-lang="python" data-label="Python">
<p class="api-tab-label">Python</p>

```python
jacobian_bands(obj, model_name: str | None = None, wrt: str = 'states') -> list[JacEntry]
```

Differentiate a model's right-hand side into a list of Jacobian bands.

**Parameters**

- `obj` — an `EsmFile`, a flattened system or a [`SysView`](/generated/api/ref/sys_view).
- `model_name` — which model of an `EsmFile`; may be omitted when the
  file has exactly one.
- `wrt` — `"states"` (default), `"parameters"` or `"time"`.

**Returns**

One [`JacEntry`](/generated/api/ref/jac_entry) per (row variable, column variable) pair that occurs
in the model. The derivative is symbolic, so the result does not depend on
the point where it is later evaluated.

[Source](https://github.com/EarthSciML/EarthSciASTDiff/blob/6f151823f210d8951df2266333997c204f54e68e/pkg/earthsci-astdiff-py/src/earthsci_astdiff/jacobian.py#L27)

**Example** (`bands`)

```python
file = earthsci_ast.load_path(FIXTURES / "bd_chem.esm")
entries = earthsci_astdiff.jacobian_bands(file)
# One entry per (row variable, column variable) pair that occurs in the model.
assert sorted({(en.u, en.v) for en in entries}) == [
    ("A", "A"), ("A", "B"), ("A", "C"), ("B", "A"), ("B", "B"), ("C", "B"),
]
view = earthsci_astdiff.sysview(file)
```

<p class="api-src">From <a href="https://github.com/EarthSciML/EarthSciASTDiff/blob/6f151823f210d8951df2266333997c204f54e68e/docs/api/examples/python/jacobians.py#L13-L19">docs/api/examples/python/jacobians.py</a></p>

</div>

<div class="api-tab" data-lang="rust" data-label="Rust">
<p class="api-tab-label">Rust</p>

```rust
pub fn jacobian_bands(file: &EsmFile, model_name: Option<&str>, wrt: Wrt) -> Result<Vec<JacEntry>, BandError>
```

Differentiate a model's right-hand side into Jacobian bands with respect
to `wrt` (states, parameters or time). `model_name` may be `None` when the
file has exactly one model.

One [`JacEntry`](/generated/api/ref/jac_entry) is returned for each (row variable, column variable) pair
that occurs in the model; the derivative is symbolic, so it does not depend
on the point where it is later evaluated.

[Source](https://github.com/EarthSciML/EarthSciASTDiff/blob/6f151823f210d8951df2266333997c204f54e68e/pkg/earthsci-astdiff-rs/src/jacobian.rs#L47) · [Rust docs](https://docs.rs/earthsci-astdiff/latest/earthsci_astdiff/jacobian/fn.jacobian_bands.html)

**Example** (`bands`)

```rust
let file = load_path(fixture("bd_chem.esm"))?;
let entries = jacobian_bands(&file, None, Wrt::States)?;
// One entry per (row variable, column variable) pair that occurs in the model.
let mut pairs: Vec<(&str, &str)> =
    entries.iter().map(|en| (en.u.as_str(), en.v.as_str())).collect();
pairs.sort();
pairs.dedup();
assert_eq!(
    pairs,
    [("A", "A"), ("A", "B"), ("A", "C"), ("B", "A"), ("B", "B"), ("C", "B")]
);
let view = sysview(&file, None)?;
```

<p class="api-src">From <a href="https://github.com/EarthSciML/EarthSciASTDiff/blob/6f151823f210d8951df2266333997c204f54e68e/docs/api/examples/rust/jacobians.rs#L16-L27">docs/api/examples/rust/jacobians.rs</a></p>

</div>

</div>
