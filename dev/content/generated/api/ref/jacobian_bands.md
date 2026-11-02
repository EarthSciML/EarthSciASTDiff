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


**Note:** Julia and Python also take a view; Rust differentiates a view (a flattened system's, from `sysview_flat`) through `jacobian_bands_view`.


<div class="api-tabs">

<div class="api-tab" data-lang="julia" data-label="Julia">
<p class="api-tab-label">Julia</p>

```julia
jacobian_bands(file::EarthSciAST.EsmFile, model_name; kw...)
jacobian_bands(flat::EarthSciAST.FlattenedSystem; kw...)
jacobian_bands(sv::SysView; wrt, forcing)
```

```julia
jacobian_bands(file::EsmFile, model_name; wrt = :states, forcing = nothing) -> Vector{JacEntry}
jacobian_bands(flat::FlattenedSystem;    wrt = :states, forcing = nothing) -> Vector{JacEntry}
jacobian_bands(sv::SysView;              wrt = :states, forcing = nothing) -> Vector{JacEntry}
```

Differentiate every differential equation of the system with respect to every
state (`wrt = :states`), every parameter outside the forcing set
(`wrt = :parameters`, scalar and array-valued alike), every forcing buffer
(`wrt = :forcing`, the cells of the buffers [`forcing_names`](/generated/api/ref/forcing_names) selects,
their record axis included) that structurally occurs in its (observed-inlined)
RHS — or with respect to time itself (`wrt = :time`, the single scalar site
`t`; the integer-valued `datetime.*` calendar fields are piecewise-constant
and differentiate to 0, the continuous `datetime.julian_day` to `d[t]/86400`,
and `interp.linear`/`interp.bilinear` of `t` to the active-segment slope).

`forcing` names the forcing set explicitly; `nothing` infers it from the
document (every array parameter with a `data` update). It also decides which
parameters the `:parameters` axis leaves to the forcing axis.

The pattern implied by the result is GLOBAL: `ifelse`/`min`/`max` branches
are unioned, so it is valid at every point and is a superset of any local
(point-wise) pattern.

On the `:parameters` and `:forcing` axes every column index must be closed at
build time (literals, folded metaparameters, `const` gathers, and the entry's
row and contracted index names); an unclosed one throws
(esm-jacobian-spec.md §4.3). A model whose right-hand sides read the forcing
set only where the derivative is identically zero (an `ifelse` condition, a
piecewise-constant op) has an empty forcing list.

[Source](https://github.com/EarthSciML/EarthSciASTDiff/blob/2a9b4a92b2b429cd94fc2ddac236c438425d94f9/pkg/EarthSciASTDiff.jl/src/jacobian.jl#L89)

**Example** (`bands`)

```julia
file = EarthSciAST.load_path(joinpath(FIXTURES, "bd_chem.esm"))
view = sysview(file, "Chem")
entries = jacobian_bands(view)
# One entry per (row variable, column variable) pair that occurs in the model.
@assert sort(unique([(en.u, en.v) for en in entries])) ==
    [("A", "A"), ("A", "B"), ("A", "C"), ("B", "A"), ("B", "B"), ("C", "B")]
```

<p class="api-src">From <a href="https://github.com/EarthSciML/EarthSciASTDiff/blob/2a9b4a92b2b429cd94fc2ddac236c438425d94f9/docs/api/examples/julia/jacobians.jl#L12-L17">docs/api/examples/julia/jacobians.jl</a></p>

</div>

<div class="api-tab" data-lang="python" data-label="Python">
<p class="api-tab-label">Python</p>

```python
jacobian_bands(obj, model_name: str | None = None, wrt: str = 'states', forcing=None) -> list[JacEntry]
```

Differentiate a model's right-hand side into a list of Jacobian bands.

**Parameters**

- `obj` — an `EsmFile`, a flattened system or a [`SysView`](/generated/api/ref/sys_view).
- `model_name` — which model of an `EsmFile`; may be omitted when the
  file has exactly one.
- `wrt` — `"states"` (default), `"parameters"` (every parameter
  outside the forcing set, array-valued ones cell by cell), `"forcing"`
  (the cells of the forcing buffers, record axis included) or `"time"`.
- `forcing` — the forcing set; `None` infers it (every array
  parameter with a `data` update, see [`forcing_names`](/generated/api/ref/forcing_names)).

**Returns**

One [`JacEntry`](/generated/api/ref/jac_entry) per (row variable, column variable) pair that occurs
in the model. The derivative is symbolic, so the result does not depend on
the point where it is later evaluated.

On the `"parameters"` and `"forcing"` axes every column index must be
closed at build time; an unclosed one raises [`BandError`](/generated/api/ref/band_error) (spec
§4.3). A model that reads the forcing set only where the derivative is
identically zero (an `ifelse` condition, a piecewise-constant op) has an
empty forcing list.

[Source](https://github.com/EarthSciML/EarthSciASTDiff/blob/2a9b4a92b2b429cd94fc2ddac236c438425d94f9/pkg/earthsci-astdiff-py/src/earthsci_astdiff/jacobian.py#L111)

**Example** (`bands`)

```python
file = earthsci_ast.load_path(FIXTURES / "bd_chem.esm")
entries = earthsci_astdiff.jacobian_bands(file)
# One entry per (row variable, column variable) pair that occurs in the model.
assert sorted({(en.u, en.v) for en in entries}) == [
    ("A", "A"),
    ("A", "B"),
    ("A", "C"),
    ("B", "A"),
    ("B", "B"),
    ("C", "B"),
]
view = earthsci_astdiff.sysview(file)
```

<p class="api-src">From <a href="https://github.com/EarthSciML/EarthSciASTDiff/blob/2a9b4a92b2b429cd94fc2ddac236c438425d94f9/docs/api/examples/python/jacobians.py#L13-L24">docs/api/examples/python/jacobians.py</a></p>

</div>

<div class="api-tab" data-lang="rust" data-label="Rust">
<p class="api-tab-label">Rust</p>

```rust
pub fn jacobian_bands(file: &EsmFile, model_name: Option<&str>, wrt: Wrt) -> Result<Vec<JacEntry>, BandError>
```

Differentiate a model's right-hand side into Jacobian bands with respect
to `wrt` (states, parameters, forcing or time). `model_name` may be
`None` when the file has exactly one model. The forcing set is inferred
(see `jacobian_bands_with_forcing` to pass it explicitly).

One [`JacEntry`](/generated/api/ref/jac_entry) is returned for each (row variable, column variable) pair
that occurs in the model; the derivative is symbolic, so it does not depend
on the point where it is later evaluated.

[Source](https://github.com/EarthSciML/EarthSciASTDiff/blob/2a9b4a92b2b429cd94fc2ddac236c438425d94f9/pkg/earthsci-astdiff-rs/src/jacobian.rs#L189) · [Rust docs](https://docs.rs/earthsci-astdiff/latest/earthsci_astdiff/jacobian/fn.jacobian_bands.html)

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

<p class="api-src">From <a href="https://github.com/EarthSciML/EarthSciASTDiff/blob/2a9b4a92b2b429cd94fc2ddac236c438425d94f9/docs/api/examples/rust/jacobians.rs#L13-L24">docs/api/examples/rust/jacobians.rs</a></p>

</div>

</div>
