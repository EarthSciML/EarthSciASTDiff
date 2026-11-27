---
title: "sysview"
description: "Stable function · Differentiate a model"
type: "api"
api_tabs: true
---


**Tier:** stable · **Kind:** function · **Topic:** [Differentiate a model](/generated/api/topics/differentiate)

| Julia | Python | Rust |
|---|---|---|
| `sysview` | `sysview` | `sysview` |


**Note:** Julia and Python take a document's model or a flattened system through one `sysview`; Rust views a flattened system through `sysview_flat`. The view holds the document's `data_sources` (a document's model takes its own; a flattened system has none, so Julia and Python take a `data_sources` keyword and Rust `SysView::with_data_sources`), which the derived documents copy for their data-fed parameters. Not given (`nothing`, `None`, Rust `with_data_sources(None)`) keeps the view's own in every binding.


<div class="api-tabs">

<div class="api-tab" data-lang="julia" data-label="Julia">
<p class="api-tab-label">Julia</p>

```julia
sysview(file::EarthSciAST.EsmFile, model_name; data_sources)
sysview(flat::EarthSciAST.FlattenedSystem; data_sources)
```

```julia
sysview(file::EsmFile, model_name; data_sources = file.data_sources) -> SysView
sysview(flat::FlattenedSystem; data_sources = nothing)              -> SysView
```

`data_sources` (a document's `data_sources`, name → `DataSource`) is what
the derived documents copy for each data-fed parameter (esm-jacobian-spec.md
§8.1). A `FlattenedSystem` does not carry them, so pass the flattened
document's own when its parameters read data.

[Source](https://github.com/EarthSciML/EarthSciASTDiff/blob/2c83701848fda145f8d65f939fb8cf382ee2681e/pkg/EarthSciASTDiff.jl/src/system.jl#L46)

**Example** (`bands`)

```julia
file = EarthSciAST.load_path(joinpath(FIXTURES, "bd_chem.esm"))
view = sysview(file, "Chem")
entries = jacobian_bands(view)
# One entry per (row variable, column variable) pair that occurs in the model.
@assert sort(unique([(en.u, en.v) for en in entries])) ==
    [("A", "A"), ("A", "B"), ("A", "C"), ("B", "A"), ("B", "B"), ("C", "B")]
```

<p class="api-src">From <a href="https://github.com/EarthSciML/EarthSciASTDiff/blob/2c83701848fda145f8d65f939fb8cf382ee2681e/docs/api/examples/julia/jacobians.jl#L12-L17">docs/api/examples/julia/jacobians.jl</a></p>

</div>

<div class="api-tab" data-lang="python" data-label="Python">
<p class="api-tab-label">Python</p>

```python
sysview(obj, model_name: str | None = None, data_sources=None) -> SysView
```

View a model of an `EsmFile`, or a `FlattenedSystem`, for differentiation.

**Parameters**

- `obj` — an `EsmFile`, a flattened system, or a [`SysView`](/generated/api/ref/sys_view)
  (returned as is, `model_name` ignored).
- `model_name` — which model of an `EsmFile`; may be omitted when the
  file has exactly one.
- `data_sources` — a document's `data_sources` (name → `DataSource`),
  the entries the derived documents copy for each data-fed parameter
  (spec §8.1). `None` keeps the input's own: an `EsmFile`'s, none for
  a flattened system, which does not carry them, so pass the flattened
  document's when its parameters read data.

[Source](https://github.com/EarthSciML/EarthSciASTDiff/blob/2c83701848fda145f8d65f939fb8cf382ee2681e/pkg/earthsci-astdiff-py/src/earthsci_astdiff/system.py#L42)

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

<p class="api-src">From <a href="https://github.com/EarthSciML/EarthSciASTDiff/blob/2c83701848fda145f8d65f939fb8cf382ee2681e/docs/api/examples/python/jacobians.py#L13-L24">docs/api/examples/python/jacobians.py</a></p>

</div>

<div class="api-tab" data-lang="rust" data-label="Rust">
<p class="api-tab-label">Rust</p>

```rust
pub fn sysview<'a>(file: &'a EsmFile, model_name: Option<&str>) -> Result<SysView<'a>, BandError>
```

View a model of `file` for differentiation. `model_name` may be `None`
when the file has exactly one model.

[Source](https://github.com/EarthSciML/EarthSciASTDiff/blob/2c83701848fda145f8d65f939fb8cf382ee2681e/pkg/earthsci-astdiff-rs/src/system.rs#L77) · [Rust docs](https://docs.rs/earthsci-astdiff/latest/earthsci_astdiff/system/fn.sysview.html)

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

<p class="api-src">From <a href="https://github.com/EarthSciML/EarthSciASTDiff/blob/2c83701848fda145f8d65f939fb8cf382ee2681e/docs/api/examples/rust/jacobians.rs#L13-L24">docs/api/examples/rust/jacobians.rs</a></p>

</div>

</div>
