---
title: "evaluation_document"
description: "Stable function · Evaluate the Jacobian"
type: "api"
api_tabs: true
---


**Tier:** stable · **Kind:** function · **Topic:** [Evaluate the Jacobian](/generated/api/topics/evaluate)

| Julia | Python | Rust |
|---|---|---|
| `evaluation_document` | `evaluation_document` | `evaluation_document` |


<div class="api-tabs">

<div class="api-tab" data-lang="julia" data-label="Julia">
<p class="api-tab-label">Julia</p>

```julia
evaluation_document(input; wrt, model_name, forcing, cse, cse_min_nodes)
```

```julia
evaluation_document(input; wrt = :states, model_name = nothing, forcing = nothing,
                    cse = true) -> (doc, fields)
```

The derived evaluation document of the first-order bands on axis `wrt`
(esm-jacobian-spec.md §8, model `"JacobianEval"`): the source's observed the
coefficients read, every source derivative zeroed, and one band-field state
`J_k` per entry whose `D` equation is that band's coefficient field. One
right-hand-side call of an unmodified EarthSciAST evaluator at `(u, p, t)`
leaves every band value in `du`; `fields[k]` names entry `k`'s field.
Placing the values in a matrix is a scatter plan's ([`bind_plan`](/generated/api/ref/bind_plan)).
With `cse`, repeated coefficient subtrees are hoisted into observed arrays
(`hoist_observed`).

[Source](https://github.com/EarthSciML/EarthSciASTDiff/blob/2a7603b0bd3d2f1c582d4b8dc2f8f9c2960584df/pkg/EarthSciASTDiff.jl/src/emit.jl#L483)

**Example** (`evaluate`)

```julia
# EarthSciASTDiff evaluates nothing: it emits ESM documents that EarthSciAST
# evaluates. The band fields J_k, as a derived model whose right-hand side,
# run once, leaves every band value in `du`. `fields[k]` names entry k's field.
doc, fields = evaluation_document(file)
@assert length(fields) == length(entries)
prob = EarthSciAST.esm_problem(
    JSON3.read(stable_json(doc)), (0.0, 1.0)
)
du = zero(prob.u0)
prob.f!(du, prob.u0, prob.p, 0.0)            # at the document's initial state
k = findfirst(en -> (en.u, en.v) == ("A", "A"), entries)
@assert du[prob.var_map["JacobianEval.$(fields[k])[1]"]] == -0.04   # -k1
# J·v as a derived model: inputs `_v_<C>`, outputs `_jvp_<R>`.
jdoc, (model, v, jvp) = jvp_document(file)
@assert model == "JvpEval"
J = jac_prototype(file)                      # sparse matrix on the fixed pattern
@assert size(J) == (12, 12) && nnz(J) == 24
# A scatter plan places the band fields in your own storage (here CSR),
# checked against the source before it is used.
bound = bind_plan(file, scatter_plan(file, Csr()))
@assert bound.len == 24
@assert all(1 <= k <= length(fields) for (k, _, _) in bound.pairs)
```

<p class="api-src">From <a href="https://github.com/EarthSciML/EarthSciASTDiff/blob/2a7603b0bd3d2f1c582d4b8dc2f8f9c2960584df/docs/api/examples/julia/jacobians.jl#L36-L57">docs/api/examples/julia/jacobians.jl</a></p>

</div>

<div class="api-tab" data-lang="python" data-label="Python">
<p class="api-tab-label">Python</p>

```python
evaluation_document(obj, model_name: str | None = None, wrt: str = 'states', forcing=None, cse: bool = True, cse_min_nodes: int = 12)
```

The derived evaluation document of the first-order bands on axis
`wrt` (spec §8, model `"JacobianEval"`): the source's observed the
coefficients read, every source derivative zeroed, and one band-field
state `J_k` per entry whose `D` equation is that band's coefficient
field. One right-hand-side call of an unmodified EarthSciAST evaluator at
`(u, p, t)` leaves every band value in `du`.

Returns `(doc, fields)`, `fields[k]` naming entry `k`'s field. With
`cse`, repeated coefficient subtrees are hoisted into observed arrays
(`hoist_observed`).

[Source](https://github.com/EarthSciML/EarthSciASTDiff/blob/2a7603b0bd3d2f1c582d4b8dc2f8f9c2960584df/pkg/earthsci-astdiff-py/src/earthsci_astdiff/emit.py#L596)

**Example** (`evaluate`)

```python
# EarthSciASTDiff evaluates nothing: it emits the band fields `J_k` as a derived
# ESM model that EarthSciAST evaluates.
doc, fields = earthsci_astdiff.evaluation_document(file)
assert "JacobianEval" in doc["models"]
assert len(fields) == len(entries)
# `J v` likewise: write `v` into the inputs io["v"], read `J v` from the
# derivatives of the outputs io["jvp"].
jdoc, io = earthsci_astdiff.jvp_document(file)
assert io["model"] in jdoc["models"]
assert io["v"]["A"] == "_v_A" and io["jvp"]["A"] == "_jvp_A"
```

<p class="api-src">From <a href="https://github.com/EarthSciML/EarthSciASTDiff/blob/2a7603b0bd3d2f1c582d4b8dc2f8f9c2960584df/docs/api/examples/python/jacobians.py#L45-L54">docs/api/examples/python/jacobians.py</a></p>

</div>

<div class="api-tab" data-lang="rust" data-label="Rust">
<p class="api-tab-label">Rust</p>

```rust
pub fn evaluation_document(sv: &SysView<'_>, entries: &[JacEntry], cse: bool, cse_min_nodes: usize) -> Result<(Value, Vec<String>), BandError>
```

The derived evaluation document and its band-state names (`J_1`, …).
With `cse`, repeated coefficient subtrees are first hoisted into observed
(`crate::eval_cse::hoist_observed`).

[Source](https://github.com/EarthSciML/EarthSciASTDiff/blob/2a7603b0bd3d2f1c582d4b8dc2f8f9c2960584df/pkg/earthsci-astdiff-rs/src/emit.rs#L1633) · [Rust docs](https://docs.rs/earthsci-astdiff/latest/earthsci_astdiff/emit/fn.evaluation_document.html)

**Example** (`evaluate`)

```rust
// EarthSciASTDiff evaluates nothing: it emits the band fields `J_k` as a
// derived ESM model that EarthSciAST evaluates.
let (doc, fields) = evaluation_document(&view, &entries, true, 12)?;
assert!(doc["models"]["JacobianEval"].is_object());
assert_eq!(fields.len(), entries.len());
// `J v` likewise: write `v` into the inputs `io.v`, read `J v` from the
// derivatives of the outputs `io.jvp`.
let (jdoc, io) = jvp_document(&view, &entries, Wrt::States, &ProductOptions::default())?;
assert!(jdoc["models"][io.model.as_str()].is_object());
assert_eq!(io.v["A"], "_v_A");
assert_eq!(io.jvp["A"], "_jvp_A");
```

<p class="api-src">From <a href="https://github.com/EarthSciML/EarthSciASTDiff/blob/2a7603b0bd3d2f1c582d4b8dc2f8f9c2960584df/docs/api/examples/rust/jacobians.rs#L41-L51">docs/api/examples/rust/jacobians.rs</a></p>

</div>

</div>
