---
title: "jvp_document"
description: "Stable function · Evaluate the Jacobian"
type: "api"
api_tabs: true
---


**Tier:** stable · **Kind:** function · **Topic:** [Evaluate the Jacobian](/generated/api/topics/evaluate)

| Julia | Python | Rust |
|---|---|---|
| `jvp_document` | `jvp_document` | `jvp_document` |


<div class="api-tabs">

<div class="api-tab" data-lang="julia" data-label="Julia">
<p class="api-tab-label">Julia</p>

```julia
jvp_document(input; wrt, model_name, forcing, cse, cse_min_nodes)
```

```julia
jvp_document(input; wrt = :states, model_name = nothing, forcing = nothing,
             cse = true) -> (doc, io)
vjp_document(input; …) -> (doc, io)
```

The product documents of esm-jacobian-spec.md §9.2 / §9.3 for one
first-order axis: a derived ESM document (model `"JvpEval"` / `"VjpEval"`)
whose right-hand side, evaluated by an unmodified EarthSciAST evaluator at
`(u, inputs, p, t)`, leaves `J v` (resp. `Jᵀ λ`) in the derivative slots of
its outputs. `input` is an `EsmFile` or a `FlattenedSystem`, as for
[`jacobian_bands`](/generated/api/ref/jacobian_bands).

`io` names the inputs and outputs: `io.v` maps each column variable to its
direction input and `io.jvp` each state to its output (`io.lam` / `io.vjp`
for `vjp_document`). The consumer writes the inputs into the evaluator's
state slots beside the source states and reads the outputs from `du`
through the evaluator's variable map.

A `VjpEval` document is an equality-filtered `faq` scatter: correct under
any evaluator, linear-time only under one that implements §9.3's
equality-filter scatter.

[Source](https://github.com/EarthSciML/EarthSciASTDiff/blob/2a9b4a92b2b429cd94fc2ddac236c438425d94f9/pkg/EarthSciASTDiff.jl/src/products.jl#L519)

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

<p class="api-src">From <a href="https://github.com/EarthSciML/EarthSciASTDiff/blob/2a9b4a92b2b429cd94fc2ddac236c438425d94f9/docs/api/examples/julia/jacobians.jl#L36-L57">docs/api/examples/julia/jacobians.jl</a></p>

</div>

<div class="api-tab" data-lang="python" data-label="Python">
<p class="api-tab-label">Python</p>

```python
jvp_document(obj, model_name: str | None = None, wrt: str = 'states', forcing=None, cse: bool = True, cse_min_nodes: int = 12)
```

The `J v` product document of spec §9.2 for one first-order axis.

A derived ESM document (model `"JvpEval"`) whose right-hand side,
evaluated by an unmodified EarthSciAST evaluator at `(u, inputs, p, t)`, leaves `J v` in the derivative slots of its outputs. `obj` is
an `EsmFile`, a flattened system or a [`SysView`](/generated/api/ref/sys_view), as for
[`jacobian_bands`](/generated/api/ref/jacobian_bands).

Returns `(doc, io)`: `io["model"]` is the model name,
`io["v"]` maps each column variable to its direction input and
`io["jvp"]` each state to its output. The consumer writes the inputs
into the evaluator's state slots beside the source states and reads the
outputs from `du` through the evaluator's variable map.

[Source](https://github.com/EarthSciML/EarthSciASTDiff/blob/2a9b4a92b2b429cd94fc2ddac236c438425d94f9/pkg/earthsci-astdiff-py/src/earthsci_astdiff/products.py#L945)

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

<p class="api-src">From <a href="https://github.com/EarthSciML/EarthSciASTDiff/blob/2a9b4a92b2b429cd94fc2ddac236c438425d94f9/docs/api/examples/python/jacobians.py#L45-L54">docs/api/examples/python/jacobians.py</a></p>

</div>

<div class="api-tab" data-lang="rust" data-label="Rust">
<p class="api-tab-label">Rust</p>

```rust
pub fn jvp_document(sv: &SysView<'_>, entries: &[JacEntry], wrt: Wrt, opts: &ProductOptions<'_>) -> Result<(Value, JvpIo), BandError>
```

The `J v` product document of spec §9.2 for the first-order `entries` on
axis `wrt` (as `crate::jacobian_bands_view` gives them): a derived ESM
document (model `"JvpEval"`) whose right-hand side, evaluated by an
unmodified EarthSciAST evaluator at `(u, inputs, p, t)`, leaves `J v` in
the derivative slots of its outputs. The consumer writes the inputs
(`JvpIo::v`) into the evaluator's state slots beside the source states
and reads the outputs (`JvpIo::jvp`) from `du`.

`entries` must be built under the same forcing set as `opts.forcing`
(`None` on both sides infers it): the document's columns come from
`wrt` and `opts.forcing`, and an entry reading a variable outside them is
refused. A column of that set the entries never read comes out zero.

Through an active recurrence (spec §4.5) the document is §9.7's tangent
sweep over the extended system, which this function builds itself:
`entries` then only serve that forcing-set check. Pass the extended
system's rows ([`crate::extended_bands`](/generated/api/ref/extended_bands)`(sv, wrt, forcing,
None)?.rows`, whose columns may also be a recurrence) rather than the
materialized bands, so the transfer recurrence is never composed.

[Source](https://github.com/EarthSciML/EarthSciASTDiff/blob/2a9b4a92b2b429cd94fc2ddac236c438425d94f9/pkg/earthsci-astdiff-rs/src/products.rs#L895) · [Rust docs](https://docs.rs/earthsci-astdiff/latest/earthsci_astdiff/products/fn.jvp_document.html)

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

<p class="api-src">From <a href="https://github.com/EarthSciML/EarthSciASTDiff/blob/2a9b4a92b2b429cd94fc2ddac236c438425d94f9/docs/api/examples/rust/jacobians.rs#L41-L51">docs/api/examples/rust/jacobians.rs</a></p>

</div>

</div>
