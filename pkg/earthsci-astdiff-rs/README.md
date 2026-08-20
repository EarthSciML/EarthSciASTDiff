# earthsci-astdiff (Rust)

Rust binding of [EarthSciASTDiff](../../README.md): analytical sparse
Jacobians for `.esm` models by rule-based differentiation of the expression
AST, against the [`earthsci-ast`](../earthsci-ast-rs) types. Implements
[`esm-jacobian-spec.md`](../../esm-jacobian-spec.md) §3 (derivative rules)
and §4 (band entries incl. `contracted` dimensions, template CSE, structure
classification, block-LU factorization plans), gated byte-for-byte by the
same goldens as the Julia reference and the Python binding
(`tests/goldens/*.bands.json`, `*.block.json`) and by the cross-runtime
value goldens (`*.jvals.json`, exact key set + 1e-10 relative).

```rust
let file = earthsci_ast::load_path("model.esm")?;
let entries = earthsci_astdiff::jacobian_bands(&file, None, Wrt::States)?;
let text = earthsci_astdiff::stable_json(
    &serde_json::Value::Array(entries.iter().map(serialize_band).collect()));
```
