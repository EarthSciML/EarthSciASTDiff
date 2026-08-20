# earthsci-astdiff (Python)

Python binding of [EarthSciASTDiff](../../README.md): analytical sparse
Jacobians for `.esm` models, by rule-based differentiation of the expression
AST. Implements the derivative semantics of
[`esm-jacobian-spec.md`](../../esm-jacobian-spec.md) (§3 per-operator rules,
§4 band entries incl. `contracted` dimensions) against the
[`earthsci-ast`](../earthsci-ast-py) types, and is gated by the same
band-list goldens as the Julia reference (`tests/goldens/*.bands.json`,
byte-identical deterministic JSON).

```python
import earthsci_ast
from earthsci_astdiff import jacobian_bands, serialize_band, stable_json

file = earthsci_ast.load("model.esm")
entries = jacobian_bands(file, "MyModel")            # ∂f/∂u as Band entries
entries_p = jacobian_bands(file, "MyModel", wrt="parameters")
text = stable_json([serialize_band(e) for e in entries])
```

Scope: the band calculus (`jacobian_bands`), the §4 entry serialization
(`serialize_band` / `parse_band` / `stable_json`), the structural sparsity
pattern (`jacobian_pattern`, plain COO index lists — no scipy dependency),
and numeric assembly (`assemble_jacobian` → `{(row, col): value}` summed per
§4 additivity) — a correctness surface evaluated per cell through the
official numpy-interpreter runtime and gated against the Julia reference's
value goldens (`tests/goldens/*.jvals.json`, exact key set + 1e-10 relative).
The emission side is in as well: `jacobian_block` / `parse_jacobian_block`
with the §4.1 template CSE (`cse_templates` / `expand_templates`), the
structure classification, and the factorization plans (per-cell block LU,
Markowitz ordering + fill-in) — byte-identical to the reference's emitted
blocks (`tests/goldens/*.block.json`). Production-rate assembly follows.
