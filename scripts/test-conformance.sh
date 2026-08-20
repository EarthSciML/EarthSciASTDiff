#!/bin/bash
# Cross-language conformance runner for EarthSciASTDiff.
# Each language binding runs its test suite against the shared fixtures in
# tests/valid and the goldens in tests/goldens. Currently: Julia only; add a
# section per binding as they land (mirrors EarthSciAST's runner).
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "=== Julia (reference) ==="
julia --project="$ROOT/pkg/EarthSciASTDiff.jl" -e 'using Pkg; Pkg.test()'

echo "=== Python ==="
(cd "$ROOT/pkg/earthsci-astdiff-py" && python3 -m pytest tests/ -q)
echo "=== Rust ==="
(cd "$ROOT/pkg/earthsci-astdiff-rs" && cargo test --quiet)
# echo "=== Go ==="       # cd "$ROOT/pkg/earthsci-astdiff-go" && go test ./...
# echo "=== TS ==="       # cd "$ROOT/pkg/earthsci-astdiff-ts" && npm test
echo "All conformance suites passed."
