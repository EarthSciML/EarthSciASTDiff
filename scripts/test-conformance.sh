#!/bin/bash
# Cross-language conformance runner for EarthSciASTDiff.
# Each language binding runs its test suite against the shared fixtures in
# tests/valid and the goldens in tests/goldens: Julia (reference), Python and
# Rust (mirrors EarthSciAST's runner). Every binding also carries the faq-tag
# non-empty gate, so a future wire-tag rename fails here instead of silently
# producing empty band lists.
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
