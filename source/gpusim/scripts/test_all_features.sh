#!/usr/bin/env bash
# Runs the gpusim test suite under all supported feature combinations.
# Requires a GPU-capable machine. Intended for CI or pre-release validation.
set -euo pipefail

cd "$(dirname "$0")/.."

echo "=== Testing f32 mode ==="
cargo test -p qdk-gpu-sim --features gpu-tests

echo "=== Testing f64 emulation mode ==="
cargo test -p qdk-gpu-sim --features gpu-tests,f64_emulation

echo "=== All feature combinations passed ==="
