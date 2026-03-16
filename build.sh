#!/usr/bin/env bash
set -euo pipefail

cmake --preset dev-debug
cmake --build build/debug --parallel

echo ""
echo "=== C++ tests ==="
./build/debug/moran_tests

echo ""
echo "=== Python tests ==="
python3 -m pytest tests/ -q
