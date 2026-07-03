#!/bin/bash
# Test runner with OpenMP process isolation.
#
# torch and lightgbm each bundle their own libomp.dylib; importing both into
# one process crashes on macOS (EXC_BAD_ACCESS in __kmp_suspend_initialize_
# thread — two OpenMP runtimes fighting). pytest imports every test module
# into a single process at collection time, so torch-importing tests live in
# tests/torch_isolated/ and run in their own interpreter.
#
# The same rule applies to experiment scripts: never import torch and
# lightgbm in one process. GRU/TSFM forecasts are produced by separate
# processes that write parquet, then merged (see src/forecasters/neural.py).

set -e
cd "$(dirname "$0")/.."

echo "=== main suite (no torch) ==="
.venv/bin/python -m pytest tests/ --ignore=tests/torch_isolated -q "$@"

echo "=== torch-isolated suite ==="
.venv/bin/python -m pytest tests/torch_isolated -q "$@"

echo "ALL TEST GROUPS PASSED"
