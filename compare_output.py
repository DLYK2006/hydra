#!/usr/bin/env python3
"""
compare_output.py -- compares worker round-trip output against PyTorch reference.

Usage:
  python compare_output.py <worker_output.txt>

where worker_output.txt contains the orchestrator's stdout from a run that
printed the response data line, e.g.:
  Received response for layer 0 from 'worker-mac', shape [128], data = [0.123, ...]

Reads data/expected_y.bin (written by train_mnist_layer0.py) and reports
max abs difference. Tolerance: < 1e-4.
"""

import sys
import struct
import re

if len(sys.argv) != 2:
    print("Usage: python compare_output.py <orchestrator_stdout.txt>")
    sys.exit(1)

with open("data/expected_y.bin", "rb") as f:
    raw = f.read()
expected = list(struct.unpack(f"<{len(raw)//4}f", raw))

text = open(sys.argv[1]).read()
m = re.search(r"data = \[([^\]]+)\]", text)
if not m:
    print("ERROR: could not find 'data = [...]' in the provided file")
    sys.exit(1)

actual = [float(x.strip()) for x in m.group(1).split(",")]

if len(actual) != len(expected):
    print(f"ERROR: expected {len(expected)} values, got {len(actual)}")
    sys.exit(1)

diffs = [abs(a - e) for a, e in zip(actual, expected)]
max_diff = max(diffs)
max_idx  = diffs.index(max_diff)

print(f"Values compared : {len(actual)}")
print(f"Max abs diff    : {max_diff:.2e}  (index {max_idx})")
print(f"  expected[{max_idx}] = {expected[max_idx]:.6f}")
print(f"  actual[{max_idx}]   = {actual[max_idx]:.6f}")
print(f"Tolerance check : {'PASS' if max_diff < 1e-4 else 'FAIL (> 1e-4)'}")
