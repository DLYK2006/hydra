#!/usr/bin/env python3
"""
compare_ln.py -- diff C++ LayerNorm output against PyTorch reference.

Usage:
  python compare_ln.py <T>

  T  -- sequence length used when running the orchestrator (1 or 4).

Reads:
  data/ln_expected_T{T}.bin   -- PyTorch reference (from extract_gpt2_ln.py)
  data/ln_response_T{T}.bin   -- C++ worker output (written by orchestrator)

Tolerance: < 1e-5 (tighter than the 1e-4 used for the linear layer because
LayerNorm has no matmul accumulation -- just mean, variance, scale, shift).
"""

import sys
import struct

if len(sys.argv) != 2:
    print("Usage: python compare_ln.py <T>")
    sys.exit(1)

T = int(sys.argv[1])
HIDDEN_DIM = 768
TOLERANCE = 1e-5

def read_bin(path):
    with open(path, "rb") as f:
        raw = f.read()
    n = len(raw) // 4
    return list(struct.unpack(f"<{n}f", raw))

try:
    expected = read_bin(f"data/ln_expected_T{T}.bin")
except FileNotFoundError:
    print(f"ERROR: data/ln_expected_T{T}.bin not found -- run extract_gpt2_ln.py {T} first")
    sys.exit(1)

try:
    actual = read_bin(f"data/ln_response_T{T}.bin")
except FileNotFoundError:
    print(f"ERROR: data/ln_response_T{T}.bin not found -- run the orchestrator first")
    sys.exit(1)

n_expected = T * HIDDEN_DIM
if len(expected) != n_expected:
    print(f"ERROR: expected file has {len(expected)} values, want {n_expected}")
    sys.exit(1)
if len(actual) != n_expected:
    print(f"ERROR: response file has {len(actual)} values, want {n_expected}")
    sys.exit(1)

diffs = [abs(a - e) for a, e in zip(actual, expected)]
max_diff = max(diffs)
max_idx  = diffs.index(max_diff)
avg_diff = sum(diffs) / len(diffs)

row = max_idx // HIDDEN_DIM
col = max_idx %  HIDDEN_DIM

print(f"Values compared   : {len(actual)}  (T={T}, H={HIDDEN_DIM})")
print(f"Max abs diff      : {max_diff:.2e}  at [row={row}, col={col}]")
print(f"  expected[{max_idx}] = {expected[max_idx]:.8f}")
print(f"  actual[{max_idx}]   = {actual[max_idx]:.8f}")
print(f"Avg abs diff      : {avg_diff:.2e}")
print(f"Tolerance check   : {'PASS' if max_diff < TOLERANCE else 'FAIL'} (threshold {TOLERANCE:.0e})")
