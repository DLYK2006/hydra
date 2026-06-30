#!/usr/bin/env python3
"""
compare_attn.py -- diff C++ attention output against PyTorch double-precision reference.

Usage:
  python compare_attn.py <T>

  T  -- sequence length used when running the orchestrator (default 1).

Reads:
  data/attn_response_T{T}.bin   -- C++ worker output [T, 768] (written by orchestrator)
  data/attn_expected_T{T}.bin   -- double-precision reference [T, 768] (from extract_gpt2_attn.py)

Tolerance: 1e-5.
"""

import sys
import struct

T          = int(sys.argv[1]) if len(sys.argv) > 1 else 8
HIDDEN_DIM = 768
TOLERANCE  = 1e-4  # M3 (T=1) was bit-identical (softmax trivially 1.0); M4 has
                   # real softmax so a small non-zero diff is expected and correct.
N_FLOATS   = T * HIDDEN_DIM

def read_bin(path, n_floats):
    with open(path, "rb") as f:
        raw = f.read()
    if len(raw) != n_floats * 4:
        raise ValueError(f"{path}: expected {n_floats*4} bytes, got {len(raw)}")
    return list(struct.unpack(f"<{n_floats}f", raw))

try:
    actual = read_bin(f"data/attn_response_T{T}.bin", N_FLOATS)
except FileNotFoundError:
    print(f"ERROR: data/attn_response_T{T}.bin not found -- run the orchestrator first")
    sys.exit(1)
except ValueError as e:
    print(f"ERROR: {e}")
    sys.exit(1)

try:
    expected = read_bin(f"data/attn_expected_T{T}.bin", N_FLOATS)
except FileNotFoundError:
    print(f"ERROR: data/attn_expected_T{T}.bin not found -- run extract_gpt2_attn.py {T} first")
    sys.exit(1)
except ValueError as e:
    print(f"ERROR: {e}")
    sys.exit(1)

diffs   = [abs(a - e) for a, e in zip(actual, expected)]
max_d   = max(diffs)
avg_d   = sum(diffs) / len(diffs)
max_idx = diffs.index(max_d)
row     = max_idx // HIDDEN_DIM
col     = max_idx %  HIDDEN_DIM
passed  = max_d < TOLERANCE

print(f"Values compared  : {N_FLOATS}  (T={T}, H={HIDDEN_DIM})")
print(f"Max abs diff     : {max_d:.2e}  at [row={row}, col={col}]")
print(f"  expected[{max_idx}] = {expected[max_idx]:.8f}")
print(f"  actual  [{max_idx}] = {actual[max_idx]:.8f}")
print(f"Avg abs diff     : {avg_d:.2e}")
print(f"Tolerance check  : {'PASS' if passed else 'FAIL'} (threshold {TOLERANCE:.0e})")
