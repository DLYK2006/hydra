#!/usr/bin/env python3
"""
compare_block.py -- diff C++ full block output against double-precision reference.

Usage:
  python compare_block.py [T]

  T -- sequence length used when running the orchestrator (default 8).

Reads:
  data/block_response_T{T}.bin   -- C++ worker output [T, 768]
  data/block_expected_T{T}.bin   -- double-precision reference [T, 768]

Tolerance: 1e-4.  Residual adds between sub-blocks accumulate float32 error
  that the per-stage cast no longer resets, so M6 will not be bit-identical.
  1e-4 is the expected ceiling for two attention matmuls + two MLP matmuls
  with float32 residuals connecting them.
"""

import sys
import struct

T          = int(sys.argv[1]) if len(sys.argv) > 1 else 8
HIDDEN_DIM = 768
TOLERANCE  = 1e-4
N_FLOATS   = T * HIDDEN_DIM


def read_bin(path, n_floats):
    with open(path, "rb") as f:
        raw = f.read()
    if len(raw) != n_floats * 4:
        raise ValueError(f"{path}: expected {n_floats * 4} bytes, got {len(raw)}")
    return list(struct.unpack(f"<{n_floats}f", raw))


try:
    actual = read_bin(f"data/block_response_T{T}.bin", N_FLOATS)
except FileNotFoundError:
    print(f"ERROR: data/block_response_T{T}.bin not found -- run the orchestrator first")
    sys.exit(1)
except ValueError as e:
    print(f"ERROR: {e}")
    sys.exit(1)

try:
    expected = read_bin(f"data/block_expected_T{T}.bin", N_FLOATS)
except FileNotFoundError:
    print(f"ERROR: data/block_expected_T{T}.bin not found -- run extract_gpt2_block.py {T} first")
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
if not passed:
    print()
    print("If diff > 1e-3, this is a wiring bug (skipped residual or wrong LN placement),")
    print("not an accumulation issue.  Check: is ln_2 applied AFTER the first residual add?")
    print("Is the second residual add applied AFTER c_proj_mlp, not before?")
