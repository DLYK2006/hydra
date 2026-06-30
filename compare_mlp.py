#!/usr/bin/env python3
"""
compare_mlp.py -- diff C++ MLP output against double-precision PyTorch reference.

Usage:
  python compare_mlp.py [T]

  T -- sequence length used when running the orchestrator (default 8).

Reads:
  data/mlp_response_gelu_T{T}.bin    -- C++ post-GELU intermediate [T, 3072]
  data/mlp_expected_gelu_T{T}.bin    -- double-prec reference post-GELU [T, 3072]
  data/mlp_response_T{T}.bin         -- C++ final output [T, 768]
  data/mlp_expected_T{T}.bin         -- double-prec reference final output [T, 768]

Reports max abs diff for BOTH stages independently so a GELU bug is isolated
  from a second-matmul bug without re-running the worker.

Tolerance: 1e-5.
"""

import sys
import struct

T          = int(sys.argv[1]) if len(sys.argv) > 1 else 8
HIDDEN_DIM = 768
MLP_HIDDEN = 3072
TOLERANCE  = 1e-5


def read_bin(path, n_floats):
    with open(path, "rb") as f:
        raw = f.read()
    if len(raw) != n_floats * 4:
        raise ValueError(f"{path}: expected {n_floats * 4} bytes, got {len(raw)}")
    return list(struct.unpack(f"<{n_floats}f", raw))


def compare(actual_path, expected_path, n_floats, col_width, label):
    try:
        actual   = read_bin(actual_path,   n_floats)
        expected = read_bin(expected_path, n_floats)
    except FileNotFoundError as e:
        print(f"\n--- {label} ---")
        print(f"ERROR: {e}")
        return False
    except ValueError as e:
        print(f"\n--- {label} ---")
        print(f"ERROR: {e}")
        return False

    diffs   = [abs(a - e) for a, e in zip(actual, expected)]
    max_d   = max(diffs)
    avg_d   = sum(diffs) / len(diffs)
    max_idx = diffs.index(max_d)
    row     = max_idx // col_width
    col     = max_idx %  col_width
    passed  = max_d < TOLERANCE

    print(f"\n--- {label} ---")
    print(f"Values compared  : {n_floats}  (T={T})")
    print(f"Max abs diff     : {max_d:.2e}  at [row={row}, col={col}]")
    print(f"  expected[{max_idx}] = {expected[max_idx]:.8f}")
    print(f"  actual  [{max_idx}] = {actual[max_idx]:.8f}")
    print(f"Avg abs diff     : {avg_d:.2e}")
    print(f"Tolerance check  : {'PASS' if passed else 'FAIL'} (threshold {TOLERANCE:.0e})")
    return passed


gelu_ok = compare(
    f"data/mlp_response_gelu_T{T}.bin",
    f"data/mlp_expected_gelu_T{T}.bin",
    T * MLP_HIDDEN, MLP_HIDDEN,
    f"post-GELU intermediate  [T={T}, MLP_HIDDEN={MLP_HIDDEN}]")

final_ok = compare(
    f"data/mlp_response_T{T}.bin",
    f"data/mlp_expected_T{T}.bin",
    T * HIDDEN_DIM, HIDDEN_DIM,
    f"final output            [T={T}, HIDDEN_DIM={HIDDEN_DIM}]")

print(f"\nOverall: {'PASS' if gelu_ok and final_ok else 'FAIL'}")
