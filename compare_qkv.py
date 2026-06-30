#!/usr/bin/env python3
"""
compare_qkv.py -- diff C++ QKV projection output against PyTorch reference.

Usage:
  python compare_qkv.py <T>

  T  -- sequence length used when running the orchestrator (default 4).

Reads:
  data/qkv_response_T{T}.bin   -- C++ worker output [T, 2304] (written by orchestrator)
  data/qkv_expected_Q_T{T}.bin -- PyTorch Q reference [T, 768]
  data/qkv_expected_K_T{T}.bin -- PyTorch K reference [T, 768]
  data/qkv_expected_V_T{T}.bin -- PyTorch V reference [T, 768]

Q, K, V are compared independently: any error localised to one head
indicates a split indexing bug; all three failing uniformly points to the
matmul kernel itself.

Tolerance: 1e-5.
"""

import sys
import struct

T = int(sys.argv[1]) if len(sys.argv) > 1 else 4
HIDDEN_DIM = 768
OUT_DIM    = 3 * HIDDEN_DIM  # 2304
TOLERANCE  = 1e-5

def read_bin(path, n_floats):
    with open(path, "rb") as f:
        raw = f.read()
    if len(raw) != n_floats * 4:
        raise ValueError(f"{path}: expected {n_floats*4} bytes, got {len(raw)}")
    return list(struct.unpack(f"<{n_floats}f", raw))

try:
    resp_flat = read_bin(f"data/qkv_response_T{T}.bin", T * OUT_DIM)
except FileNotFoundError:
    print(f"ERROR: data/qkv_response_T{T}.bin not found -- run the orchestrator first")
    sys.exit(1)
except ValueError as e:
    print(f"ERROR: {e}")
    sys.exit(1)

# Response is [T, 2304] row-major; split into Q, K, V along the out axis.
def extract_head(flat, t, col_start, col_end):
    width = col_end - col_start
    out = []
    for row in range(t):
        for j in range(col_start, col_end):
            out.append(flat[row * OUT_DIM + j])
    return out

Q_resp = extract_head(resp_flat, T, 0,          HIDDEN_DIM)
K_resp = extract_head(resp_flat, T, HIDDEN_DIM, 2*HIDDEN_DIM)
V_resp = extract_head(resp_flat, T, 2*HIDDEN_DIM, OUT_DIM)

all_pass = True
for name, resp, exp_path in [
    ("Q", Q_resp, f"data/qkv_expected_Q_T{T}.bin"),
    ("K", K_resp, f"data/qkv_expected_K_T{T}.bin"),
    ("V", V_resp, f"data/qkv_expected_V_T{T}.bin"),
]:
    try:
        expected = read_bin(exp_path, T * HIDDEN_DIM)
    except FileNotFoundError:
        print(f"ERROR: {exp_path} not found -- run extract_gpt2_qkv.py {T} first")
        sys.exit(1)
    except ValueError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    diffs   = [abs(a - e) for a, e in zip(resp, expected)]
    max_d   = max(diffs)
    avg_d   = sum(diffs) / len(diffs)
    max_idx = diffs.index(max_d)
    row     = max_idx // HIDDEN_DIM
    col     = max_idx %  HIDDEN_DIM
    passed  = max_d < TOLERANCE

    print(f"--- {name} ({T}x{HIDDEN_DIM}) ---")
    print(f"  Max abs diff : {max_d:.2e}  at [row={row}, col={col}]")
    print(f"  Avg abs diff : {avg_d:.2e}")
    print(f"  Tolerance    : {'PASS' if passed else 'FAIL'} (threshold {TOLERANCE:.0e})")

    all_pass = all_pass and passed

print()
print("Overall:", "PASS" if all_pass else "FAIL")
