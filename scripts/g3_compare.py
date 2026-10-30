# G3 compare. Hand-reviewed; do not edit. Called by scripts/g3_check.sh.
# Usage: g3_compare.py RESPONSE.bin EXPECTED.bin TOL LAYER
# Both files: raw float32 [8, 768]. Fails if max abs diff > TOL, on size
# mismatch, or on any NaN (NaN never compares <= TOL).
import sys
import numpy as np

resp_path, exp_path, tol, layer = sys.argv[1], sys.argv[2], float(sys.argv[3]), int(sys.argv[4])
T, H = 8, 768

r = np.fromfile(resp_path, dtype=np.float32)
e = np.fromfile(exp_path, dtype=np.float32)
if r.size != T * H or e.size != T * H:
    print(f"FAIL h{layer:02d}: size response={r.size} expected={e.size}, want {T * H}")
    sys.exit(1)

d = np.abs(r.astype(np.float64) - e.astype(np.float64))
# Relative diff floor 1e-6 avoids dividing by ~0 expected values; reported only.
rel = d / np.maximum(np.abs(e.astype(np.float64)), 1e-6)
i = int(np.argmax(d))
max_abs = float(d.max())
ok = max_abs <= tol
print(f"{'PASS' if ok else 'FAIL'} h{layer:02d}: max_abs={max_abs:.3e} at [{i // H},{i % H}] "
      f"max_rel={float(rel.max()):.3e}  (tol abs {tol:.1e})")
sys.exit(0 if ok else 1)
