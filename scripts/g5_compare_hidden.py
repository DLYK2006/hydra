# G5 hidden-state gate. Hand-reviewed; do not edit. Called by scripts/g5_check.sh.
# Usage: g5_compare_hidden.py STAGE_DIR
# Requires STAGE_DIR/hidden_f32.bin (written by gen_hydra.py: every step's full
# block-11 output, float32, steps concatenated) to be byte-identical, step by step,
# to the committed chained-mirror oracle data/g5_mirror_hidden.bin, using the row
# offsets in data/g5_mirror_hidden_index.txt. Exit 0 only if all 20 steps match and
# the two files are the same size. Compares hidden states (the input to ln_f), not logits.
import sys
import numpy as np

HIDDEN, N_GEN = 768, 20
stage = sys.argv[1]

index = []
with open("data/g5_mirror_hidden_index.txt") as f:
    for line in f:
        if line.strip() and not line.startswith("#"):
            step, T, off = (int(v) for v in line.split())
            index.append((step, T, off))
if [s for s, _, _ in index] != list(range(N_GEN)):
    raise SystemExit(f"FAIL: index has steps {[s for s, _, _ in index]}, want 0..{N_GEN - 1}")
rows = index[-1][2] + index[-1][1]
for (s, T, off), (_, _, nxt) in zip(index, index[1:] + [(None, None, rows)]):
    if off + T != nxt:
        raise SystemExit(f"FAIL: index not contiguous at step {s}")

mirror = np.fromfile("data/g5_mirror_hidden.bin", dtype=np.float32)
hydra = np.fromfile(f"{stage}/hidden_f32.bin", dtype=np.float32)
if mirror.size != rows * HIDDEN or hydra.size != rows * HIDDEN:
    raise SystemExit(f"FAIL: floats mirror={mirror.size} hydra={hydra.size}, want {rows * HIDDEN}")
mirror, hydra = mirror.reshape(rows, HIDDEN), hydra.reshape(rows, HIDDEN)

ok = True
print(f"{'step':>4}  {'T':>2}  {'rows':>9}  {'diff_floats':>11}  {'max_abs':>10}  result")
for step, T, off in index:
    m, h = mirror[off:off + T], hydra[off:off + T]
    same = m.tobytes() == h.tobytes()
    ok &= same
    nd = 0 if same else int((m.view(np.uint32) != h.view(np.uint32)).sum())
    mx = 0.0 if same else float(np.abs(m.astype(np.float64) - h.astype(np.float64)).max())
    print(f"{step:>4}  {T:>2}  {off:>4}..{off + T - 1:<4}  {nd:>11}  {mx:>10.3e}  "
          f"{'IDENTICAL' if same else 'DIFFERS'}")
print("G5 hidden-state gate:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
