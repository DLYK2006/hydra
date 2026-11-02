# G5 compare (logit gate). Hand-reviewed; do not edit. Called by scripts/g5_check.sh.
# Usage: g5_compare.py STAGE_DIR
# Reads STAGE_DIR/tokens.txt and STAGE_DIR/logits_f64.bin (written by gen_hydra.py) and
# compares them to the G1 references. Exit 0 only if every gate passes:
#   1. the 20 data rows of tokens.txt are byte-identical to data/gen_ref_tokens.txt's;
#   2. per step, max|hydra_logits - gen_ref_logits_f64| <= G5 tolerance, parsed from the
#      "# G5 tolerance = ..." header line of data/gen_ref_summary.txt (1.170157e-04);
#   3. per step, both the reference and Hydra f64 top-1 minus top-2 margins > 2 x tolerance.
# NaN anywhere fails (NaN never compares <= / >).
import re
import sys
import numpy as np

N_GEN, VOCAB = 20, 50257
stage = sys.argv[1]


def data_rows(path):
    with open(path, "rb") as f:
        return [l for l in f.read().split(b"\n") if l and not l.startswith(b"#")]


tol = None
with open("data/gen_ref_summary.txt") as f:
    for line in f:
        m = re.match(r"# G5 tolerance = 1\.6 x max\(max_abs_diff\) = ([0-9.eE+-]+)\s*$", line)
        if m:
            tol = float(m.group(1))
if tol is None:
    raise SystemExit("FAIL: no '# G5 tolerance = ...' line in data/gen_ref_summary.txt")

ref_rows, hyd_rows = data_rows("data/gen_ref_tokens.txt"), data_rows(f"{stage}/tokens.txt")
ref = np.fromfile("data/gen_ref_logits_f64.bin", dtype=np.float64)
hyd = np.fromfile(f"{stage}/logits_f64.bin", dtype=np.float64)
if len(ref_rows) != N_GEN or len(hyd_rows) != N_GEN:
    raise SystemExit(f"FAIL: token rows ref={len(ref_rows)} hydra={len(hyd_rows)}, want {N_GEN}")
if ref.size != N_GEN * VOCAB or hyd.size != N_GEN * VOCAB:
    raise SystemExit(f"FAIL: logits floats ref={ref.size} hydra={hyd.size}, want {N_GEN * VOCAB}")
ref, hyd = ref.reshape(N_GEN, VOCAB), hyd.reshape(N_GEN, VOCAB)


def margin(row):
    top2 = np.sort(row)[-2:]
    return float(top2[1] - top2[0])


ok = True
print(f"G5 tolerance {tol:.6e}  (margin floor 2 x tol = {2 * tol:.6e})")
print(f"{'step':>4}  {'tok':>5}  {'ref_tok':>7}  {'max_abs':>10}  {'margin_hyd':>11}  {'margin_ref':>11}  verdict")
for s in range(N_GEN):
    d = float(np.abs(hyd[s] - ref[s]).max())
    mh, mr = margin(hyd[s]), margin(ref[s])
    tok_ok = hyd_rows[s] == ref_rows[s]
    step_ok = tok_ok and d <= tol and mh > 2 * tol and mr > 2 * tol
    ok &= step_ok
    th = int(hyd_rows[s].split()[1]); tr = int(ref_rows[s].split()[1])
    print(f"{s:>4}  {th:>5}  {tr:>7}  {d:>10.3e}  {mh:>11.4e}  {mr:>11.4e}  {'OK' if step_ok else 'FAIL'}"
          + ("" if tok_ok else "  <- token row differs"))
print("G5 compare:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
