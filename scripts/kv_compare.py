# KV-cache compare. Hand-reviewed; do not edit. Called by scripts/kv_check.sh.
# Usage:
#   kv_compare.py b DIR          stage B (cache ON, fixed prompt, 20 tokens)
#   kv_compare.py c ONDIR OFFDIR stage C (64 tokens, cache ON vs OFF)
# DIR holds gen_hydra.py outputs: tokens.txt, logits_f64.bin, hidden_f32.bin, sends.txt
# ("step layer sequence_id start_pos T" per packet).
# hidden_f32.bin layouts:
#   cache OFF: every step's full block-11 output, steps concatenated (T_prompt+s rows at step s)
#   cache ON:  only the rows the worker returned: the T_prompt prefill rows (step 0), then one
#              newest row per later step -> T_prompt + n_gen - 1 rows
# All hidden checks are byte-for-byte (hard). Logits (cache ON head = newest row only) are
# checked per element against the a-priori float64 dot-product bound
#   |a - b| <= 2 * n * u * sum_i |x_i * w_i|,  n = 768, u = 2^-53,
# x = float64 ln_f output of the step's newest block-11 row (byte-identical on both sides by the
# hidden gate), w = the token's wte row. Not a tuned constant. max(|a-b|/bound) is reported.
import sys
import numpy as np
import torch
import torch.nn.functional as F
from transformers import GPT2Model

H, VOCAB, N_BLOCKS = 768, 50257, 12
U = 2.0 ** -53

_model = None


def model():
    global _model
    if _model is None:
        _model = GPT2Model.from_pretrained("gpt2").eval()
    return _model


def data_rows(path):
    with open(path, "rb") as f:
        return [l for l in f.read().split(b"\n") if l and not l.startswith(b"#")]


def read_sends(path):
    with open(path) as f:
        return [tuple(int(v) for v in l.split()) for l in f if l.strip() and not l.startswith("#")]


def check_cache_used(d, n_gen):
    """Prefill at step 0 (start_pos 0, T=t_prompt), then T=1 at start_pos t_prompt+s-1, every
    block 0..11, one nonzero sequence_id throughout. Returns (ok, t_prompt)."""
    sends = read_sends(f"{d}/sends.txt")
    t_prompt = sends[0][4] if sends else -1
    want = [(s, n, 0 if s == 0 else t_prompt + s - 1, t_prompt if s == 0 else 1)
            for s in range(n_gen) for n in range(N_BLOCKS)]
    got = [(s, n, p, T) for (s, n, _seq, p, T) in sends]
    seqs = {seq for (_s, _n, seq, _p, _T) in sends}
    ok = got == want and len(seqs) == 1 and 0 not in seqs
    print(f"{'PASS' if ok else 'FAIL'} cache used: {len(sends)} packets, sequence_ids {sorted(seqs)}, "
          f"prefill T={t_prompt} then T=1 decode (expected {len(want)} packets)")
    return ok, t_prompt


def load(path, dtype, rows):
    a = np.fromfile(path, dtype)
    if a.size != rows * (H if dtype == np.float32 else VOCAB):
        raise SystemExit(f"FAIL {path}: {a.size} values, want {rows} rows")
    return a.reshape(rows, -1)


def logit_bound_check(label, newest, a, b):
    """newest: [n_gen, 768] float32 rows; a, b: [n_gen, VOCAB] float64 logits."""
    m = model()
    W = m.wte.weight.detach().double()
    absW = W.abs()
    worst, viol = 0.0, []
    with torch.no_grad():
        for s in range(newest.shape[0]):
            x = F.layer_norm(torch.from_numpy(newest[s:s + 1].copy()).double(), [H],
                             weight=m.ln_f.weight.double(), bias=m.ln_f.bias.double())[0]
            bound = (2 * H * U * (x.abs() @ absW.T)).numpy()
            d = np.abs(a[s] - b[s])
            if np.isnan(d).any() or np.isnan(bound).any() or ((bound == 0) & (d != 0)).any():
                viol.append(s)
                continue
            r = np.where(bound > 0, d / np.where(bound > 0, bound, 1.0), 0.0)
            worst = max(worst, float(r.max()))
            if (d > bound).any():
                viol.append(s)
    ok = not viol
    print(f"{'PASS' if ok else 'FAIL'} {label}: every element |a-b| <= 2*768*2^-53*sum|x*w| "
          f"(max ratio {worst:.3e})" + (f"; violating steps {viol}" if viol else ""))
    return ok


def stage_b(d):
    ok = True
    n_gen = 20
    rows = data_rows(f"{d}/tokens.txt")
    tok_ok = rows == data_rows("data/gen_ref_tokens.txt") and len(rows) == n_gen
    print(f"{'PASS' if tok_ok else 'FAIL'} tokens: 20 rows byte-identical to data/gen_ref_tokens.txt")
    ok &= tok_ok
    used, tp = check_cache_used(d, n_gen)
    ok &= used
    if tp != 11:
        print(f"FAIL prefill T={tp}, want 11")
        return False
    idx = [tuple(int(v) for v in l.split()) for l in open("data/g5_mirror_hidden_index.txt")
           if l.strip() and not l.startswith("#")]
    total = idx[-1][2] + idx[-1][1]
    mirror = load("data/g5_mirror_hidden.bin", np.float32, total)
    off_run = load("data/live/g5_run1/hidden_f32.bin", np.float32, total)   # frozen cache-off run
    hyd = load(f"{d}/hidden_f32.bin", np.float32, tp + n_gen - 1)
    pre_ok = hyd[:tp].tobytes() == mirror[:tp].tobytes()
    print(f"{'PASS' if pre_ok else 'FAIL'} prefill rows 0..{tp - 1} byte-identical to the mirror (step 0)")
    ok &= pre_ok
    newest = np.stack([hyd[tp - 1] if s == 0 else hyd[tp - 1 + s] for s in range(n_gen)])
    bad_m = [s for s, T, off in idx if newest[s].tobytes() != mirror[off + T - 1].tobytes()]
    bad_o = [s for s, T, off in idx if newest[s].tobytes() != off_run[off + T - 1].tobytes()]
    print(f"{'PASS' if not bad_m else 'FAIL'} newest block-11 row per step byte-identical to the "
          f"mirror: {n_gen - len(bad_m)}/{n_gen}" + (f" (differ: {bad_m})" if bad_m else ""))
    print(f"{'PASS' if not bad_o else 'FAIL'} newest block-11 row per step byte-identical to the "
          f"frozen cache-off run: {n_gen - len(bad_o)}/{n_gen}" + (f" (differ: {bad_o})" if bad_o else ""))
    ok &= not bad_m and not bad_o
    a = load(f"{d}/logits_f64.bin", np.float64, n_gen)
    b = load("data/g5_hydra_logits.bin", np.float64, n_gen)
    ok &= logit_bound_check("logits vs data/g5_hydra_logits.bin", newest, a, b)
    return ok


def stage_c(on, off, n_gen=64):
    ok = True
    r_on, r_off = data_rows(f"{on}/tokens.txt"), data_rows(f"{off}/tokens.txt")
    tok_ok = r_on == r_off and len(r_on) == n_gen
    print(f"{'PASS' if tok_ok else 'FAIL'} {n_gen} token rows identical, cache ON vs OFF")
    ok &= tok_ok
    used, tp = check_cache_used(on, n_gen)
    ok &= used
    if tp < 1:
        return False
    off_rows = sum(tp + s for s in range(n_gen))
    h_off = load(f"{off}/hidden_f32.bin", np.float32, off_rows)
    h_on = load(f"{on}/hidden_f32.bin", np.float32, tp + n_gen - 1)
    pre_ok = h_on[:tp].tobytes() == h_off[:tp].tobytes()
    print(f"{'PASS' if pre_ok else 'FAIL'} prefill rows byte-identical, cache ON vs OFF (step 0)")
    ok &= pre_ok
    off_last, cum = [], 0
    for s in range(n_gen):
        cum += tp + s
        off_last.append(h_off[cum - 1])
    newest = np.stack([h_on[tp - 1] if s == 0 else h_on[tp - 1 + s] for s in range(n_gen)])
    bad = [s for s in range(n_gen) if newest[s].tobytes() != off_last[s].tobytes()]
    print(f"{'PASS' if not bad else 'FAIL'} newest block-11 row per step byte-identical, ON vs OFF: "
          f"{n_gen - len(bad)}/{n_gen}" + (f" (differ: {bad})" if bad else ""))
    ok &= not bad
    a = load(f"{on}/logits_f64.bin", np.float64, n_gen)
    b = load(f"{off}/logits_f64.bin", np.float64, n_gen)
    d = np.abs(a - b)
    print(f"REPORT final-step max|logits_on - logits_off| = {float(d[-1].max()):.3e}; "
          f"all steps max = {float(d.max()):.3e}")
    ok &= logit_bound_check("logits ON vs OFF", newest, a, b)
    return ok


if __name__ == "__main__":
    if sys.argv[1] == "b":
        sys.exit(0 if stage_b(sys.argv[2]) else 1)
    if sys.argv[1] == "c":
        sys.exit(0 if stage_c(sys.argv[2], sys.argv[3]) else 1)
    raise SystemExit(f"unknown stage {sys.argv[1]!r}")
