#!/usr/bin/env python3
"""
extract_gpt2_g3_refs.py  --  G3 references.

For each GPT-2 block N in 0..11, compute the double-precision stage-boundary
reference output of block N applied to the fixed input data/block_input_T8.bin,
with exactly the convention of extract_gpt2_block.py lines 106-155, and write
data/g3_expected_hNN_T8.bin (raw float32 [8, 768]).

Weights come from the HuggingFace model (transformer.h[N]), NOT from
data/hNN_*.bin, so a worker that loads the wrong h-file is caught: the
reference stays tied to the true block N.  (scripts/g2_verify.py already proves
data/hNN_* == transformer.h[N].)

Gate before writing anything: block 0's output bytes must equal
data/block_expected_T8.bin byte-for-byte; otherwise nothing is written, exit 1.

Also writes data/g3_ref_summary.txt (LF, tracked): per block, max|f32 - f64|
where f64 = this double-convention reference and f32 = PyTorch's own float32
GPT2Block forward on the same input.  Informational only: it never affects
pass/fail or which files are written.

Refuses to run if any data/g3_expected_h*_T8.bin or data/g3_ref_summary.txt
exists.
"""

import glob
import sys
import numpy as np
import torch
import torch.nn.functional as F
from transformers import GPT2Model

T, HIDDEN_DIM, N_HEAD = 8, 768, 12
HEAD_DIM = HIDDEN_DIM // N_HEAD
SCALE = 1.0 / (HEAD_DIM ** 0.5)   # 0.125 exactly, same expression as extract_gpt2_block.py

SUMMARY_PATH = "data/g3_ref_summary.txt"
existing = sorted(glob.glob("data/g3_expected_h*_T8.bin") + glob.glob(SUMMARY_PATH))
if existing:
    print(f"ERROR: {len(existing)} G3 reference output(s) exist, e.g. {existing[0]}. "
          "Refusing to overwrite.", file=sys.stderr)
    sys.exit(1)

x_orig = torch.from_numpy(np.fromfile("data/block_input_T8.bin", dtype=np.float32))
if x_orig.numel() != T * HIDDEN_DIM:
    raise ValueError(f"block_input_T8.bin: {x_orig.numel()} floats, want {T * HIDDEN_DIM}")
x_orig = x_orig.view(T, HIDDEN_DIM)

model = GPT2Model.from_pretrained("gpt2")
model.eval()


def block_ref(blk, x_in):
    """Verbatim op sequence of extract_gpt2_block.py:106-155, weights from blk."""
    ln1_gamma = blk.ln_1.weight.detach().float(); ln1_beta = blk.ln_1.bias.detach().float()
    ln2_gamma = blk.ln_2.weight.detach().float(); ln2_beta = blk.ln_2.bias.detach().float()
    W_attn = blk.attn.c_attn.weight.detach().float(); b_attn = blk.attn.c_attn.bias.detach().float()
    W_proj = blk.attn.c_proj.weight.detach().float(); b_proj = blk.attn.c_proj.bias.detach().float()
    c_fc_W = blk.mlp.c_fc.weight.detach().float();    c_fc_b = blk.mlp.c_fc.bias.detach().float()
    c_proj_W = blk.mlp.c_proj.weight.detach().float(); c_proj_b = blk.mlp.c_proj.bias.detach().float()
    with torch.no_grad():
        x = x_in.clone()
        residual = x.clone()
        x_n = F.layer_norm(x, [HIDDEN_DIM], weight=ln1_gamma, bias=ln1_beta)
        qkv = (x_n.double() @ W_attn.double() + b_attn.double()).float()
        qd = qkv.double()
        Q = qd[:, :HIDDEN_DIM].view(T, N_HEAD, HEAD_DIM).permute(1, 0, 2)
        K = qd[:, HIDDEN_DIM:2*HIDDEN_DIM].view(T, N_HEAD, HEAD_DIM).permute(1, 0, 2)
        V = qd[:, 2*HIDDEN_DIM:].view(T, N_HEAD, HEAD_DIM).permute(1, 0, 2)
        scores = torch.bmm(Q, K.transpose(1, 2)) * SCALE
        causal_mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(causal_mask.unsqueeze(0), -1e9)
        attn_w = torch.softmax(scores, dim=-1)
        ctx = torch.bmm(attn_w, V).permute(1, 0, 2).contiguous().view(T, HIDDEN_DIM)
        attn_out = ctx.float()
        attn_proj = (attn_out.double() @ W_proj.double() + b_proj.double()).float()
        x = residual + attn_proj
        residual = x.clone()
        x_n = F.layer_norm(x, [HIDDEN_DIM], weight=ln2_gamma, bias=ln2_beta)
        h = (x_n.double() @ c_fc_W.double() + c_fc_b.double()).float()
        h_gelu = F.gelu(h, approximate='tanh')
        mlp_out = (h_gelu.double() @ c_proj_W.double() + c_proj_b.double()).float()
        return residual + mlp_out


outs = [block_ref(model.h[n], x_orig) for n in range(12)]

# Gate: h00 must reproduce the June M6 reference byte-for-byte.
june = open("data/block_expected_T8.bin", "rb").read()
h00 = outs[0].numpy().astype(np.float32).tobytes()
if h00 != june:
    a = np.frombuffer(h00, dtype=np.float32); b = np.frombuffer(june, dtype=np.float32)
    n_diff = int((a != b).sum()) if a.size == b.size else -1
    print(f"ERROR: h00 != data/block_expected_T8.bin ({n_diff} differing floats); "
          "nothing written.", file=sys.stderr)
    sys.exit(1)
print("GATE: h00 byte-identical to data/block_expected_T8.bin")

# Informational f32-vs-f64 comparison, all 12 BEFORE any write, so nothing it
# raises can leave a half-written oracle set behind the overwrite guard. Wrapped
# like extract_gpt2_block.py:166-174: GPT2Block's forward signature/return varies
# across transformers versions. A failure is recorded, never fatal.
summary_rows = []
for n, out in enumerate(outs):
    try:
        with torch.no_grad():
            f32 = model.h[n](x_orig.unsqueeze(0))[0].squeeze(0).float()
        d = (f32 - out).abs().max().item()
        row = f"{n:4d}  {d:.6e}"
        print(f"h{n:02d}: max|f32 - f64| = {d:.2e}  max|f64| = {out.abs().max().item():.3e}")
    except Exception as e:
        row = f"{n:4d}  unavailable ({type(e).__name__}: {e})"
        print(f"h{n:02d}: float32 comparison skipped: {e}")
    summary_rows.append(row)

for n, out in enumerate(outs):
    out.numpy().astype(np.float32).tofile(f"data/g3_expected_h{n:02d}_T8.bin")
print("Wrote 12 files data/g3_expected_h{00..11}_T8.bin")

with open(SUMMARY_PATH, "w", newline="\n") as f:
    f.write("# G3 reference summary -- informational only, not a gate\n")
    f.write("# input: data/block_input_T8.bin [8, 768]\n")
    f.write("# f64 = double-convention reference (data/g3_expected_hNN_T8.bin)\n")
    f.write("# f32 = PyTorch float32 GPT2Block forward, same weights and input\n")
    f.write("# block  max|f32 - f64|\n")
    for row in summary_rows:
        f.write(row + "\n")
print(f"Wrote {SUMMARY_PATH}")
