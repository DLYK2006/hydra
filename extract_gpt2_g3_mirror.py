#!/usr/bin/env python3
"""
extract_gpt2_g3_mirror.py  --  G3 bit-exact references ("C++ mirror").

For each GPT-2 block N in 0..11: the op sequence of extract_gpt2_g3_refs.py
(itself extract_gpt2_block.py:106-155) on data/block_input_T8.bin, except that
LayerNorm and GELU are computed exactly as worker.cpp does:
  - layernorm(), worker.cpp: mean, variance, normalize, *gamma, +beta all in
    float64, one float32 cast per output element;
  - gelu(), worker.cpp: tanh approximation evaluated in float64, one float32
    cast.
Everything else (double-accumulated matmuls with float32 stage casts, float64
attention, -1e9 causal mask, float32 residual adds) is unchanged. Writes
data/g3_mirror_hNN_T8.bin (raw float32 [8, 768]).

Purpose: g3_check_v2.sh requires each Hydra block response to be
cmp-identical to this file. Weights come from HF transformer.h[N], not from the
data/hNN_* files, so a worker serving the wrong h-file, or dispatching
off-by-one, cannot match. Bit-identity relies on torch's float64 reductions
(mean/sum) rounding to the same float32 as worker.cpp's sequential loops; this
held for all 12 blocks in the 2026-10-30 diagnostic and is re-checked by the
h00 gate below and by g3_check_v2.sh on every block.

Gate before writing anything: block 0's output must be byte-identical to the
June C++ response data/block_response_T8.bin; otherwise nothing is written,
exit 1. Refuses to run if any data/g3_mirror_h*_T8.bin exists.
"""

import glob
import sys
import numpy as np
import torch
from transformers import GPT2Model

T, HIDDEN_DIM, N_HEAD = 8, 768, 12
HEAD_DIM = HIDDEN_DIM // N_HEAD
SCALE = 1.0 / (HEAD_DIM ** 0.5)   # 0.125 exactly, same expression as extract_gpt2_block.py
LN_EPS = 1e-5                     # worker.cpp layernorm(): constexpr double eps = 1e-5
GELU_K = 0.7978845608028654       # worker.cpp gelu(): kSqrt2OverPi
GELU_B = 0.044715                 # worker.cpp gelu(): kBeta

existing = sorted(glob.glob("data/g3_mirror_h*_T8.bin"))
if existing:
    print(f"ERROR: {len(existing)} g3_mirror file(s) exist, e.g. {existing[0]}. "
          "Refusing to overwrite.", file=sys.stderr)
    sys.exit(1)

x_orig = torch.from_numpy(np.fromfile("data/block_input_T8.bin", dtype=np.float32))
if x_orig.numel() != T * HIDDEN_DIM:
    raise ValueError(f"block_input_T8.bin: {x_orig.numel()} floats, want {T * HIDDEN_DIM}")
x_orig = x_orig.view(T, HIDDEN_DIM)

model = GPT2Model.from_pretrained("gpt2")
model.eval()


def layernorm_cpp(x, gamma, beta):
    """worker.cpp layernorm(), row-wise: float64 throughout, float32 cast once."""
    # Same expressions as the 2026-10-30 diagnostic that was bit-identical to
    # Hydra on all 12 blocks; keep them as-is.
    xd = x.double()
    mean = xd.mean(-1, keepdim=True)
    var = ((xd - mean) ** 2).sum(-1, keepdim=True) / HIDDEN_DIM
    inv_std = 1.0 / torch.sqrt(var + LN_EPS)
    return ((xd - mean) * inv_std * gamma.double() + beta.double()).float()


def gelu_cpp(h):
    """worker.cpp gelu(): tanh approximation in float64, float32 cast once."""
    xd = h.double()
    inner = GELU_K * (xd + GELU_B * xd * xd * xd)
    return (0.5 * xd * (1.0 + torch.tanh(inner))).float()


def block_mirror(blk, x_in):
    """extract_gpt2_g3_refs.py block_ref() with LayerNorm/GELU swapped for the C++ forms."""
    ln1_gamma = blk.ln_1.weight.detach().float(); ln1_beta = blk.ln_1.bias.detach().float()
    ln2_gamma = blk.ln_2.weight.detach().float(); ln2_beta = blk.ln_2.bias.detach().float()
    W_attn = blk.attn.c_attn.weight.detach().float(); b_attn = blk.attn.c_attn.bias.detach().float()
    W_proj = blk.attn.c_proj.weight.detach().float(); b_proj = blk.attn.c_proj.bias.detach().float()
    c_fc_W = blk.mlp.c_fc.weight.detach().float();    c_fc_b = blk.mlp.c_fc.bias.detach().float()
    c_proj_W = blk.mlp.c_proj.weight.detach().float(); c_proj_b = blk.mlp.c_proj.bias.detach().float()
    with torch.no_grad():
        x = x_in.clone()
        residual = x.clone()
        x_n = layernorm_cpp(x, ln1_gamma, ln1_beta)
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
        x_n = layernorm_cpp(x, ln2_gamma, ln2_beta)
        h = (x_n.double() @ c_fc_W.double() + c_fc_b.double()).float()
        h_gelu = gelu_cpp(h)
        mlp_out = (h_gelu.double() @ c_proj_W.double() + c_proj_b.double()).float()
        return residual + mlp_out


outs = [block_mirror(model.h[n], x_orig) for n in range(12)]

# Gate: h00 must reproduce the June C++ block response byte-for-byte.
june = open("data/block_response_T8.bin", "rb").read()
h00 = outs[0].numpy().astype(np.float32).tobytes()
if h00 != june:
    a = np.frombuffer(h00, dtype=np.float32); b = np.frombuffer(june, dtype=np.float32)
    n_diff = int((a != b).sum()) if a.size == b.size else -1
    print(f"ERROR: mirror h00 != data/block_response_T8.bin ({n_diff} differing floats); "
          "nothing written.", file=sys.stderr)
    sys.exit(1)
print("GATE: mirror h00 byte-identical to June data/block_response_T8.bin")

for n, out in enumerate(outs):
    out.numpy().astype(np.float32).tofile(f"data/g3_mirror_h{n:02d}_T8.bin")
print("Wrote 12 files data/g3_mirror_h{00..11}_T8.bin")
