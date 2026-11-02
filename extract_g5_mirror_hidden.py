#!/usr/bin/env python3
"""
extract_g5_mirror_hidden.py  --  G5 hidden-state oracle (chained C++ mirror).

For each of the 20 G1 step contexts (prompt + the first `step` gold tokens from
data/gen_ref_tokens.txt; T = 11..30), compute the GPT-2 embedding exactly as
gen_gpt2_ref.py does (float32 wte + wpe), then chain the G3 C++-mirror block
(extract_gpt2_g3_mirror.py's op sequence, generalised from T=8 to any T) through
blocks 0..11, and record ALL T rows of block 11's output (float32, the input to
ln_f).

Writes:
  data/g5_mirror_hidden.bin        raw float32, steps concatenated: 410 x 768
                                   (sum of T over steps = 11 + 12 + ... + 30 = 410)
  data/g5_mirror_hidden_index.txt  per step: step, T, row offset into the .bin (LF)

Independence from Hydra: inputs are only the HF checkpoint (transformer.h[N],
wte, wpe), the tokenizer and the committed G1 gold tokens. Nothing Hydra
produces is read, and this script is committed with its outputs before any G4/G5
Hydra generation runs.

Self-check before writing anything (else exit 1, nothing written):
  - the prompt token ids equal the "# prompt_ids" header of data/gen_ref_tokens.txt;
  - float64 ln_f + LM head (gen_gpt2_ref.py head_f64 ops) on each step's last row
    gives argmax == the gold token for all 20 steps, and
    max over steps of max|logits - data/gen_ref_logits_f64.bin| prints as 1.495e-05
    (the value the G4/G5 plan predicted with the same chain).
Refuses to run if either output exists.
"""

import ast
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from transformers import GPT2Model, GPT2Tokenizer

PROMPT = "Hello, how are you doing today? I hope everything"   # fixed for G1-G5
N_GEN, HIDDEN, N_HEAD, N_BLOCKS, VOCAB = 20, 768, 12, 12, 50257
HEAD_DIM = HIDDEN // N_HEAD
SCALE = 1.0 / (HEAD_DIM ** 0.5)   # same expression as extract_gpt2_block.py
LN_EPS = 1e-5                     # worker.cpp layernorm(): constexpr double eps = 1e-5
GELU_K = 0.7978845608028654       # worker.cpp gelu(): kSqrt2OverPi
GELU_B = 0.044715                 # worker.cpp gelu(): kBeta
EXPECTED_MAX_DIFF = "1.495e-05"   # planned prediction (scratchpad g4_predict.py), printed %.3e

OUT_BIN = "data/g5_mirror_hidden.bin"
OUT_IDX = "data/g5_mirror_hidden_index.txt"
for p in (OUT_BIN, OUT_IDX):
    if os.path.exists(p):
        sys.exit(f"ERROR: {p} exists; refusing to overwrite")


def layernorm_cpp(x, gamma, beta):
    """worker.cpp layernorm(), row-wise. Same expressions as extract_gpt2_g3_mirror.py."""
    xd = x.double()
    mean = xd.mean(-1, keepdim=True)
    var = ((xd - mean) ** 2).sum(-1, keepdim=True) / HIDDEN
    inv_std = 1.0 / torch.sqrt(var + LN_EPS)
    return ((xd - mean) * inv_std * gamma.double() + beta.double()).float()


def gelu_cpp(h):
    """worker.cpp gelu(). Same expressions as extract_gpt2_g3_mirror.py."""
    xd = h.double()
    inner = GELU_K * (xd + GELU_B * xd * xd * xd)
    return (0.5 * xd * (1.0 + torch.tanh(inner))).float()


def block_mirror(blk, x_in):
    """extract_gpt2_g3_mirror.py block_mirror(), with T taken from the input."""
    T = x_in.shape[0]
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
        Q = qd[:, :HIDDEN].view(T, N_HEAD, HEAD_DIM).permute(1, 0, 2)
        K = qd[:, HIDDEN:2*HIDDEN].view(T, N_HEAD, HEAD_DIM).permute(1, 0, 2)
        V = qd[:, 2*HIDDEN:].view(T, N_HEAD, HEAD_DIM).permute(1, 0, 2)
        scores = torch.bmm(Q, K.transpose(1, 2)) * SCALE
        causal_mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(causal_mask.unsqueeze(0), -1e9)
        attn_w = torch.softmax(scores, dim=-1)
        ctx = torch.bmm(attn_w, V).permute(1, 0, 2).contiguous().view(T, HIDDEN)
        attn_out = ctx.float()
        attn_proj = (attn_out.double() @ W_proj.double() + b_proj.double()).float()
        x = residual + attn_proj
        residual = x.clone()
        x_n = layernorm_cpp(x, ln2_gamma, ln2_beta)
        h = (x_n.double() @ c_fc_W.double() + c_fc_b.double()).float()
        h_gelu = gelu_cpp(h)
        mlp_out = (h_gelu.double() @ c_proj_W.double() + c_proj_b.double()).float()
        return residual + mlp_out


# --- gold contexts ---------------------------------------------------------------
gold, header_prompt_ids = [], None
with open("data/gen_ref_tokens.txt") as f:
    for line in f:
        if line.startswith("# prompt_ids = "):
            header_prompt_ids = ast.literal_eval(line.split("=", 1)[1].strip())
        elif line.strip() and not line.startswith("#"):
            gold.append(int(line.split()[1]))
if len(gold) != N_GEN:
    raise SystemExit(f"ERROR: {len(gold)} gold tokens in data/gen_ref_tokens.txt, want {N_GEN}")

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2Model.from_pretrained("gpt2").eval()
prompt_ids = tokenizer.encode(PROMPT)
if prompt_ids != header_prompt_ids:
    raise SystemExit(f"ERROR: prompt ids {prompt_ids} != gen_ref_tokens.txt header {header_prompt_ids}")

ref = np.fromfile("data/gen_ref_logits_f64.bin", dtype=np.float64)
if ref.size != N_GEN * VOCAB:
    raise SystemExit(f"ERROR: gen_ref_logits_f64.bin has {ref.size} floats, want {N_GEN * VOCAB}")
ref = ref.reshape(N_GEN, VOCAB)


def head_f64(x):   # gen_gpt2_ref.py head_f64(), same ops
    x = F.layer_norm(x.double(), [HIDDEN],
                     weight=model.ln_f.weight.double(), bias=model.ln_f.bias.double())
    return x @ model.wte.weight.double().T


# --- chain the mirror for every step ------------------------------------------------
hidden, index, worst, ok_tokens = [], [], 0.0, True
offset = 0
with torch.no_grad():
    for step in range(N_GEN):
        ids = torch.tensor([prompt_ids + gold[:step]], dtype=torch.long)
        T = ids.shape[1]
        pos = torch.arange(T, dtype=torch.long).unsqueeze(0)
        x = (model.wte(ids) + model.wpe(pos)).squeeze(0).float()   # gen_gpt2_ref.py embedding
        for n in range(N_BLOCKS):
            x = block_mirror(model.h[n], x)
        logits = head_f64(x)[-1].numpy()
        d = float(np.abs(logits - ref[step]).max())
        worst = max(worst, d)
        tok = int(logits.argmax())
        ok_tokens &= tok == gold[step]
        print(f"step {step:2d}  T={T:2d}  rows {offset:3d}..{offset + T - 1:3d}  "
              f"max|logits - ref| = {d:.3e}  token {tok} (gold {gold[step]})")
        hidden.append(x.numpy().astype(np.float32))
        index.append((step, T, offset))
        offset += T

print(f"max over steps = {worst:.3e} (planned {EXPECTED_MAX_DIFF}); tokens == gold: {ok_tokens}; rows = {offset}")
if not ok_tokens or f"{worst:.3e}" != EXPECTED_MAX_DIFF:
    sys.exit("ERROR: self-check failed; nothing written")

np.concatenate(hidden, axis=0).astype(np.float32).tofile(OUT_BIN)
with open(OUT_IDX, "w", newline="\n") as f:
    f.write("# G5 chained-mirror hidden-state index: block-11 output rows per step\n")
    f.write(f"# file: {OUT_BIN}, float32, {offset} x {HIDDEN}, row-major\n")
    f.write("# step  T  row_offset\n")
    for step, T, off in index:
        f.write(f"{step:4d}  {T:2d}  {off:4d}\n")
print(f"SELF-CHECK PASS; wrote {OUT_BIN} ({offset} x {HIDDEN} float32) and {OUT_IDX}")
