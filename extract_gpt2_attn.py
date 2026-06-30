#!/usr/bin/env python3
"""
extract_gpt2_attn.py -- GPT-2 block-0 attention weights, real-token inputs,
and double-precision causal-masked reference output.

Usage:
  python extract_gpt2_attn.py [T]

  T -- sequence length (number of tokens), default 8.
       M3 used T=1 with torch.randn (causal mask trivially a no-op).
       M4 uses T=8 with real GPT-2 token+position embeddings so softmax
       and causal masking do real work.

Computation (mirrors worker.cpp --mode attention):
  1. QKV projection:  x @ W_attn + b_attn            -> qkv [T, 2304]  (double-accum)
  2. Reshape Q,K,V to [N_HEAD, T, HEAD_DIM]; scaled QK^T in double
  3. Causal mask: scores[h, qt, kt] = -1e9 for kt > qt (pre-softmax, same
     constant as C++ so exp(-1e9 - row_max) == 0.0 in float64 on both sides)
  4. Numerically-stable softmax over the kt dimension (subtract row max)
  5. Weighted V sum per head in double -> attn_out [T, 768] cast to float32
  6. c_proj:  attn_out @ W_proj + b_proj              -> [T, 768]  (double-accum)

Input (M4+): tokenize a fixed English sentence, look up
  wte(token_ids) + wpe(pos_ids) from GPT-2 weights (float32), take the
  first T tokens.  More representative than randn; exercises real embedding
  magnitudes and correlations.

Double-precision convention (established M2): all matmul stages accumulated
  in float64, cast back to float32 at each inter-stage boundary, mirroring
  C++ matmul_nt / attn_heads.  Both sides in double -> comparison verifies
  operational correctness, not float32-BLAS bit-parity.

Writes (all under data/):
  attn_proj_W.bin           -- 768*768 float32  [in=768, out=768]
  attn_proj_b.bin           -- 768 float32
  attn_input_T{T}.bin       -- T*768 float32  real GPT-2 token+pos embeddings
  attn_expected_T{T}.bin    -- T*768 float32  causal-masked double-prec reference

Note: qkv_W.bin / qkv_b.bin (c_attn weights) must already exist (written by
      extract_gpt2_qkv.py).  The worker loads all four in --mode attention.
"""

import sys
import os
import numpy as np
import torch
from transformers import GPT2Model, GPT2Tokenizer

T          = int(sys.argv[1]) if len(sys.argv) > 1 else 8
HIDDEN_DIM = 768
N_HEAD     = 12
HEAD_DIM   = HIDDEN_DIM // N_HEAD  # 64
SCALE      = 1.0 / (HEAD_DIM ** 0.5)  # 0.125 exactly (64 is a perfect square)

print("Loading GPT-2 small from HuggingFace (cached after first run)...")
model     = GPT2Model.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model.eval()

# c_attn: Conv1D, weight [in=768, out=2304]
W_attn = model.h[0].attn.c_attn.weight.detach().float()  # [768, 2304]
b_attn = model.h[0].attn.c_attn.bias.detach().float()    # [2304]

# c_proj: Conv1D, weight [in=768, out=768]
W_proj = model.h[0].attn.c_proj.weight.detach().float()  # [768, 768]
b_proj = model.h[0].attn.c_proj.bias.detach().float()    # [768]

print(f"c_attn.weight.shape = {list(W_attn.shape)}  (expect [768, 2304])")
print(f"c_proj.weight.shape = {list(W_proj.shape)}  (expect [768, 768])")
assert W_attn.shape == (768, 2304), f"unexpected c_attn shape {W_attn.shape}"
assert b_attn.shape == (2304,),     f"unexpected c_attn bias {b_attn.shape}"
assert W_proj.shape == (768, 768),  f"unexpected c_proj shape {W_proj.shape}"
assert b_proj.shape == (768,),      f"unexpected c_proj bias {b_proj.shape}"

# Real token embeddings: tokenize a fixed sentence, take the first T token IDs,
# look up wte (word token) + wpe (position) embeddings from GPT-2 weights.
# This produces a [T, 768] input drawn from GPT-2's actual embedding space --
# more representative than randn and exercises realistic activation magnitudes.
TEXT = "Hello, how are you doing today? I hope everything"
token_ids_list = tokenizer.encode(TEXT)[:T]
# Pad to exactly T by repeating the last token if the sentence tokenizes shorter.
while len(token_ids_list) < T:
    token_ids_list.append(token_ids_list[-1])

token_ids = torch.tensor([token_ids_list], dtype=torch.long)  # [1, T]
pos_ids   = torch.arange(T, dtype=torch.long).unsqueeze(0)    # [1, T]

with torch.no_grad():
    x = (model.wte(token_ids) + model.wpe(pos_ids)).squeeze(0).float()  # [T, 768]

print(f"\nToken IDs (first {T}): {token_ids_list}")
print(f"Tokens: {[tokenizer.decode([t]) for t in token_ids_list]}")
print(f"Input  x[0, 0:4] = {x[0, :4].tolist()}")

with torch.no_grad():
    # Stage 1: QKV projection -- double accumulation, float32 output.
    # Mirrors C++ matmul_nt: acc stays double through 768 k steps, cast once.
    qkv = (x.double() @ W_attn.double() + b_attn.double()).float()  # [T, 2304]

    # Stage 2: Scaled QK^T in double.
    # Re-promote float32 qkv to double so scores match C++ attn_heads double accumulators.
    qd = qkv.double()
    Q = qd[:, :HIDDEN_DIM].view(T, N_HEAD, HEAD_DIM).permute(1, 0, 2)              # [12, T, 64]
    K = qd[:, HIDDEN_DIM:2*HIDDEN_DIM].view(T, N_HEAD, HEAD_DIM).permute(1, 0, 2)
    V = qd[:, 2*HIDDEN_DIM:].view(T, N_HEAD, HEAD_DIM).permute(1, 0, 2)

    scores = torch.bmm(Q, K.transpose(1, 2)) * SCALE  # [12, T, T]

    # Causal mask: token qt cannot attend to future tokens kt > qt.
    # Use -1e9 (not -inf) to match C++'s `scores[qt*T+kt] = -1e9` exactly:
    # exp(-1e9 - row_max) underflows to 0.0 in float64 on both sides, so
    # masked positions contribute zero to the denominator and the V weighted sum.
    causal_mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
    scores = scores.masked_fill(causal_mask.unsqueeze(0), -1e9)  # [12, T, T]

    attn_w  = torch.softmax(scores, dim=-1)          # [12, T, T]
    ctx     = torch.bmm(attn_w, V)                   # [12, T, 64]
    ctx     = ctx.permute(1, 0, 2).contiguous().view(T, HIDDEN_DIM)  # [T, 768]
    attn_out = ctx.float()  # cast to float32, mirroring C++ attn_heads

    # Stage 3: c_proj -- double accumulation, float32 output.
    y = (attn_out.double() @ W_proj.double() + b_proj.double()).float()  # [T, 768]

print(f"\ny[0, 0:4]        = {y[0, :4].tolist()}")
print(f"y[{T-1}, 0:4]        = {y[T-1, :4].tolist()}")

# Print attention weight distribution for row T-1 (the last token, which can
# attend to all previous tokens) -- confirms the causal mask is active.
# At qt=T-1 all T positions are unmasked; weights should be non-uniform.
# At qt=0 only position 0 is unmasked; weight[0][0] should be 1.0.
print(f"\nCausal mask sanity checks (all 12 heads averaged over T={T} rows):")
attn_w_np = attn_w.float().numpy()  # [12, T, T]
first_row_avg = attn_w_np[:, 0, :].mean(axis=0)   # mean over heads, qt=0
last_row_avg  = attn_w_np[:, T-1, :].mean(axis=0) # mean over heads, qt=T-1
print(f"  attn_w[h, qt=0,   :]  head-avg = {first_row_avg.tolist()}  (expect [1.0, 0.0, ...])")
print(f"  attn_w[h, qt={T-1}, :]  head-avg = {[f'{v:.3f}' for v in last_row_avg.tolist()]}  (expect T non-zero values)")

# At T=1 the causal mask is empty (triu(diagonal=1) of a 1x1 matrix is all-False),
# so attn_out == V per head (softmax([s]) = 1.0 exactly).
if T == 1:
    V_flat = qkv[0, 2 * HIDDEN_DIM:]
    structural_err = (attn_out[0].double() - qd[0, 2*HIDDEN_DIM:]).abs().max().item()
    print(f"\nT=1 structural check: max|attn_out - V| = {structural_err:.2e}  (expect 0.00e+00)")

# HuggingFace cross-check: GPT-2 attention applies its own causal mask internally.
# Expected diff vs our double-precision reference: ~1e-4
# (float32 BLAS vs double accumulation across two 768-term matmuls).
try:
    with torch.no_grad():
        hf_out = model.h[0].attn(x.unsqueeze(0))[0].squeeze(0).float()
        hf_diff = (hf_out - y).abs()
        print(f"\nHuggingFace cross-check (float32 BLAS vs double ref, T={T}):")
        print(f"  max|double_ref - hf_f32| = {hf_diff.max():.2e}  (expect ~1e-4)")
        print(f"  avg diff                 = {hf_diff.mean():.2e}")
except Exception as e:
    print(f"HuggingFace cross-check skipped: {e}")

os.makedirs("data", exist_ok=True)

def write_bin(path, tensor):
    arr = tensor.numpy().astype(np.float32)
    arr.tofile(path)
    print(f"  Written {path}  ({arr.nbytes} bytes, {arr.size} float32s)")

write_bin("data/attn_proj_W.bin",         W_proj)
write_bin("data/attn_proj_b.bin",         b_proj)
write_bin(f"data/attn_input_T{T}.bin",    x)
write_bin(f"data/attn_expected_T{T}.bin", y)

print()
print(f"T={T}, N_HEAD={N_HEAD}, HEAD_DIM={HEAD_DIM}, SCALE={SCALE}")
print(f"Input  shape: {list(x.shape)}")
print(f"Output shape: {list(y.shape)}")
