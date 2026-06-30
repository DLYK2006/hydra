#!/usr/bin/env python3
"""
extract_gpt2_block.py -- GPT-2 block-0 full transformer block weights and
double-precision stage-boundary reference output.

Usage:
  python extract_gpt2_block.py [T]

  T -- sequence length (default 8).

Block ordering (Pre-LN, GPT-2 convention):
  residual = x
  x_n      = LayerNorm(x,   ln_1.weight, ln_1.bias)
  attn_out = attention(x_n)    (c_attn QKV proj -> multi-head causal attn -> c_proj)
  x        = residual + attn_out                      (float32 residual add)
  residual = x
  x_n      = LayerNorm(x,   ln_2.weight, ln_2.bias)
  mlp_out  = mlp(x_n)          (c_fc -> GELU(tanh) -> c_proj)
  x        = residual + mlp_out                       (float32 residual add)
  output   = x

Precision convention (established M2, extended through M5):
  - Each matmul: double-accumulated k-loop, float32 cast at each stage boundary.
    Mirrors C++ matmul_nt (acc[] stays double through all in_dim k steps).
  - LayerNorm: F.layer_norm in float32.  C++ double accumulators for mean/var
    match PyTorch float32 to ~1e-7 per M1 verification.
  - Attention heads: double QK^T, double softmax, double V weighted sum, float32
    cast once per output element.  Mirrors C++ attn_heads.
  - Causal mask: -1e9 constant (not -inf) on both sides so exp(-1e9 - row_max)
    underflows identically to 0.0 in float64.
  - GELU: F.gelu(approximate='tanh') in float32.  C++ gelu() promotes float->double
    for the polynomial+tanh then casts back to float32; the float32 tanh in PyTorch
    matches to within 1e-7 per M5 verification.
  - Residual adds: float32 on both sides.  These are NOT matmul boundaries, so the
    per-stage cast does NOT reset divergence -- residual adds accumulate sub-float32
    error across sub-blocks.  Expect M6 max diff ~1e-4 (NOT the 0.00e+00 of M3/M4).

Weights written (only the new ones -- all others already on disk from M1-M5):
  data/block_ln2_gamma.bin     [768]   float32   ln_2.weight
  data/block_ln2_beta.bin      [768]   float32   ln_2.bias

Inputs/outputs written:
  data/block_input_T{T}.bin    [T*768] float32   raw token+pos embeddings (same as M4/M5)
  data/block_expected_T{T}.bin [T*768] float32   full block output reference
"""

import sys
import os
import numpy as np
import torch
import torch.nn.functional as F
from transformers import GPT2Model, GPT2Tokenizer

T          = int(sys.argv[1]) if len(sys.argv) > 1 else 8
HIDDEN_DIM = 768
N_HEAD     = 12
HEAD_DIM   = HIDDEN_DIM // N_HEAD  # 64
MLP_HIDDEN = 3072
SCALE      = 1.0 / (HEAD_DIM ** 0.5)  # 0.125 exactly

print("Loading GPT-2 small from HuggingFace (cached after first run)...")
model     = GPT2Model.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model.eval()

# --- Weights ------------------------------------------------------------------

# ln_1 (already on disk as data/ln_gamma.bin / data/ln_beta.bin)
ln1_gamma = model.h[0].ln_1.weight.detach().float()   # [768]
ln1_beta  = model.h[0].ln_1.bias.detach().float()     # [768]

# ln_2 (new for M6 -- written to block_ln2_gamma.bin / block_ln2_beta.bin)
ln2_gamma = model.h[0].ln_2.weight.detach().float()   # [768]
ln2_beta  = model.h[0].ln_2.bias.detach().float()     # [768]

# Attention weights (already on disk from M2/M3/M4)
W_attn = model.h[0].attn.c_attn.weight.detach().float()  # [768, 2304]
b_attn = model.h[0].attn.c_attn.bias.detach().float()    # [2304]
W_proj = model.h[0].attn.c_proj.weight.detach().float()  # [768, 768]
b_proj = model.h[0].attn.c_proj.bias.detach().float()    # [768]

# MLP weights (already on disk from M5)
c_fc_W   = model.h[0].mlp.c_fc.weight.detach().float()   # [768, 3072]
c_fc_b   = model.h[0].mlp.c_fc.bias.detach().float()     # [3072]
c_proj_W = model.h[0].mlp.c_proj.weight.detach().float() # [3072, 768]
c_proj_b = model.h[0].mlp.c_proj.bias.detach().float()   # [768]

print(f"ln_1.weight.shape = {list(ln1_gamma.shape)}  (expect [768])")
print(f"ln_2.weight.shape = {list(ln2_gamma.shape)}  (expect [768])")

# --- Input: same wte+wpe token embeddings as M4/M5 ---------------------------
TEXT = "Hello, how are you doing today? I hope everything"
token_ids_list = tokenizer.encode(TEXT)[:T]
while len(token_ids_list) < T:
    token_ids_list.append(token_ids_list[-1])

token_ids = torch.tensor([token_ids_list], dtype=torch.long)
pos_ids   = torch.arange(T, dtype=torch.long).unsqueeze(0)
with torch.no_grad():
    x_orig = (model.wte(token_ids) + model.wpe(pos_ids)).squeeze(0).float()  # [T, 768]

print(f"\nTokens (first {T}): {[tokenizer.decode([t]) for t in token_ids_list]}")
print(f"x[0, 0:4] = {x_orig[0, :4].tolist()}")

# --- Reference computation ----------------------------------------------------
with torch.no_grad():
    x = x_orig.clone()

    # ---- Sub-block 1: attention sub-layer ----

    # Step 1a: ln_1 (float32 -- C++ double accumulators match to ~1e-7 per M1)
    residual = x.clone()
    x_n = F.layer_norm(x, [HIDDEN_DIM], weight=ln1_gamma, bias=ln1_beta)

    # Step 1b: QKV projection (double accumulated, float32 cast at boundary)
    qkv = (x_n.double() @ W_attn.double() + b_attn.double()).float()  # [T, 2304]

    # Step 1c: Multi-head attention with causal mask.
    # Re-promote to double so scores match C++ attn_heads() double accumulators.
    qd = qkv.double()
    Q  = qd[:, :HIDDEN_DIM].view(T, N_HEAD, HEAD_DIM).permute(1, 0, 2)               # [12, T, 64]
    K  = qd[:, HIDDEN_DIM:2*HIDDEN_DIM].view(T, N_HEAD, HEAD_DIM).permute(1, 0, 2)
    V  = qd[:, 2*HIDDEN_DIM:].view(T, N_HEAD, HEAD_DIM).permute(1, 0, 2)

    scores = torch.bmm(Q, K.transpose(1, 2)) * SCALE            # [12, T, T]
    causal_mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
    scores = scores.masked_fill(causal_mask.unsqueeze(0), -1e9)  # matches C++ -1e9
    attn_w = torch.softmax(scores, dim=-1)                       # [12, T, T], float64
    ctx    = torch.bmm(attn_w, V).permute(1, 0, 2).contiguous().view(T, HIDDEN_DIM)
    attn_out = ctx.float()   # float32 cast -- mirrors C++ attn_heads() final cast

    # Step 1d: c_proj (double accumulated, float32 cast at boundary)
    attn_proj = (attn_out.double() @ W_proj.double() + b_proj.double()).float()  # [T, 768]

    # Step 1e: first residual add (float32 -- divergence begins accumulating here)
    x = residual + attn_proj

    # ---- Sub-block 2: MLP sub-layer ----

    # Step 2a: ln_2 (float32)
    residual = x.clone()
    x_n = F.layer_norm(x, [HIDDEN_DIM], weight=ln2_gamma, bias=ln2_beta)

    # Step 2b: c_fc (double accumulated, float32 cast at boundary)
    h = (x_n.double() @ c_fc_W.double() + c_fc_b.double()).float()  # [T, 3072]

    # Step 2c: GELU tanh approximation (float32)
    h_gelu = F.gelu(h, approximate='tanh')

    # Step 2d: c_proj (double accumulated, float32 cast at boundary)
    mlp_out = (h_gelu.double() @ c_proj_W.double() + c_proj_b.double()).float()  # [T, 768]

    # Step 2e: second residual add (float32)
    x = residual + mlp_out
    output = x

print(f"\noutput[0,   0:4] = {output[0,   :4].tolist()}")
print(f"output[{T-1}, 0:4] = {output[T-1, :4].tolist()}")
print(f"output[0, max] = {output[0].abs().max().item():.6f}")

# --- HuggingFace cross-check --------------------------------------------------
# model.h[0](x_orig.unsqueeze(0)) returns (hidden_states, ...) -- unpack first.
# Expected diff vs our reference: ~1e-3 because GPT-2 uses float32 BLAS for
# two 768-term matmuls and two 3072-term matmuls, each contributing ~1e-5; plus
# the residual adds are float32 on both sides so error is not reset between stages.
try:
    with torch.no_grad():
        hf_out = model.h[0](x_orig.unsqueeze(0))[0].squeeze(0).float()
        hf_diff = (hf_out - output).abs()
        print(f"\nHuggingFace cross-check (float32 BLAS vs double ref, T={T}):")
        print(f"  max|double_ref - hf_f32| = {hf_diff.max():.2e}  (expect ~1e-3)")
        print(f"  avg diff                 = {hf_diff.mean():.2e}")
except Exception as e:
    print(f"HuggingFace cross-check skipped: {e}")

# --- Write files --------------------------------------------------------------
os.makedirs("data", exist_ok=True)

def write_bin(path, tensor):
    arr = tensor.numpy().astype(np.float32)
    arr.tofile(path)
    print(f"  Written {path}  ({arr.nbytes} bytes, {arr.size} float32s)")

# Only new weight files -- all others already on disk from M1-M5.
write_bin("data/block_ln2_gamma.bin",          ln2_gamma)
write_bin("data/block_ln2_beta.bin",           ln2_beta)
write_bin(f"data/block_input_T{T}.bin",        x_orig)
write_bin(f"data/block_expected_T{T}.bin",     output)

print()
print(f"T={T}, HIDDEN_DIM={HIDDEN_DIM}, N_HEAD={N_HEAD}, HEAD_DIM={HEAD_DIM}")
print(f"Input  shape: {list(x_orig.shape)}")
print(f"Output shape: {list(output.shape)}")
