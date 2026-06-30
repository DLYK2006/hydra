#!/usr/bin/env python3
"""
extract_gpt2_mlp.py -- GPT-2 block-0 MLP weights and double-precision reference.

Usage:
  python extract_gpt2_mlp.py [T]

  T -- sequence length, default 8.  Same wte+wpe embeddings as M4 so the
       MLP input is a realistic continuation of the attention output.

Computation (mirrors worker.cpp --mode mlp):
  h      = x @ c_fc.W   + c_fc.b    [T,768]  -> [T,3072]  (double-accum, float32 cast)
  h_gelu = GELU(h, tanh approximation)         [T,3072]  elementwise
  y      = h_gelu @ c_proj.W + c_proj.b        [T,3072] -> [T,768]  (double-accum, float32 cast)

GELU variant -- CRITICAL: GPT-2 uses the tanh approximation, not erf.
  gelu(x) = 0.5*x*(1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3)))
  Python:  F.gelu(h, approximate='tanh')  -- must NOT use the default erf form.

Stage-boundary convention (same as M2-M4): each matmul accumulated in float64,
  cast to float32 once per stage.  GELU input is already float32 (from c_fc
  cast); F.gelu operates in float32.  The second matmul is double-accumulated
  from the float32 GELU output.

Writes (under data/):
  mlp_c_fc_W.bin              [768 * 3072] float32   c_fc weight  [in=768, out=3072]
  mlp_c_fc_b.bin              [3072]       float32   c_fc bias
  mlp_c_proj_W.bin            [3072 * 768] float32   c_proj weight [in=3072, out=768]
  mlp_c_proj_b.bin            [768]        float32   c_proj bias
  mlp_input_T{T}.bin          [T * 768]    float32   real GPT-2 token+pos embeddings
  mlp_expected_gelu_T{T}.bin  [T * 3072]   float32   post-GELU intermediate
  mlp_expected_T{T}.bin       [T * 768]    float32   final output

The post-GELU intermediate is written separately so compare_mlp.py can diff
  the GELU stage independently from the second matmul, localising any failure.
"""

import sys
import os
import math
import numpy as np
import torch
import torch.nn.functional as F
from transformers import GPT2Model, GPT2Tokenizer

T          = int(sys.argv[1]) if len(sys.argv) > 1 else 8
HIDDEN_DIM = 768
MLP_HIDDEN = 3072

print("Loading GPT-2 small from HuggingFace (cached after first run)...")
model     = GPT2Model.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model.eval()

# MLP weights: Conv1D [in, out] layout (same convention as c_attn/c_proj in attention).
c_fc_W   = model.h[0].mlp.c_fc.weight.detach().float()    # [768, 3072]
c_fc_b   = model.h[0].mlp.c_fc.bias.detach().float()      # [3072]
c_proj_W = model.h[0].mlp.c_proj.weight.detach().float()  # [3072, 768]
c_proj_b = model.h[0].mlp.c_proj.bias.detach().float()    # [768]

print(f"c_fc.weight.shape   = {list(c_fc_W.shape)}    (expect [768, 3072])")
print(f"c_proj.weight.shape = {list(c_proj_W.shape)}  (expect [3072, 768])")
assert c_fc_W.shape   == (768, 3072), f"unexpected c_fc shape {c_fc_W.shape}"
assert c_fc_b.shape   == (3072,),     f"unexpected c_fc bias {c_fc_b.shape}"
assert c_proj_W.shape == (3072, 768), f"unexpected c_proj shape {c_proj_W.shape}"
assert c_proj_b.shape == (768,),      f"unexpected c_proj bias {c_proj_b.shape}"

# Same input as M4: real GPT-2 token+position embeddings.
TEXT = "Hello, how are you doing today? I hope everything"
token_ids_list = tokenizer.encode(TEXT)[:T]
while len(token_ids_list) < T:
    token_ids_list.append(token_ids_list[-1])

token_ids = torch.tensor([token_ids_list], dtype=torch.long)
pos_ids   = torch.arange(T, dtype=torch.long).unsqueeze(0)
with torch.no_grad():
    x = (model.wte(token_ids) + model.wpe(pos_ids)).squeeze(0).float()  # [T, 768]

print(f"\nTokens (first {T}): {[tokenizer.decode([t]) for t in token_ids_list]}")
print(f"x[0, 0:4] = {x[0, :4].tolist()}")

# Reference computation.
with torch.no_grad():
    # Stage 1: c_fc -- double-accumulated matmul, float32 cast.
    # matmul_nt convention: x [T,768] @ W [768,3072] + b -> [T,3072]
    h = (x.double() @ c_fc_W.double() + c_fc_b.double()).float()  # [T, 3072]

    # Stage 2: GELU (tanh approximation) -- elementwise on float32.
    # approximate='tanh' is mandatory: GPT-2 was trained with the tanh form,
    # not the default erf form.  Using the wrong variant gives close-but-wrong
    # outputs with no obvious error signal.
    h_gelu = F.gelu(h, approximate='tanh')  # [T, 3072], float32

    # Stage 3: c_proj -- double-accumulated matmul, float32 cast.
    # matmul_nt convention: h_gelu [T,3072] @ W [3072,768] + b -> [T,768]
    y = (h_gelu.double() @ c_proj_W.double() + c_proj_b.double()).float()  # [T, 768]

print(f"\nStage values (row 0, first 4 elements):")
print(f"  h      [0, 0:4] = {h[0, :4].tolist()}")
print(f"  h_gelu [0, 0:4] = {h_gelu[0, :4].tolist()}")
print(f"  y      [0, 0:4] = {y[0, :4].tolist()}")
print(f"  y      [{T-1}, 0:4] = {y[T-1, :4].tolist()}")

# HuggingFace cross-check: model.h[0].mlp applies the full MLP in float32.
# Expected diff vs our double-accumulated reference: ~1e-5 from float32 BLAS.
try:
    with torch.no_grad():
        hf_out = model.h[0].mlp(x).float()
        hf_diff = (hf_out - y).abs()
        print(f"\nHuggingFace cross-check (float32 BLAS vs double ref, T={T}):")
        print(f"  max|double_ref - hf_f32| = {hf_diff.max():.2e}  (expect ~1e-5)")
        print(f"  avg diff                 = {hf_diff.mean():.2e}")
except Exception as e:
    print(f"HuggingFace cross-check skipped: {e}")

os.makedirs("data", exist_ok=True)

def write_bin(path, tensor):
    arr = tensor.numpy().astype(np.float32)
    arr.tofile(path)
    print(f"  Written {path}  ({arr.nbytes} bytes, {arr.size} float32s)")

write_bin("data/mlp_c_fc_W.bin",              c_fc_W)
write_bin("data/mlp_c_fc_b.bin",              c_fc_b)
write_bin("data/mlp_c_proj_W.bin",            c_proj_W)
write_bin("data/mlp_c_proj_b.bin",            c_proj_b)
write_bin(f"data/mlp_input_T{T}.bin",         x)
write_bin(f"data/mlp_expected_gelu_T{T}.bin", h_gelu)
write_bin(f"data/mlp_expected_T{T}.bin",      y)

print()
print(f"T={T}, HIDDEN_DIM={HIDDEN_DIM}, MLP_HIDDEN={MLP_HIDDEN}")
print(f"Input  shape: {list(x.shape)}")
print(f"Output shape: {list(y.shape)}")
print(f"GELU intermediate shape: {list(h_gelu.shape)}")
