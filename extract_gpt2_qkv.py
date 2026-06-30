#!/usr/bin/env python3
"""
extract_gpt2_qkv.py -- extract GPT-2 block-0 QKV projection weights and
generate a deterministic test input + PyTorch reference output.

Usage:
  python extract_gpt2_qkv.py [T]

  T  -- sequence length (number of tokens), default 4.

GPT-2 stores c_attn as Conv1D with weight shape [in_features, 3*embed_dim] =
[768, 2304] and computes y = x @ W + b (input left, weight right).  This is
the OPPOSITE of the MNIST convention where W is [out, in] and the kernel does
W @ x.  The binary file written here preserves the [in, out] layout so the
C++ kernel can use y[t*out + j] += x[t*in + k] * W[k*out + j] directly.

Writes (all under data/):
  qkv_W.bin              -- 768*2304 float32, row-major [in=768, out=2304]
  qkv_b.bin              -- 2304 float32 bias
  qkv_input_T{T}.bin     -- T*768 float32, row-major input
  qkv_expected_Q_T{T}.bin -- T*768 float32 Q reference
  qkv_expected_K_T{T}.bin -- T*768 float32 K reference
  qkv_expected_V_T{T}.bin -- T*768 float32 V reference
"""

import sys
import os
import numpy as np
import torch
from transformers import GPT2Model

T = int(sys.argv[1]) if len(sys.argv) > 1 else 4
HIDDEN_DIM = 768
OUT_DIM    = 3 * HIDDEN_DIM  # 2304

print("Loading GPT-2 small from HuggingFace (cached after first run)...")
model = GPT2Model.from_pretrained("gpt2")
model.eval()

# c_attn is Conv1D: weight stored [in_features, out_features] = [768, 2304]
W = model.h[0].attn.c_attn.weight.detach().float()  # [768, 2304]
b = model.h[0].attn.c_attn.bias.detach().float()    # [2304]

print(f"c_attn.weight.shape = {list(W.shape)}  (expect [768, 2304] = [in, out])")
print(f"c_attn.bias.shape   = {list(b.shape)}")
assert W.shape == (HIDDEN_DIM, OUT_DIM), f"unexpected W shape {W.shape}"
assert b.shape == (OUT_DIM,),            f"unexpected b shape {b.shape}"

# Deterministic input: same seed every run
torch.manual_seed(0)
x = torch.randn(T, HIDDEN_DIM, dtype=torch.float32)

# Reference forward pass in double precision so the expected files match the
# double-accumulated C++ kernel rather than PyTorch's float32 BLAS.
# Comparing double vs double isolates correctness; comparing double vs float32
# BLAS adds ~1e-5 of comparison noise for a 768-term sum (42 ULPs at worst).
with torch.no_grad():
    y = x.double() @ W.double() + b.double()  # [T, 2304] in float64
    y = y.float()  # cast back to float32 for binary storage

# Split QKV along the output axis
Q = y[:, :HIDDEN_DIM]           # [T, 768]
K = y[:, HIDDEN_DIM:2*HIDDEN_DIM]  # [T, 768]
V = y[:, 2*HIDDEN_DIM:]         # [T, 768]

os.makedirs("data", exist_ok=True)

def write_bin(path, tensor):
    arr = tensor.numpy().astype(np.float32)
    arr.tofile(path)
    print(f"  Written {path}  ({arr.nbytes} bytes, {arr.size} float32s)")

write_bin("data/qkv_W.bin",                W)
write_bin("data/qkv_b.bin",                b)
write_bin(f"data/qkv_input_T{T}.bin",      x)
write_bin(f"data/qkv_expected_Q_T{T}.bin", Q)
write_bin(f"data/qkv_expected_K_T{T}.bin", K)
write_bin(f"data/qkv_expected_V_T{T}.bin", V)

print()
print(f"T={T}, IN={HIDDEN_DIM}, OUT={OUT_DIM}")
print(f"x shape    : {list(x.shape)}")
print(f"y=x@W+b    : {list(y.shape)}")
print(f"Q row0[0:4]: {Q[0, :4].tolist()}")
print(f"K row0[0:4]: {K[0, :4].tolist()}")
print(f"V row0[0:4]: {V[0, :4].tolist()}")
