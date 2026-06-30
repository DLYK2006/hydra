#!/usr/bin/env python3
"""
extract_gpt2_ln.py -- extract GPT-2 LayerNorm (ln_1, block 0) weights and
generate a deterministic test input + PyTorch reference output.

Usage:
  python extract_gpt2_ln.py [T]

  T  -- sequence length (number of rows), default 1.
        Run with T=1 for M1, T=4 for M1b.

Writes (all under data/):
  ln_input_T{T}.bin      -- T*768 float32, row-major
  ln_gamma.bin           -- 768 float32 (shared across T values)
  ln_beta.bin            -- 768 float32
  ln_expected_T{T}.bin   -- T*768 float32 reference output from F.layer_norm
"""

import sys
import os
import struct
import numpy as np
import torch
import torch.nn.functional as F
from transformers import GPT2Model

T = int(sys.argv[1]) if len(sys.argv) > 1 else 1
HIDDEN_DIM = 768
EPS = 1e-5

print(f"Loading GPT-2 small from HuggingFace (downloads ~548 MB on first run)...")
model = GPT2Model.from_pretrained("gpt2")
model.eval()

# Access weights via module attribute -- avoids state_dict key naming ambiguity
# between GPT2Model ('h.0.ln_1.weight') and GPT2LMHeadModel ('transformer.h.0...')
gamma = model.h[0].ln_1.weight.detach().float()  # [768]
beta  = model.h[0].ln_1.bias.detach().float()    # [768]

assert gamma.shape == (HIDDEN_DIM,), f"unexpected gamma shape {gamma.shape}"
assert beta.shape  == (HIDDEN_DIM,), f"unexpected beta shape {beta.shape}"

# Deterministic input: same seed every run so the binary files are reproducible
torch.manual_seed(0)
x = torch.randn(T, HIDDEN_DIM, dtype=torch.float32)

# Reference forward pass -- this is what the C++ worker must match
y_ref = F.layer_norm(x, [HIDDEN_DIM], gamma, beta, eps=EPS)

os.makedirs("data", exist_ok=True)

def write_bin(path, tensor):
    arr = tensor.numpy().astype(np.float32)
    arr.tofile(path)
    print(f"  Written {path}  ({arr.nbytes} bytes, {arr.size} float32s)")

write_bin(f"data/ln_input_T{T}.bin",    x)
write_bin("data/ln_gamma.bin",          gamma)
write_bin("data/ln_beta.bin",           beta)
write_bin(f"data/ln_expected_T{T}.bin", y_ref)

print()
print(f"T={T}, HIDDEN_DIM={HIDDEN_DIM}, eps={EPS}")
print(f"Input  shape : {list(x.shape)}")
print(f"Output shape : {list(y_ref.shape)}")
print(f"First output row (first 6 values): {y_ref[0, :6].tolist()}")
if T > 1:
    print(f"Last output row  (first 6 values): {y_ref[-1, :6].tolist()}")
