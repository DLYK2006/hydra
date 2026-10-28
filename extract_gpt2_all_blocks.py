#!/usr/bin/env python3
"""
extract_gpt2_all_blocks.py  --  G2: write all 12 GPT-2 small blocks' weights to disk.

Usage (from repo root):
  python extract_gpt2_all_blocks.py

Writes 144 files, data/h{NN}_{name}.bin for NN = 00..11, raw float32,
row-major, no header -- the same encoding as the M1-M6 extract scripts:

  name          tensor               shape         Conv1D layout
  ln1_gamma     ln_1.weight          [768]
  ln1_beta      ln_1.bias            [768]
  c_attn_W      attn.c_attn.weight   [768, 2304]   [in, out]
  c_attn_b      attn.c_attn.bias     [2304]
  attn_proj_W   attn.c_proj.weight   [768, 768]    [in, out]
  attn_proj_b   attn.c_proj.bias     [768]
  ln2_gamma     ln_2.weight          [768]
  ln2_beta      ln_2.bias            [768]
  c_fc_W        mlp.c_fc.weight      [768, 3072]   [in, out]
  c_fc_b        mlp.c_fc.bias        [3072]
  mlp_proj_W    mlp.c_proj.weight    [3072, 768]   [in, out]
  mlp_proj_b    mlp.c_proj.bias      [768]

Per block 28,351,488 bytes; all 12 blocks 340,217,856 bytes.

Checks (run after this script):
  bash scripts/g2_canary.sh    -- h00_* byte-identical to the June block-0 files
  python scripts/g2_verify.py  -- every h{NN}_* byte-identical to transformer.h[NN]
"""

import glob
import os
import sys
import numpy as np
from transformers import GPT2Model

N_BLOCKS = 12

# (file name, state_dict key within transformer.h[i], expected shape)
TENSORS = [
    ("ln1_gamma",   "ln_1.weight",        (768,)),
    ("ln1_beta",    "ln_1.bias",          (768,)),
    ("c_attn_W",    "attn.c_attn.weight", (768, 2304)),
    ("c_attn_b",    "attn.c_attn.bias",   (2304,)),
    ("attn_proj_W", "attn.c_proj.weight", (768, 768)),
    ("attn_proj_b", "attn.c_proj.bias",   (768,)),
    ("ln2_gamma",   "ln_2.weight",        (768,)),
    ("ln2_beta",    "ln_2.bias",          (768,)),
    ("c_fc_W",      "mlp.c_fc.weight",    (768, 3072)),
    ("c_fc_b",      "mlp.c_fc.bias",      (3072,)),
    ("mlp_proj_W",  "mlp.c_proj.weight",  (3072, 768)),
    ("mlp_proj_b",  "mlp.c_proj.bias",    (768,)),
]

# Refuse to overwrite: existing h* files are outputs someone has already
# verified (or not); regenerating must be an explicit delete-then-run.
existing = sorted(glob.glob("data/h*.bin"))
if existing:
    print(f"ERROR: {len(existing)} data/h*.bin file(s) already exist, e.g. {existing[0]}.",
          file=sys.stderr)
    print("Refusing to overwrite. Delete data/h*.bin first if regeneration is intended.",
          file=sys.stderr)
    sys.exit(1)

print("Loading GPT-2 small from HuggingFace (cached after first run)...")
model = GPT2Model.from_pretrained("gpt2")
model.eval()

os.makedirs("data", exist_ok=True)

total = 0
for i in range(N_BLOCKS):
    sd = model.h[i].state_dict()
    for name, key, shape in TENSORS:
        t = sd[key].detach().float()
        # Conv1D [in, out] layout confirmed in M2; a transposed checkpoint
        # would silently produce a wrong-but-same-size file, so fail here
        # (explicit raise, not assert, so `python -O` cannot skip it).
        if tuple(t.shape) != shape:
            raise ValueError(f"h{i:02d} {key}: shape {tuple(t.shape)} != {shape}")
        arr = np.ascontiguousarray(t.numpy(), dtype=np.float32)
        path = f"data/h{i:02d}_{name}.bin"
        arr.tofile(path)
        total += arr.nbytes
    print(f"  block {i:02d}: 12 files written")

print(f"Wrote {N_BLOCKS * len(TENSORS)} files, {total} bytes total")
