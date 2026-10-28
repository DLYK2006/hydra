# G2 verify. Hand-written; the agent must not edit this file. Run from repo root.
# Checks every h{NN}_*.bin against the model's own block-NN tensor, byte for byte.
import sys
import numpy as np
import torch
from transformers import GPT2LMHeadModel

NAMES = {
    "ln1_gamma": "ln_1.weight",       "ln1_beta": "ln_1.bias",
    "c_attn_W": "attn.c_attn.weight", "c_attn_b": "attn.c_attn.bias",
    "attn_proj_W": "attn.c_proj.weight", "attn_proj_b": "attn.c_proj.bias",
    "ln2_gamma": "ln_2.weight",       "ln2_beta": "ln_2.bias",
    "c_fc_W": "mlp.c_fc.weight",      "c_fc_b": "mlp.c_fc.bias",
    "mlp_proj_W": "mlp.c_proj.weight", "mlp_proj_b": "mlp.c_proj.bias",
}

m = GPT2LMHeadModel.from_pretrained("gpt2").eval()
ref = [{k: v.detach().to(torch.float32).numpy().reshape(-1)
        for k, v in m.transformer.h[i].state_dict().items()} for i in range(12)]

ok = bad = 0
for i in range(12):
    for short, key in NAMES.items():
        path = f"data/h{i:02d}_{short}.bin"
        try:
            got = np.fromfile(path, dtype=np.float32)
        except FileNotFoundError:
            print(f"MISSING {path}")
            bad += 1
            continue
        if got.size == ref[i][key].size and np.array_equal(got, ref[i][key]):
            ok += 1
        else:
            hits = [j for j in range(12)
                    if got.size == ref[j][key].size and np.array_equal(got, ref[j][key])]
            print(f"BAD {path} size={got.size} matches_block={hits if hits else 'none'}")
            bad += 1

print(f"VERIFY {ok}/144 OK, {bad} BAD")
passed = ok == 144 and bad == 0
print("G2_VERIFY_PASS" if passed else "G2_VERIFY_FAIL")
sys.exit(0 if passed else 1)
