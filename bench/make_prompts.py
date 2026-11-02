# Writes the benchmark prompts as GPT-2 token-id files (one id per line):
#   prompt_L<n>.txt  first n ids of ONE fixed-seed stream of 1024 ids, n in 32,128,512,1024-N
#   prompt_g5.txt    the fixed G1-G5 PROMPT tokenized (anchor: must reproduce gen_ref tokens)
#   SHA256SUMS
# Refuses to overwrite. Commit the outputs before any timed run.
# Usage (WSL2, repo root):
#   /mnt/c/Users/ASUS/AppData/Local/Programs/Python/Python313/python.exe bench/make_prompts.py N_DECODE
import hashlib
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from transformers import GPT2Tokenizer   # noqa: E402
from gen_hydra import PROMPT             # noqa: E402

VOCAB, CTX, SEED = 50257, 1024, 20261102
n_decode = int(sys.argv[1])
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts")
if os.path.exists(out):
    sys.exit(f"ERROR: {out} exists; refusing to overwrite")
ids = np.random.default_rng(SEED).integers(0, VOCAB, size=CTX)
files = {f"prompt_L{L}.txt": (f"# seed={SEED} vocab={VOCAB} first {L} of {CTX} ids\n", ids[:L])
         for L in (32, 128, 512, CTX - n_decode)}
files["prompt_g5.txt"] = (f"# gen_hydra.PROMPT = {PROMPT!r}\n", GPT2Tokenizer.from_pretrained("gpt2").encode(PROMPT))
os.makedirs(out)
sums = []
for name, (hdr, seq) in files.items():
    body = hdr + "".join(f"{int(i)}\n" for i in seq)
    with open(os.path.join(out, name), "w", newline="\n") as f:
        f.write(body)
    sums.append(f"{hashlib.sha256(body.encode()).hexdigest()}  {name}\n")
with open(os.path.join(out, "SHA256SUMS"), "w", newline="\n") as f:
    f.writelines(sums)
print("".join(sums), end="")
