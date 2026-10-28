#!/usr/bin/env python3
"""
gen_gpt2_ref.py  --  G-series G1: Python-only gold reference for GPT-2 greedy
generation.

FIXED INPUT (frozen for G1-G5, do not change):
  PROMPT = "Hello, how are you doing today? I hope everything"
  N_GEN  = 20   (new tokens generated beyond the prompt)

Two forward passes are run at each generation step, both receiving the same
context (the float32-greedy token sequence so far):

  float32 pass: HuggingFace model() forward, float32 BLAS throughout.
                Must match model.generate(do_sample=False) exactly -- verified
                step-by-step below.

  double pass:  mirrors the planned C++ worker + Python gen.py convention:
                  wte + wpe lookup: float32 (Python will compute and send to C++)
                  12 blocks: double-accumulated matmuls (x.double() @ W.double()),
                             float32 cast at each stage boundary (matches C++
                             matmul_nt / attn_heads / layernorm convention)
                             float32 residual adds (matches C++ block mode)
                  ln_f:  float32 F.layer_norm (C++ double accumulators match to
                         ~1e-7 per M1 -- negligible at this scale)
                  LM head: float32 x @ wte.T (matches what gen.py will do in G4)

G1 obligations (must be met before G2 starts):
  (a) Verify float32 forward-pass argmax == model.generate() at every step.
  (b) Report double-pass argmax vs float32 argmax at every step; document any
      disagreements with logit margin (near-tie vs real precision error).
  (c) Measure and report max|logits_f32 - logits_double| across all N_GEN steps.
      This number sets the G5 logit-check tolerance -- not a pre-committed
      constant; derived from the measurement with written justification.

Writes:
  data/gen_ref_tokens.txt   -- gold token IDs (float32-greedy), one per line,
                               with decoded strings.  Frozen reference for G4/G5.
  data/gen_ref_logits_f64.bin -- double-pass logits per step, raw float64,
                               [N_GEN, 50257] row-major.  The LM head itself is
                               float32 (see forward_double), so these values are
                               float32-exact, upcast for storage.
  data/gen_ref_summary.txt  -- per step: token, max|f32 - f64| of logits,
                               f64 top-1 minus top-2 margin; plus the sha256
                               of gen_ref_logits_f64.bin.
"""

import hashlib
import os
import time
import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# ── Fixed input -- do not change between G1 and G5 ───────────────────────────
PROMPT = "Hello, how are you doing today? I hope everything"
N_GEN  = 20
# ─────────────────────────────────────────────────────────────────────────────

HIDDEN_DIM = 768
N_HEADS    = 12
HEAD_DIM   = 64      # HIDDEN_DIM // N_HEADS
SCALE      = 0.125   # 1.0 / sqrt(64), exact

print("Loading GPT-2 (124M) from HuggingFace...")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model     = GPT2LMHeadModel.from_pretrained("gpt2")
model.eval()

# Confirm weight tying: lm_head.weight must be the same object as wte.weight.
# If this fails, the script's LM head computation would differ from model().
assert model.lm_head.weight is model.transformer.wte.weight, \
    "weight tying broken -- lm_head.weight is not wte.weight"


# ── Double-precision single-block forward ─────────────────────────────────────
# Mirrors C++ worker --mode block exactly per CLAUDE.md M6.
#
# Precision contract (matches C++ matmul_nt / attn_heads / layernorm):
#   LayerNorm:  float32 F.layer_norm  (C++ double accumulators match to ~1e-7;
#               verified in M1 -- negligible at 12-block scale)
#   Matmul:     (x.double() @ W.double() + b.double()).float()
#               acc stays double through all k steps, cast once at end
#   Attention:  QK^T, softmax, V weighted sum all in float64;
#               cast to float32 once per output element
#   GELU:       F.gelu(approximate='tanh') in float32
#               (matches C++ gelu() which promotes to double for tanh then
#               casts back; difference is sub-ULP at float32 per M5 ~1e-7)
#   Residuals:  float32 (no precision boost; matches C++ x[i] += ...)
#
def block_double(x: torch.Tensor, blk, T: int) -> torch.Tensor:
    """One GPT-2 transformer block in the double-precision convention."""
    # Conv1D weight layout: [in_dim, out_dim] throughout GPT-2 (confirmed M2)
    W_ca = blk.attn.c_attn.weight   # [768, 2304]
    b_ca = blk.attn.c_attn.bias     # [2304]
    W_cp = blk.attn.c_proj.weight   # [768, 768]
    b_cp = blk.attn.c_proj.bias     # [768]
    W_fc = blk.mlp.c_fc.weight      # [768, 3072]
    b_fc = blk.mlp.c_fc.bias        # [3072]
    W_mp = blk.mlp.c_proj.weight    # [3072, 768]
    b_mp = blk.mlp.c_proj.bias      # [768]

    # ── Attention sub-block ───────────────────────────────────────────────────
    residual = x
    x_n = F.layer_norm(x, [HIDDEN_DIM], weight=blk.ln_1.weight, bias=blk.ln_1.bias)

    # QKV projection: double-accumulated, float32 cast at boundary
    qkv = (x_n.double() @ W_ca.double() + b_ca.double()).float()   # [T, 2304]

    # Multi-head causal attention in double
    # GPT-2 QKV layout within qkv: Q | K | V, each [T, HIDDEN_DIM] contiguous
    qd  = qkv.double()
    Q   = qd[:, :HIDDEN_DIM].view(T, N_HEADS, HEAD_DIM).permute(1, 0, 2)                # [12, T, 64]
    K   = qd[:, HIDDEN_DIM:2*HIDDEN_DIM].view(T, N_HEADS, HEAD_DIM).permute(1, 0, 2)   # [12, T, 64]
    V   = qd[:, 2*HIDDEN_DIM:].view(T, N_HEADS, HEAD_DIM).permute(1, 0, 2)             # [12, T, 64]

    scores = torch.bmm(Q, K.transpose(1, 2)) * SCALE           # [12, T, T], float64
    # Causal mask: -1e9 matches C++ attn_heads() constant so exp(-1e9 - row_max)
    # underflows to 0.0 in float64 identically on both sides (confirmed M4)
    causal = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
    scores = scores.masked_fill(causal.unsqueeze(0), -1e9)
    attn_w = torch.softmax(scores, dim=-1)                      # [12, T, T], float64
    ctx    = torch.bmm(attn_w, V)                               # [12, T, 64], float64
    ctx    = ctx.permute(1, 0, 2).contiguous().view(T, HIDDEN_DIM).float()  # [T, 768]

    # c_proj: double-accumulated, float32 cast
    attn_proj = (ctx.double() @ W_cp.double() + b_cp.double()).float()  # [T, 768]
    x = residual + attn_proj   # float32 residual add

    # ── MLP sub-block ─────────────────────────────────────────────────────────
    residual = x
    x_n = F.layer_norm(x, [HIDDEN_DIM], weight=blk.ln_2.weight, bias=blk.ln_2.bias)

    # c_fc: double-accumulated, float32 cast
    h = (x_n.double() @ W_fc.double() + b_fc.double()).float()  # [T, 3072]
    # GELU tanh approximation: mandatory for GPT-2 (trained with this variant;
    # erf form gives wrong outputs with no obvious error signal -- confirmed M5)
    h = F.gelu(h, approximate='tanh')                           # [T, 3072], float32
    # c_proj: double-accumulated, float32 cast
    mlp_out = (h.double() @ W_mp.double() + b_mp.double()).float()  # [T, 768]
    x = residual + mlp_out   # float32 residual add

    return x


def forward_double(ctx_ids: list) -> torch.Tensor:
    """
    Full 12-block GPT-2 forward in the double-precision convention.
    Returns float32 logits [50257] at the last token position only.
    """
    T   = len(ctx_ids)
    ids = torch.tensor([ctx_ids], dtype=torch.long)
    pos = torch.arange(T, dtype=torch.long).unsqueeze(0)

    with torch.no_grad():
        # Embedding: float32 (Python gen.py will compute this and send to C++)
        x = (model.transformer.wte(ids) + model.transformer.wpe(pos)).squeeze(0).float()

        # 12 transformer blocks, double-precision convention
        for i in range(12):
            x = block_double(x, model.transformer.h[i], T)

        # Final layer norm: float32 (gen.py will call model.transformer.ln_f)
        x = F.layer_norm(x, [HIDDEN_DIM],
                         weight=model.transformer.ln_f.weight,
                         bias=model.transformer.ln_f.bias)

        # LM head: float32 x @ wte.T (weight-tied; same as model.lm_head(x))
        logits = (x @ model.transformer.wte.weight.T)   # [T, 50257], float32

    return logits[-1]   # [50257], last position only


def forward_f32(ctx_ids: list) -> torch.Tensor:
    """
    Float32 forward via HuggingFace model().
    Returns logits [50257] at the last position.
    Must match model.generate(do_sample=False) argmax exactly.
    """
    ids = torch.tensor([ctx_ids], dtype=torch.long)
    with torch.no_grad():
        out = model(ids)
    return out.logits[0, -1, :]   # [50257]


# ── Step 1: Gold sequence from model.generate() ───────────────────────────────
prompt_ids = tokenizer.encode(PROMPT)
T_prompt   = len(prompt_ids)

print(f"\nPrompt      : {repr(PROMPT)}")
print(f"Token count : {T_prompt}")
print(f"Token IDs   : {prompt_ids}")
print(f"Decoded toks: {[repr(tokenizer.decode([t])) for t in prompt_ids]}")
print(f"\nRunning model.generate(do_sample=False, max_new_tokens={N_GEN})...")

t0 = time.time()
with torch.no_grad():
    gen_out = model.generate(
        torch.tensor([prompt_ids], dtype=torch.long),
        max_new_tokens=N_GEN,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
print(f"  Done in {time.time()-t0:.1f}s")

gold_new_ids = gen_out[0, T_prompt:].tolist()

print(f"\nGold token sequence ({N_GEN} new tokens from model.generate):")
for step, tid in enumerate(gold_new_ids):
    print(f"  step {step:2d}: id={tid:5d}  {repr(tokenizer.decode([tid]))}")
print(f"\nFull decoded output: {repr(tokenizer.decode(gold_new_ids))}")

# ── Step 2: Verify float32 forward() matches model.generate() ─────────────────
# model() and model.generate() must agree on argmax at every step to confirm
# our forward_f32 helper is a faithful implementation of greedy decoding.
# Any mismatch here would mean forward_f32 is wrong, not that generation diverges.
print("\n[Step 2: Verifying forward_f32() == model.generate() at every step]")
f32_mismatch = False
for step in range(N_GEN):
    ctx  = prompt_ids + gold_new_ids[:step]
    lf   = forward_f32(ctx)
    pred = lf.argmax().item()
    exp  = gold_new_ids[step]
    if pred != exp:
        f32_mismatch = True
        top2   = lf.topk(2).values
        margin = (top2[0] - top2[1]).item()
        print(f"  step {step:2d}: MISMATCH  pred={pred} ({repr(tokenizer.decode([pred]))})"
              f"  expected={exp} ({repr(tokenizer.decode([exp]))})"
              f"  top2-margin={margin:.4e}")
if not f32_mismatch:
    print("  All steps match -- forward_f32() == model.generate() ✓")
else:
    print("  WARNING: mismatch above means forward_f32() is not faithful to model.generate()")
    print("           Investigate before treating float32 logits as the reference.")

# ── Step 3: Per-step float32 vs double comparison ─────────────────────────────
# Context at each step: the float32-greedy token sequence so far.
# Both passes receive identical context, so any logit diff is purely precision.
print(f"\n[Step 3: Per-step logit comparison: float32 (HF BLAS) vs double (C++ convention)]")
print(f"  Context length grows from T={T_prompt} to T={T_prompt + N_GEN - 1}")
print()
hdr = f"{'step':>4}  {'max|f32-dbl|':>14}  {'avg|f32-dbl|':>14}  {'f32_tok':>8}  {'dbl_tok':>8}  agree"
print(hdr)
print("-" * len(hdr))

max_diffs   = []
n_agree     = 0
mismatches  = []
dbl_logits  = []   # per-step double-pass logits, saved for G5
f64_margins = []   # per-step f64 top-1 minus top-2

t_start = time.time()
for step in range(N_GEN):
    ctx = prompt_ids + gold_new_ids[:step]

    lf  = forward_f32(ctx)
    ld  = forward_double(ctx)

    diff  = (lf - ld).abs()
    max_d = diff.max().item()
    avg_d = diff.mean().item()
    max_diffs.append(max_d)

    ld64 = ld.double()
    dbl_logits.append(ld64)
    top2_d = ld64.topk(2).values
    f64_margins.append((top2_d[0] - top2_d[1]).item())

    tok_f = lf.argmax().item()
    tok_d = ld.argmax().item()
    agree = (tok_f == tok_d)
    if agree:
        n_agree += 1

    note = ""
    if not agree:
        # Classify: near-tie (margin < 2×diff) vs real precision error
        top2   = lf.topk(2).values
        margin = (top2[0] - top2[1]).item()
        tie    = (margin < 2.0 * max_d)
        note   = (f"  ← MISMATCH  f32-margin={margin:.2e} "
                  f"{'[near-tie: not a bug]' if tie else '[margin >> diff: investigate]'}")
        mismatches.append((step, tok_f, tok_d, margin, max_d, tie))

    # Abbreviate token strings to keep columns aligned
    def tok_str(tid):
        s = tokenizer.decode([tid])
        return repr(s[:6] + '..' if len(s) > 6 else s)

    print(f"  {step:2d}   {max_d:14.3e}   {avg_d:14.3e}  "
          f"{tok_f:5d}    {tok_d:5d}  {'Y' if agree else 'N'}{note}")

elapsed = time.time() - t_start
print(f"\n  ({elapsed:.1f}s for {N_GEN} steps × 2 forward passes = {elapsed/N_GEN:.1f}s/step avg)")

# ── Step 4: Summary and G5 tolerance derivation ───────────────────────────────
overall_max = max(max_diffs)
overall_avg = sum(max_diffs) / len(max_diffs)
step_max_idx = max_diffs.index(overall_max)

print()
print("=" * 70)
print("G1 SUMMARY")
print("=" * 70)
print(f"PROMPT   = {repr(PROMPT)}")
print(f"N_GEN    = {N_GEN}")
print(f"T_prompt = {T_prompt}  (context grows to T={T_prompt + N_GEN - 1} at last step)")
print()
print(f"float32 forward vs model.generate() : {'ALL MATCH ✓' if not f32_mismatch else 'MISMATCH -- see Step 2'}")
print(f"Token agreement (f32 vs double)     : {n_agree}/{N_GEN} steps agree")

if mismatches:
    print(f"\nToken mismatches between float32 and double ({len(mismatches)} total):")
    for step, tf, td, margin, maxd, tie in mismatches:
        print(f"  step {step:2d}: f32={tf} ({repr(tokenizer.decode([tf]))}) "
              f"dbl={td} ({repr(tokenizer.decode([td]))}) "
              f"margin={margin:.2e} max_diff={maxd:.2e} "
              f"{'[near-tie]' if tie else '[investigate]'}")

print()
print("Logit diff (float32 HF BLAS vs double C++ convention):")
print(f"  Overall max|diff| across {N_GEN} steps : {overall_max:.4e}  (at step {step_max_idx})")
print(f"  Mean of per-step max|diff|             : {overall_avg:.4e}")
print(f"  Min  of per-step max|diff|             : {min(max_diffs):.4e}  (at step {max_diffs.index(min(max_diffs))})")
print()
print("G5 tolerance derivation:")
print(f"  Measured max|logits_f32 - logits_double| = {overall_max:.4e}")
print(f"  Source: {overall_max:.4e} = accumulated error across 12 blocks + ln_f + LM head")
print(f"  At M6 single-block level: max diff was 3.81e-06 per block.")
print(f"  12 blocks compound additively; LM head (768-term matmul x 50257) adds ~1e-5 more.")
print(f"  Safety factor 3x applied: {3.0 * overall_max:.2e}  <-- provisional G5 tolerance")
print(f"  IMPORTANT: this is provisional. The G5 tolerance must be set to the SMALLEST")
print(f"  value that passes all N_GEN steps, with written justification. If the measured")
print(f"  max is implausibly large (> 1e-2), diagnose before setting the tolerance.")

# ── Step 5: Write frozen gold reference ───────────────────────────────────────
os.makedirs("data", exist_ok=True)
ref_path = "data/gen_ref_tokens.txt"
with open(ref_path, "w", newline="\n") as f:
    f.write("# Gold token sequence -- float32 greedy (model.generate, do_sample=False)\n")
    f.write(f"# PROMPT = {repr(PROMPT)}\n")
    f.write(f"# N_GEN  = {N_GEN}\n")
    f.write(f"# T_prompt = {T_prompt}\n")
    f.write(f"# prompt_ids = {prompt_ids}\n")
    f.write("#\n")
    f.write("# step  token_id  decoded\n")
    for step, tid in enumerate(gold_new_ids):
        f.write(f"{step:4d}  {tid:7d}  {repr(tokenizer.decode([tid]))}\n")
print(f"\nGold reference written to {ref_path}")

# ── Step 6: Per-step double-pass logits + summary for G5 ──────────────────────
logits_path = "data/gen_ref_logits_f64.bin"
torch.stack(dbl_logits).numpy().tofile(logits_path)   # [N_GEN, 50257] float64
with open(logits_path, "rb") as f:
    logits_sha = hashlib.sha256(f.read()).hexdigest()

summary_path = "data/gen_ref_summary.txt"
with open(summary_path, "w", newline="\n") as f:
    f.write("# G1 per-step summary -- context = float32-greedy tokens so far\n")
    f.write(f"# PROMPT = {repr(PROMPT)}\n")
    f.write(f"# N_GEN  = {N_GEN}\n")
    f.write("# max_abs_diff = max|logits_f32 - logits_f64| over the vocab\n")
    f.write("# f64_margin   = top-1 minus top-2 of logits_f64\n")
    f.write("# step  token  max_abs_diff  f64_margin\n")
    for step in range(N_GEN):
        f.write(f"{step:4d}  {gold_new_ids[step]:5d}  {max_diffs[step]:.6e}  {f64_margins[step]:.6e}\n")
    f.write(f"# sha256({logits_path}) = {logits_sha}\n")
print(f"Per-step logits written to {logits_path} (sha256 {logits_sha})")
print(f"Summary written to {summary_path}")
