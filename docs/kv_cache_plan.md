# KV-cache plan (approved for build, 2026-11-01)

**Scope:**
- The Mac worker keeps per-layer K/V for all 12 GPT-2 blocks.
- Prefill sends the prompt once; decode sends one new row and attends over
  the cache.
- One minimal, backward-compatible proto change.
- `gen_hydra.py --kv-cache on|off`, where off is byte-identical to today.
- A new checker, `scripts/kv_check.sh`, committed before the first cached
  run.
- Benchmarking and profiling are out of scope (see the last section).
- `skills/protobuf-workflow.md` was read first. This change is a schema
  change, so its full procedure applies (Protocol section).

## 0. Byte identity: what is exact, and what is bounded
Measured read-only before any code, from `scratchpad/kv_probe.py` and
`kv_probe2.py` (data: `data/g5_mirror_hidden.bin`, which is byte-identical to
Hydra's G5 hidden states, and `data/g5_hydra_logits.bin`):

- **Prefix stability: 19/19 step pairs byte-identical.** In a full recompute,
  rows 0..T-1 of block 11's output do not change when the context grows.
  - Why: every op is row-wise (LayerNorm, the matmuls, GELU, residuals),
    except attention.
  - In attention, the causal mask sets `scores = -1e9`; `exp(-1e9 - max)`
    is exactly `0.0`, and adding `+0.0` or `0.0*V` to a sum leaves it
    bitwise unchanged.
  - **Consequence:** K/V rows cached at an earlier step equal what a full
    recompute would produce, and so does every returned row.
- **The C++ decode path can be bitwise-identical to full recompute** for the
  new row, provided the cached attention uses the same loop orders as
  `attn_heads` (worker.cpp:178):
  - scores: for h, then kt ascending, then a double dot over d ascending,
    times `scale`;
  - `mx` = row[0], then kt=1..;
  - sum of exp, kt ascending; divide;
  - V accumulation, kt ascending.

  The only difference is that decode skips the masked keys kt > p. Those
  contribute exact zeros in the full path, so nothing changes. [Likely, high;
  verified empirically by gate B.]
- **The op that does NOT preserve arithmetic order is the Python LM head.**
  `head_f64` in gen_hydra.py computes `x @ wte.weight.double().T`:
  - On the same normalized row, an `[T,768] @ W.T` last row and a
    `[1,768] @ W.T` row are byte-identical at only **15/20** steps (max diff
    4.832e-13).
  - Matrix-vector `W @ y` is also 15/20.
  - `F.layer_norm` is shape-independent: 20/20.
  - [Likely] cause: torch/BLAS picks a different kernel or blocking for
    M=1, which changes the reduction order over k=768.
  - **Owner decision (2026-11-01):** with the cache ON, the client applies
    ln_f + LM head to the **newest row only**; cache OFF stays exactly as
    today.
    - Reason: the full-matrix head is about 40e9 multiply-adds per token at
      T=1024, about 380× the cached worker's work.
    - Consequence: cache-ON logits are **not** byte-identical to the frozen
      full-matrix logits (they differ at the 1e-13 level at 5/20 steps).
  - **Exact (hard, byte-identical):** tokens; the prefill rows; each step's
    newest block-11 row (against the mirror, the frozen cache-off run, and in
    gate C the cache-off run).
  - **Bounded (hard, per element, not a tuned constant):** logits must satisfy
    `|a-b| <= 2*n*u*sum_i|x_i*w_i|` with n=768 and u=2^-53. Here x is the
    float64 ln_f output of the step's newest block-11 row (byte-identical on
    both sides by the hidden gate) and w is that token's wte row. It is
    computed in float64 in `kv_compare.py`, which reports max(|a-b|/bound).
  - **Measured on the 20 G5 steps** (read-only `scratchpad/kv_bound_probe.py`;
    newest-row head vs the frozen logits):
    - honest: max ratio **1.613e-02**, 0/20 steps violate;
    - float32 head: at least **6.520e+06** × the bound, 20/20 steps violate;
    - wrong wte row: at least **1.002e+12** ×, 20/20 violate.
    - The bound itself is about 2.3e-11 to 3.0e-11 at step 0.

## 1. Design: cache in the Mac worker (block mode only)
- **Capacity:** `KV_CAPACITY = 1024` positions (GPT-2 n_ctx).
- **Storage**, allocated once at startup in block mode, after the weights
  load and before `bind`:
  - per layer L (12): `K_L` and `V_L`, each `[1024, 768]` float32, holding
    exactly the float32 values in `qkv_buf` columns 768..1535 (K) and
    1536..2303 (V) for each cached position;
  - **exact bytes: 12 layers × 2 (K, V) × 1024 × 768 × 4 B = 75,497,472 B
    (72.0 MiB)**;
  - plus `len[12]` (int) and `cached_seq` (uint64).
  - With the 340,217,856 B of weights, about 396.5 MiB (415,715,328 B) resident on the 8 GiB
    Mac.
- **Per packet in block mode:**
  - `sequence_id == 0`: today's stateless path, byte-for-byte; the cache is
    never touched.
  - `sequence_id != 0`, validated **before any state change**. Error contract
    on this path: **every** rejection (checks 1-5, including shape and data
    size) sends the existing "no data, no shape" reply, keeps the connection,
    and the worker stays up. The `sequence_id == 0` path keeps today's
    behaviour unchanged: shape and size errors close that connection without
    a reply (worker.cpp:809-829). That divergence is deliberate, so existing
    clients and checkers see no behaviour change.
    1. `layer_index` in 0..11 (as today).
    2. Shape `[T,768]` with `1 <= T <= KV_CAPACITY` (checked first, so T is
       small), then data size == T·768 computed in `int64_t`. No int overflow
       for any wire T. The `sequence_id == 0` path keeps today's
       `pkt_T * pkt_H` int check (worker.cpp:823) unchanged, as stated.
    3. **Bounds:** `start_pos >= 0` (int32 from the wire) and, in 64-bit
       arithmetic, `(int64_t)start_pos + T <= 1024`. Otherwise reject, logging
       `KV bounds: start_pos=... T=... capacity 1024`. This rules out signed
       overflow (UB) near INT_MAX and negative positions. It comes before (4). (An oversize prefill, T > 1024, is already
       rejected by check 2's `T <= KV_CAPACITY`, so it never reaches here.)
    4. **Sequence:** if `sequence_id != cached_seq`, `start_pos` must be 0;
       then reset (`cached_seq = sequence_id`, all 12 `len = 0`). Otherwise
       reject: `unknown sequence`.
    5. **Alignment:** `start_pos == len[L]`, else reject:
       `start_pos X != cached length Y (layer L)`. This rejects gaps,
       rewrites, replays, and a stale or wrong position.
  - Then compute:
    - LN1 on the T new rows;
    - `matmul_nt` → `qkv_buf[T,2304]`;
    - copy the K and V slices of each new row into `K_L/V_L[start_pos + i]`;
    - `len[L] += T`;
    - `attn_heads_cached`: query row i at position p = start_pos + i attends
      over keys kt = 0..p, read from the cache, with the loop orders above;
    - `c_proj`, residual, LN2, MLP, residual — unchanged per-row code;
    - reply with the T new rows only, shape `[T,768]`.
- **Prefill** is `start_pos = 0`, T = prompt length. **Decode** is
  `start_pos = len`, T = 1.
- **Single active sequence:** one cache; any new `sequence_id` resets it. A
  failure part-way through a step (some layers appended, a later one
  rejected) leaves the `len[]` values mismatched. The next packet of that
  sequence is then rejected by (5), and the client must start a new sequence.
  Fail-closed.

### worker.cpp changes (file:line at HEAD 8b58b4e)
| # | Where | Change |
|---|---|---|
| K1 | after :271 (`MAX_MSG_LEN`) | `constexpr int KV_CAPACITY = 1024;` |
| K2 | after `struct BlockWeights` (:275-283) | `struct KVCache { std::vector<float> K[N_BLOCKS], V[N_BLOCKS]; int len[N_BLOCKS]; uint64_t seq; };` |
| K3 | after `attn_heads()` (:178-232) | `attn_heads_cached(const float* qkv_new, int T_new, int start_pos, const float* K, const float* V, float* out, …)`: the same loop and accumulation order as `attn_heads`, keys 0..p per query row, reusing its `scores` and `acc` buffer pattern. `attn_heads` itself is untouched, so the sequence-0 path is byte-identical by construction. |
| K4 | block load branch (:407-436) | After the weights load, allocate the cache (`assign(KV_CAPACITY*HIDDEN_DIM, 0.f)` per layer and per K/V), log `KV cache: 75497472 bytes`. Fail-fast before `bind`. |
| K5 | block branch (:789-807, after the L range check) | If `packet.sequence_id() != 0`: validations 2-5 with error replies (send, then `continue`, as in the existing L-range rejection), then the cached compute path. Else: today's code, unchanged. |
| K6 | log line :891 | When the cache is used, the existing "Computed block layer L [T,768]" line gains `seq=... start_pos=... cache_len=...`. (No timing in this milestone; see the last section.) |

The other modes are unchanged, and the orchestrator is unchanged (fields
default to 0). Both are rebuilt only because the generated .pb files change.

## 2. Protocol: minimal, backward-compatible proto change
```protobuf
message TensorPacket {
    int32          layer_index = 1;
    repeated float data        = 2;
    string         device_id   = 3;
    repeated int32 shape       = 4;
    uint64         sequence_id = 5;  // 0 = no cache (today's behaviour); !=0 = KV-cached sequence
    int32          start_pos   = 6;  // absolute position of data's first row within the sequence
}
```
- **`sequence_id`:**
  - It switches caching on and identifies the sequence. A value the worker
    hasn't seen means a new sequence and a cache reset (it must come with
    `start_pos` 0).
  - 0 keeps every existing client and checker on today's path. In proto3 a
    0 value isn't serialized, so their wire bytes are unchanged.
  - Its name differs from CLAUDE.md's planned `sequence_num` (drop and
    reorder detection) on purpose: a different purpose and a different
    field.
- **`start_pos`:** the worker uses it to:
  - verify alignment with its cached length, which catches K/V appended at
    the wrong position, gaps, and replays;
  - place the new K/V rows;
  - apply the causal limit p = start_pos + i.

  Without it, a wrong position would be silently appended. 0 means prefill.
- **Compatibility:**
  - An old client talking to a new worker: both fields are 0, so today's
    path.
  - A new client with the cache off talking to an old worker: identical
    bytes.
  - A new client with the cache on talking to an OLD worker: the old worker
    ignores the unknown fields and runs stateless on 1 row, which gives wrong
    output. Gate B's last-row and logit byte identity catches this.
- Field numbers 5 and 6 are new; no number is reused or changed (skill rules).

**Regeneration, in the skill file's order (each needs approval when run):**
1. Edit `hydra.proto` on WSL2 only.
2. WSL2: `make proto` (`protoc --cpp_out=. hydra.proto`); `ls -lh hydra.pb.h hydra.pb.cc`
   shows a new timestamp; `grep -c sequence_id hydra.pb.h` > 0.
3. WSL2: `make clean && make`. The orchestrator must build with zero
   warnings.
4. Python: `protoc --python_out=. hydra.proto`. Record the new
   `sha256sum hydra_pb2.py` in `skills/protobuf-workflow.md` (gitignored;
   replacing `dfc9e1f2…`). Then `python -W error -c "import hydra_pb2;
   hydra_pb2.TensorPacket(sequence_id=1, start_pos=2)"`.
5. Mac: scp **only** `hydra.proto` (and `worker.cpp`). Never `.pb.*`.
6. Mac: `make proto`. The Mac Makefile (contents read 2026-10-28) sets
   `BREW_PREFIX := $(shell brew --prefix protobuf 2>/dev/null || echo /usr/local)`
   and `PROTOC := $(BREW_PREFIX)/bin/protoc`. Over non-interactive SSH, `brew`
   is not on PATH, so this falls back to `/usr/local/bin/protoc`; `proto_exit`
   is captured either way;
   `ls -lh hydra.pb.h hydra.pb.cc`; `grep -c sequence_id hydra.pb.h` > 0;
   `/usr/local/bin/protoc --version` (full path: the SSH PATH lacks
   /usr/local/bin).
7. Mac: `make clean && make`. Zero errors. **Deviation from the skill's
   "zero warnings":** the known Homebrew Abseil `-Wsign-compare` and ld dylib
   version warnings are allowed (CLAUDE.md Known Weaknesses). Any warning from
   project code fails.
8. Round trip: gate B, where the worker must honour `sequence_id` and
   `start_pos` sent by Python.

## 3. Client: gen_hydra.py
New flags: `--kv-cache on|off` (default **off**), `--n-gen N` (default 20)
and `--prompt TEXT` (default the fixed prompt). `--n-gen` and `--prompt` are
needed by gates C and D.

- **Off:** the code path is unchanged, and the packets don't set fields 5 and
  6, so the wire bytes are identical to today. The output files are
  identical too (gate A, plus a `cmp` against `data/live/g5_run1/*`).
  `tokens.txt`'s header uses PROMPT and N_GEN, so it is unchanged at the
  defaults.
- **On:**
  - `sequence_id = secrets.randbits(63) | 1`, fresh per run.
  - Step 0: prefill all prompt rows (`start_pos` 0).
  - Step s ≥ 1: send only the new token's embedding row (wte + wpe at
    position p = T-1) with `start_pos = p`, T = 1.
  - Logits: `head_f64(newest_row)` only: float64 ln_f, then the
    `[1,768] @ wte.T` matmul (owner decision, section 0).
  - Keep only the block-11 rows the hidden gates need: the prefill reply's
    rows (step 0), then each step's newest row. No full-matrix
    reconstruction.
  - One connection per step: 12 packets, as today.
- **New output `sends.txt`**, written in **both** modes (off: `sequence_id`
  0, `start_pos` 0, full T), one line per packet:
  `step layer sequence_id start_pos T`.
  - The values come from the constructed packet's fields
    (`pkt.sequence_id`, `pkt.start_pos`, `pkt.shape[0]`), not from loop
    variables, after `SerializeToString`.
  - The checker uses it to prove the client used the cache.
  - gen_hydra.py's refuse-to-overwrite list includes `sends.txt`.
- The `tokens.txt` header records the **actual** prompt and n-gen; it is
  unchanged at the defaults.
- `hidden_f32.bin` layout:
  - cache OFF: unchanged (every step's full block-11 matrix);
  - cache ON: exactly the rows received, i.e. T_prompt prefill rows, then
    one newest row per later step: T_prompt + n_gen − 1 rows (30 for the
    fixed prompt, 20 tokens).

## 4. Gates: scripts/kv_check.sh (+ kv_compare.py, kv_overflow.py), drafts in repo
They run against a freshly launched KV-build worker in `--mode block`.

- **Before running:**
  - PRECHECK refuses if `data/live/kv` or `data/live/g5` exists (both absent
    after the approved renames).
  - The oracles, the frozen logits and pin, and the client/helpers must be
    committed and unmodified.
- **A, cache off:** `bash scripts/g5_check.sh` must exit 0 and print
  `PASS logits byte-identical to frozen` and `G5_CHECK_PASS`. **Plus**
  `cmp` of `data/live/g5/{tokens.txt,logits_f64.bin,hidden_f32.bin}` against
  the pre-KV client's `data/live/g5_run1/` (tokens header included). The
  precheck requires `g5_run1`'s three files to exist.
- **B, cache on, cold, fixed prompt, 20 tokens:**
  - the token rows are byte-identical to `data/gen_ref_tokens.txt`;
  - the **prefill rows** (11) are byte-identical to the mirror's step-0 rows;
  - **each step's newest block-11 row** is byte-identical to that step's last
    row in `data/g5_mirror_hidden.bin` **and** in the frozen cache-off run
    `data/live/g5_run1/hidden_f32.bin`;
  - `logits_f64.bin` is within the per-element bound of
    `data/g5_hydra_logits.bin` (section 0); max ratio reported;
  - `sends.txt` shows prefill (`start_pos` 0, T=11), then T=1 at
    `start_pos = 10+s` for all 12 layers, with one nonzero `sequence_id`;
  - **worker-side cross-check** (by the loop, approved log read): after
    kv_check, `worker_kv_block_<RUN>.log` must show, for the stage-B sequence,
    "Computed block layer L [1,768] ... seq=..." lines (19 decode steps × 12
    layers) and 12 prefill lines with [11,768]. A client that silently sends
    full context cannot produce those.
- **C, 64 tokens, cache on vs off (hard, owner decision):**
  - the token IDs must be identical;
  - the prefill rows and each step's newest block-11 row are byte-identical
    (on vs off);
  - the logits are within the per-element bound (section 0); it reports
    max|Δlogits| at the final step and overall, and the max ratio;
  - it checks the cache was used in the on-run.
- **D, stale cache:** cache on with the prompt "The quick brown fox jumps
  over the lazy dog" (9 tokens; +5 generated = 13 cached positions, so
  positions 11-12 stay stale after the fixed prompt's 11-row prefill) for 5
  tokens, then the fixed prompt with cache on.
  `tokens.txt`, `logits_f64.bin` and `hidden_f32.bin` must be byte-identical
  to B's cold run.
- **E, protocol and overflow:** `kv_overflow.py`, layer 0 only, one
  connection per request; it puts the repo root on `sys.path` to import
  gen_hydra's framing. Timeout 600 s, because a T=1000 prefill through one
  block is slow on the Mac.
  - Every rejection is followed by an accepted, aligned packet on the
    already-established sequence D. That proves no rejection reset, adopted
    or partly appended the cache, including on the reset path (check 4).
  - **E0:** D prefill 4.
  - **E1:** a new sequence's prefill T=1025 → rejected; then D at 4 →
    accepted.
  - **E2:** D overflow append (start 5, T=1020) → rejected; then D at 5 →
    accepted.
  - **E3:** D replay at 0 and gap at 7 → rejected; aligned at 6 → accepted.
  - **E4:** new sequence E at `start_pos` 5 → rejected; then D at 7 →
    accepted (no reset/adoption); then E at 0 → accepted (reset to E).
  - **E5:** an 8-row single prefill vs 4 + 4 chunked; the reply rows must be
    byte-identical. That checks values, not just shapes, for T>1 at
    `start_pos` > 0.
  - **E6:** fill to exactly 1024 (1000 + 24) → accepted; one more row →
    rejected.
  - **E7:** a new sequence → accepted (worker alive).
- **F, G3 regression on the rebuilt worker** (run by the loop, not
  kv_check.sh):
  - `bash scripts/g3_check_v2.sh` with a block worker → `G3V2_CHECK_PASS`;
  - `bash scripts/regress_mode.sh {layernorm,qkv,attention,mlp}`, each with
    its mode's worker → 4 × `REGRESS_<MODE>_PASS`;
  - each needs its staging dir absent: `data/live/g3v2` and
    `data/live/regress_*` were renamed to `<name>_pre_kv_20261101`
    (owner-approved, 2026-11-01), so they are absent now.
- **The whole set is committed before the first cached run**, with headers
  changed from "DRAFT pending approval" to "Hand-reviewed; do not edit" at
  commit, so checklist item 2 applies afterwards. It ends in
  `KV_CHECK_PASS` or `KV_CHECK_FAIL`.

## 5. Mac commands (each one listed, to be approved)
With `M=yonghowlee@10.0.0.62` and `D='"$HOME/Documents/DLYK stuff"'`:
1. `scp -q worker.cpp hydra.proto "yonghowlee@10.0.0.62:Documents/DLYK stuff/"`
   (only these two files; never `.pb.*`).
2. Hashes: `ssh $M "cd $D && shasum -a 256 worker.cpp hydra.proto worker"`
   (compare with WSL2 `sha256sum worker.cpp hydra.proto`).
3. Backup: `ssh $M "cd $D && cp -p worker worker.pre_kv"`
4. Regenerate:
   `ssh $M "cd $D && make proto > live_logs/make_proto_kv_<RUN>.log 2>&1; echo proto_exit=\$?; ls -lh hydra.pb.h hydra.pb.cc; grep -c sequence_id hydra.pb.h; /usr/local/bin/protoc --version"`
   (`<RUN>` is a fresh UTC timestamp per attempt, e.g. 20261102T1530Z, so
   earlier logs are never overwritten. The same applies to steps 5 and 8.)
5. Build:
   `ssh $M "cd $D && make clean && make > live_logs/make_kv_<RUN>.log 2>&1; echo make_exit=\$?; grep -c 'error:' live_logs/make_kv_<RUN>.log"`
6. New binary hash: `ssh $M "cd $D && shasum -a 256 worker"`
7. Pre-launch: `ssh $M "pgrep -lx worker; lsof -nP -iTCP:50051 -sTCP:LISTEN"`
8. Launch, with the gate (PID non-empty plus `listening` before any checker):
   - KV block worker:
     `ssh $M "cd $D && (nohup ./worker --mode block > live_logs/worker_kv_block_<RUN>.log 2>&1 < /dev/null & echo \$!)"`
   - Gate F reuses the approved mode launch commands, but with per-run log names
     (`worker_g3_block_<RUN>.log`, `worker_g3_<mode>_<RUN>.log`) so the G3
     evidence logs `worker_g3_block.log` / `worker_g3_<mode>.log` are not overwritten.
9. Wait / read log: `ssh $M "cd $D; for i in \$(seq 1 40); do grep -q listening live_logs/<log> && break; sleep 0.5; done; cat live_logs/<log>"`
10. Stop: `ssh $M "kill -INT <PID>; sleep 1; pgrep -lx worker; echo pgrep_exit=\$?; tail -3 \"\$HOME/Documents/DLYK stuff/live_logs/<log>\""`
11. Rollback, only if needed: `ssh $M "cd $D && cp -p worker.pre_kv worker"`.
    That restores the binary only, so the Mac's `worker.cpp`, `hydra.proto`
    and `hydra.pb.*` would no longer match it. Full source parity needs scp of
    the HEAD `worker.cpp` and `hydra.proto`, then `make proto`. Each step would
    be listed and approved at the time.
12. After all gates pass: `ssh $M "cd $D && rm worker.pre_kv"`. This is a
    deletion and needs a separate yes.

## 6. Timing note (for the later benchmarking milestone; not built now)
- **Worker:** `std::chrono::steady_clock` around the compute only (after
  parse, before serialize/send). Log `compute_us=…` per packet; optionally a
  new reply field in a later schema change.
- **Client:** time per packet, send-to-receive. Network plus serialization ≈
  client round trip − worker `compute_us`.
- Kept out of this milestone so gates A-F stay unchanged.

## Open decisions
Resolved 2026-11-01 (owner):
- **Renames:** done. `data/live/g5`, `data/live/g3v2` and
  `data/live/regress_{attention,layernorm,mlp,qkv}` →
  `<name>_pre_kv_20261101`.
- **Mac commands approved:** scp of `hydra.proto`, `make proto`, the
  `worker_kv_block_<RUN>.log` launch line, the `worker.pre_kv` backup and
  rollback.
- **Gate C is hard** (section 4). The cache-on head runs on the newest row
  only, with a logit bound (section 0).
- **Run-1 baseline** (`data/live/g5_run1/`, WSL `sha256sum`, recorded before
  any KV work; gitignored, so pinned here):
  - `tokens.txt` `b8d8ed19d875156cf4e43126a98c419e0fe48390c7b0d3ca9f5a9c006ab00ddf`
  - `logits_f64.bin` `a16e5e833c87430262f1bcb6c80d1a54cebc11d41975f8b8ed4083bdbb5cdf17`
    (= committed frozen pin)
  - `hidden_f32.bin` `9fb4edec449fb9ecc88f3e79f6d8005102f4372a068e36ba6278439c4cea065d`
    (= committed mirror oracle)
- `kv_overflow.py` checks each accepted reply's data length (T·768).
