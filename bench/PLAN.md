# KV benchmark + profiling plan (approved 2026-11-01: N=12, 6 reps, interleaved, 30 s cooldown,
# warmup pair, anchor pair; matrix run detached and resumable)

Files in `bench/`:
- `worker_timing.diff` (applied in the timing commit), `make_prompts.py`, `bench_run.py`,
  `bench_matrix.sh`, `summarize.py`, `profile.sh`;
- tests: `test_summarize.py`, `test_matrix_resume.sh`;
- probe evidence: `bench/probe/`, hashed in `bench/probe/SHA256SUMS`.

Python everywhere: `/mnt/c/Users/ASUS/AppData/Local/Programs/Python/Python313/python.exe`
(WSL2) / `C:\Users\ASUS\AppData\Local\Programs\Python\Python313\python.exe` (Windows).

Rule for this file: every number is quoted from a file in `bench/probe/` (named in place), is a
plan parameter, or is the printed output of a named script.

## 1. Worker timing (`bench/worker_timing.diff`, a real unified diff)
- `git apply --check` is clean on b7d24cb. The patched `worker.cpp` passes
  `g++ -std=c++17 -O2 -Wall -Wextra -fsyntax-only` with zero warnings.
- Adds `#include <chrono>` and `t_parsed = steady_clock::now()` right after a successful
  `ParseFromString`. `compute_us` is taken right before the generic `sendPacket`, so the window
  covers dispatch, compute, response build, serialize and the existing per-packet cout lines.
- After the send it prints one line:
  `TIMING dev=<device_id> layer=<L> T=<shape0> seq=<sequence_id> start_pos=<p> compute_us=<n>`.
- Logging only. Not timed: error replies and `--mode mlp`'s two-packet path (neither is benchmarked).
- Gates rerun on the timing binary: `kv_check.sh` A-E, `g3_check_v2.sh`, and
  `regress_mode.sh layernorm|qkv|attention|mlp`. Staging dirs are renamed to
  `<name>_pre_timing_20261101` first.

## 2. Cost probe (ran once, 2026-11-02T01:57Z)
Sources:
- client `bench/probe/cost_probe_20261102T0157Z.log`, from `bench/probe/cost_probe.py`;
- worker log copy `bench/probe/worker_kv_probe_20261102T0157Z.mac_copy.log`, via the approved `cat`.

The worker log shows "Worker listening", then exactly 24 `Received` lines (12 `shape [960,768]
seq=0 start_pos=0` and 12 `shape [960,768] seq=2635022799589855401 start_pos=0`), then
`Shutdown signal received`.

| run | packets | bytes per packet, req / reply (incl. 4 B prefix) | wall per packet | total wall |
|---|---|---|---|---|
| uncached step, T=960 | 12 | 2,949,149 / 2,949,149 (layer 0: 2,949,147 / 2,949,147) | 5.428-5.892 s | **68.088 s** |
| cached prefill, T=960 | 12 | 2,949,159 / 2,949,149 (layer 0: 2,949,157 / 2,949,147) | 4.652-5.022 s | **58.610 s** |

RTT, from `bench/probe/ping_20261102T0209Z.txt` (20 pings, about 12 min after the probe):
`rtt min/avg/max/mdev = 13.380/72.022/170.394/34.146 ms`.

### Runtime model
`bench/probe/estimate.py` → `estimate_out.txt`. [Likely]-grade: one probe point.
Model terms:
- full attention at T=960 = 2 × (U − C) = 1.58 s/packet;
- pair cost 1.71 µs; row cost 4.26 ms per block;
- fixed cost per packet = ping average, 72.022 ms;
- client startup = 15 s per run [Guessing];
- 30 s cooldown per run; anchor and warmup pairs included.

```
N=12 longest L=1012: rep=34.0 min; total 5 reps=2.93 h, 6 reps=3.49 h
cached decode per token ~ 926 ms (12 x RTT = 864 ms); uncached step T=512 ~ 32.5 s, T=1008 ~ 73.4 s, T=1024 ~ 74.8 s
```
[Likely] cached decode is network-bound on this link: about 864 of 926 ms per token is RTT.

**First-pair check** (the goal's stop rule): `bench/probe/first_pair_estimate.py 12 32` →
`first_pair_estimate_out.txt`:
`model: L=32 N=12 off=36.91 s on=13.51 s pair=50.42 s (step walls only, no process startup)`.
Called with the two run dirs, it prints the measured value and the ratio. Outside 0.5-1.5 means
DECISION NEEDED. The rep 1 stop rule remains: over 1.5× its estimate of 34.0 min means stop and re-plan.

## 3. Matrix (`bench/bench_matrix.sh BATCH 12 6`), detached and resumable
- Cells: prompt lengths 32, 128, 512 and 1012, cache on and off, 6 runs each: 48 timed runs.
- The schedule is 52 rows, also printed by `--dry-run`:
  - **anchor pair**: `prompt_g5.txt`, 19 decode steps, on and off. Both must reproduce
    `data/gen_ref_tokens.txt`, or the matrix stops. The gate is checked before the first
    non-anchor row in every invocation, including a resume where the anchors are skipped as done.
  - **warmup pair**: L=32, discarded.
  - Anchor and warmup are rep 0 and are excluded from the results.
- Order: each rep rotates the length order; each length runs its on/off pair back to back; the
  first side is `(rep + length index) % 2`.
  - Measured on `--dry-run` at N=12, 6 reps: every length starts on-first 3 times and off-first
    3 times. The mean position among the 48 timed runs is 24.5 for on and 24.5 for off
    (equivalently 28.5 / 28.5 in schedule.csv's `order` column, which also counts the 4 rep-0 rows).
  - The same 30 s cooldown follows every run.
  - One worker process serves the whole matrix after the launch gate.
- Detached: `tmux new-session -d -s hydra_bench 'bash bench/bench_matrix.sh BATCH 12 6 >> bench/raw/BATCH.console 2>&1'`.
  A detached tmux session survived with no WSL client attached. Source:
  `bench/probe/tmux_survival_heartbeat_20261102.txt`, 18 UTC times of day from 02:27:22 to
  02:28:57, then `END`. The interval is nominally 5 s (`sleep 5`); the measured gaps are 14 × 5 s,
  2 × 8 s and 1 × 9 s, so there were short scheduling stalls but no lost beats. The file carries
  times only; the date comes from the filename and the session. Only a Git Bash reader watched the file, and no `wsl.exe` was running.
- Resumable: rerunning the same command skips every run whose `done` line is in `matrix.log`.
  - A started-but-unfinished run has its partial dir and log renamed `*.aborted_<UTC>`, never
    deleted, and is rerun.
  - Every attempt gets a unique worker `device_id`, so lines from aborted attempts never mix
    with the retry's.
  - A different schedule for an existing batch is refused, as is a completed batch, and a second
    instance (`flock`).
  - Any run failure stops the matrix; rerunning resumes it.
- Stdin safety: every Windows `python.exe`/`powershell.exe` and every `ssh` call gets
  `< /dev/null`, and the schedule is read on fd 3. Under WSL interop a Windows process otherwise
  consumes the loop's stdin.
- On completion the matrix writes `SHA256SUMS` over all raw files, then `matrix.log.sha256` over
  the final `matrix.log`.
- Specs (`specs_{wsl,windows,mac}_<UTC>.txt`) and every ping are checked: a failure stops the
  matrix. A resume re-collects specs if no earlier attempt recorded them. The first ping after a
  resume is labelled `rep<r>_resume`, since it may fall mid-rep.
- Batch name for the real run: `20261102_n12r6`. `bench/raw/B` holds only a leftover client log
  from an aborted `profile.sh` invocation that failed before any network contact.
- `bench/test_matrix_resume.sh` (stubbed client, ssh and ping; throwaway git repo) checks that:
  - the first pass stops on an injected failure;
  - a concurrent instance is refused;
  - the resume completes 52/52;
  - exactly the 6 runs done before the failure are skipped;
  - the partial attempt is renamed;
  - the done order equals `schedule.csv`;
  - `matrix.log.sha256` is correct;
  - a completed batch refuses a rerun;
  - `--dry-run` prints 52 rows;
  - bad anchors stop both the first pass and the resume, before any warmup or timed run;
  - specs are recorded once, with a timestamp;
  - the first ping after a resume is labelled `rep1_resume`.

  Result: 12 PASS lines (one per check above), `MATRIX_RESUME_TEST PASS`.

### Metrics (`summarize.py` → `bench/results.csv`, one row per run)
- Prefill time (step 0), split into worker and network+client.
- Per-run decode mean, on the total, worker and network+client series.
- Tokens/s, worker-only tokens/s, worker and network+client decode seconds, bytes.
- Worker = Σ `compute_us` of the step's 12 packets, from the worker log. Network+client =
  step wall − worker. That includes the connection setup per step, embedding, float64 head and
  argmax: everything not inside the worker.
- **p50/p95 are defined only over pooled per-cell tokens**: every decode step of every run in the
  cell, 6 × 12 = 72 samples, nearest rank. Each is computed separately for the total, worker and
  network+client series, since percentiles are not additive. `results.csv` has no per-run
  percentiles: with 12 samples, a per-run p95 would just be the maximum.
- `RESULTS.md` (generated): per-run values (median [min-max]), the pooled percentile table, and
  an order/drift table (decode mean when a run went first vs second in its pair).
- Per-run sanity, hard: tokens equal the on/off partner's. On mismatch, check the top-2 margin
  first. On/off logits differ by at most 1.080e-12, per `bench/probe/kv_check_20261102T0136Z.log`:
  `REPORT ... all steps max = 1.080e-12`.

## 4. Prompts (`bench/make_prompts.py 12`, committed before any timed run)
- `prompt_L{32,128,512,1012}.txt`: prefixes of one `default_rng(20261102)` stream of 1024 IDs.
- `prompt_g5.txt`: `gen_hydra.PROMPT` tokenized.
- `SHA256SUMS`.
- `bench_matrix.sh` refuses to start unless the prompts, `SHA256SUMS`, the bench scripts,
  `gen_hydra.py` and `data/gen_ref_tokens.txt` are committed and unmodified.

## 5. Specs and RTT
Recorded by `bench_matrix.sh` into `bench/raw/BATCH/specs_*.txt`, and once standalone for the goal.
- WSL2: `lscpu | grep -E 'Model name|^CPU\(s\)|Thread|Core|Socket'; free -b; uname -a; head -2 /etc/os-release`.
- Windows host: `Get-CimInstance Win32_Processor|Win32_ComputerSystem|Win32_OperatingSystem`.
- Mac (approved): `sysctl machdep.cpu.brand_string hw.ncpu hw.physicalcpu hw.memsize; sw_vers`.
- RTT: `ping -c 20 10.0.0.62`, full output with a UTC stamp, before and after every rep, into
  `ping_<tag>_<UTC>.txt`. Files are never overwritten across resumes.

## 6. Profiler (`bench/profile.sh`), after the matrix; never summarized
- Existence check (approved): `ssh $M "ls -l /usr/bin/sample; sample 2>&1 | head -15"`.
- `bash bench/profile.sh BATCH <PID> off512 off 2`, then `bash bench/profile.sh BATCH <PID> on512 on 24`.
  It refuses if any `bench/raw/BATCH/profile/<label>*` exists.
- The Mac folder is defined once (`MDIR='"$HOME/Documents/DLYK stuff"'`, expanded by the remote
  shell), so the write and the readback use the same folder. `--print-cmd` shows the exact
  remote commands:
  ```
  cd "$HOME/Documents/DLYK stuff" && sample <PID> 10 -file "$HOME/Documents/DLYK stuff"/live_logs/sample_<label>_<RUN>.log
  cd "$HOME/Documents/DLYK stuff" && cat live_logs/sample_<label>_<RUN>.log
  ```
- Top 5 by self time: `awk '/^Sort by top of stack/{f=1;next} f&&NF' <copy> | head -5`.
  - [Likely] uncached: `matmul_nt`/`attn_heads`.
  - [Likely] cached: blocked in `recv`/`accept`.
  - Caveat: `-O2` inlining may fold small functions into `main`.

## 7. Outputs and traceability (`summarize.py`; checks W0-W6)
- **W0:** every raw file matches `SHA256SUMS`, and `matrix.log` matches `matrix.log.sha256`.
  It stops before parsing anything.
- **W1:** the worker log copy equals the Mac's `shasum`.
- **W2:** each run's worker TIMING lines match `packets.csv` 1:1 in order, and the Received
  count equals the TIMING count for that device.
- **W3:** steps.csv and tokens.txt have exactly n_decode + 1 rows, and prompt identity holds: `run.json` prompt file, prompt sha256 and `n_decode` match the
  schedule and `bench/prompts/SHA256SUMS`, and `prompt_len` equals the file's ID count. Then the
  WORKER-side T/seq/start_pos pattern must fit the mode. An uncached run that silently used the
  cache fails here, on worker data.
- **W4:** on/off tokens identical (a missing partner is reported as such).
- **W5:** every scheduled run completed exactly once, and the order of the `done` lines equals
  the schedule. Aborted, renamed attempts are allowed.
- **W6:** the anchors' tokens equal `data/gen_ref_tokens.txt`.
- `--verify [--out-dir]`:
  - regenerates into a temp dir;
  - requires `results.csv` and the generated part of `RESULTS.md` to be byte-identical;
  - requires every signed number in the hand-written Findings to appear as a whole token in the
    generated tables or `results.csv`;
  - lists integers under 3 characters as only weakly checked.
- `bench/test_summarize.py` asserts that each negative case fails on the expected check
  (W0 ×2, W2 ×2, W3 ×2, W4, W5, W6, invented number, overwrite refusal), and that an aborted retry
  passes. Result: `TEST PASS`. Writing UTF-8 explicitly was a fix the test found: under cp1252
  the em dash broke `--verify`.
