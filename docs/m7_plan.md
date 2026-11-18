# M7 plan — two-worker pipeline over a star topology (DRAFT, 2026-11-18)

Status: planning only. Nothing here is committed. The only runs so far are against a local
WSL2 worker (no Mac). Evidence paths are under `data/live/` (gitignored).

## 0. Linux worker parity (done, local runs only)

**Build.** `Makefile` gains a WSL2-only target `make worker_linux` (binary `worker_linux`, not
part of `make`/`all`; `make clean` also removes it and `orchestrator_local`; both binaries are
gitignored). The Mac's own Makefile is untouched.
- Toolchain: `g++ (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0`, `ldd (Ubuntu GLIBC 2.39-0ubuntu8.12) 2.39`,
  `libprotoc 3.21.12`, WSL2 kernel `6.6.87.2-microsoft-standard-WSL2`, Ubuntu 24.04.4.
- Command: `g++ -O2 -std=c++17 -Wall -Wextra worker.cpp hydra.pb.cc -lprotobuf -o worker_linux`.
  Zero warnings. Same flags as the WSL2 orchestrator. No `-march`/`-mfma`/`-ffast-math`: baseline
  x86-64 has no FMA instructions, so neither compiler can contract `a*b+c`.
- `worker.cpp` sha256 `cfa075e4…5f797d` (= HEAD). [Unverified today] that it equals the Mac's
  copy: the Mac's `shasum` gave the same hash on 2026-11-11 (session output only, not saved to any
  file), and the Mac was not contacted in this step. Step 3 re-checks it first with approved #2
  and saves the output (section 5).
  `worker_linux` sha256 `a89a9991…725fa`.
- [Unverified] the Mac Makefile's CXXFLAGS (not read: no Mac access in this step). The Mac and
  Linux outputs agree bit for bit anyway (below), so any flag difference has no visible effect.

**Checkers against localhost.** The locked checkers honor no host override: `orchestrator.cpp:255`
(compiled-in `10.0.0.62`), `g3_check_v2.sh` `WORKER_HOST`, `gen_hydra.py`'s `--host` default (the
checkers never pass `--host`) and `kv_overflow.py:36` are all hardcoded, and each refuses to run
if its staging dir exists. Method, with no original edited: `scripts/m7_local_checkers.sh CHECKER
[ARGS]` (draft). Every invocation regenerates the copies from the locked originals as they are on
disk at that moment, into a fresh `data/live/linux_checkers_<UTC>/`, by 18 exact substitutions
(each must change exactly the stated number of lines, else it stops before running anything);
rebuilds `orchestrator_local` from a copy of `orchestrator.cpp` with only the IP literal changed;
prints the full `diff -u` (36 changed lines: host, orchestrator binary, run-tagged staging dirs
`data/live/linux_<UTC>_*`, helper-script paths, and `kv_overflow.py`'s repo-root `sys.path` line,
which assumed the file lives in `scripts/`); then runs the requested copy. All of it (hashes,
diff, checker output, `== checker exit N`) is also saved to `$OUT/run.log`, and the runner exits
with the checker's status. Verified: `data/live/linux_checkers_20261118T193103Z/run.log`
(`regress_mode qkv`: 36 changed lines, `REGRESS_QKV_PASS`, `== checker exit 0`, runner exit 0).

**Results** (step 0 used the first, one-shot version of the generator, whose copies differed in the
same 18 lines except that the staging dirs were `data/live/linux_*` without a run tag; driver:
launch `worker_linux --mode M` with `nohup`, gate on `listening`, tee the checker, stop with
SIGINT, `pgrep -x worker_linux` empty after every run):

| check | result | log |
|---|---|---|
| `g3_check_v2.sh` (block) | `G3V2_CHECK_PASS`: 12/12 IDENTICAL to the mirror, all within the 1.6x bound; h00 = June; 12/-1 rejected; oversize prefix survived | `data/live/g3_check_v2_linux_20261118.log` |
| `regress_mode.sh layernorm` | `REGRESS_LAYERNORM_PASS` | `data/live/regress_layernorm_linux_20261118.log` |
| `regress_mode.sh qkv` | `REGRESS_QKV_PASS` | `data/live/regress_qkv_linux_20261118.log` |
| `regress_mode.sh attention` | `REGRESS_ATTENTION_PASS` | `data/live/regress_attention_linux_20261118.log` |
| `regress_mode.sh mlp` | `REGRESS_MLP_PASS` | `data/live/regress_mlp_linux_20261118.log` |
| `kv_check.sh` (block, fresh) | `KV_CHECK_PASS` (line 71): stage A runs `g5_check.sh` (G5_CHECK_PASS, logits byte-identical to frozen, outputs = `g5_run1`); B, C (64/64 rows, final-step logits on-off 0.000e+00), D, E (all overflow/protocol cases) pass. Max logit ratio 1.613e-02 (B), 2.484e-02 (C), the same as the Mac's | `data/live/kv_check_linux_20261118.log` |

**Answer:** the Linux build is bit-exact to the mirror for all 12 blocks (and to the June mode
oracles). No op differs: glibc 2.39's `exp`/`tanh`/`sqrt` and macOS libm give the same float32
results after each double-to-float stage cast on every tested input. [Likely] this holds because
`sqrt` is correctly rounded on both, and any last-ulp difference in a double `exp`/`tanh` is far
below the float32 cast that follows; it is an observation on these inputs, not a guarantee for all
inputs, which is why gates A/B re-check byte identity end to end.

## 1. Design

**Split: WSL2 worker layers 0-5, Mac worker layers 6-11** (contiguous, 6/6).
- Star topology (Option A): the client sends every layer and receives every reply, so a token
  costs 12 sequential round trips whatever the split. Per-token time = sum of block compute +
  6 localhost round trips + 6 Wi-Fi round trips + client work.
- Measured compute, identical work on both machines (G5 prompt, cache off, 20 steps, T=11..30,
  240 packets, worker `compute_us`): WSL2 worker 19.049 s total, Mac worker 18.862 s, ratio
  Mac/WSL2 0.99x. Per step, layers 0-5 cost 482.5 ms (WSL2) / 472.1 ms (Mac) and layers 6-11
  469.9 / 471.0 ms. Sources: WSL2 `data/live/linux_worker_logs/worker_linux_block_20261118T184814Z.log`
  (kv_check stage A, device `gen-hydra-py`, seq 0, first 240 TIMING lines); Mac
  `bench/raw/20261111_n12r6/worker_log.txt` (benchmark `anchor_off`). The (layer, T, seq,
  start_pos) sequences were asserted identical. Caveat: one run per machine; the WSL2 run was the
  first after its worker launched, the Mac run followed the benchmark's warmup.
- So the machines and the layers are about equally fast: a 6/6 split balances compute. Only the
  network is asymmetric (localhost vs Wi-Fi), and it favors putting more layers on WSL2.
- Why 6/6 and not the faster all-on-WSL2: M7's point is a real two-machine pipeline; an even split
  gives each worker half the weights and half the K/V traffic, and makes the per-worker range
  gates symmetric. The split is a config value; an uneven split (e.g. 0-8 local) is a
  measurement for later, not a design change. Order of the halves does not matter in a star.

**Client** (`gen_hydra.py`, proposed changes; defaults keep today's behavior and wire bytes):
- `--route "0-5=127.0.0.1:50051/worker-wsl,6-11=10.0.0.62:50051/worker-mac"`: layer -> (host,
  port, expected reply `device_id`). Must cover 0..11 exactly once, contiguous per worker, else a
  startup error. Absent: all 12 layers on `--host:--port`, device `worker-mac` (today).
- `--client-id ID` (default `gen-hydra-py`, the current constant), so each run's packets can be
  told apart in the worker logs. The checker uses unique ids per run.
- `--connect-timeout S` (default 3 s) on connect; the existing `--timeout` (10 s) on every recv.
- Markers, unbuffered (`print(..., flush=True)`; the checker also runs Python with `-u`):
  `READY` after imports and the model load, before connecting; `CONNECTED <endpoint> ...` once
  every worker connection is open, before the first send. Failure time limits count from these
  markers, never from process start (torch/transformers import time is excluded).
- Connection reuse: one TCP connection per worker for the whole run, all opened before step 0
  (fail fast if a worker is down), `TCP_NODELAY` on each. Per step, layers go over their owner's
  connection. Today each step opens a new connection; over Wi-Fi that costs a handshake per token.
  On exit (success or error) the client closes every connection; each worker sees EOF and returns
  to `accept()`.
- Reply validation adds: `device_id` = the layer owner's expected id; no unknown protobuf fields
  (`google.protobuf.unknown_fields.UnknownFieldSet(reply)` must be empty; verified on protobuf
  6.33.4: 0 for a clean reply, 1 when an unknown field 7 is appended).
- Errors: `ERROR: worker HOST:PORT (layers A-B, DEVICE): REASON` on stdout/stderr, nonzero exit,
  no output files written (`tokens.txt`, `logits_f64.bin`, `hidden_f32.bin`, `sends.txt`,
  `route.txt`; today's client already writes only after all steps succeed).
- New `DIR/route.txt` (only with `--route`), one row per packet:
  `step layer endpoint reply_device_id T req_floats reply_floats req_bytes reply_bytes`.
  `sends.txt` keeps its format, so `kv_compare.py` works unchanged.

**Worker** (`worker.cpp`, proposed; defaults keep single-worker behavior and wire bytes):
- `--layers A-B` (default `0-11`): the block-mode range check at `worker.cpp:867`
  (`L < 0 || L >= N_BLOCKS`) becomes `L < lo || L > hi`; out-of-range packets get the existing
  error reply. Startup log line (block mode): `Worker layers A-B device_id NAME`.
- `--device-id NAME` (default `worker-mac`) replaces the literal at `worker.cpp:615/802/824`.
- One log line per request, after its reply (or error reply) is sent:
  `WIRE from=CLIENT_ID peer=IP:PORT layer=L T=T rx_bytes=N tx_bytes=M replies=K`, where
  rx = 4-byte prefix + request payload, tx = sum over reply packets of prefix + payload, and the
  peer comes from `accept()`. Wire bytes and peers are then proven from the workers, not from the
  client's own report. The startup line and every WIRE line end with `std::endl` (flushed): the
  checker polls the live logs (D3, D4, D6), so a buffered line would fail a gate spuriously.
- Weights for all 12 blocks still load (28 MB each); K/V is only ever written for layers that
  pass the range check, so a worker holds no K/V for other layers. The listen port stays
  hardcoded (`worker.cpp:528`, 50051): one worker per machine.
- `worker.cpp` creates exactly one socket, the listener (`worker.cpp:519`,
  `socket(AF_INET, SOCK_STREAM, 0)`), and has no `connect(`, `sendto(`, `sendmsg(` or
  `SOCK_DGRAM`. The checker re-asserts all of this on the committed source, so a worker can only
  answer connections it accepted; it cannot open one to another worker or send datagrams.

**Timeouts and one worker down.**
- Refused (nothing listening): `ConnectionRefusedError` at startup, at once.
- Unroutable/offline host: connect timeout, within `--connect-timeout`.
- Stalled (alive, not answering): recv timeout, within `--timeout`.
- Died mid-run or between steps: EOF/reset on the next send or recv, at once (a SIGKILLed process's
  sockets are closed by its kernel, so a peer on a live host gets FIN/RST); if the host itself
  vanished, the recv timeout still bounds it.
In every case the other worker sees its connection close and keeps serving. With the cache on, its
K/V for the dead run's sequence stays stale until the next run's new `sequence_id` resets it (the
validated path from KV stage D).
- Caveat: SIGINT does not stop a worker that is mid-connection (the per-packet loop does not check
  the shutdown flag; CLAUDE.md Known Weaknesses), so stop a worker only between runs; fault
  injection uses SIGSTOP/SIGCONT/SIGKILL.

**Network and build facts measured for the design** (local runs, 2026-11-18; saved under
`data/live/m7_evidence_20261118/` unless noted):
- Windows Python reaches the WSL2 worker on `127.0.0.1:50051` (no `.wslconfig`, NAT mode, default
  localhost forwarding): `('127.0.0.1', 50085) -> ('127.0.0.1', 50051)`, 64.6 ms first connect,
  `data/live/winreach_linux_20261118.log`; `kv_check` then ran its whole Windows-Python client
  over this path.
- The WSL2 worker sees that client as peer `127.0.0.1`: `peer_probe.txt` (Windows client socket
  `('127.0.0.1', 60036) -> ('127.0.0.1', 50051)`; WSL2 `ss`: `127.0.0.1:50051 127.0.0.1:56938`).
  Windows' source address toward the Mac is `10.0.0.113` (`peer_probe.txt`; also
  `bench/raw/20261111_n12r6/net_route_windows_20261111.txt:13`). Hence the peer gate: every WSL2
  peer must be 127.0.0.1, every Mac peer the client's source address (computed at check time).
  [Likely, not measured] the Mac also sees the checker's probes, which `orchestrator` sends from
  inside WSL2, as `10.0.0.113` (WSL2 NAT translates to the Windows host address). If that is wrong
  the peer gate fails spuriously; it cannot pass wrongly.
- `timeout` from WSL2 terminates a Windows `python.exe`: `timeout_probe.txt` (`timeout
  --kill-after=5 3` -> rc 124 after 3.009 s; `tasklist` no longer shows the Windows PID 55280).
- Builds are reproducible: `build_repro.txt` (two builds of `worker.cpp` and the existing
  `worker_linux` all hash `a89a9991…725fa`), so the checker can pin a running worker to the
  committed source by rebuilding it and comparing with `/proc/<PID>/exe`.
- Exact wire size: `4 + TensorPacket(layer_index, device_id, data = T*768 floats, shape = [T,768],
  sequence_id/start_pos when set).ByteSize()`, with the request's `device_id` = that run's client
  id (`run.json` `device_id`) and the reply's = `worker-mac`, equals the recorded
  `req_bytes`/`reply_bytes` of all 8280 packets in `bench/raw/20261111_n12r6/runs/*/packets.csv`
  (52 runs, cache on/off, T = 1..1024): 0 mismatches (re-run independently by the reviewer).

## 2. Protocol

No proto change. `layer_index` already routes; `sequence_id`/`start_pos` already drive a per-layer
cache, and each worker sees the same values for its own layers; `device_id` (string, field 3)
already carries the worker identity. K/V never needs to cross the wire: in a star each layer's K/V
lives with that layer's owner, and the only cross-machine data is the [T,768] activation between
layers, which already goes through the client. New log lines and client flags are not wire changes.

## 3. Gates (`scripts/m7_check.sh`, draft; reviewed and committed before the first distributed run)

Bounded: the checker re-runs itself under `timeout --foreground` (1800 s overall; `--foreground`
keeps it in the terminal's process group so Ctrl-C still stops it) and every client run under
`timeout` (300 s), so a real hang ends in FAIL, never in a stuck checker. With `--foreground`, an
overall timeout stops only the checker's own shell: a client still running is ended by its own
300 s `timeout`, and a worker the checker relaunched (D4) keeps running until the section 5
cleanup stops it. Client ids are unique
per checker run (`m7-<UTC>-<stage>`); the staging dir is `data/live/m7_<UTC>`. Inputs that reach a
remote shell (`MAC_LOG_NAME`, `MAC_PID`) are validated before any remote command.
Precheck pins everything that runs: the committed sources (`git diff --quiet HEAD`), the running
WSL2 worker (`/proc/<PID>/exe` byte-identical to a fresh build of the committed `worker.cpp`, also
after the D4 relaunch), the probe binaries (built by the checker from the committed
`orchestrator.cpp`; the local one differs in exactly the IP line), the Mac source (#2:
`worker.cpp` and `hydra.proto` hashes = `HEAD`'s; the binary hash is recorded in
`$STAGE/mac_sha256.txt` and is proven only by gate E, which ran that same binary), exactly one Mac
worker with the given PID (#6), and the static socket rules for `worker.cpp`. A failed ssh
is a FAIL, never a pass.
- A: distributed, cache off: 20 tokens = G1 (`g5_compare.py`), hidden byte-identical to
  `data/g5_mirror_hidden.bin` (`g5_compare_hidden.py`), logits byte-identical to the frozen
  `data/g5_hydra_logits.bin`. Allowed because step 0 shows parity.
- B: distributed, cache on: `kv_compare.py b` unchanged (gate-B equivalent).
- C, ranges: each worker's startup line names its configured range and device id (read from that
  worker's own log); WSL2 rejects layer 6 and serves 0 and 5; the Mac rejects 5 and serves 6 and
  11 (served blocks byte-identical to the G3 mirror).
- C, routing: in runs A and B every packet went to its layer's owner and was answered by that
  owner's `device_id` (route.txt, cross-checked row by row with sends.txt); 20 packets per layer
  per run; every request and reply carries T*768 floats.
- C, wire (from the workers' logs): for every packet of runs A and B, the worker's `rx_bytes` and
  `tx_bytes` equal the exact activation-only size for that layer, T, client id, device id,
  sequence_id and start_pos, with exactly one reply; the worker saw exactly the layers it owns,
  the same (layer, T) sequence the client sent; every Received line's shape is exactly [T,768]
  with T*768 floats. Every peer of every request (all client ids, probes included) is 127.0.0.1
  on WSL2 and the client's source address on the Mac.
- D: the client exits nonzero (not via `timeout`), names the failed owner in its `ERROR: worker`
  line, writes none of its five outputs, and the other worker then serves a byte-identical block:
  - D1 WSL2 owner refused (127.0.0.1:50052; first checked to refuse from Windows) and
    D2 unroutable (TEST-NET-1 192.0.2.1): within 3 + 3 s of READY.
  - D3 WSL2 worker SIGSTOPped after its first packet of the run: within 10 + 3 s of the signal;
    SIGCONT after.
  - D5 Mac worker SIGSTOPped after the client finishes step 0: within 10 + 3 s; SIGCONT after.
  - D4 WSL2 worker SIGKILLed after its first packet: within 10 + 3 s; then relaunched from the
    checker's own build of the committed `worker.cpp` (`$STAGE/bin/worker_linux`, same flags),
    behind the same gate and pin check.
  - D6 Mac worker SIGKILLed between steps: when the WSL2 worker has served step 2's layer 0 (its
    13th packet of the run), the checker SIGSTOPs the WSL2 worker, so the client waits on WSL2
    while the Mac is idle with steps 0-1 served and nothing in flight; then #14 kills the Mac and
    the WSL2 worker gets SIGCONT. The client must fail within 10 + 3 s of the SIGCONT, naming the
    Mac. "Between steps" is proven twice: at the pause, WSL2 has served 13-17 packets of that run
    (step 2's layer 0 but not its layer 5, so the client is still waiting on WSL2 and has sent the
    Mac nothing for step 2); and the Mac's own log shows exactly 12 Received and 12 WIRE lines (2
    whole steps) for that run. Otherwise FAIL. The pause is one ssh round trip, far below the 10 s
    recv timeout.
  - Order D1, D2, D3, D5, D4, relaunch, D6, so each kill leaves the other worker up to check.
- E: single-worker mode (all 12 on the Mac, no `--layers`): `scripts/g5_check.sh` and
  `scripts/kv_check.sh` unchanged, plus `g3_check_v2.sh` and `regress_mode.sh` x4 for the rebuilt
  worker binary (section 5, step 3).

## 4. Review questions: would the gates catch it?

- (a) A layer routed to the wrong worker: yes. C/wire: the receiving worker's own log shows a
  layer outside its range for that client id (exact layer set required), even if both workers
  hold all weights and the output is still correct; this does not depend on the client's
  route.txt or on the reply `device_id`. C/routing additionally catches it from the client side,
  and the client itself rejects a reply whose `device_id` is not the owner's.
- (b) A worker serving a layer outside its range: yes. C/ranges probes both sides of each boundary
  (WSL2: 6 rejected, 0 and 5 served; Mac: 5 rejected, 6 and 11 served), so an off-by-one in
  `--layers` in either direction fails; the startup line must name the configured range; C/wire
  requires each worker's served layers to be exactly its range.
- (c) K/V leaking onto the network: client<->worker, yes: each worker's own rx/tx byte counts
  must equal the exact activation-only size, one reply per request (an extra field, extra reply
  packet or larger payload changes the count), and the client rejects unknown fields.
  Worker<->worker: the guarantee is static, on the committed and pinned source: one socket, the
  `SOCK_STREAM` listener; no `connect(`, `sendto(`, `sendmsg(` or `SOCK_DGRAM`, so a worker can only
  answer connections it accepted. The peer check is a second, weaker layer: it would catch a
  connection from the Mac into WSL2 (peer not 127.0.0.1), but a connection from WSL2 to the Mac
  would arrive from the same NATed 10.0.0.113 as the client, so it cannot tell those apart.
  Residual: traffic from a process other than the worker is out of scope.
- (d) A hang when a worker dies or stalls: yes. D3/D5 stall each worker mid-run, D4 kills WSL2
  mid-run, D6 kills the Mac between steps, D1/D2 cover a worker that is down before the run; each
  must end within its limit, measured from READY or the fault, not process start. The checker
  itself cannot hang: overall and per-client `timeout`, verified to terminate a Windows
  `python.exe` from WSL2.

## 5. Commands

`M=yonghowlee@10.0.0.62`, `D='"$HOME/Documents/DLYK stuff"'`. Every Mac command runs as
`ssh -o BatchMode=yes $M "<command>" < /dev/null` with exactly the command text below.

**Approved Mac command list for M7** (numbered 1-15 for this plan only; it does not follow the
numbering of `docs/kv_cache_plan.md` §5. NEW = added by the owner's 2026-11-18 decisions):
1. `scp -q worker.cpp "yonghowlee@10.0.0.62:Documents/DLYK stuff/"` (worker.cpp only; never `.pb.*`).
2. `cd $D && shasum -a 256 worker.cpp hydra.proto worker`
3. `cd $D && cp -p worker worker.pre_m7`
4. Build, in the background with a log (a Mac build has taken over 10 min):
   `cd $D && (nohup sh -c 'make clean && make; echo make_exit=\$?' > live_logs/make_m7_<RUN>.log 2>&1 < /dev/null &)`;
   progress and result only through #11 on `make_m7_<RUN>.log` (done when its last line is `make_exit=N`).
5. `cd $D && shasum -a 256 worker`
6. `pgrep -lx worker; lsof -nP -iTCP:50051 -sTCP:LISTEN`
7. `cd $D && (nohup ./worker --mode <mode> > live_logs/<log> 2>&1 < /dev/null & echo \$!)`, `<mode>` in
   block|layernorm|qkv|attention|mlp, `<log>` = `worker_kv_block_<RUN>.log` or `worker_g3_<mode>_<RUN>.log`.
8. NEW: `cd $D && (nohup ./worker --mode block --layers 6-11 --device-id worker-mac > live_logs/worker_m7_block_<RUN>.log 2>&1 < /dev/null & echo \$!)`
9. `cd $D; for i in \$(seq 1 40); do grep -q listening live_logs/<log> && break; sleep 0.5; done; cat live_logs/<log>`
10. `kill -INT <PID>; sleep 1; pgrep -lx worker; echo pgrep_exit=\$?; tail -3 \"\$HOME/Documents/DLYK stuff/live_logs/<log>\"`
11. `cd $D && cat live_logs/<log>` (`<log>` must match `^[A-Za-z0-9_.-]+\.log$`).
12. NEW: `kill -STOP <PID>`
13. NEW: `kill -CONT <PID>`
14. NEW: `kill -KILL <PID>`
    (#10 and #12-#14: `<PID>` is ONLY the PID printed by that run's launch command (#7 or #8) and
    confirmed by its #9 gate; never a name, pattern or other PID.)
15. Rollback, only if needed: `cd $D && cp -p worker.pre_m7 worker` (restores the binary saved by #3).
Removing `worker.pre_m7` afterwards is a deletion and needs a separate yes.

**Step 1: code changes (separate approval, diff per file).** `worker.cpp` and `gen_hydra.py` as in
section 1.

**Step 2: WSL2, rebuild and re-prove parity with default flags (local only).**
```
make worker_linux
# per check: launch worker_linux --mode <mode> (nohup, gate on "listening"), then
bash scripts/m7_local_checkers.sh g3_check_v2              # block
bash scripts/m7_local_checkers.sh regress_mode <mode>      # layernorm|qkv|attention|mlp
bash scripts/m7_local_checkers.sh kv_check                 # fresh block worker
# stop: kill -INT <PID>; sleep 1; pgrep -a -x worker_linux; echo pgrep_exit=$?
```

**Step 3: Mac, parity first, then single-worker gate E (approved commands only).**
By step 3 the WSL2 `worker.cpp` has the step-1 changes, so the pre-M7 reference is the committed
file at `8c6c38a` (the HEAD this plan was written on; sha256 `cfa075e4…5f797d`), not the working
copy. Runs as a subshell so an existing file does not close the terminal:
```
( F=data/live/m7_mac_parity_$(date -u +%Y%m%dT%H%M%SZ).txt
  if [ -e "$F" ]; then echo "ERROR: $F exists"; exit 1; fi
  { date -u +%FT%TZ
    echo "== Mac (#2)"; ssh -o BatchMode=yes $M "cd $D && shasum -a 256 worker.cpp hydra.proto worker" < /dev/null
    echo "== pre-M7 reference"; git show 8c6c38a:worker.cpp | sha256sum | sed 's/-$/8c6c38a:worker.cpp/'
    git show 8c6c38a:hydra.proto | sha256sum | sed 's/-$/8c6c38a:hydra.proto/'; } | tee "$F" )
```
The Mac's `worker.cpp` and `hydra.proto` must equal the pre-M7 reference before anything else runs.
After #1 (scp), #2 must show the Mac's `worker.cpp` = the new committed `worker.cpp`.
Then #1, #2, #3, #4 (wait for `make_exit=0` via #11), #5, and for each check #6 (must be empty),
#7, #9, the unchanged checker, #10:
`bash scripts/kv_check.sh` (fresh block worker; its stage A is `scripts/g5_check.sh`),
`bash scripts/g3_check_v2.sh` (block), `bash scripts/regress_mode.sh <mode>` x4. The existing staging
dirs `data/live/{kv,g5,g3v2,regress_*}` must be renamed first (needs a yes).

**Step 4:** hand-review and commit `scripts/m7_check.sh`, the client and the worker changes.

**Step 5: distributed run.**
WSL2, in a WSL terminal kept open (the worker must outlive the shell that starts it):
```
cd /mnt/c/Users/ASUS/Documents/hydra
pgrep -a -x worker_linux; ss -ltnH 'sport = :50051'          # both empty
LOG=data/live/linux_worker_logs/worker_wsl_m7_$(date -u +%Y%m%dT%H%M%SZ).log
nohup ./worker_linux --mode block --layers 0-5 --device-id worker-wsl > "$LOG" 2>&1 < /dev/null & WPID=$!
for i in $(seq 1 40); do grep -q listening "$LOG" && break; sleep 0.5; done; cat "$LOG"; echo "WPID=$WPID"
```
Mac: #6 (empty), #8 (prints MPID), #9 with `worker_m7_block_<RUN>.log`.
Checker (uses only #2, #6, #11, #12-#14 on the Mac):
```
bash scripts/m7_check.sh "$WPID" "$LOG" "$MPID" worker_m7_block_<RUN>.log
```
Afterwards: the checker stops the relaunched WSL2 worker with SIGINT, and D6 killed the Mac worker;
confirm with `pgrep -a -x worker_linux` (empty) and #6 (empty). If the checker stops early (FAIL,
timeout or Ctrl-C), a worker may be left SIGSTOPped (D3, D5, D6), and a stopped process does not
act on SIGINT until continued, so: WSL2 `kill -CONT <PID>; kill -INT <PID>; sleep 1; pgrep -a -x
worker_linux`; Mac #13 (CONT) then #10, each with that worker's own PID.
