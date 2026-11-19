#!/usr/bin/env bash
# M7 checker. DRAFT (2026-11-18): must be hand-reviewed and committed before the first distributed
# run. It targets the PROPOSED interfaces in docs/m7_plan.md section 1, which do not exist yet:
#   gen_hydra.py --route SPEC --client-id ID --connect-timeout S
#     prints "READY" (imports + model loaded, before connecting) and "CONNECTED ..." (every worker
#     connection open, before the first send), unbuffered; writes DIR/route.txt; rejects replies
#     with unknown protobuf fields; on any worker failure prints
#     "ERROR: worker HOST:PORT (layers A-B, DEVICE): REASON", exits nonzero, writes nothing.
#   worker --layers A-B --device-id NAME
#     logs "Worker layers A-B device_id NAME" at startup; rejects any layer outside A-B with the
#     existing error reply; logs one line per request after its reply is sent:
#     "WIRE from=CLIENT_ID peer=IP:PORT layer=L T=T rx_bytes=N tx_bytes=M replies=K"
#     (rx = 4-byte prefix + request payload; tx = sum over reply packets of prefix + payload).
# Run from repo root on WSL2, both workers freshly launched behind their launch gates
# (docs/m7_plan.md section 5):
#   bash scripts/m7_check.sh WSL_PID WSL_LOG MAC_PID MAC_LOG_NAME
# The whole run is bounded by OVERALL_S (the script re-runs itself under `timeout`), and every
# client run by CLIENT_MAX_S, so a real hang ends in FAIL, never in a stuck checker.
# Mac commands: only the exact forms in docs/m7_plan.md section 5 (#2 shasum, #6 pgrep, #11 cat of
# MAC_LOG_NAME, #12-#14 kill -STOP/-CONT/-KILL of MAC_PID).
# Gates:
#   [A] distributed, cache OFF: 20 tokens = G1; hidden byte-identical to data/g5_mirror_hidden.bin;
#       logits byte-identical to the frozen data/g5_hydra_logits.bin.
#   [B] distributed, cache ON: scripts/kv_compare.py b unchanged.
#   [C] ranges: each worker's startup line names the configured range and device id; WSL2 rejects
#       layer 6 and serves 0 and 5, the Mac rejects 5 and serves 6 and 11 (byte-identical to the
#       G3 mirror). Routing: every packet of runs A and B went to its layer's owner, was answered
#       by that owner, 20 packets per layer per run. Wire, proven from the WORKERS' logs: for every
#       packet, rx_bytes and tx_bytes equal the exact size of an activation-only TensorPacket for
#       that T (computed here with hydra_pb2), one reply per request, every Received shape exactly
#       [T,768]; every peer of the WSL2 worker is 127.0.0.1 and every peer of the Mac worker is
#       the client's own source address (no worker-to-worker traffic); worker.cpp has no connect().
#   [D] failures: WSL2 owner refused (D1) / unroutable (D2) at connect; WSL2 stalled mid-run (D3,
#       SIGSTOP); Mac stalled mid-run (D5, SIGSTOP); WSL2 killed mid-run (D4, SIGKILL); Mac killed
#       between steps (D6, SIGKILL). Each time the client exits nonzero within its limit, names
#       the owner, writes nothing, and the other worker then serves a byte-identical block.
#       Limits count from the client's READY marker (D1, D2) or from the fault injection (D3-D6),
#       never from process start (torch/transformers import time is excluded).
# Gate E (single-worker mode, all 12 layers on the Mac: scripts/g5_check.sh and
# scripts/kv_check.sh unchanged) is a separate run; see docs/m7_plan.md.
set -u
OVERALL_S=1800
if [ -z "${M7_UNDER_TIMEOUT:-}" ]; then
  # --foreground: the checker stays in the terminal's process group, so Ctrl-C still stops it.
  M7_UNDER_TIMEOUT=1 timeout --foreground --kill-after=30 "$OVERALL_S" bash "$0" "$@"; rc=$?
  if [ "$rc" -eq 124 ] || [ "$rc" -eq 137 ]; then
    echo "FAIL overall timeout: the checker did not finish within ${OVERALL_S}s"; echo "M7_CHECK_FAIL"
  fi
  exit "$rc"
fi

PY=/mnt/c/Users/ASUS/AppData/Local/Programs/Python/Python313/python.exe
M=yonghowlee@10.0.0.62
D='"$HOME/Documents/DLYK stuff"'
SSH="ssh -o BatchMode=yes $M"
RUN=$(date -u +%Y%m%dT%H%M%SZ); ID="m7-$RUN"      # unique client ids per checker run
STAGE=data/live/m7_$RUN
WSL_EP=127.0.0.1:50051; WSL_DEV=worker-wsl; WSL_RANGE=0-5
MAC_EP=10.0.0.62:50051; MAC_DEV=worker-mac; MAC_RANGE=6-11
ROUTE="$WSL_RANGE=$WSL_EP/$WSL_DEV,$MAC_RANGE=$MAC_EP/$MAC_DEV"
N_GEN=20; TIMEOUT_S=10; CONNECT_S=3; SLACK_S=3; CLIENT_MAX_S=300; READY_MAX_S=180
fail=0
WSL_PID=${1:?WSL_PID}; WSL_LOG=${2:?WSL_LOG}; MAC_PID=${3:?MAC_PID}; MAC_LOG=${4:?MAC_LOG_NAME}
die() { echo "$1"; echo "M7_CHECK_FAIL"; exit 1; }
# Inputs that reach a remote shell are validated before any remote command runs.
[[ "$MAC_LOG" =~ ^[A-Za-z0-9_.-]+\.log$ ]] || die "ERROR: MAC_LOG_NAME '$MAC_LOG' must match ^[A-Za-z0-9_.-]+\.log\$"
[[ "$MAC_PID" =~ ^[0-9]+$ ]] || die "ERROR: MAC_PID '$MAC_PID' is not a number"
[[ "$WSL_PID" =~ ^[0-9]+$ ]] || die "ERROR: WSL_PID '$WSL_PID' is not a number"

mac_pgrep() { $SSH "pgrep -lx worker; lsof -nP -iTCP:50051 -sTCP:LISTEN" < /dev/null; }   # #6
mac_cat()   { $SSH "cd $D && cat live_logs/$MAC_LOG" < /dev/null; }                         # #11
mac_sig()   { $SSH "kill -$1 $MAC_PID" < /dev/null; }                                       # #12-#14
now() { date +%s.%N; }
# wait_for FILE PATTERN SECONDS: returns 0 once the extended regex PATTERN is in FILE.
wait_for() { local i; for i in $(seq 1 $(( $3 * 10 ))); do grep -Eq "$2" "$1" 2>/dev/null && return 0; sleep 0.1; done; return 1; }

echo "[stage PRECHECK] run id $ID"
[ -e "$STAGE" ] && die "ERROR: $STAGE exists"
for f in gen_hydra.py worker.cpp orchestrator.cpp hydra.proto scripts/m7_check.sh scripts/g5_compare.py scripts/g5_compare_hidden.py \
         scripts/kv_compare.py data/g5_mirror_hidden.bin data/g5_mirror_hidden_index.txt \
         data/g5_hydra_logits.bin data/g5_hydra_logits.sha256 data/gen_ref_tokens.txt; do
  git cat-file -e "HEAD:$f" 2>/dev/null || die "ERROR: $f is not committed"
  git diff --quiet HEAD -- "$f" || die "ERROR: $f differs from its committed version"
done
sha256sum -c --quiet data/g5_hydra_logits.sha256 || die "ERROR: frozen logits != pin"   # pin paths are repo-root relative
# Static guarantee that a worker cannot talk to anything but the client that connected to it:
# exactly one socket() (the SOCK_STREAM listener), no connect(), no datagram send paths.
grep -nE '\bconnect\s*\(|\bsendto\s*\(|\bsendmsg\s*\(|SOCK_DGRAM' worker.cpp \
  && die "ERROR: worker.cpp has an outbound or datagram socket path; workers must only accept()"
[ "$(grep -cE '\bsocket\s*\(' worker.cpp)" -eq 1 ] && grep -qE '\bsocket\s*\(AF_INET, SOCK_STREAM' worker.cpp \
  || die "ERROR: worker.cpp must create exactly one socket, the SOCK_STREAM listener"
echo "PASS worker.cpp: one SOCK_STREAM listener socket, no connect/sendto/sendmsg/SOCK_DGRAM"
# Binaries pinned to the committed sources (builds are reproducible: data/live/m7_evidence_20261118/build_repro.txt).
mkdir -p "$STAGE/bin" "$STAGE/probe/data" || die "ERROR: cannot create $STAGE"
g++ -O2 -std=c++17 -Wall -Wextra worker.cpp hydra.pb.cc -lprotobuf -o "$STAGE/bin/worker_linux" \
  || die "ERROR: cannot build worker.cpp"
g++ -O2 -std=c++17 -Wall -Wextra orchestrator.cpp hydra.pb.cc -lprotobuf -o "$STAGE/bin/orchestrator" \
  || die "ERROR: cannot build orchestrator.cpp"
sed 's|"10\.0\.0\.62"|"127.0.0.1"|' orchestrator.cpp > "$STAGE/bin/orchestrator_local.cpp"
[ "$(diff orchestrator.cpp "$STAGE/bin/orchestrator_local.cpp" | grep -c '^>')" -eq 1 ] \
  || die "ERROR: orchestrator_local must differ from orchestrator.cpp in exactly the IP line"
g++ -O2 -std=c++17 -Wall -Wextra -I. "$STAGE/bin/orchestrator_local.cpp" hydra.pb.cc -lprotobuf \
  -o "$STAGE/bin/orchestrator_local" || die "ERROR: cannot build orchestrator_local"
# wsl_pinned PID: the running WSL2 worker is exactly the binary built from the committed worker.cpp.
wsl_pinned() { cmp -s "/proc/$1/exe" "$STAGE/bin/worker_linux" \
  || die "ERROR: running worker_linux (PID $1) is not the build of the committed worker.cpp"; }
[ "$(pgrep -x worker_linux)" = "$WSL_PID" ] || die "ERROR: the only worker_linux PID must be $WSL_PID"
wsl_pinned "$WSL_PID"
grep -q "^Worker layers $WSL_RANGE device_id $WSL_DEV$" "$WSL_LOG" || die "ERROR: $WSL_LOG lacks 'Worker layers $WSL_RANGE device_id $WSL_DEV'"
grep -q "listening" "$WSL_LOG" || die "ERROR: $WSL_LOG has no 'listening' line"
out=$(mac_pgrep) || die "ERROR: Mac #6 failed (ssh exit $?)"
[ "$(printf '%s\n' "$out" | grep -c ' worker$')" -eq 1 ] && printf '%s\n' "$out" | grep -qx "$MAC_PID worker" \
  || die "ERROR: Mac must run exactly one worker, PID $MAC_PID (#6 said: $out)"
# Mac source = committed worker.cpp (#2); the binary hash is recorded with it.
mac_sum=$($SSH "cd $D && shasum -a 256 worker.cpp hydra.proto worker" < /dev/null) || die "ERROR: Mac #2 failed"
printf '%s\n' "$mac_sum" > "$STAGE/mac_sha256.txt"
[ "$(printf '%s\n' "$mac_sum" | awk '$2 == "worker.cpp" {print $1}')" = "$(git show HEAD:worker.cpp | sha256sum | cut -d' ' -f1)" ] \
  || die "ERROR: the Mac's worker.cpp is not the committed worker.cpp (see $STAGE/mac_sha256.txt)"
[ "$(printf '%s\n' "$mac_sum" | awk '$2 == "hydra.proto" {print $1}')" = "$(git show HEAD:hydra.proto | sha256sum | cut -d' ' -f1)" ] \
  || die "ERROR: the Mac's hydra.proto is not the committed hydra.proto (see $STAGE/mac_sha256.txt)"
echo "PASS Mac worker.cpp and hydra.proto = committed; Mac binary $(printf '%s\n' "$mac_sum" | awk '$2 == "worker" {print $1}')"
mac_cat | grep -q "^Worker layers $MAC_RANGE device_id $MAC_DEV$" || die "ERROR: Mac log lacks 'Worker layers $MAC_RANGE device_id $MAC_DEV'"
# The client's source address toward the Mac, as the Mac will see it (a UDP connect sends nothing).
CLIENT_IP=$("$PY" -c "import socket; u=socket.socket(socket.AF_INET, socket.SOCK_DGRAM); u.connect(('10.0.0.62', 9)); print(u.getsockname()[0])" | tr -d '\r')
[[ "$CLIENT_IP" =~ ^[0-9.]+$ ]] || die "ERROR: cannot determine the client's source address ('$CLIENT_IP')"
echo "client source address toward the Mac: $CLIENT_IP"
cp data/block_input_T8.bin "$STAGE/probe/data/" || die "ERROR: staging failed"

# probe BIN LAYER accept|reject LABEL: one block round trip with the checker's own builds of the
# committed orchestrator.cpp (orchestrator -> Mac, orchestrator_local -> WSL2), T=8, the G3 input.
probe() { local bin=$1 L=$2 want=$3 label=$4 rc n; n=$(printf %02d "$L")
  rm -f "$STAGE/probe/data/block_response_T8.bin"
  (cd "$STAGE/probe" && timeout 60 "../bin/$bin" --mode block --T 8 --layer "$L" > "orch_${bin}_$L.txt" 2>&1); rc=$?
  if [ "$want" = reject ]; then
    if [ "$rc" -eq 2 ] && grep -q "Worker rejected layer_index $L" "$STAGE/probe/orch_${bin}_$L.txt" \
       && [ ! -f "$STAGE/probe/data/block_response_T8.bin" ]; then echo "PASS $label: layer $L rejected"
    else echo "FAIL $label: layer $L not rejected (exit $rc)"; fail=1; fi
  else
    if [ "$rc" -eq 0 ] && cmp -s "$STAGE/probe/data/block_response_T8.bin" "data/g3_mirror_h${n}_T8.bin"; then
      echo "PASS $label: layer $L served byte-identical to data/g3_mirror_h${n}_T8.bin"
    else echo "FAIL $label: layer $L not served correctly (exit $rc)"; fail=1; fi
  fi; }
wsl_alive() { probe orchestrator_local 0 accept "WSL2 alive $1"; }
mac_alive() { probe orchestrator 11 accept "Mac alive $1"; }

# gen DIR ARGS...: one distributed client run in the foreground, bounded by CLIENT_MAX_S.
gen() { local out=$1; shift
  timeout --kill-after=10 "$CLIENT_MAX_S" "$PY" -u gen_hydra.py --route "$ROUTE" --timeout "$TIMEOUT_S" \
      --connect-timeout "$CONNECT_S" --out "$out" "$@" > "$out.log" 2>&1 < /dev/null; local rc=$?
  tr -d '\r' < "$out.log" | grep -E "^Wrote |^ERROR"; return $rc; }

echo "[stage A] distributed, cache OFF"
if gen "$STAGE/a_off" --kv-cache off --client-id "$ID-a"; then
  "$PY" scripts/g5_compare_hidden.py "$STAGE/a_off" 2>&1 | tr -d '\r'; [ "${PIPESTATUS[0]}" -eq 0 ] || fail=1
  "$PY" scripts/g5_compare.py "$STAGE/a_off" 2>&1 | tr -d '\r'; [ "${PIPESTATUS[0]}" -eq 0 ] || fail=1
  if cmp -s "$STAGE/a_off/logits_f64.bin" data/g5_hydra_logits.bin; then
    echo "PASS stage A: logits byte-identical to frozen data/g5_hydra_logits.bin"
  else echo "FAIL stage A: logits differ from frozen data/g5_hydra_logits.bin"; fail=1; fi
else echo "FAIL stage A: client failed, see $STAGE/a_off.log"; fail=1; fi

echo "[stage B] distributed, cache ON (cold)"
if gen "$STAGE/b_on" --kv-cache on --client-id "$ID-b"; then
  "$PY" scripts/kv_compare.py b "$STAGE/b_on" 2>&1 | tr -d '\r'; [ "${PIPESTATUS[0]}" -eq 0 ] || fail=1
else echo "FAIL stage B: client failed, see $STAGE/b_on.log"; fail=1; fi

echo "[stage C] ranges, routing, wire bytes, peers"
probe orchestrator_local 6 reject "WSL2 boundary"; probe orchestrator_local 0 accept "WSL2 range start"
probe orchestrator_local 5 accept "WSL2 range end"
probe orchestrator 5 reject "Mac boundary"; probe orchestrator 6 accept "Mac range start"
probe orchestrator 11 accept "Mac range end"
cp "$WSL_LOG" "$STAGE/worker_wsl_C.log"
mac_cat > "$STAGE/worker_mac_C.log" || die "ERROR: cannot read the Mac log"
"$PY" - "$STAGE" "$ROUTE" "$N_GEN" "$ID-a" "$ID-b" "$CLIENT_IP" <<'PYEOF' || fail=1
import re, sys
import hydra_pb2                              # cwd = repo root
stage, route, n_gen, id_a, id_b, client_ip = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5], sys.argv[6]
H = 768
owner, ranges = {}, {}                        # layer -> (endpoint, device); device -> "A-B"
for part in route.split(","):
    rng, dest = part.split("=")
    ep, dev = dest.split("/")
    a, b = map(int, rng.split("-"))
    ranges[dev] = rng
    for l in range(a, b + 1):
        owner[l] = (ep, dev)
logs = {"worker-wsl": f"{stage}/worker_wsl_C.log", "worker-mac": f"{stage}/worker_mac_C.log"}
peer_ip = {"worker-wsl": "127.0.0.1", "worker-mac": client_ip}
all_ok = True
def check(label, problems):
    global all_ok
    all_ok &= not problems
    for p in problems[:10]:
        print(f"FAIL stage C {label}: {p}")
    if len(problems) > 10:
        print(f"FAIL stage C {label}: ... {len(problems) - 10} more")
    if not problems:
        print(f"PASS stage C {label}")

def size(layer, dev, T, seq=0, sp=0):         # exact wire bytes of an activation-only packet
    p = hydra_pb2.TensorPacket(layer_index=layer, device_id=dev)
    if seq:
        p.sequence_id, p.start_pos = seq, sp
    p.data.extend([0.0] * (T * H)); p.shape.extend([T, H])
    return 4 + p.ByteSize()                   # packed fixed32 floats: size does not depend on values

# T=-1 is a request without a shape (worker.cpp logs -1); it must still be parsed so its peer is checked.
wire = re.compile(r"^WIRE from=(\S+) peer=([0-9.]+):(\d+) layer=(-?\d+) T=(-?\d+) rx_bytes=(\d+) tx_bytes=(\d+) replies=(\d+)$")
recv = re.compile(r"^Received TensorPacket #\d+ from '([^']*)' \(layer (-?\d+)\), (\d+) floats, shape \[([^\]]*)\]")
W, R = {}, {}                                 # device -> list of parsed WIRE / Received lines
for dev, path in logs.items():
    lines = [l.rstrip("\r\n") for l in open(path, errors="replace")]
    check(f"{dev} startup range", [] if f"Worker layers {ranges[dev]} device_id {dev}" in lines
          else [f"no 'Worker layers {ranges[dev]} device_id {dev}' line"])
    W[dev] = [m.groups() for m in map(wire.match, lines) if m]
    R[dev] = [m.groups() for m in map(recv.match, lines) if m]
    # every peer of every request, any client id (probes included): no worker-to-worker traffic
    check(f"{dev} peers", [f"peer {w[1]}:{w[2]} (from={w[0]} layer {w[3]}), want {peer_ip[dev]}"
                           for w in W[dev] if w[1] != peer_ip[dev]])

for run, cid in (("a_off", id_a), ("b_on", id_b)):
    sends = [l.split() for l in open(f"{stage}/{run}/sends.txt") if l.strip() and not l.startswith("#")]
    rt = [l.split() for l in open(f"{stage}/{run}/route.txt") if l.strip() and not l.startswith("#")]
    probs, per_layer = [], {}
    if len(rt) != len(sends):
        probs.append(f"route.txt {len(rt)} rows vs sends.txt {len(sends)}")
    for s, r in zip(sends, rt):
        step, layer, seq, sp, T = map(int, s)
        r_step, r_layer, ep, dev, r_T, req_f, rep_f, req_b, rep_b = r
        per_layer[layer] = per_layer.get(layer, 0) + 1
        if (int(r_step), int(r_layer), int(r_T)) != (step, layer, T):
            probs.append(f"step {step} layer {layer}: route.txt row {r[:5]} does not match sends.txt")
        if (ep, dev) != owner[layer]:
            probs.append(f"step {step} layer {layer} went to {ep}/{dev}, owner {owner[layer]}")
        if (int(req_f), int(rep_f)) != (T * H, T * H):
            probs.append(f"step {step} layer {layer}: {req_f}/{rep_f} floats, want {T * H}")
    if per_layer != {l: n_gen for l in range(12)}:
        probs.append(f"packets per layer {per_layer}, want {n_gen} for each of 0..11")
    check(f"client routing {run} ({len(rt)} packets)", probs)
    # worker side, from each worker's own log
    for dev in logs:
        mine = [(int(s[0]), int(s[1]), int(s[2]), int(s[3]), int(s[4])) for s in sends if owner[int(s[1])][1] == dev]
        ws = [w for w in W[dev] if w[0] == cid]
        rs = [r for r in R[dev] if r[0] == cid]
        probs = []
        if len(ws) != len(mine):
            probs.append(f"{len(ws)} WIRE lines, client sent {len(mine)} packets to this worker")
        if len(rs) != len(ws):
            probs.append(f"{len(rs)} Received lines vs {len(ws)} WIRE lines")
        for (step, layer, seq, sp, T), w in zip(mine, ws):
            wl, wT, rx, tx, nrep = int(w[3]), int(w[4]), int(w[5]), int(w[6]), int(w[7])
            want_rx, want_tx = size(layer, cid, T, seq, sp), size(layer, dev, T)
            if (wl, wT) != (layer, T):
                probs.append(f"step {step}: worker logged layer {wl} T={wT}, client sent layer {layer} T={T}")
            if (rx, tx, nrep) != (want_rx, want_tx, 1):
                probs.append(f"step {step} layer {layer}: rx {rx} tx {tx} replies {nrep}, want rx {want_rx} tx {want_tx} replies 1")
        for r in rs:
            shape = [x.strip() for x in r[3].split(",")]
            if len(shape) != 2 or shape[1] != str(H) or int(r[2]) != int(shape[0]) * H:
                probs.append(f"Received layer {r[1]}: {r[2]} floats, shape [{r[3]}], want [T,{H}] with T*{H} floats")
        layers = sorted({int(w[3]) for w in ws})
        want_layers = sorted(l for l, (_, d) in owner.items() if d == dev)
        if layers != want_layers:
            probs.append(f"layers {layers}, want exactly {want_layers}")
        check(f"{dev} wire {run} ({len(ws)} packets, exact sizes)", probs)
sys.exit(0 if all_ok else 1)
PYEOF

echo "[stage D] failures"
# client_bg LABEL ARGS...: one client run in the background, bounded by CLIENT_MAX_S.
client_bg() { local label=$1; shift; COUT="$STAGE/$label"
  timeout --kill-after=10 "$CLIENT_MAX_S" "$PY" -u gen_hydra.py --timeout "$TIMEOUT_S" \
      --connect-timeout "$CONNECT_S" --kv-cache off --client-id "$ID-$label" --out "$COUT" "$@" \
      > "$COUT.log" 2>&1 < /dev/null &
  CPID=$!; }
# judge LABEL T0 LIMIT NEEDLE: the client must exit nonzero (not via timeout) within LIMIT s of
# T0, name NEEDLE (the failed owner's endpoint) in an ERROR line, and write none of its outputs.
judge() { local label=$1 t0=$2 limit=$3 needle=$4 rc dt f wrote=""
  wait "$CPID"; rc=$?; dt=$(echo "$(now) - $t0" | bc)
  for f in tokens.txt logits_f64.bin hidden_f32.bin sends.txt route.txt; do [ -e "$COUT/$f" ] && wrote="$wrote $f"; done
  if [ "$rc" -ne 0 ] && [ "$rc" -ne 124 ] && [ "$rc" -ne 137 ] && [ "$(echo "$dt <= $limit" | bc)" -eq 1 ] \
     && tr -d '\r' < "$COUT.log" | grep -q "^ERROR: worker $needle " && [ -z "$wrote" ]; then
    echo "PASS $label: exit $rc ${dt}s after the reference point (limit ${limit}s), names $needle, wrote nothing"
  else
    echo "FAIL $label: exit $rc ${dt}s after the reference point (limit ${limit}s), wrote:${wrote:- nothing}; see $COUT.log"; fail=1
  fi; }
# ready_t0 LABEL: wait for the client's READY marker; sets T0 to the time it was seen. Not called in
# a $(...) subshell, so its fail=1 reaches this shell.
ready_t0() { wait_for "$COUT.log" '^READY' "$READY_MAX_S" || { echo "FAIL $1: no READY marker within ${READY_MAX_S}s"; fail=1; }; T0=$(now); }

# From the client's side of the WSL2/Windows boundary: Windows 127.0.0.1:50052 must refuse.
free=$("$PY" -c "
import socket
try:
    socket.create_connection(('127.0.0.1', 50052), 5)   # Windows reports a localhost refusal after ~2.05 s: data/live/m7_evidence_20261118/windows_refuse_timing.txt
    print('accepted')
except ConnectionRefusedError:
    print('refused')
except Exception as e:
    print('other:', type(e).__name__)" < /dev/null | tr -d '\r')
[ "$free" = refused ] || die "ERROR: Windows 127.0.0.1:50052 must refuse connections for D1 (got: $free)"
client_bg d1 --route "$WSL_RANGE=127.0.0.1:50052/$WSL_DEV,$MAC_RANGE=$MAC_EP/$MAC_DEV"
ready_t0 d1; judge d1_refused "$T0" $((CONNECT_S + SLACK_S)) "127.0.0.1:50052"
client_bg d2 --route "$WSL_RANGE=192.0.2.1:50051/$WSL_DEV,$MAC_RANGE=$MAC_EP/$MAC_DEV"   # RFC 5737 TEST-NET-1
ready_t0 d2; judge d2_unroutable "$T0" $((CONNECT_S + SLACK_S)) "192.0.2.1:50051"
wsl_alive "after D1/D2"; mac_alive "after D1/D2"

echo "[D3] WSL2 worker stalled mid-run"
client_bg d3 --route "$ROUTE"
if wait_for "$WSL_LOG" "^WIRE from=$ID-d3 " "$READY_MAX_S"; then
  kill -STOP "$WSL_PID" || { echo "FAIL D3: kill -STOP failed"; fail=1; }; t0=$(now)
  judge d3_wsl_stall "$t0" $((TIMEOUT_S + SLACK_S)) "$WSL_EP"
  kill -CONT "$WSL_PID" || { echo "FAIL D3: kill -CONT failed"; fail=1; }
else echo "FAIL D3: no packet from $ID-d3 reached the WSL2 worker"; fail=1; wait "$CPID"; fi
wsl_alive "after D3"; mac_alive "after D3"

echo "[D5] Mac worker stalled mid-run"
client_bg d5 --route "$ROUTE"
if wait_for "$COUT.log" '^step +0 ' "$READY_MAX_S"; then
  mac_sig STOP || { echo "FAIL D5: kill -STOP $MAC_PID failed"; fail=1; }; t0=$(now)
  judge d5_mac_stall "$t0" $((TIMEOUT_S + SLACK_S)) "$MAC_EP"
  mac_sig CONT || { echo "FAIL D5: kill -CONT $MAC_PID failed"; fail=1; }
else echo "FAIL D5: client never finished step 0"; fail=1; wait "$CPID"; fi
wsl_alive "after D5"; mac_alive "after D5"

echo "[D4] WSL2 worker killed mid-run"
client_bg d4 --route "$ROUTE"
if wait_for "$WSL_LOG" "^WIRE from=$ID-d4 " "$READY_MAX_S"; then
  kill -KILL "$WSL_PID" || { echo "FAIL D4: kill -KILL failed"; fail=1; }; t0=$(now)
  judge d4_wsl_killed "$t0" $((TIMEOUT_S + SLACK_S)) "$WSL_EP"
else echo "FAIL D4: no packet from $ID-d4 reached the WSL2 worker"; fail=1; wait "$CPID"; fi
mac_alive "after D4 (WSL2 worker killed)"
sleep 1; pgrep -x worker_linux > /dev/null && { echo "FAIL D4: a worker_linux process is still running"; fail=1; }

echo "[D4b] relaunch the WSL2 worker (the checker's own build of the committed worker.cpp, same flags)"
WSL_LOG="data/live/linux_worker_logs/worker_wsl_m7_${RUN}_relaunch.log"
[ -e "$WSL_LOG" ] && die "ERROR: $WSL_LOG exists"
nohup "$STAGE/bin/worker_linux" --mode block --layers "$WSL_RANGE" --device-id "$WSL_DEV" > "$WSL_LOG" 2>&1 < /dev/null &
WSL_PID=$!
wait_for "$WSL_LOG" 'listening' 60 && grep -q "^Worker layers $WSL_RANGE device_id $WSL_DEV$" "$WSL_LOG" \
  || die "ERROR: WSL2 relaunch failed its launch gate; see $WSL_LOG"
wsl_pinned "$WSL_PID"
echo "WSL2 worker relaunched: PID $WSL_PID, log $WSL_LOG"

echo "[D6] Mac worker killed between steps"
# Making the kill land between steps: once the WSL2 worker has served step 2's layer 0 (its 13th
# packet of this run), SIGSTOP it. The client is then waiting on WSL2 and the Mac is idle with
# exactly steps 0 and 1 served and nothing in flight. Kill the Mac, then SIGCONT WSL2; the client
# finishes step 2's WSL2 half and its first Mac send/recv fails. The pause (one ssh) must stay well
# under the 10 s recv timeout, else the client times out on WSL2 and the needle check fails.
client_bg d6 --route "$ROUTE"
hit=0
for i in $(seq 1 $((READY_MAX_S * 10))); do
  [ "$(grep -c "^WIRE from=$ID-d6 " "$WSL_LOG" 2>/dev/null)" -ge 13 ] && { hit=1; break; }
  sleep 0.1
done
if [ "$hit" -eq 1 ]; then
  kill -STOP "$WSL_PID" || { echo "FAIL D6: kill -STOP WSL2 failed"; fail=1; }; ts=$(now)
  # Proof the client is still waiting on WSL2 (so nothing is in flight to the Mac): WSL2 has served
  # 13..17 packets of this run, i.e. step 2's layer 0 but not yet its layer 5.
  w=$(grep -c "^WIRE from=$ID-d6 " "$WSL_LOG")
  if [ "$w" -ge 13 ] && [ "$w" -le 17 ]; then echo "PASS D6: paused with WSL2 at $w packets (inside step 2's WSL2 half)"
  else echo "FAIL D6: paused with WSL2 at $w packets, want 13-17; the Mac may have had a request in flight"; fail=1; fi
  mac_sig KILL || { echo "FAIL D6: kill -KILL $MAC_PID failed"; fail=1; }
  kill -CONT "$WSL_PID" || { echo "FAIL D6: kill -CONT WSL2 failed"; fail=1; }; t0=$(now)
  echo "D6 WSL2 paused $(echo "$t0 - $ts" | bc)s while the Mac was killed"
  judge d6_mac_killed "$t0" $((TIMEOUT_S + SLACK_S)) "$MAC_EP"
  # Proven from the Mac's own log: it had served exactly steps 0 and 1 (12 packets) when killed.
  mac_cat > "$STAGE/worker_mac_D6.log" || { echo "FAIL D6: cannot read the Mac log"; fail=1; }
  n=$(grep -c "^WIRE from=$ID-d6 " "$STAGE/worker_mac_D6.log" || true)
  r=$(grep -c "^Received TensorPacket #[0-9]* from '$ID-d6'" "$STAGE/worker_mac_D6.log" || true)
  if [ "$n" -eq 12 ] && [ "$r" -eq 12 ]; then echo "PASS D6: the Mac had received and served exactly 2 whole steps (12 packets) when killed"
  else echo "FAIL D6: the Mac had received $r and served $n packets for $ID-d6, want 12 and 12; fault not between steps"; fail=1; fi
else echo "FAIL D6: the WSL2 worker never served step 2's layer 0 for $ID-d6"; fail=1; wait "$CPID"; fi
wsl_alive "after D6 (Mac worker killed)"
out=$(mac_pgrep); rc=$?     # #6 exits with lsof's status: 1 = nothing listening; 255 = ssh failed
if [ "$rc" -eq 255 ]; then echo "FAIL D6: Mac #6 failed (ssh exit 255)"; fail=1
elif printf '%s\n' "$out" | grep -q "^$MAC_PID "; then echo "FAIL D6: Mac PID $MAC_PID still running"; fail=1
else echo "PASS D6: Mac PID $MAC_PID gone (#6)"; fi

echo "[stop] WSL2 worker (the Mac worker was killed by D6)"
kill -INT "$WSL_PID"; sleep 1
pgrep -x worker_linux > /dev/null && { echo "FAIL stop: worker_linux still running"; fail=1; }

if [ "$fail" -eq 0 ]; then echo "M7_CHECK_PASS"; else echo "M7_CHECK_FAIL"; exit 1; fi
