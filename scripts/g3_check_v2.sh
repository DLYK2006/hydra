#!/usr/bin/env bash
# G3 checker v2. Hand-reviewed; do not edit. Run from repo root on WSL2, with
# the Mac worker already running the G3 build in --mode block. Reads data/;
# writes only its own staging dir data/live/g3v2/ (ignored via data/live/).
#
# Supersedes g3_check.sh's flat 6.1e-6 gate for blocks 01-11: that bound is
# below one float32 ULP for |values| >= 64, and the remaining Hydra-vs-reference
# difference is the reference's float32 LayerNorm/GELU vs worker.cpp's float64
# (reports/2026-10-28_g3.md). Per block N it requires:
#   1. the response cmp-identical to data/g3_mirror_hNN_T8.bin (bit-exact);
#   2. max|Hydra - data/g3_expected_hNN_T8.bin| <= PYTORCH_PARITY_FACTOR x
#      block N's PyTorch-f32 max_abs from the committed data/g3_ref_summary.txt;
#   3. layer_index N sent and echoed.
# Plus: block 0 byte-identical to the June data/block_response_T8.bin;
# layer_index 12 and -1 rejected (orchestrator exit 2) with the worker still
# serving; an oversize length prefix does not wedge the worker.
set -u
# Windows Python: WSL2's native Python has no numpy (CLAUDE.md, M1 notes).
# Relative paths work only because the repo lives on /mnt/c (interop maps cwd).
PY=/mnt/c/Users/ASUS/AppData/Local/Programs/Python/Python313/python.exe
WORKER_HOST=10.0.0.62   # same hardcoded worker address as orchestrator.cpp
WORKER_PORT=50051
PYTORCH_PARITY_FACTOR=1.6   # Hydra must be within 1.6x PyTorch float32's own error
STAGE=data/live/g3v2
SUMMARY=data/g3_ref_summary.txt
fail=0

die() { echo "$1"; echo "G3V2_CHECK_FAIL"; exit 1; }

[ -e "$STAGE" ] && die "ERROR: $STAGE exists; remove it to rerun (refusing to overwrite)"
for f in orchestrator data/block_input_T8.bin data/block_response_T8.bin "$SUMMARY"; do
  [ -f "$f" ] || die "MISSING $f"
done
for i in $(seq 0 11); do
  n=$(printf %02d "$i")
  [ -f "data/g3_expected_h${n}_T8.bin" ] || die "MISSING data/g3_expected_h${n}_T8.bin"
  [ -f "data/g3_mirror_h${n}_T8.bin" ]   || die "MISSING data/g3_mirror_h${n}_T8.bin"
done
mkdir -p "$STAGE/data" && cp data/block_input_T8.bin "$STAGE/data/" || die "ERROR: staging failed"

# run <layer_index> <tag>: one orchestrator round trip from inside $STAGE, so its
# hardcoded data/block_response_T8.bin lands in $STAGE/data/, never in data/.
run() { (cd "$STAGE" && ../../../orchestrator --mode block --T 8 --layer "$1" > "orch_$2.txt" 2>&1); }

# parity <resp> <expected> <block>: prints "<max_abs> <bound> OK|OVER"; exit 0 only if OK.
# A missing/unparsable summary row raises (nonzero exit), never a silent pass.
parity() {
  "$PY" - "$1" "$2" "$SUMMARY" "$3" "$PYTORCH_PARITY_FACTOR" <<'PYEOF'
import sys
import numpy as np
resp, exp, summ, n, factor = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), float(sys.argv[5])
pt = None
for line in open(summ):
    f = line.split()
    if f and not line.startswith("#") and int(f[0]) == n:
        pt = float(f[1])
if pt is None:
    raise SystemExit(f"no row for block {n} in {summ}")
r = np.fromfile(resp, np.float32).astype(np.float64)
e = np.fromfile(exp, np.float32).astype(np.float64)
if r.size != 8 * 768 or e.size != 8 * 768:
    raise SystemExit(f"size response={r.size} expected={e.size}, want {8 * 768}")
d = float(np.abs(r - e).max())     # NaN -> comparison below is False -> OVER
bound = factor * pt
ok = d <= bound
print(f"{d:.3e} {bound:.3e} {'OK' if ok else 'OVER'}")
sys.exit(0 if ok else 1)
PYEOF
}

TABLE="$STAGE/table.txt"
printf '%-5s  %-13s  %-22s  %-10s\n' block hydra_max_abs "bound(${PYTORCH_PARITY_FACTOR}xPT-f32)" mirror_cmp > "$TABLE"

# --- 12 valid layers ---------------------------------------------------------
for i in $(seq 0 11); do
  n=$(printf %02d "$i")
  run "$i" "h$n"; rc=$?
  if [ "$rc" -ne 0 ] || [ ! -f "$STAGE/data/block_response_T8.bin" ]; then
    echo "FAIL h$n: orchestrator exit $rc, see $STAGE/orch_h$n.txt"; fail=1
    [ -f "$STAGE/data/block_response_T8.bin" ] && mv "$STAGE/data/block_response_T8.bin" "$STAGE/failed_h$n.bin"
    printf '%-5s  %-13s  %-22s  %-10s\n' "h$n" "-" "-" "NO_RESPONSE" >> "$TABLE"
    continue
  fi
  mv "$STAGE/data/block_response_T8.bin" "$STAGE/resp_h$n.bin"
  grep -q "layer $i, " "$STAGE/orch_h$n.txt" || { echo "FAIL h$n: sent layer_index != $i"; fail=1; }
  grep -q "Received response for layer $i from" "$STAGE/orch_h$n.txt" || { echo "FAIL h$n: reply layer_index != $i"; fail=1; }

  if cmp -s "$STAGE/resp_h$n.bin" "data/g3_mirror_h${n}_T8.bin"; then
    mirror=IDENTICAL
  else
    mirror=DIFFERS; fail=1; echo "FAIL h$n: not bit-identical to data/g3_mirror_h${n}_T8.bin"
  fi

  out=$(parity "$STAGE/resp_h$n.bin" "data/g3_expected_h${n}_T8.bin" "$i" 2>&1); prc=$?
  out=$(printf '%s' "$out" | tr -d '\r')          # Windows Python writes CRLF
  read -r hmax bound verdict <<< "$out"
  if [ "$prc" -ne 0 ] || [ "${verdict:-}" != "OK" ]; then
    fail=1; echo "FAIL h$n: PyTorch-parity check: $out"
  fi
  printf '%-5s  %-13s  %-22s  %-10s\n' "h$n" "${hmax:-?}" "${bound:-?} ${verdict:-ERR}" "$mirror" >> "$TABLE"
done
cat "$TABLE"

# --- block 0 regression vs June ----------------------------------------------
if [ -f "$STAGE/resp_h00.bin" ] && cmp -s "$STAGE/resp_h00.bin" data/block_response_T8.bin; then
  echo "PASS h00 byte-identical to June data/block_response_T8.bin"
else
  echo "FAIL h00 not byte-identical to June data/block_response_T8.bin"; fail=1
fi

# --- bad indices: rejected, worker survives ----------------------------------
for bad in 12 -1; do
  run "$bad" "bad$bad"; rc=$?
  if [ "$rc" -eq 2 ] && [ ! -f "$STAGE/data/block_response_T8.bin" ] \
     && grep -q "Worker rejected layer_index $bad" "$STAGE/orch_bad$bad.txt"; then
    echo "PASS layer_index $bad rejected (orchestrator exit 2)"
  else
    echo "FAIL layer_index $bad: orchestrator exit $rc (want 2), see $STAGE/orch_bad$bad.txt"; fail=1
  fi
  [ -f "$STAGE/data/block_response_T8.bin" ] && mv "$STAGE/data/block_response_T8.bin" "$STAGE/unexpected_bad$bad.bin"
  run 0 "alive_after$bad"; rc=$?
  if [ "$rc" -eq 0 ] && [ -f "$STAGE/data/block_response_T8.bin" ] \
     && cmp -s "$STAGE/data/block_response_T8.bin" data/block_response_T8.bin; then
    echo "PASS worker alive after layer_index $bad (follow-up layer 0 byte-identical)"
  else
    echo "FAIL worker not serving after layer_index $bad: exit $rc"; fail=1
  fi
  [ -f "$STAGE/data/block_response_T8.bin" ] && mv "$STAGE/data/block_response_T8.bin" "$STAGE/resp_alive_after$bad.bin"
done

# --- MAX_MSG_LEN cap: oversize length prefix must not wedge the worker ---------
# Hold a connection open after sending only a 0xFFFFFFFF length prefix. With the
# cap, the worker drops that connection at once and serves the follow-up. Without
# it, the worker either dies on a ~4 GiB resize or blocks reading the payload on
# this held connection; either way the follow-up (5 s SO_RCVTIMEO) fails.
if exec 3<>"/dev/tcp/$WORKER_HOST/$WORKER_PORT"; then
  printf '\377\377\377\377' >&3
  run 0 alive_after_oversize; rc=$?
  exec 3>&-
  if [ "$rc" -eq 0 ] && [ -f "$STAGE/data/block_response_T8.bin" ] \
     && cmp -s "$STAGE/data/block_response_T8.bin" data/block_response_T8.bin; then
    echo "PASS worker served a request while an oversize-length connection was held open"
  else
    echo "FAIL worker not serving after oversize length prefix: exit $rc"; fail=1
  fi
  [ -f "$STAGE/data/block_response_T8.bin" ] && mv "$STAGE/data/block_response_T8.bin" "$STAGE/resp_alive_after_oversize.bin"
else
  echo "FAIL could not open raw connection to $WORKER_HOST:$WORKER_PORT"; fail=1
fi

if [ "$fail" -eq 0 ]; then echo "G3V2_CHECK_PASS"; else echo "G3V2_CHECK_FAIL"; exit 1; fi
