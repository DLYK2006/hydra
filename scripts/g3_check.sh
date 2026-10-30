#!/usr/bin/env bash
# G3 checker. Hand-reviewed; do not edit. Run from repo root on WSL2, with the Mac
# worker already running the G3 build in --mode block. Reads data/; writes only
# its own staging dir data/live/g3/ (ignored via .gitignore's data/live/).
#
# For N in 0..11: send data/block_input_T8.bin with layer_index=N, compare the
# response to data/g3_expected_hNN_T8.bin (fail if max abs diff > TOL).
# Also: block 0 response must be byte-identical to the June
# data/block_response_T8.bin; layer_index 12 and -1 must be rejected (orchestrator
# exit 2, no response file) and a follow-up valid request must still succeed.
set -u
# Windows Python: WSL2's native Python has no numpy (CLAUDE.md, M1 notes).
# Relative paths work only because the repo lives on /mnt/c (interop maps cwd).
PY=/mnt/c/Users/ASUS/AppData/Local/Programs/Python/Python313/python.exe
WORKER_HOST=10.0.0.62   # same hardcoded worker address as orchestrator.cpp:235
WORKER_PORT=50051
TOL=6.1e-6   # compare_block.py tolerance: 1.6 x measured 3.81e-06 (block 0, T=8)
STAGE=data/live/g3
fail=0

die() { echo "$1"; echo "G3_CHECK_FAIL"; exit 1; }

[ -e "$STAGE" ] && die "ERROR: $STAGE exists; remove it to rerun (refusing to overwrite)"
for f in orchestrator data/block_input_T8.bin data/block_response_T8.bin; do
  [ -f "$f" ] || die "MISSING $f"
done
for i in $(seq 0 11); do
  n=$(printf %02d "$i")
  [ -f "data/g3_expected_h${n}_T8.bin" ] || die "MISSING data/g3_expected_h${n}_T8.bin"
done
mkdir -p "$STAGE/data" && cp data/block_input_T8.bin "$STAGE/data/" || die "ERROR: staging failed"

# run <layer_index> <tag>: one orchestrator round trip from inside $STAGE, so its
# hardcoded data/block_response_T8.bin lands in $STAGE/data/, never in data/.
run() { (cd "$STAGE" && ../../../orchestrator --mode block --T 8 --layer "$1" > "orch_$2.txt" 2>&1); }

# --- 12 valid layers ---------------------------------------------------------
for i in $(seq 0 11); do
  n=$(printf %02d "$i")
  run "$i" "h$n"; rc=$?
  if [ "$rc" -ne 0 ] || [ ! -f "$STAGE/data/block_response_T8.bin" ]; then
    echo "FAIL h$n: orchestrator exit $rc, see $STAGE/orch_h$n.txt"; fail=1
    # Move any partial/stale response aside so no later step can read it.
    [ -f "$STAGE/data/block_response_T8.bin" ] && mv "$STAGE/data/block_response_T8.bin" "$STAGE/failed_h$n.bin"
    continue
  fi
  mv "$STAGE/data/block_response_T8.bin" "$STAGE/resp_h$n.bin"
  # The packet must have gone out with layer N, and the reply must echo N.
  grep -q "layer $i, " "$STAGE/orch_h$n.txt" || { echo "FAIL h$n: sent layer_index != $i"; fail=1; }
  grep -q "Received response for layer $i from" "$STAGE/orch_h$n.txt" || { echo "FAIL h$n: reply layer_index != $i"; fail=1; }
  "$PY" scripts/g3_compare.py "$STAGE/resp_h$n.bin" "data/g3_expected_h${n}_T8.bin" "$TOL" "$i" || fail=1
done

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

if [ "$fail" -eq 0 ]; then echo "G3_CHECK_PASS"; else echo "G3_CHECK_FAIL"; exit 1; fi
