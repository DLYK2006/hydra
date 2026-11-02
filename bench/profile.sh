#!/usr/bin/env bash
# One profiled run, AFTER the timed matrix (sampling perturbs timing; these runs are never
# summarized). Run from repo root on WSL2 against the already-gated worker PID.
# Usage: bash bench/profile.sh BATCH WORKER_PID LABEL on|off N_DECODE [--print-cmd]
#   e.g. ... off512 off 2   (uncached decode step at context 513, ~32 s, contains the 10 s sample)
#        ... on512  on 24   (~22 s of cached decode, the 10 s sample covers ~11 tokens)
#   --print-cmd: print the exact remote sample and readback commands and exit (nothing runs).
# Starts bench_run.py (prompt_L512) in the background, waits for "STEP 1 START" (first decode
# step), then runs macOS `sample` for 10 s into live_logs/sample_<LABEL>_<RUN>.log inside the
# Mac folder (a .log, so it is read back with the approved `cat live_logs/<name>.log`), waits for
# the client, copies the report to bench/raw/BATCH/profile/ and prints the top 5 by self time.
set -u
PY=/mnt/c/Users/ASUS/AppData/Local/Programs/Python/Python313/python.exe
M=yonghowlee@10.0.0.62
BATCH=${1:?}; PID=${2:?}; LABEL=${3:?}; MODE=${4:?}; N=${5:?}
OUT=bench/raw/$BATCH/profile; RUN=$(date -u +%Y%m%dT%H%MZ)
# The Mac folder (no "hydra" subdir), expanded by the REMOTE shell. One definition for the sample
# write and the readback: the same folder the approved launch and cat commands cd into.
MDIR='"$HOME/Documents/DLYK stuff"'
MLOG="live_logs/sample_${LABEL}_$RUN.log"
SAMPLE_CMD="cd $MDIR && sample $PID 10 -file $MDIR/$MLOG"
READ_CMD="cd $MDIR && cat $MLOG"
if [ "${6:-}" = --print-cmd ]; then printf '%s\n%s\n' "$SAMPLE_CMD" "$READ_CMD"; exit 0; fi
[ -n "${6:-}" ] && { echo "ERROR: unknown 6th argument '$6'"; exit 1; }
compgen -G "$OUT/$LABEL*" > /dev/null && { echo "ERROR: $OUT/$LABEL* exists; refusing to overwrite"; exit 1; }
mkdir -p "$OUT"
"$PY" bench/bench_run.py --prompt-file bench/prompts/prompt_L512.txt --kv-cache "$MODE" --n-decode "$N" \
    --run-id "prof_$LABEL" --out "$OUT/$LABEL" > "$OUT/$LABEL.client.log" 2>&1 < /dev/null &
CLIENT=$!
until grep -q "STEP 1 START" "$OUT/$LABEL.client.log" 2>/dev/null; do
  kill -0 $CLIENT 2>/dev/null || { echo "ERROR: client exited before decode; see $OUT/$LABEL.client.log"; exit 1; }
  sleep 0.2
done
ssh -o BatchMode=yes $M "$SAMPLE_CMD" < /dev/null > "$OUT/$LABEL.sample_cmd.txt" 2>&1 \
  || { echo "ERROR: sample failed; see $OUT/$LABEL.sample_cmd.txt"; exit 1; }
wait $CLIENT || { echo "ERROR: profiled client run failed; see $OUT/$LABEL.client.log"; exit 1; }
ssh -o BatchMode=yes $M "$READ_CMD" < /dev/null > "$OUT/$LABEL.sample.log" 2>/dev/null \
  || { echo "ERROR: cannot read $MLOG"; exit 1; }
echo "== $LABEL: top 5 by self time (sample's 'Sort by top of stack' section) from $OUT/$LABEL.sample.log"
awk '/^Sort by top of stack/{f=1;next} f&&NF' "$OUT/$LABEL.sample.log" | head -5
